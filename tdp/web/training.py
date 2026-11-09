"""Train one new board from a browser upload.

The flow mirrors the desktop U-Net experiment: a bare recording is the input, a
high-emissivity (painted or taped) recording of the same board is the label, and
the residual correction net learns only the low-emissivity correction.  What is
new is that nothing about the board is assumed beforehand:

1. both uploads are parsed at whatever sensor resolution they carry;
2. each session's board is located and rectified to the shared grid chosen
   from the bare session;
3. the low-emissivity footprint is detected on the bare reference frame;
4. frames are paired by trusted-pixel thermal state, never by frame index;
5. the network trains on every pair.

There is no held-out session here.  The server therefore reports training-set
fit error against the high-emissivity label and says so in the board record.
A user who wants an independent check must upload a third session.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from tdp.unet.train import resolve_device, train_arrays
from tdp.web.inference import apply_operator, load_operator
from tdp.web.registry import (
    BOARD_ROOT,
    Board,
    build_trained_board,
    next_board_id,
)
from tdp.web.sessions import (
    RectifiedSession,
    UploadError,
    detect_unknown_mask,
    detect_window,
    fit_window_to_frame,
    grid_for_board,
    load_uploads,
    rectify_session,
    reference_frame_index,
    register_window,
    state_pairs,
)

# (key, Chinese label, share of total progress).  The browser maps keys to its
# own step labels; the shares only decide how fast the ring moves.
STAGES: tuple[tuple[str, str, float], ...] = (
    ("read", "读取上传数据", 0.07),
    ("window", "定位板体", 0.05),
    ("rectify", "配准到统一网格", 0.07),
    ("mask", "识别不可信区域", 0.04),
    ("pair", "按热状态配对", 0.07),
    ("train", "训练校正模型", 0.58),
    ("evaluate", "评估训练误差", 0.07),
    ("save", "保存板型", 0.05),
)


def stage_labels() -> list[dict]:
    return [{"key": key, "label": label} for key, label, _ in STAGES]


class _Progress:
    """Map a stage and its local fraction onto one global percentage."""

    def __init__(self, report: Callable[..., None]) -> None:
        self._report = report
        self._starts: dict[str, tuple[float, float]] = {}
        cursor = 0.0
        for key, _label, share in STAGES:
            self._starts[key] = (cursor, share)
            cursor += share

    def __call__(
        self, stage: str, message: str, fraction: float = 1.0, metrics: dict | None = None
    ) -> None:
        start, share = self._starts[stage]
        overall = min(100.0, 100.0 * (start + share * max(0.0, min(1.0, float(fraction)))))
        self._report(stage, message, overall, metrics)


def _best_preview_pair(pairs: dict) -> int:
    """The last painted frame, which is normally the hot end of the run."""
    return int(len(pairs["x_theta"]) - 1)


def _preview(
    pairs: dict,
    checkpoint: Path,
    board_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> dict:
    index = _best_preview_pair(pairs)
    model = load_operator(checkpoint, device="cpu")
    raw = np.asarray(pairs["x_theta"][index], dtype=np.float64)
    predicted = apply_operator(model, raw, board_mask, unknown_mask, device="cpu")
    return {
        "raw": raw.astype(np.float32),
        "predicted": predicted.astype(np.float32),
        "ambient_c": np.float32(pairs["bare_ambient_c"][index]),
        "board_mask": np.asarray(board_mask, dtype=bool),
        "unknown_mask": np.asarray(unknown_mask, dtype=bool),
        "signature_score": np.float32(1.0),
    }


def _cleanup(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


def _manual_window(
    window: dict,
    frames: Sequence,
    board_size_mm: tuple[float, float] | None,
) -> dict:
    """Turn an explicit window into the same record the detector produces.

    An automatic edge fit can stop at a thermal discontinuity inside the board.
    On the VisionFive 2 the right-hand connector bank is low-emissivity, so the
    bare session measured 107 px wide where the painted session measured 149.
    The caller may then pass ``center_rc`` / ``size_hw`` / ``angle_deg`` from a
    frame where the outline is visible.  The painted session is still registered
    to this window, so the two sessions are paired on the same physical area.
    """
    try:
        center = [float(window["center_rc"][0]), float(window["center_rc"][1])]
        size = [float(window["size_hw"][0]), float(window["size_hw"][1])]
        angle = float(window.get("angle_deg", 0.0))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise UploadError("窗口格式应为 center_rc、size_hw、angle_deg") from exc
    if not all(np.isfinite(center + size + [angle])):
        raise UploadError("窗口数值必须是有限数")
    if min(size) < 8.0:
        raise UploadError(f"窗口尺寸过小：{size[0]:.1f} x {size[1]:.1f} px")
    index = int(reference_frame_index(frames))
    center, size = fit_window_to_frame(center, size, angle, frames[index].shape_hw)
    rows, cols = grid_for_board(board_size_mm, (size[0], size[1]))
    return {
        "center_rc": [float(center[0]), float(center[1])],
        "size_hw": [float(size[0]), float(size[1])],
        "angle_deg": float(angle),
        "measured_size_hw": [float(size[0]), float(size[1])],
        "edge_source": "manual",
        "aspect_residual": None,
        "aspect_forced": board_size_mm is not None,
        "grid_hw": [int(rows), int(cols)],
        "reference_index": index,
        "source": "manual",
    }


def train_board(
    *,
    name: str,
    bare_files: Iterable[tuple[str, bytes]],
    painted_files: Iterable[tuple[str, bytes]],
    board_size_mm: tuple[float, float] | None,
    epochs: int,
    report: Callable[..., None],
    device: str = "auto",
    window: dict | None = None,
) -> Board:
    """Preprocess, pair, train, and register one uploaded board."""
    name = name.strip()
    if not name:
        raise UploadError("请填写板型名称")
    board_size_mm = None if board_size_mm is None else tuple(float(v) for v in board_size_mm)
    if board_size_mm is not None and (len(board_size_mm) != 2 or min(board_size_mm) <= 0):
        raise UploadError("板体尺寸需要两个正数（长边、短边，单位 mm）")
    epochs = int(epochs)
    if not 1 <= epochs <= 2000:
        raise UploadError("训练轮次需要介于 1 和 2000 之间")

    bare_files = list(bare_files)
    painted_files = list(painted_files)
    if not bare_files or not painted_files:
        raise UploadError("需要同时提供裸板与黑化板 CSV 或 JPEG")

    progress = _Progress(report)
    resolved = resolve_device(device)

    # ---- read -----------------------------------------------------------
    progress("read", "读取裸板数据", 0.0)
    bare_frames, bare_failures = load_uploads(bare_files)
    progress("read", "读取黑化板数据", 0.5)
    painted_frames, painted_failures = load_uploads(painted_files)
    if len(bare_frames) < 4 or len(painted_frames) < 4:
        raise UploadError("每组数据至少需要 4 帧；当前裸板 "
                          f"{len(bare_frames)} 帧、黑化板 {len(painted_frames)} 帧")
    progress(
        "read",
        f"裸板 {len(bare_frames)} 帧，黑化板 {len(painted_frames)} 帧"
        + (f"；跳过 {len(bare_failures) + len(painted_failures)} 个无法解析的文件"
           if bare_failures or painted_failures else ""),
        1.0,
    )

    # ---- window ---------------------------------------------------------
    if window is not None:
        progress("window", "使用给定的板体窗口", 0.0)
        bare_window = _manual_window(window, bare_frames, board_size_mm)
    else:
        progress("window", "在裸板参考帧上定位板体", 0.0)
        bare_window = detect_window(bare_frames, board_size_mm=board_size_mm)
    grid_hw = (int(bare_window["grid_hw"][0]), int(bare_window["grid_hw"][1]))
    progress("window", "在黑化板参考帧上定位板体", 0.6)
    painted_window = detect_window(painted_frames, board_size_mm=board_size_mm)
    progress(
        "window",
        f"网格 {grid_hw[0]} × {grid_hw[1]} px，裸板角度 {bare_window['angle_deg']:+.1f}°",
        1.0,
    )

    # ---- rectify --------------------------------------------------------
    progress("rectify", "把裸板帧配准到统一网格", 0.0)
    bare: RectifiedSession = rectify_session(bare_frames, bare_window, grid_hw)
    progress("rectify", "按结构对齐黑化板与裸板", 0.35)
    painted_reference = painted_frames[int(painted_window["reference_index"])]
    registered = register_window(
        bare.theta_k[bare.reference_index],
        bare_window,
        painted_reference,
        grid_hw,
        initial_window=painted_window,
    )
    registration_warnings: list[str] = []
    if registered["score"] < 0.25:
        registration_warnings.append(
            "裸板与黑化板的结构匹配度只有 "
            f"{registered['score']:.2f}；请确认两组数据拍的是同一块板"
        )
    progress(
        "rectify",
        f"黑化板窗口已对齐（结构匹配 {registered['score']:.2f}，"
        f"缩放 {registered['scale_from_initial']:.2f}）",
        0.75,
    )
    painted: RectifiedSession = rectify_session(painted_frames, registered, grid_hw)

    # ---- mask -----------------------------------------------------------
    progress("mask", "在裸板上识别低发射率区域", 0.0)
    board = bare.valid & painted.valid
    unknown = detect_unknown_mask(bare.theta_k[bare.reference_index], board)
    unknown_fraction = float(unknown[board].mean()) if board.any() else 0.0
    if unknown_fraction > 0.75:
        raise UploadError(f"不可信区域占板体 {unknown_fraction:.0%}，请检查裸板数据")
    progress("mask", f"不可信区域占板体 {unknown_fraction:.1%}", 1.0)

    # ---- pair -----------------------------------------------------------
    progress("pair", "按可信区温度水平匹配两段序列", 0.0)
    pairs = state_pairs(bare, painted, unknown_mask=unknown)
    diagnostics = pairs["diagnostics"]
    progress(
        "pair",
        f"{diagnostics['n_pairs']} 对；可信区配对残差 {diagnostics['trusted_mae_k']:.3f} K",
        1.0,
    )

    # ---- train ----------------------------------------------------------
    board_id = next_board_id(name)
    directory = BOARD_ROOT / board_id
    directory.mkdir(parents=True, exist_ok=True)
    report_every = max(1, epochs // 12)
    history: list[dict] = []
    if resolved == "cpu" and diagnostics["n_pairs"] > 60:
        progress(
            "train",
            f"当前只有 CPU；{epochs} 轮、{diagnostics['n_pairs']} 对会比较慢。"
            "有 MPS 或 CUDA 时服务会自动使用。",
            0.0,
        )

    def on_epoch(epoch: int, total: int, metrics: dict | None) -> None:
        if metrics is not None:
            history.append({**metrics, "epoch": int(epoch)})
            message = (
                f"第 {epoch}/{total} 轮 · 板体 MAE {metrics['board_mae_k']:.3f} K"
                f" · 不可信区 MAE {metrics['masked_mae_k']:.3f} K"
            )
        else:
            message = f"第 {epoch}/{total} 轮"
        progress("train", message, epoch / total, None if metrics is None else {**metrics, "epoch": epoch})

    try:
        result = train_arrays(
            pairs["x_theta"],
            pairs["y_theta"],
            pairs["board_mask"],
            pairs["unknown_mask"],
            directory,
            epochs=epochs,
            device=resolved,
            report_every=report_every,
            progress=on_epoch,
            save_predictions=False,
            extra_config={"source": "browser upload"},
        )
        final = result["final"]
        best = result["best"]
        progress(
            "evaluate",
            f"训练集板体 MAE {final['board_mae_k']:.3f} K，不可信区 MAE "
            f"{final['masked_mae_k']:.3f} K",
            1.0,
        )
        # The trainer saves the best epoch; make the stable name the board uses.
        shutil.move(str(directory / "model_best.pt"), str(directory / "model.pt"))
        preview = _preview(
            pairs, directory / "model.pt", pairs["board_mask"], pairs["unknown_mask"]
        )
        progress("save", "写入板型与预览", 0.5)
        metrics = {
            "final": final,
            "best": best,
            "config": result["config"],
            "basis": "training-set fit against the high-emissivity label; no held-out session",
            "pairing": diagnostics,
            "label": "训练集拟合误差（非留出泛化）",
        }
        training = {
            "history": history,
            "epochs": epochs,
            "device": resolved,
            "n_pairs": diagnostics["n_pairs"],
            "n_bare_frames": bare.n_frames,
            "n_painted_frames": painted.n_frames,
            "unknown_fraction_of_board": unknown_fraction,
            "bare_failures": bare_failures,
            "painted_failures": painted_failures,
            "warnings": list(bare.warnings) + list(painted.warnings) + registration_warnings,
            "window": bare_window,
            "note": (
                "上传数据训练；报告的是训练集误差。留出泛化需要另一次独立会话，"
                "本服务没有把它冒充成验证误差。"
            ),
        }
        board = build_trained_board(
            board_id=board_id,
            name=name,
            grid_hw=grid_hw,
            board_size_mm=board_size_mm,
            window=bare_window,
            board_mask=pairs["board_mask"],
            unknown_mask=pairs["unknown_mask"],
            checkpoint=directory / "model.pt",
            metrics=metrics,
            pair_count=diagnostics["n_pairs"],
            training=training,
            preview=preview,
            note=(
                f"{name}：浏览器上传训练。上报误差为训练集拟合误差，不是留出泛化。"
            ),
        )
    except Exception:
        _cleanup(directory)
        raise
    progress("save", "板型已加入板型库", 1.0)
    return board
