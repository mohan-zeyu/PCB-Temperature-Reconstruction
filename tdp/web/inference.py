"""Apply a saved board operator to one arbitrary HIKMICRO CSV or JPEG.

This is real model inference, not a lookup: the upload is parsed, the board is
located, the frame is rectified through the board's own window, and the residual
correction net predicts a correction field.  The returned temperature is

    predicted_surface = bare_reading + predicted_correction

in kelvin above that frame's own ambient.  The correction was learned from a
high-emissivity recording of the same board, so it removes the low-emissivity
suppression in the areas the board's mask marks.  It is still an estimate: the
model reports a structure-match score, and the caller must show it.

The code deliberately does not fall back to "nearest saved frame".  If a board
is missing, that is an error the user must see.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tdp.io import parse_thermal_upload
from tdp.data.crop import estimate_ambient, rectify_quad, rotated_rect_corners
from tdp.unet.model import build_residual_unet
from tdp.unet.train import THETA_SCALE_K
from tdp.web.sessions import (
    UploadError,
    UploadedFrame,
    detect_window,
    fit_window_to_frame,
    register_window,
    signature_score,
)

_MODEL_CACHE: dict[tuple[str, int, str], torch.nn.Module] = {}
_MODEL_LOCK = threading.Lock()

MIN_STRUCTURE_SCORE = 0.30


def load_operator(checkpoint: Path | str, device: str = "cpu") -> torch.nn.Module:
    """Load one board checkpoint and keep it warm for the next request."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"board checkpoint missing: {path}")
    key = (str(path.resolve()), path.stat().st_mtime_ns, str(device))
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = build_residual_unet(in_channels=3)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        model.to(torch.device(device))
        _MODEL_CACHE[key] = model
        return model


def apply_operator(
    model: torch.nn.Module,
    theta_k: np.ndarray,
    board_mask: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """One correction field for one frame, in kelvin."""
    theta = np.asarray(theta_k, dtype=np.float64)
    board = np.asarray(board_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    x = np.where(board, theta, 0.0)
    static = np.stack([board.astype(np.float32), unknown.astype(np.float32)])[None]
    with torch.no_grad():
        channel = torch.tensor(
            (x / THETA_SCALE_K)[None, None], dtype=torch.float32, device=device
        )
        correction = model(
            torch.cat([channel, torch.tensor(static, device=device)], dim=1)
        )[0, 0].cpu().numpy().astype(np.float64)
    return (x / THETA_SCALE_K + correction) * THETA_SCALE_K


def resolve_window(board, frame: UploadedFrame) -> tuple[dict, list[str]]:
    """Use the saved window, or re-detect for a board trained that way."""
    warnings: list[str] = []
    if getattr(board, "window_mode", "fixed") == "fixed":
        center, size = fit_window_to_frame(
            board.window["center_rc"], board.window["size_hw"],
            float(board.window["angle_deg"]), frame.shape_hw,
        )
        return {
            "center_rc": [center[0], center[1]],
            "size_hw": [size[0], size[1]],
            "angle_deg": float(board.window["angle_deg"]),
            "source": "recorded",
        }, warnings
    if getattr(board, "preview", None) is not None and board.window is not None:
        registered = register_window(
            board.preview["raw"],
            board.window,
            frame,
            board.grid_hw,
            initial_window=board.window,
        )
        if registered["score"] < 0.30:
            warnings.append(
                "此帧与板型的结构匹配度只有 "
                f"{registered['score']:.2f}；结果可能不可靠"
            )
        return registered, warnings
    try:
        detected = detect_window([frame], board_size_mm=board.board_size_mm)
        return {
            "center_rc": detected["center_rc"],
            "size_hw": detected["size_hw"],
            "angle_deg": detected["angle_deg"],
            "source": "detected",
        }, warnings
    except (ValueError, UploadError, FloatingPointError) as exc:
        warnings.append(f"自动定位失败，使用训练时窗口：{exc}")
        center, size = fit_window_to_frame(
            board.window["center_rc"], board.window["size_hw"],
            float(board.window["angle_deg"]), frame.shape_hw,
        )
        return {
            "center_rc": [center[0], center[1]],
            "size_hw": [size[0], size[1]],
            "angle_deg": float(board.window["angle_deg"]),
            "source": "recorded-fallback",
        }, warnings


@dataclass
class OrientedCrop:
    """One frame rectified in the rotation that matches the board best."""

    quarter_turns: int  # counter-clockwise, 0..3
    window: dict
    theta: np.ndarray
    valid: np.ndarray
    signature: float | None
    warnings: list[str]


def _best_orientation(board, frame: UploadedFrame, ambient: float) -> OrientedCrop:
    """Try the four 90-degree orientations and keep the best structure match.

    A saved board window and the residual net both live in the orientation the
    board was trained in.  A frame held sideways or upside down still contains
    the same board, but the fixed window and the net cannot be rotated by a
    small angle search: 90, 180 and 270 degrees are different discrete cases.
    Rotating the frame before rectification is cheap, so all four are scored and
    the highest structure match wins.  Without a stored reference signature
    there is nothing to score, and the frame is used as uploaded.
    """
    reference = np.asarray(board.reference_signature)
    has_reference = reference.size > 0 and float(np.abs(reference).max()) > 0.0
    best: OrientedCrop | None = None
    best_score = -np.inf
    for turns in range(4):
        temps = np.ascontiguousarray(np.rot90(frame.temps, turns)) if turns else frame.temps
        candidate = UploadedFrame(
            name=frame.name,
            digest=frame.digest,
            timestamp=frame.timestamp,
            temps=temps,
            ambient_c=ambient,
        )
        try:
            window, warnings = resolve_window(board, candidate)
            corners = rotated_rect_corners(
                window["center_rc"], window["size_hw"], window["angle_deg"]
            )
            crop = rectify_quad(temps, corners, tuple(board.grid_hw))
        except (UploadError, ValueError, FloatingPointError):
            continue
        theta = np.asarray(crop, dtype=np.float64) - ambient
        valid = np.isfinite(theta)
        if not valid.any():
            continue
        theta = np.where(valid, theta, 0.0)
        signature = None
        if has_reference:
            signature = max(-1.0, min(1.0, signature_score(reference, theta)))
        score = signature if signature is not None else 0.0
        if best is None or score > best_score:
            best = OrientedCrop(turns, window, theta, valid, signature, list(warnings))
            best_score = score
    if best is None:
        raise UploadError("配准后的板体没有任何有效像素")
    return best


def predict_frame(
    board,
    name: str,
    data: bytes,
    *,
    device: str = "cpu",
) -> dict:
    """Parse one upload, rectify it, and apply the board's operator."""
    frame = parse_thermal_upload(data, name, expect_shape=None)
    ambient = estimate_ambient(frame.temps)
    uploaded = UploadedFrame(
        name=name,
        digest="",
        timestamp=frame.timestamp,
        temps=np.asarray(frame.temps, dtype=np.float32),
        ambient_c=ambient,
    )
    oriented = _best_orientation(board, uploaded, ambient)
    window = oriented.window
    theta = oriented.theta
    valid = oriented.valid
    signature = oriented.signature
    warnings = oriented.warnings
    if oriented.quarter_turns:
        note = f"此帧旋转 {90 * oriented.quarter_turns}° 后与板型一致"
        if signature is not None:
            note += f"（结构匹配 {signature:.2f}）"
        warnings.insert(0, note)
    if signature is not None and signature < MIN_STRUCTURE_SCORE:
        warnings.append(
            f"此帧与板型的结构匹配度仅 {signature:.2f}；结果可能不可靠"
        )
    if float(valid.mean()) < 0.98:
        warnings.append(f"板体网格仅 {valid.mean() * 100:.1f}% 有有效采样")
    if board.board_size_mm is None:
        warnings.append("板体物理尺寸未知；位置与长度只能按像素报告")

    model = load_operator(board.checkpoint, device=device)
    predicted = apply_operator(
        model, theta, board.board_mask, board.unknown_mask, device=device
    )
    predicted = np.where(valid, predicted, np.nan)
    return {
        "board_id": board.id,
        "source_name": name,
        "ambient_c": float(ambient),
        "raw": theta.astype(np.float32),
        "predicted": predicted.astype(np.float32),
        "board_mask": np.asarray(board.board_mask, dtype=bool),
        "unknown_mask": np.asarray(board.unknown_mask, dtype=bool),
        "valid_mask": valid,
        "grid_hw": [int(board.grid_hw[0]), int(board.grid_hw[1])],
        "window": window,
        "quarter_turns": int(oriented.quarter_turns),
        "signature_score": None if signature is None else float(signature),
        "warnings": warnings,
        "device": str(device),
        "model_note": (
            "predicted = bare reading + learned correction; the correction comes from "
            "paired high-emissivity recordings of this board and is an estimate, not a measurement"
        ),
    }
