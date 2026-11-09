"""The board library on disk: built-in presets plus boards trained in browser.

A board is more than a checkpoint.  It owns the physical outline, the window
that maps a raw sensor frame onto its grid, the low-emissivity mask the network
was trained to fill, a structure signature used to warn when an upload looks
like a different board, and the pair of sample fields the library card shows.
All of that lives in one directory so a board can be copied or deleted whole.

Built-in presets point at the checkpoints this repository already trained.  The
checkpoint is copied into the board directory: ``outputs/`` is disposable, and
a board that stops working when a unrelated experiment directory is cleaned is
not a board.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tdp.web.sessions import (
    DEFAULT_PIXEL_SIZE_MM,
    build_session,
    detect_unknown_mask,
    gradient_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD_ROOT = REPO_ROOT / "outputs" / "web_boards"

# The presets are the two boards this repository actually measured.  Both
# entries name the window recorded by Stage 1; re-detecting it on upload would
# move case10 several pixels, because its ports confuse the profile detector.
PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "raspberry-pi-4b",
        "name": "Raspberry Pi 4B",
        "checkpoint": "outputs/thermal/unet_case10/train_both/model_best.pt",
        "metrics": "outputs/thermal/unet_case10/train_both/train_metrics.json",
        "masks": "outputs/thermal/unet_case10/paired_masks.npz",
        "grid_hw": (157, 103),
        "board_size_mm": (85.0, 56.0),
        "window": {
            "center_rc": [112.93952641708474, 93.38042243802526],
            "size_hw": [163.44505411386945, 107.68144741619635],
            "angle_deg": 1.5,
        },
        "sample_dir": "web/samples/raspberry-pi-4b",
        "pair_count": 432,
        "note": (
            "case10 / case17 full load and case09 / case16 half load; trained on every pair. "
            "The reported error is fit error against the painted label, not held-out generalization."
        ),
    },
    {
        "id": "orange-pi-5-pro",
        "name": "Orange Pi 5 Pro",
        "checkpoint": "outputs/thermal/unet_opi5pro/train_both/model_best.pt",
        "metrics": "outputs/thermal/unet_opi5pro/train_both/train_metrics.json",
        "masks": "outputs/thermal/unet_opi5pro/pairs_both.npz",
        "grid_hw": (164, 103),
        "board_size_mm": (89.0, 56.0),
        "window": {
            "center_rc": [87.85556083243483, 109.08559689852547],
            "size_hw": [136.1172026495654, 85.64677919523217],
            "angle_deg": -0.6,
        },
        "sample_dir": "web/samples/orange-pi-5-pro/inference",
        "pair_count": 480,
        "note": (
            "case12 / case14 half load and case13 / case15 full load; trained on every pair. "
            "The reported error is fit error against the high-emissivity label."
        ),
    },
)

_ID_RE = re.compile(r"[^a-z0-9-]+")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _abs(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


abs_path = _abs


def slugify(name: str) -> str:
    """Filesystem-safe id; the display name is kept untouched."""
    slug = _ID_RE.sub("-", name.strip().lower()).strip("-")
    return slug[:32] or "board"


@dataclass
class Board:
    """One trained or preset board, with everything inference needs."""

    id: str
    name: str
    builtin: bool
    created_at: str
    grid_hw: tuple[int, int]
    board_size_mm: tuple[float, float] | None
    pixel_size_mm: float
    pixel_size_known: bool
    window: dict
    window_mode: str  # "fixed" or "detected"
    checkpoint: Path
    board_mask: np.ndarray
    unknown_mask: np.ndarray
    reference_signature: np.ndarray
    metrics: dict = field(default_factory=dict)
    pair_count: int = 0
    note: str = ""
    training: dict = field(default_factory=dict)
    sample_path: str | None = None
    preview: dict | None = None

    @property
    def directory(self) -> Path:
        return BOARD_ROOT / self.id

    @property
    def array_path(self) -> Path:
        return self.directory / "arrays.npz"

    @property
    def preview_path(self) -> Path:
        return self.directory / "preview.npz"

    @property
    def available(self) -> bool:
        return self.checkpoint.exists() and self.array_path.exists()

    def summary(self) -> dict:
        """Metadata the browser needs to draw the picker and library card."""
        return {
            "id": self.id,
            "name": self.name,
            "builtin": self.builtin,
            "created_at": self.created_at,
            "grid_hw": [int(self.grid_hw[0]), int(self.grid_hw[1])],
            "board_size_mm": None if self.board_size_mm is None else [float(v) for v in self.board_size_mm],
            "pixel_size_mm": float(self.pixel_size_mm),
            "pixel_size_known": bool(self.pixel_size_known),
            "pair_count": int(self.pair_count),
            "metrics": self.metrics,
            "note": self.note,
            "training": self.training,
            "available": self.available,
            "has_preview": self.preview_path.exists(),
        }

    def public_dict(self) -> dict:
        return {
            **self.summary(),
            "window": self.window,
            "window_mode": self.window_mode,
            "unknown_fraction": float(self.unknown_mask[self.board_mask].mean())
            if self.board_mask.any() else 0.0,
        }


def save_board(board: Board) -> Board:
    """Write metadata and arrays; the checkpoint is expected to be in place."""
    board.directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        board.array_path,
        board_mask=np.asarray(board.board_mask, dtype=bool),
        unknown_mask=np.asarray(board.unknown_mask, dtype=bool),
        reference_signature=np.asarray(board.reference_signature, dtype=np.float32),
    )
    (board.directory / "board.json").write_text(
        json.dumps(
            {
                "id": board.id,
                "name": board.name,
                "builtin": board.builtin,
                "created_at": board.created_at,
                "grid_hw": [int(board.grid_hw[0]), int(board.grid_hw[1])],
                "board_size_mm": None if board.board_size_mm is None else [float(v) for v in board.board_size_mm],
                "pixel_size_mm": float(board.pixel_size_mm),
                "pixel_size_known": bool(board.pixel_size_known),
                "window": board.window,
                "window_mode": board.window_mode,
                "checkpoint": _rel(board.checkpoint),
                "metrics": board.metrics,
                "pair_count": int(board.pair_count),
                "note": board.note,
                "training": board.training,
                "sample_path": board.sample_path,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return board


def load_board(directory: Path) -> Board:
    meta = json.loads((directory / "board.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz") as arrays:
        board_mask = np.asarray(arrays["board_mask"], dtype=bool)
        unknown_mask = np.asarray(arrays["unknown_mask"], dtype=bool)
        signature = np.asarray(arrays["reference_signature"], dtype=np.float32)
    preview = None
    preview_path = directory / "preview.npz"
    if preview_path.exists():
        with np.load(preview_path) as data:
            preview = {key: np.array(data[key]) for key in data.files}
    return Board(
        id=str(meta["id"]),
        name=str(meta["name"]),
        builtin=bool(meta.get("builtin", False)),
        created_at=str(meta.get("created_at", "")),
        grid_hw=(int(meta["grid_hw"][0]), int(meta["grid_hw"][1])),
        board_size_mm=(
            None if meta.get("board_size_mm") is None
            else (float(meta["board_size_mm"][0]), float(meta["board_size_mm"][1]))
        ),
        pixel_size_mm=float(meta.get("pixel_size_mm", DEFAULT_PIXEL_SIZE_MM)),
        pixel_size_known=bool(meta.get("pixel_size_known", meta.get("board_size_mm") is not None)),
        window=dict(meta["window"]),
        window_mode=str(meta.get("window_mode", "fixed")),
        checkpoint=_abs(meta["checkpoint"]),
        board_mask=board_mask,
        unknown_mask=unknown_mask,
        reference_signature=signature,
        metrics=dict(meta.get("metrics", {})),
        pair_count=int(meta.get("pair_count", 0)),
        note=str(meta.get("note", "")),
        training=dict(meta.get("training", {})),
        sample_path=meta.get("sample_path"),
        preview=preview,
    )


def list_boards() -> list[Board]:
    """Every available board, presets first."""
    if not BOARD_ROOT.exists():
        return []
    boards: list[Board] = []
    for directory in sorted(BOARD_ROOT.iterdir()):
        if not (directory / "board.json").exists():
            continue
        try:
            board = load_board(directory)
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
        if board.available:
            boards.append(board)
    order = {preset["id"]: index for index, preset in enumerate(PRESETS)}
    boards.sort(key=lambda item: (order.get(item.id, len(order)), item.builtin is False, item.created_at))
    return boards


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _board_directory(board_id: str) -> Path:
    """Resolve one id under the library root, or refuse it.

    The id reaches this function from a URL.  Joining it directly would let
    ``..`` address anything on disk, so the id must be a plain single segment
    and the resolved directory must stay inside the library.
    """
    if not _SAFE_ID.fullmatch(board_id) or board_id in {".", ".."}:
        raise KeyError(board_id)
    root = BOARD_ROOT.resolve()
    directory = (root / board_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise KeyError(board_id) from exc
    return directory


def get_board(board_id: str) -> Board:
    directory = _board_directory(board_id)
    if not (directory / "board.json").exists():
        raise KeyError(board_id)
    board = load_board(directory)
    if not board.available:
        raise FileNotFoundError(f"board {board_id!r} has no usable checkpoint or arrays")
    return board


def delete_board(board_id: str) -> None:
    """Remove one trained board.  Presets are protected."""
    directory = _board_directory(board_id)
    if not (directory / "board.json").exists():
        raise KeyError(board_id)
    board = load_board(directory)
    if board.builtin:
        raise PermissionError("built-in boards cannot be deleted")
    shutil.rmtree(directory)


def _first_sample(sample_dir: str | None) -> Path | None:
    if not sample_dir:
        return None
    folder = _abs(sample_dir)
    if not folder.exists():
        return None
    candidates = (
        sorted(folder.glob("*.csv"))
        or sorted(folder.glob("**/*.csv"))
        or sorted(folder.glob("*.jpeg"))
        or sorted(folder.glob("**/*.jpeg"))
        or sorted(folder.glob("**/*.jpg"))
    )
    return candidates[0] if candidates else None


def _board_preview(board: Board, sample: Path) -> tuple[dict, np.ndarray]:
    """Run the board's own operator on one sample frame for the library card."""
    from tdp.web.inference import predict_frame  # local import breaks the cycle

    result = predict_frame(board, sample.name, sample.read_bytes())
    preview = {
        "raw": np.asarray(result["raw"], dtype=np.float32),
        "predicted": np.asarray(result["predicted"], dtype=np.float32),
        "ambient_c": np.float32(result["ambient_c"]),
        "board_mask": np.asarray(result["board_mask"], dtype=bool),
        "unknown_mask": np.asarray(result["unknown_mask"], dtype=bool),
        "signature_score": np.float32(result["signature_score"]),
    }
    board.reference_signature = gradient_signature(result["raw"])
    np.savez_compressed(
        board.preview_path,
        **{key: np.asarray(value) for key, value in preview.items()},
    )
    board.preview = preview
    return preview, board.reference_signature


def ensure_presets() -> list[dict]:
    """Create the built-in board records from this repository's checkpoints.

    This is idempotent.  A missing source checkpoint is reported, not fatal:
    the static frontend still works, and the API simply lists fewer boards.
    """
    BOARD_ROOT.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    for preset in PRESETS:
        directory = BOARD_ROOT / preset["id"]
        checkpoint = _abs(preset["checkpoint"])
        if not checkpoint.exists():
            events.append({"board": preset["id"], "status": "missing-checkpoint", "path": str(checkpoint)})
            continue
        if (directory / "board.json").exists() and (directory / "arrays.npz").exists() \
                and (directory / "model.pt").exists():
            events.append({"board": preset["id"], "status": "present"})
            continue

        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, directory / "model.pt")
        masks_path = _abs(preset["masks"])
        try:
            with np.load(masks_path) as source:
                board_mask = np.asarray(source["board_mask"], dtype=bool)
                unknown_mask = np.asarray(source["unknown_mask"], dtype=bool)
            grid_hw = tuple(int(v) for v in preset["grid_hw"])
            if board_mask.shape != grid_hw or unknown_mask.shape != grid_hw:
                raise ValueError(
                    f"{preset['id']} masks {board_mask.shape} do not match grid {grid_hw}"
                )
        except (OSError, KeyError, ValueError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            events.append({"board": preset["id"], "status": "failed", "error": str(exc)})
            continue
        signature = np.zeros(grid_hw, dtype=np.float32)
        metrics: dict = {}
        metrics_path = _abs(preset["metrics"])
        if metrics_path.exists():
            try:
                raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            metrics = {
                "final": raw.get("final", {}),
                "best": raw.get("best", {}),
                "config": {
                    key: raw.get("config", {}).get(key)
                    for key in ("n_frames", "epochs", "loss", "n_parameters", "theta_scale_k")
                },
                "basis": "training-set fit against the high-emissivity label",
            }
        sample = _first_sample(preset["sample_dir"])
        board = Board(
            id=str(preset["id"]),
            name=str(preset["name"]),
            builtin=True,
            created_at=datetime.fromtimestamp(
                (directory / "model.pt").stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            grid_hw=grid_hw,
            board_size_mm=tuple(float(v) for v in preset["board_size_mm"]),
            pixel_size_mm=DEFAULT_PIXEL_SIZE_MM,
            pixel_size_known=True,
            window={**preset["window"], "grid_hw": list(grid_hw), "source": "recorded"},
            window_mode="fixed",
            checkpoint=directory / "model.pt",
            board_mask=board_mask,
            unknown_mask=unknown_mask,
            reference_signature=signature,
            metrics=metrics,
            pair_count=int(preset["pair_count"]),
            note=str(preset["note"]),
            sample_path=None if sample is None else str(_rel(sample)),
        )
        save_board(board)
        if sample is not None:
            try:
                _board_preview(board, sample)
                save_board(board)
                events.append({"board": preset["id"], "status": "created-with-preview"})
            except Exception as exc:  # a preview must not stop the server
                events.append({"board": preset["id"], "status": "created-without-preview", "error": str(exc)})
        else:
            events.append({"board": preset["id"], "status": "created-without-preview"})
    return events


def next_board_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(name)}-{stamp}"
    candidate = base
    suffix = 1
    while (BOARD_ROOT / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def build_trained_board(
    *,
    board_id: str,
    name: str,
    grid_hw: tuple[int, int],
    board_size_mm: tuple[float, float] | None,
    window: dict,
    board_mask: np.ndarray,
    unknown_mask: np.ndarray,
    checkpoint: Path,
    metrics: dict,
    pair_count: int,
    training: dict,
    preview: dict | None,
    note: str = "",
    sample_path: str | None = None,
) -> Board:
    """Assemble and persist a board trained from browser uploads."""
    reference = np.zeros(grid_hw, dtype=np.float32)
    if preview is not None:
        reference = gradient_signature(preview["raw"])
    board = Board(
        id=board_id,
        name=name,
        builtin=False,
        created_at=_now(),
        grid_hw=(int(grid_hw[0]), int(grid_hw[1])),
        board_size_mm=board_size_mm,
        pixel_size_mm=DEFAULT_PIXEL_SIZE_MM,
        pixel_size_known=board_size_mm is not None,
        window=dict(window),
        window_mode="detected",
        checkpoint=checkpoint,
        board_mask=np.asarray(board_mask, dtype=bool),
        unknown_mask=np.asarray(unknown_mask, dtype=bool),
        reference_signature=reference,
        metrics=metrics,
        pair_count=int(pair_count),
        note=note,
        training=training,
        sample_path=sample_path,
    )
    board.directory.mkdir(parents=True, exist_ok=True)
    save_board(board)
    if preview is not None:
        np.savez_compressed(
            board.preview_path,
            **{key: np.asarray(value) for key, value in preview.items()},
        )
        board.preview = preview
    return board


def build_board_from_upload(
    *,
    name: str,
    bare_files: Iterable[tuple[str, bytes]],
    painted_files: Iterable[tuple[str, bytes]],
    board_size_mm: tuple[float, float] | None,
    epochs: int,
    report,
    device: str = "auto",
) -> Board:
    """Preprocess, pair, train and register one new board.

    Imported lazily by the server so a plain library import does not pull in
    torch.  The work itself lives in :mod:`tdp.web.training`.
    """
    from tdp.web.training import train_board

    return train_board(
        name=name,
        bare_files=bare_files,
        painted_files=painted_files,
        board_size_mm=board_size_mm,
        epochs=epochs,
        report=report,
        device=device,
    )
