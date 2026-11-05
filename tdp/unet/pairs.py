"""Build per-pixel supervised pairs between a bare-board and a painted-board run.

The two runs describe the same physical board at the same load, one before and
one after matte-black paint.  On the canonical board grid (same 85 x 56 mm
outline, same 157 x 103 pixels) a pixel is the same physical point in both
sessions.  The bare reading is suppressed on low-emissivity metal; the painted
reading is treated as the true temperature (the 0.95-emissivity definition of
truth the project uses).

A pair is one *powered* bare frame and the painted field for the same thermal
state:

* the state key is elapsed time since each session's own load onset, measured
  from the CPU-load logs in absolute time (never random frame indices);
* the painted field at that elapsed time is a per-pixel linear interpolation of
  the two bracketing painted powered frames (6 s cadence, 146 s response);
* where the bare session stays powered after the painted powered frames end,
  the target is the painted steady-mean field (mean of its last powered
  frames), exactly the steady clamp the SoC anchor calibration already applies.
  Such pairs are flagged, never silently mixed with interpolated pairs.

All temperatures are theta = raw field minus that frame's own ambient, so the
~2 K session-to-session ambient difference never leaks into the labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tdp.thermal.calibration import (
    longest_high_load_interval,
    read_timestamped_values,
    sequence_start_epoch,
)

def level_temperature_series(seq: dict, powered: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Median theta over trusted pixels per frame - the state key used to
    match sessions whose load history is not logged (case16)."""
    theta = theta_series(seq)
    return np.median(theta[powered][:, pixels], axis=1)


def state_matched_pairs(
    bare_theta: np.ndarray,
    painted_theta: np.ndarray,
    bare_powered: np.ndarray,
    painted_powered: np.ndarray,
    bare_level: np.ndarray,
    painted_level: np.ndarray,
    trusted: np.ndarray,
    masked: np.ndarray,
    *,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pair every painted powered frame with the bare frame whose trusted-pixel
    temperature level is closest.

    Direction choice: one painted frame per pair, so all of the painted
    session's trustworthy frames (including its stepped-load segments) become
    labels, and no painted frame is double-counted.  Matching by state rather
    than elapsed time is the only sound key for case16, which has no logs and a
    stepped load.  The trusted-pixel residual of each pair is returned so the
    pairing noise is measured, never assumed.
    """
    bare_level = np.asarray(bare_level, dtype=np.float64)
    painted_level = np.asarray(painted_level, dtype=np.float64)
    if bare_level.shape != (len(bare_powered),) or painted_level.shape != (len(painted_powered),):
        raise ValueError("level series must have one value per powered frame")
    bare_idx: list[int] = []
    painted_idx: list[int] = []
    for j, target in enumerate(painted_level):
        i = int(np.argmin(np.abs(bare_level - target)))
        bare_idx.append(i)
        painted_idx.append(j)
    bare_idx = np.asarray(bare_idx, dtype=np.int64)
    painted_idx = np.asarray(painted_idx, dtype=np.int64)
    x = bare_theta[bare_powered][bare_idx]
    y = painted_theta[painted_powered][painted_idx]
    residual = y - x
    diagnostics = {
        "label": label,
        "n_pairs": int(len(bare_idx)),
        "trusted_mae_k": float(np.abs(residual[:, trusted]).mean()),
        "trusted_bias_k": float(residual[:, trusted].mean()),
        "masked_mae_k": float(np.abs(residual[:, masked]).mean()),
        "masked_bias_k": float(residual[:, masked].mean()),
        "level_gap_k_p95": float(np.percentile(np.abs(bare_level[bare_idx] - painted_level), 95)),
    }
    return bare_idx, painted_idx, diagnostics


# case10's powered stretch is fixed by project decision (CLAUDE.md): frames
# 0-239, the whole powered recording.  Painted powered frames are cut where the
# reference's CPU log leaves its >=90 % plateau.
CASE10_POWERED_FRAMES = 240

# Last powered painted frames averaged for the steady target beyond painted
# coverage.  The painted run is near its settled state by then (response 146 s,
# powered > 5 response times).
STEADY_MEAN_LAST_FRAMES = 25

# Guard gap around every contiguous validation/test block (frames are ~6 s
# apart and near-duplicates; random splits are forbidden in this project).
GUARD_FRAMES = 5

# Loss/board-region convention: the shared, physically valid board pixels.
CONVENTIONS = {
    "theta": "temperature above that frame's ambient, kelvin",
    "pair_key": "elapsed seconds since each session's own >=90 % load onset",
    "painted_past_coverage": "steady-mean painted field (flagged, not extrapolated)",
    "split": "contiguous powered blocks separated by guard gaps; never random",
}


def load_sequence_arrays(sequence: str) -> dict:
    """Return the prepared export arrays for one sequence name."""
    base = Path("outputs/dynamic_inverse") / sequence / "latest"
    with np.load(base / "temperature_corrected_sequence.npz") as data:
        arrays = {k: data[k] for k in data.files}
    arrays["clock_start_epoch_s"] = sequence_start_epoch(base / "run_metadata.json")
    return arrays


def theta_series(arrays: dict) -> np.ndarray:
    """raw_temperature_c minus that frame's own ambient, in kelvin."""
    raw = np.asarray(arrays["raw_temperature_c"], dtype=np.float64)
    ambient = np.asarray(arrays["ambient_c"], dtype=np.float64)
    return raw - ambient[:, None, None]


def load_interval_from_measurements(
    sequence: str, measurements, threshold_percent: float = 90.0
) -> object:
    """Longest sustained load interval for a sequence, read from its CPU log.

    Full-load sessions reach >=90 %; half-load sessions plateau near 50 % and
    must be read with a lower threshold, otherwise the log never qualifies.
    """
    spec = measurements.sequences[sequence]
    log = read_timestamped_values(
        Path(measurements.dataset_root) / spec.logs["cpu_load"],
        value_columns=("cpu_usage_percent", "total_percent"),
    )
    return longest_high_load_interval(log, threshold_percent=threshold_percent)


def powered_frames_by_clock(
    arrays: dict, load_stop_epoch_s: float, tolerance_s: float = 1.0
) -> np.ndarray:
    """Powered frame indices: a contiguous prefix ending at the load stop."""
    epochs = arrays["clock_start_epoch_s"] + np.asarray(
        arrays["timestamps_s"], dtype=np.float64
    )
    last = int(np.searchsorted(epochs, load_stop_epoch_s + tolerance_s, side="right") - 1)
    indices = np.arange(last + 1)
    if last + 1 < len(epochs) and epochs[last + 1] <= load_stop_epoch_s:
        raise ValueError("powered prefix is not contiguous up to the load stop")
    return indices


def contiguous_split(
    n_frames: int,
    *,
    train: tuple[int, int] = (0, 135),
    val: tuple[int, int] = (140, 182),
    test: tuple[int, int] = (187, 240),
    guard: int = GUARD_FRAMES,
) -> dict[str, np.ndarray]:
    """Contiguous powered blocks separated by guard gaps (never random).

    Each block is [start, stop).  The caller must guarantee guard >= 1 between
    consecutive blocks; that is enforced here.
    """
    blocks = {"train": train, "val": val, "test": test}
    starts = [start for start, _ in blocks.values()]
    stops = [stop for _, stop in blocks.values()]
    if starts != sorted(starts) or any(a >= b for a, b in blocks.values()):
        raise ValueError("split blocks must be ordered, non-empty [start, stop) ranges")
    for earlier, later in zip(list(blocks.values())[:-1], list(blocks.values())[1:]):
        if later[0] - earlier[1] < guard:
            raise ValueError(
                f"guard gap must be >= {guard} frames between blocks; got "
                f"{earlier[1]} -> {later[0]}"
            )
    for _, stop in blocks.values():
        if stop > n_frames:
            raise ValueError(f"block ends past {n_frames} frames")
    return {name: np.arange(start, stop) for name, (start, stop) in blocks.items()}


def _interpolate_painted_fields(
    target_epochs: np.ndarray,
    painted_powered: np.ndarray,
    painted_epochs_powered: np.ndarray,
    painted_theta_powered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-pixel linear interpolation of painted theta at target epochs.

    Returns (fields, flags, nearest_source, first_over_index):
    - flags: 'interp' inside coverage, 'steady' beyond the last painted powered
      frame (steady-mean field), 'first' before the first (nearest frame).
    - first_over_index: first bare-frame index whose target lies beyond the
      painted powered coverage, or len(target_epochs) if none.
    """
    n = len(target_epochs)
    flags = np.empty(n, dtype="U8")
    fields = np.empty((n,) + painted_theta_powered.shape[1:], dtype=np.float64)
    positions = np.searchsorted(
        painted_epochs_powered, target_epochs, side="right"
    ).astype(np.int64)
    first_over = n
    steady_field = painted_theta_powered[-STEADY_MEAN_LAST_FRAMES:].mean(axis=0)

    for i, (target, pos) in enumerate(zip(target_epochs, positions)):
        if pos == 0:
            flags[i] = "first"
            fields[i] = painted_theta_powered[0]
            continue
        if pos >= len(painted_epochs_powered):
            if first_over == n:
                first_over = i
            flags[i] = "steady"
            fields[i] = steady_field
            continue
        left, right = int(pos - 1), int(pos)
        t_left = painted_epochs_powered[left]
        width = painted_epochs_powered[right] - t_left
        weight = (target - t_left) / width if width > 0.0 else 0.0
        flags[i] = "interp"
        fields[i] = (1.0 - weight) * painted_theta_powered[left] + weight * painted_theta_powered[right]
    return fields, flags, positions, first_over


def build_paired_archive(
    *,
    bare_arrays: dict,
    painted_arrays: dict,
    bare_powered: np.ndarray,
    painted_powered: np.ndarray,
    bare_load_start_epoch_s: float,
    painted_load_start_epoch_s: float,
    board: np.ndarray,
    unknown_mask: np.ndarray,
    out_path: str | Path,
    load_value: float = 1.0,
    label: str = "",
) -> dict:
    """Build the bare->painted supervised archive for one paired load level.

    Returns a diagnostics dictionary (not the arrays): the arrays are written
    to out_path as an .npz plus out_path.with_suffix('.json') metadata.
    """
    b_epochs = bare_arrays["clock_start_epoch_s"] + np.asarray(
        bare_arrays["timestamps_s"], dtype=np.float64
    )
    p_epochs = painted_arrays["clock_start_epoch_s"] + np.asarray(
        painted_arrays["timestamps_s"], dtype=np.float64
    )
    bare_theta = theta_series(bare_arrays)
    painted_theta = theta_series(painted_arrays)
    if bare_theta.shape[1:] != board.shape or painted_theta.shape[1:] != board.shape:
        raise ValueError("sequences do not share the canonical board grid")

    bare_powered = np.asarray(bare_powered, dtype=np.int64)
    painted_powered = np.asarray(painted_powered, dtype=np.int64)
    bare_p_epochs = b_epochs[bare_powered]
    painted_epochs_powered = p_epochs[painted_powered]
    if painted_epochs_powered.ndim != 1 or not np.all(np.diff(painted_epochs_powered) > 0):
        raise ValueError("painted powered epochs must be strictly increasing")

    # State key: elapsed time since each session's own load onset.
    bare_elapsed = bare_p_epochs - bare_load_start_epoch_s
    target_epochs = painted_load_start_epoch_s + bare_elapsed

    painted_fields, flags, _, first_over = _interpolate_painted_fields(
        target_epochs, painted_powered, painted_epochs_powered,
        painted_theta[painted_powered],
    )

    x_theta = bare_theta[bare_powered]
    y_theta = painted_fields
    if x_theta.shape != y_theta.shape:
        raise ValueError("paired shapes disagree")

    board_bool = np.asarray(board, dtype=bool)
    unknown_bool = np.asarray(unknown_mask, dtype=bool)
    trusted = board_bool & ~unknown_bool
    if trusted.sum() == 0:
        raise ValueError("no trusted pixels to score the pairing")

    # Label-noise diagnostics: on trusted pixels y and x should agree (both
    # sessions read the truth there); on masked pixels y - x is the correction
    # the model must learn.  Masked statistics stay on the shared board: a few
    # footprint pixels can sit where the bare session has no valid surface and
    # no label exists.
    residual = y_theta - x_theta
    masked_bool = board_bool & unknown_bool
    trusted_mae = np.abs(residual[:, trusted]).mean(axis=1)
    trusted_bias = residual[:, trusted].mean(axis=1)
    masked_mae = np.abs(residual[:, masked_bool]).mean(axis=1)
    board_mae = np.abs(residual[:, board_bool]).mean(axis=1)

    try:
        splits = contiguous_split(len(bare_powered))
    except ValueError:
        # Short sequences (e.g. the tape sessions) cannot hold the default
        # 240-frame blocks.  This experiment trains on the whole archive anyway,
        # so one train block with empty val/test is recorded rather than
        # inventing a split the owner did not ask for.
        full = np.arange(len(bare_powered))
        splits = {"train": full, "val": full[:0], "test": full[:0]}
    np.savez_compressed(
        out_path,
        x_theta=x_theta.astype(np.float32),
        y_theta=y_theta.astype(np.float32),
        board_mask=board_bool,
        unknown_mask=unknown_bool,
        elapsed_s=bare_elapsed.astype(np.float64),
        painted_flag=flags,
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
        load_value=np.float64(load_value),
    )

    meta = {
        "label": label,
        "conventions": CONVENTIONS,
        "n_pairs": int(len(bare_powered)),
        "n_interp": int(np.sum(flags == "interp")),
        "n_steady_flag": int(np.sum(flags == "steady")),
        "n_first_flag": int(np.sum(flags == "first")),
        "first_steady_index": int(first_over) if first_over < len(bare_powered) else None,
        "painted_powered_frames": int(len(painted_powered)),
        "load_value": float(load_value),
        "board_px": int(board_bool.sum()),
        "trusted_px": int(trusted.sum()),
        "masked_px": int(masked_bool.sum()),
        "split_blocks": {
            name: [int(splits[name][0]), int(splits[name][-1]) + 1] if len(splits[name]) else []
            for name in splits
        },
        "trusted_label_noise_mae_k_by_split": {
            name: float(np.abs(residual[splits[name]][:, trusted]).mean()) if len(splits[name]) else 0.0
            for name in splits
        },
        "masked_correction_mae_k_by_split": {
            name: float(np.abs(residual[splits[name]][:, masked_bool]).mean()) if len(splits[name]) else 0.0
            for name in splits
        },
        "steady_trusted_label_noise_mae_k": float(np.abs(residual[first_over:][:, trusted]).mean())
        if first_over < len(bare_powered)
        else 0.0,
    }
    Path(out_path).with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
