"""Cross-session evidence for temperatures hidden by low-emissivity surfaces.

An infrared temperature alone cannot separate surface temperature from emissivity
and reflected radiance.  The image-only reconstruction can therefore locate a
source under a metal lid from the surrounding board, but it cannot determine the
lid's absolute temperature reliably.  This module handles the extra evidence used
for that missing level:

* timestamped load and SoC-temperature logs;
* a high-emissivity recording of the same board and load; and
* a paired attenuation mask that audits the image-only footprint detector.

The functions are deliberately small and data-format agnostic.  The case10 driver
chooses the files and records every calibration assumption in its JSON report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, label, median_filter

from tdp.data.emissivity import CROSS


@dataclass(frozen=True)
class TimestampedValues:
    """One numeric CSV column indexed by absolute Unix time."""

    epoch_s: np.ndarray
    values: np.ndarray
    timestamp_column: str
    value_column: str


@dataclass(frozen=True)
class LoadInterval:
    """Longest sustained high-load interval in one timestamped log."""

    start_epoch_s: float
    stop_epoch_s: float
    n_samples: int


@dataclass(frozen=True)
class PairedAttenuation:
    """Pixels whose bare-board reading is suppressed relative to painted truth."""

    mask: np.ndarray
    attenuation_k: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class SurfaceAnchorSeries:
    """A target surface-temperature series transferred from a painted reference."""

    surface_temperature_c: np.ndarray
    uncertainty_k: np.ndarray
    report: dict[str, Any]


def sequence_start_epoch(metadata_path: Path | str) -> float:
    """Read the first infrared frame's absolute timestamp from run metadata."""

    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    try:
        text = payload["sequence"]["quality"]["clock"]["start"]
    except KeyError as exc:
        raise ValueError(f"{metadata_path} has no absolute sequence clock") from exc
    if not text:
        raise ValueError(f"{metadata_path} has an empty absolute sequence clock")
    return float(datetime.fromisoformat(str(text)).timestamp())


def read_timestamped_values(
    path: Path | str,
    *,
    value_columns: Iterable[str],
    timestamp_columns: Iterable[str] = ("datetime", "timestamp"),
) -> TimestampedValues:
    """Read the first matching timestamp and numeric-value columns from a CSV."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        timestamp_column = next((name for name in timestamp_columns if name in fields), None)
        value_column = next((name for name in value_columns if name in fields), None)
        if timestamp_column is None or value_column is None:
            raise ValueError(
                f"{source} needs one of timestamp columns {tuple(timestamp_columns)} and "
                f"one of value columns {tuple(value_columns)}; found {fields}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source} is empty")
    epoch_s = np.asarray(
        [datetime.fromisoformat(row[timestamp_column]).timestamp() for row in rows],
        dtype=np.float64,
    )
    values = np.asarray([float(row[value_column]) for row in rows], dtype=np.float64)
    if not np.isfinite(epoch_s).all() or not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite timestamps or values")
    if len(epoch_s) > 1 and not np.all(np.diff(epoch_s) > 0.0):
        raise ValueError(f"{source} timestamps are not strictly increasing")
    return TimestampedValues(epoch_s, values, timestamp_column, value_column)


def align_to_frames(
    log: TimestampedValues,
    frame_epoch_s: np.ndarray,
    *,
    maximum_extrapolation_s: float = 0.0,
) -> np.ndarray:
    """Linearly align a timestamped log to infrared frames without silent extrapolation."""

    frames = np.asarray(frame_epoch_s, dtype=np.float64)
    if frames.ndim != 1 or not np.isfinite(frames).all():
        raise ValueError("frame_epoch_s must be a finite one-dimensional array")
    margin = float(maximum_extrapolation_s)
    if margin < 0.0:
        raise ValueError("maximum_extrapolation_s must be non-negative")
    if frames.size and (
        frames.min() < log.epoch_s[0] - margin or frames.max() > log.epoch_s[-1] + margin
    ):
        raise ValueError(
            "temperature log does not cover all requested infrared frames: "
            f"log [{log.epoch_s[0]:.0f}, {log.epoch_s[-1]:.0f}], "
            f"frames [{frames.min():.0f}, {frames.max():.0f}]"
        )
    return np.interp(frames, log.epoch_s, log.values)


def longest_high_load_interval(
    log: TimestampedValues,
    *,
    threshold_percent: float = 90.0,
    maximum_gap_s: float = 10.0,
) -> LoadInterval:
    """Return the longest contiguous run whose reported load clears a threshold."""

    high = np.flatnonzero(log.values >= float(threshold_percent))
    if high.size == 0:
        raise ValueError(f"load log never reaches {threshold_percent:g}%")
    groups: list[np.ndarray] = []
    start = 0
    for slot in range(1, len(high)):
        separated = high[slot] != high[slot - 1] + 1
        time_gap = log.epoch_s[high[slot]] - log.epoch_s[high[slot - 1]]
        if separated or time_gap > float(maximum_gap_s):
            groups.append(high[start:slot])
            start = slot
    groups.append(high[start:])
    best = max(
        groups,
        key=lambda group: (
            log.epoch_s[group[-1]] - log.epoch_s[group[0]],
            len(group),
        ),
    )
    return LoadInterval(
        start_epoch_s=float(log.epoch_s[best[0]]),
        stop_epoch_s=float(log.epoch_s[best[-1]]),
        n_samples=int(len(best)),
    )


def steady_frame_mask(
    frame_epoch_s: np.ndarray,
    load: LoadInterval,
    *,
    window_s: float = 180.0,
) -> np.ndarray:
    """Frames in the final requested window of a powered interval."""

    frames = np.asarray(frame_epoch_s, dtype=np.float64)
    window = float(window_s)
    if window <= 0.0:
        raise ValueError("steady window must be positive")
    mask = (frames >= load.stop_epoch_s - window) & (frames <= load.stop_epoch_s)
    if mask.sum() < 3:
        raise ValueError(f"only {int(mask.sum())} infrared frames in the steady window")
    return mask


def paired_attenuation_mask(
    bare_theta_k: np.ndarray,
    bare_steady: np.ndarray,
    painted_theta_k: np.ndarray,
    painted_steady: np.ndarray,
    *,
    board_mask: np.ndarray | None = None,
    threshold_k: float = 5.0,
    smooth_px: float = 1.0,
    min_region_px: int = 6,
) -> PairedAttenuation:
    """Find suppressed bare-board readings using a matched painted recording.

    This is not a fitted emissivity.  It is a reliability audit: at corresponding
    steady states of the same board and load, a compact region that reads several
    kelvin colder only before painting cannot be declared trustworthy merely
    because it remains hotter than its immediate neighbours.
    """

    bare = np.asarray(bare_theta_k, dtype=np.float64)
    painted = np.asarray(painted_theta_k, dtype=np.float64)
    bare_select = np.asarray(bare_steady, dtype=bool)
    painted_select = np.asarray(painted_steady, dtype=bool)
    if bare.ndim != 3 or painted.ndim != 3 or bare.shape[1:] != painted.shape[1:]:
        raise ValueError("paired temperature sequences must share shape (T, H, W)")
    if bare_select.shape != (len(bare),) or painted_select.shape != (len(painted),):
        raise ValueError("steady masks must have one value per frame")
    if bare_select.sum() < 3 or painted_select.sum() < 3:
        raise ValueError("each paired sequence needs at least three steady frames")
    board = (
        np.ones(bare.shape[1:], dtype=bool)
        if board_mask is None
        else np.asarray(board_mask, dtype=bool)
    )
    if board.shape != bare.shape[1:]:
        raise ValueError("board_mask does not match paired frame shape")
    attenuation = np.median(painted[painted_select], axis=0) - np.median(
        bare[bare_select], axis=0
    )
    if smooth_px > 0.0:
        attenuation = gaussian_filter(attenuation, float(smooth_px))
    seed = binary_opening((attenuation > float(threshold_k)) & board, structure=CROSS)
    labels, count = label(seed, structure=np.ones((3, 3), dtype=bool))
    if count:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= int(min_region_px))
        keep = keep[keep != 0]
        mask = np.isin(labels, keep)
    else:
        mask = np.zeros(board.shape, dtype=bool)
    return PairedAttenuation(
        mask=mask,
        attenuation_k=attenuation,
        report={
            "kind": "paired painted-board attenuation audit; not fitted emissivity",
            "threshold_k": float(threshold_k),
            "smooth_px": float(smooth_px),
            "min_region_px": int(min_region_px),
            "n_flagged_px": int(mask.sum()),
            "n_regions": int(label(mask, structure=np.ones((3, 3), dtype=bool))[1]),
            "attenuation_p99_k": float(np.percentile(attenuation[board], 99.0)),
        },
    )


def transfer_surface_anchor(
    target_frame_epoch_s: np.ndarray,
    target_die_temperature_c: np.ndarray,
    target_load: LoadInterval,
    reference_frame_epoch_s: np.ndarray,
    reference_die_temperature_c: np.ndarray,
    reference_surface_temperature_c: np.ndarray,
    reference_load: LoadInterval,
    *,
    steady_window_s: float = 180.0,
    smoothing_frames: int = 5,
    uncertainty_floor_k: float = 1.5,
) -> SurfaceAnchorSeries:
    """Transfer a painted run's die-to-surface difference to another session.

    The transient difference is matched by elapsed time since load onset.  Once
    the reference enters its final steady window, its mean difference is used so
    log quantisation does not inject artificial frame-to-frame temperature steps.
    """

    target_frames = np.asarray(target_frame_epoch_s, dtype=np.float64)
    target_die = np.asarray(target_die_temperature_c, dtype=np.float64)
    reference_frames = np.asarray(reference_frame_epoch_s, dtype=np.float64)
    reference_die = np.asarray(reference_die_temperature_c, dtype=np.float64)
    reference_surface = np.asarray(reference_surface_temperature_c, dtype=np.float64)
    if target_frames.shape != target_die.shape or reference_frames.shape != reference_die.shape:
        raise ValueError("each die-temperature series must match its frame timestamps")
    if reference_surface.shape != reference_frames.shape:
        raise ValueError("reference surface temperature must have one value per frame")
    arrays = (target_frames, target_die, reference_frames, reference_die, reference_surface)
    if any(array.ndim != 1 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("surface-anchor inputs must be finite one-dimensional arrays")
    if smoothing_frames < 1 or smoothing_frames % 2 == 0:
        raise ValueError("smoothing_frames must be a positive odd integer")

    reference_steady = steady_frame_mask(
        reference_frames, reference_load, window_s=steady_window_s
    )
    raw_difference = reference_die - reference_surface
    steady_difference = raw_difference[reference_steady]
    offset_mean = float(np.mean(steady_difference))
    offset_sd = float(np.std(steady_difference, ddof=1))
    smooth_difference = median_filter(
        raw_difference, size=int(smoothing_frames), mode="nearest"
    )
    reference_elapsed = reference_frames - reference_load.start_epoch_s
    target_elapsed = target_frames - target_load.start_epoch_s
    steady_start_elapsed = reference_load.stop_epoch_s - steady_window_s - reference_load.start_epoch_s
    matched_elapsed = np.minimum(target_elapsed, steady_start_elapsed)
    mapped_difference = np.interp(
        matched_elapsed,
        reference_elapsed,
        smooth_difference,
        left=float(smooth_difference[0]),
        right=offset_mean,
    )
    mapped_difference[target_elapsed >= steady_start_elapsed] = offset_mean
    surface = target_die - mapped_difference
    # One independent painted session cannot measure run-to-run variation.  The
    # explicit floor carries that limitation instead of reporting the small IR
    # frame scatter as if it were a complete uncertainty estimate.
    uncertainty = max(float(uncertainty_floor_k), offset_sd)
    return SurfaceAnchorSeries(
        surface_temperature_c=surface,
        uncertainty_k=np.full(surface.shape, uncertainty, dtype=np.float64),
        report={
            "kind": "case-matched die-to-painted-surface temperature anchor",
            "reference_steady_window_s": float(steady_window_s),
            "reference_steady_frames": int(reference_steady.sum()),
            "die_to_surface_offset_mean_k": offset_mean,
            "die_to_surface_offset_sd_k": offset_sd,
            "uncertainty_floor_k": float(uncertainty_floor_k),
            "anchor_uncertainty_k": uncertainty,
            "target_surface_steady_mean_c": float(
                np.mean(surface[steady_frame_mask(target_frames, target_load, window_s=steady_window_s)])
            ),
            "note": (
                "the input log measures the die; the painted reference supplies the "
                "die-to-visible-surface difference"
            ),
        },
    )
