"""Detect heating / near-steady / load-off / invalid intervals in a sequence.

Three signals are combined, because each alone is fooled by something:

* ``mean(theta)`` -- the level.  Says where the board is, not where it is going.
* ``d mean(theta)/dt`` -- the trend.  A slow drift and a genuine plateau look
  alike at a single sample, so it is smoothed over a configurable window.
* the *fraction of regions with a positive local slope* -- the agreement.  This
  is what separates a board-wide decay from a mean that happens to fall while
  one component keeps heating, and it is the reason the load-off interval can
  be labelled without knowing the load schedule.

The load-off interval is never called "free cooling", only
``free_cooling_candidate``.  On this dataset that distinction is not pedantry:
the CPU-load log shows the load stopping while the SoC log shows the die still
30 K above ambient, so the board is idling, not unplugged, and a residual
source is still running.  Anything downstream that assumes ``q = 0`` there
would bias every cooling rate it produces.

Manual intervals in the YAML always win over detection -- :func:`detect_phases`
applies them last -- so a session the operator knows about is never overridden
by a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HEATING = "heating"
NEAR_STEADY = "near_steady"
COOLING = "free_cooling_candidate"
INVALID = "invalid_transition"
PHASE_LABELS = (HEATING, NEAR_STEADY, COOLING, INVALID)


@dataclass
class PhaseResult:
    labels: np.ndarray  # (T,) str
    mean_theta: np.ndarray  # (T,)
    slope: np.ndarray  # (T,) per unit time
    positive_fraction: np.ndarray  # (T,) share of pixels/regions still warming
    segments: list[dict[str, Any]]
    report: dict[str, Any]

    def mask(self, label: str) -> np.ndarray:
        return self.labels == label


def _smooth(values: np.ndarray, times: np.ndarray, window: float) -> np.ndarray:
    """Local linear smoothing on a possibly irregular clock."""
    if window <= 0:
        return values.astype(np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for k, centre in enumerate(times):
        near = np.abs(times - centre) <= window / 2.0
        out[k] = float(np.mean(values[near]))
    return out


def _slope(values: np.ndarray, times: np.ndarray, window: float) -> np.ndarray:
    """Local least-squares slope, in value-units per time-unit."""
    out = np.zeros_like(values, dtype=np.float64)
    for k, centre in enumerate(times):
        near = np.abs(times - centre) <= max(window, 1e-9) / 2.0
        if near.sum() < 3:
            near = np.zeros_like(near)
            lo, hi = max(0, k - 2), min(len(times), k + 3)
            near[lo:hi] = True
        t_local = times[near] - centre
        if np.ptp(t_local) < 1e-9:
            continue
        out[k] = float(np.polyfit(t_local, values[near], 1)[0])
    return out


def _segments(labels: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    edges = np.nonzero(np.diff(labels.astype(object) != labels.astype(object)))[0]
    boundaries = [0]
    for k in range(1, len(labels)):
        if labels[k] != labels[k - 1]:
            boundaries.append(k)
    boundaries.append(len(labels))
    out: list[dict[str, Any]] = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        out.append({
            "label": str(labels[lo]),
            "start_index": int(lo),
            "stop_index": int(hi - 1),
            "start_time": float(times[lo]),
            "stop_time": float(times[hi - 1]),
            "duration": float(times[hi - 1] - times[lo]),
            "n_frames": int(hi - lo),
        })
    del edges
    return out


def _merge_short(labels: np.ndarray, times: np.ndarray, minimum: float) -> np.ndarray:
    """Absorb intervals too short to be a phase into their longer neighbour."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        for segment in _segments(labels, times):
            if segment["duration"] >= minimum or segment["n_frames"] >= 4:
                continue
            lo, hi = segment["start_index"], segment["stop_index"]
            before = labels[lo - 1] if lo > 0 else None
            after = labels[hi + 1] if hi + 1 < len(labels) else None
            replacement = before if before is not None else after
            if replacement is None:
                continue
            labels[lo : hi + 1] = replacement
            changed = True
            break
    return labels


def detect_phases(
    theta: np.ndarray,
    times: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    smooth_seconds: float = 120.0,
    steady_slope: float = 1.0e-4,
    slope_threshold_relative: float = 0.20,
    min_phase_seconds: float = 120.0,
    rewarm_tolerance_k: float = 0.5,
    overrides: list[dict[str, Any]] | None = None,
    time_unit: str = "s",
) -> PhaseResult:
    """Label every frame.  ``theta`` is ``(T, N)`` -- regions or flattened pixels."""
    if theta.ndim != 2:
        raise ValueError(f"theta must be (T, N), got shape {theta.shape}")
    if len(times) != theta.shape[0]:
        raise ValueError("times and theta disagree on the number of frames")

    if weight is None:
        weight = np.ones_like(theta)
    total = weight.sum(axis=1)
    mean_theta = np.where(
        total > 0, (theta * weight).sum(axis=1) / np.maximum(total, 1e-9), np.nan
    )
    mean_theta = _smooth(mean_theta, times, smooth_seconds)
    slope = _slope(mean_theta, times, max(smooth_seconds, 2.0 * float(np.median(np.diff(times)))))

    per_unit_slope = np.gradient(theta, times, axis=0)
    positive_fraction = (per_unit_slope > 0).mean(axis=1)

    # Scale the threshold to this sequence's own dynamics: the boundary between
    # "moving" and "steady" is a fraction of how fast the board ever moves, with
    # the configured absolute value as a floor.
    reference = float(np.percentile(np.abs(slope), 95))
    threshold = max(float(steady_slope), float(slope_threshold_relative) * reference)

    labels = np.full(len(times), NEAR_STEADY, dtype=object)
    labels[slope > threshold] = HEATING
    labels[slope < -threshold] = COOLING
    labels = _merge_short(labels, times, min_phase_seconds)

    # Extend each decay to the end of the load-off interval.  A slope threshold
    # marks only the *fast* head of a decay: once the board is within a few K of
    # its idle level the mean slope drops under any sensible threshold and the
    # tail gets relabelled steady.  That tail is not steady, it is the slow
    # modes -- and those are precisely what identifies the cooling rate, so
    # cutting them off would throw away the informative part of the interval.
    # The interval therefore runs until the board warms again or the record
    # ends, judged against the running minimum rather than the last sample.
    for segment in _segments(labels, times):
        if segment["label"] != COOLING:
            continue
        stop = segment["stop_index"]
        floor = float(np.min(mean_theta[segment["start_index"] : stop + 1]))
        while stop + 1 < len(labels) and labels[stop + 1] == NEAR_STEADY:
            if mean_theta[stop + 1] > floor + rewarm_tolerance_k:
                break
            stop += 1
            floor = min(floor, float(mean_theta[stop]))
        labels[segment["start_index"] : stop + 1] = COOLING

    # A free-cooling candidate is only a candidate while the board is cooling
    # *as a whole*.  Two different things can break that, and the board mean
    # only catches one of them:
    #
    #  * the mean itself turns back up -- handled by the extension guard above;
    #  * the mean keeps falling because a few large, hot regions dominate it,
    #    while most regions are already warming again.
    #
    # The second is why the fraction of regions with a positive slope is
    # tracked at all.  A mean-only test would hand that stretch to a fit that
    # assumes decay, which is precisely the "no evidence of local sustained
    # warming" condition this label is supposed to carry.  Sustained is the
    # operative word: a couple of noisy frames above one half is not a
    # turnaround, so a majority has to hold for half a phase.
    min_run = max(3, int(np.ceil(min_phase_seconds / 2.0 / max(float(np.median(np.diff(times))), 1e-9))))
    for segment in _segments(labels, times):
        if segment["label"] != COOLING:
            continue
        lo, hi = segment["start_index"], segment["stop_index"]
        rising = positive_fraction[lo : hi + 1] > 0.5
        cut = None
        run_start = None
        for k, flag in enumerate(rising):
            if flag:
                run_start = k if run_start is None else run_start
                if k - run_start + 1 >= min_run:
                    cut = lo + run_start
                    break
            else:
                run_start = None
        if cut is None:
            continue
        # Keep the decaying head; mark the turnaround and everything after it as
        # a transition rather than trimming it away silently.
        if cut <= lo:
            labels[lo : hi + 1] = INVALID
        else:
            labels[cut : hi + 1] = INVALID

    detected = labels.copy()
    applied: list[dict[str, Any]] = []
    for override in overrides or []:
        label = override.get("label")
        if label not in PHASE_LABELS:
            raise ValueError(f"override label {label!r} must be one of {PHASE_LABELS}")
        if "start_index" in override:
            lo, hi = int(override["start_index"]), int(override["stop_index"])
        else:
            lo = int(np.searchsorted(times, float(override["start_time"])))
            hi = int(np.searchsorted(times, float(override["stop_time"]), side="right") - 1)
        lo, hi = max(0, lo), min(len(times) - 1, hi)
        if hi < lo:
            raise ValueError(f"override {override!r} selects no frames")
        labels[lo : hi + 1] = label
        applied.append({"label": label, "start_index": lo, "stop_index": hi, "source": "config"})

    counts = {name: int((labels == name).sum()) for name in PHASE_LABELS}
    report = {
        "counts": counts,
        "time_unit": time_unit,
        "steady_slope_threshold": float(threshold),
        "steady_slope_floor": float(steady_slope),
        "slope_threshold_relative": float(slope_threshold_relative),
        "slope_p95": reference,
        "smooth_window": float(smooth_seconds),
        "overrides_applied": applied,
        "n_frames_changed_by_overrides": int((labels != detected).sum()),
        "rewarm_min_run_frames": int(min_run),
        "cooling_label_note": (
            "the decay interval is labelled free_cooling_candidate, never free cooling: "
            "the CPU-load log shows the load stopping while the SoC log shows the die still "
            "far above ambient, so a residual idle source is present and q=0 is not asserted"
        ),
    }
    return PhaseResult(
        labels=labels.astype(str),
        mean_theta=mean_theta,
        slope=slope,
        positive_fraction=positive_fraction,
        segments=_segments(labels.astype(str), times),
        report=report,
    )


def load_schedule_overrides(
    path: str,
    frame_times_unix: np.ndarray,
    load_column: str = "total_percent",
    on_threshold: float = 20.0,
) -> list[dict[str, Any]]:
    """Turn a recorded CPU-load log into powered/unpowered overrides.

    This is *optional* and off by default: the log is data the campaign already
    has, but the pipeline must run without it, and it is never allowed to
    supply a physical constant -- only which frames the load was running for.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    if "unix_time" not in frame or load_column not in frame:
        raise ValueError(f"{path} lacks 'unix_time' or {load_column!r}")
    load = np.interp(
        frame_times_unix, frame["unix_time"].to_numpy(float), frame[load_column].to_numpy(float)
    )
    powered = load >= on_threshold
    overrides: list[dict[str, Any]] = []
    if powered.any():
        index = np.nonzero(powered)[0]
        overrides.append(
            {"label": NEAR_STEADY, "start_index": int(index[0]), "stop_index": int(index[-1])}
        )
    return overrides
