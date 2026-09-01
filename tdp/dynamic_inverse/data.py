"""Time-series data contract for the dynamic inversion.

This module is an *adapter*.  Frame parsing lives in :mod:`tdp.io.hm_csv`,
board geometry in :mod:`tdp.data.crop` and the low-emissivity repair in
:mod:`tdp.data.emissivity`; nothing here re-implements them.  What it adds is
everything a *sequence* needs and a single frame does not: one board window
shared by every frame, one artifact mask shared by every frame, drift
registration, timestamp auditing and a per-pixel confidence that remembers
which pixels were measured and which were reconstructed.

The output is :class:`ThermalSequence`, whose fields are the contract the rest
of the package codes against::

    temperature_c      float32 (T, H, W)   repaired board temperature
    valid_mask         bool    (H, W)      pixels usable at all
    trust_weight       float32 (H, W)      1 = measured, ->0 deep inside a repair
    emissivity         float32 (H, W)      per-region fitted, diagnostic
    timestamps_s       float64 (T,)        seconds from the first frame
    ambient_c          float32 (T,)        border-ring median per frame
    pixel_size_mm      float | None

``timestamps_s`` comes from the filename clock, which ticks in whole seconds:
that is a real 1 s quantisation on a ~6.3 s cadence and it is recorded in the
quality report rather than smoothed away.  When no clock is available at all
the loader falls back to ``dt = 1 frame`` and marks the sequence so every
downstream unit string switches to ``frame``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, shift as ndshift
from scipy.ndimage import rotate as ndrotate

from tdp.data import emissivity as em
from tdp.data.crop import estimate_ambient, rectify_quad, rotated_rect_corners
from tdp.dynamic_inverse.config import MeasurementConfig, SequenceSpec
from tdp.io import read_hm_csv

# Drift below this is at the registration estimator's own resolution.
_DRIFT_APPLY_PX = 1.5


# --------------------------------------------------------------------------
# board geometry
# --------------------------------------------------------------------------
def _axis_alignment_score(excess: np.ndarray, angle_deg: float) -> float:
    """How sharply the board's edges project onto the axes after derotation.

    A rectangular board rotated back into alignment concentrates all four of
    its edge gradients into two row peaks and two column peaks.  Off-angle,
    each edge smears across many rows, so the projected profile flattens.  The
    score is the mass of the few largest profile values above the profile mean,
    which is maximal exactly when the edges are axis-aligned.

    This deliberately replaces an orientation estimate from image moments: the
    warm air halo around the board is nearly isotropic and drags a moment-based
    major axis around by several degrees between frames.
    """
    rotated = ndrotate(excess, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    smooth = gaussian_filter(rotated, 1.0)
    total = 0.0
    for axis in (0, 1):
        profile = np.abs(np.gradient(smooth, axis=axis)).mean(axis=1 - axis)
        profile = profile - profile.mean()
        total += float(np.sort(profile)[-6:].sum())
    return total


def estimate_board_angle(
    frame: np.ndarray, ambient_c: float | None = None, search_deg: float = 20.0
) -> float:
    """In-plane rotation of the board, in degrees, by projection sharpness."""
    ambient = estimate_ambient(frame) if ambient_c is None else float(ambient_c)
    excess = np.clip(frame.astype(np.float64) - ambient, 0.0, None)
    coarse = np.arange(-search_deg, search_deg + 1e-9, 1.0)
    scores = [_axis_alignment_score(excess, a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    scores = [_axis_alignment_score(excess, a) for a in fine]
    return float(fine[int(np.argmax(scores))])


def _edge_positions(profile: np.ndarray) -> tuple[float, float]:
    """Leading and trailing edge of a plateau, from its derivative extrema."""
    derivative = np.gradient(gaussian_filter(profile, 1.5))
    half = len(derivative) // 2
    return float(np.argmax(derivative[:half])), float(half + np.argmin(derivative[half:]))


def estimate_board_window(
    frame: np.ndarray,
    ambient_c: float | None = None,
    angle_deg: float | None = None,
    aspect: float = 85.0 / 56.0,
    search_deg: float = 20.0,
    edges_derotated: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Locate the PCB as a rotated rectangle in the raw frame.

    The board is derotated and its four edges are taken either from
    ``edges_derotated`` -- row/column indices agreed once for a session -- or,
    failing that, from the row and column profiles of the excess-temperature
    field.  Either way the resulting extent is then *forced* onto the known
    85:56 outline: the two measured extents are replaced by the pair with the
    same area and the correct aspect ratio.  ``aspect_residual`` reports how
    far the raw extent was from that ratio, which is the honest measure of how
    well the window was found -- a few percent is edge-finding noise or ports
    protruding past the PCB, a large value means a cable or a halo was caught.
    """
    ambient = estimate_ambient(frame) if ambient_c is None else float(ambient_c)
    angle = estimate_board_angle(frame, ambient, search_deg) if angle_deg is None else float(angle_deg)
    excess = np.clip(frame.astype(np.float64) - ambient, 0.0, None)
    rotated = ndrotate(excess, angle, reshape=False, order=1, mode="constant", cval=0.0)

    if edges_derotated:
        missing = {"top", "bottom", "left", "right"} - set(edges_derotated)
        if missing:
            raise ValueError(f"edges_derotated missing {sorted(missing)}")
        row0, row1 = float(edges_derotated["top"]), float(edges_derotated["bottom"])
        col0, col1 = float(edges_derotated["left"]), float(edges_derotated["right"])
        edge_source = "config"
    else:
        row0, row1 = _edge_positions(rotated.mean(axis=1))
        col0, col1 = _edge_positions(rotated.mean(axis=0))
        edge_source = "detected"
    measured_h, measured_w = row1 - row0, col1 - col0
    if measured_h <= 1 or measured_w <= 1:
        raise ValueError("board edges could not be located in the excess-temperature profile")

    area = measured_h * measured_w
    height = float(np.sqrt(area * aspect))
    width = float(np.sqrt(area / aspect))
    aspect_residual = abs((measured_h / measured_w) / aspect - 1.0)

    # Centre found in the derotated frame; rotate it back about the frame centre.
    centre_rot = np.array([(row0 + row1) / 2.0, (col0 + col1) / 2.0])
    frame_centre = np.array([(frame.shape[0] - 1) / 2.0, (frame.shape[1] - 1) / 2.0])
    theta = np.deg2rad(angle)
    # scipy.ndimage.rotate(angle) maps source -> rotated; invert it here.
    back = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    centre = frame_centre + back @ (centre_rot - frame_centre)
    return {
        "center_rc": (float(centre[0]), float(centre[1])),
        "size_hw": (height, width),
        "angle_deg": float(angle),
        "measured_size_hw": (float(measured_h), float(measured_w)),
        "edges_derotated": {"top": row0, "bottom": row1, "left": col0, "right": col1},
        "edge_source": edge_source,
        "aspect_residual": float(aspect_residual),
        "ambient_c": float(ambient),
    }


# --------------------------------------------------------------------------
# sequence container
# --------------------------------------------------------------------------
@dataclass
class ThermalSequence:
    """A board-frame temperature series plus everything needed to trust it."""

    name: str
    temperature_c: np.ndarray  # (T, H, W) float32, repaired
    raw_temperature_c: np.ndarray  # (T, H, W) float32, as measured (rectified only)
    valid_mask: np.ndarray  # (H, W) bool
    trust_weight: np.ndarray  # (H, W) float32 in [0, 1]
    repair_mask: np.ndarray  # (H, W) bool, pixels the repair replaced
    emissivity: np.ndarray  # (H, W) float32
    timestamps_s: np.ndarray  # (T,) float64
    ambient_c: np.ndarray  # (T,) float32
    pixel_size_mm: float | None
    time_unit: str  # "s" or "frame"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.temperature_c.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return (int(self.temperature_c.shape[1]), int(self.temperature_c.shape[2]))

    @property
    def theta(self) -> np.ndarray:
        """Excess temperature over ambient, (T, H, W)."""
        return self.temperature_c - self.ambient_c[:, None, None]

    def board_mean_theta(self) -> np.ndarray:
        """Trust-weighted mean excess temperature per frame."""
        weight = np.where(self.valid_mask, self.trust_weight, 0.0)
        total = weight.sum()
        if total <= 0:
            raise ValueError("sequence has no trusted pixels")
        return (self.theta * weight).reshape(self.n_frames, -1).sum(1) / total


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _file_digest(paths: list[Path], limit: int = 4096) -> str:
    """Stable digest of a file list: names plus sizes, cheap and reproducible."""
    hasher = hashlib.sha256()
    for path in paths[:limit]:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
    hasher.update(str(len(paths)).encode("ascii"))
    return hasher.hexdigest()[:16]


def _timestamps(frames_ts: list[datetime | None]) -> tuple[np.ndarray, str, dict]:
    if any(ts is None for ts in frames_ts):
        n = len(frames_ts)
        return np.arange(n, dtype=np.float64), "frame", {
            "clock": "absent",
            "note": "no filename clock; dt fixed at 1 frame, never guessed from a nominal rate",
        }
    base = frames_ts[0]
    seconds = np.array([(ts - base).total_seconds() for ts in frames_ts], dtype=np.float64)
    diffs = np.diff(seconds)
    report = {
        "clock": "filename",
        "start": base.isoformat(),
        "dt_median_s": float(np.median(diffs)) if len(diffs) else float("nan"),
        "dt_min_s": float(diffs.min()) if len(diffs) else float("nan"),
        "dt_max_s": float(diffs.max()) if len(diffs) else float("nan"),
        "monotonic": bool(np.all(diffs > 0)) if len(diffs) else True,
        "quantisation_s": 1.0,
    }
    return seconds, "s", report


def _drift_shift(reference: np.ndarray, frame: np.ndarray, max_shift: int = 4) -> tuple[float, float]:
    """Integer-pixel drift of ``frame`` relative to ``reference``, by gradient ZNCC.

    Gradient magnitude is used rather than temperature so that a cold frame and
    a hot frame still correlate through the board's component pattern instead
    of through their common level.
    """

    def edges(image: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(gaussian_filter(image.astype(np.float64), 1.0))
        out = np.hypot(gy, gx)
        return out - out.mean()

    ref, cur = edges(reference), edges(frame)
    best, best_score = (0.0, 0.0), -np.inf
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            shifted = ndshift(cur, (dr, dc), order=1, mode="nearest")
            denom = np.linalg.norm(ref) * np.linalg.norm(shifted)
            score = float((ref * shifted).sum() / denom) if denom > 0 else -np.inf
            if score > best_score:
                best_score, best = score, (float(dr), float(dc))
    return best


def load_sequence(
    spec: SequenceSpec,
    measurements: MeasurementConfig,
    root: Path | str | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    register_drift: bool = True,
) -> ThermalSequence:
    """Load one recorded sequence into the :class:`ThermalSequence` contract."""
    root = Path(root) if root is not None else Path(measurements.dataset_root)
    case_dir = root / spec.case_dir
    if not case_dir.exists():
        matches = sorted(root.glob(spec.case_dir + "*"))
        if not matches:
            raise FileNotFoundError(f"case directory not found: {case_dir}")
        case_dir = matches[0]
    files = sorted((case_dir / "data").glob("HM*.csv"))
    if not files:
        files = sorted(case_dir.rglob("HM*.csv"))
    if not files:
        raise FileNotFoundError(f"no HM*.csv under {case_dir}")
    selected = files[::max(1, int(frame_stride))]
    if max_frames is not None:
        selected = selected[: int(max_frames)]

    raw_frames: list[np.ndarray] = []
    stamps: list[datetime | None] = []
    ambient: list[float] = []
    parse_failures: list[dict] = []
    for path in selected:
        try:
            frame = read_hm_csv(path)
        except Exception as exc:  # a corrupt frame must not sink the sequence
            parse_failures.append({"file": path.name, "error": str(exc)})
            continue
        raw_frames.append(frame.temps)
        stamps.append(frame.timestamp)
        ambient.append(estimate_ambient(frame.temps))
    if not raw_frames:
        raise RuntimeError(f"every frame under {case_dir} failed to parse")

    stack = np.stack(raw_frames)
    ambient_c = np.asarray(ambient, dtype=np.float32)
    timestamps_s, time_unit, clock_report = _timestamps(stamps)

    quality = _quality_report(stack, timestamps_s, time_unit)
    quality["clock"] = clock_report
    quality["parse_failures"] = parse_failures
    quality["n_files_found"] = len(files)
    quality["n_frames_used"] = int(stack.shape[0])
    quality["frame_stride"] = int(frame_stride)
    quality["input_digest"] = _file_digest(selected)

    # --- board window, agreed once from a hot reference frame ---------------
    reference_index = int(np.argsort(stack.reshape(len(stack), -1).mean(1))[
        min(len(stack) - 1, int(spec.repair.reference_quantile * (len(stack) - 1)))
    ])
    reference = stack[reference_index]
    window = spec.board
    if window.center_rc is None or window.size_hw is None:
        estimated = estimate_board_window(
            reference,
            ambient_c=float(ambient_c[reference_index]),
            angle_deg=window.angle_deg,
            aspect=measurements.board_size_mm[0] / measurements.board_size_mm[1],
            search_deg=window.search_angle_deg,
            edges_derotated=window.edges_derotated,
        )
        center = window.center_rc or estimated["center_rc"]
        size = window.size_hw or estimated["size_hw"]
        angle = estimated["angle_deg"] if window.angle_deg is None else window.angle_deg
        quality["board_window_estimated"] = estimated
    else:
        center, size, angle = window.center_rc, window.size_hw, window.angle_deg
        quality["board_window_estimated"] = None
    quality["board_window"] = {
        "center_rc": [float(center[0]), float(center[1])],
        "size_hw": [float(size[0]), float(size[1])],
        "angle_deg": float(angle),
        "output_hw": list(window.output_hw),
        "source": "config" if quality["board_window_estimated"] is None else "estimated",
    }

    # --- drift registration -------------------------------------------------
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    measured_shift = np.zeros((len(stack), 2), dtype=np.float64)
    if register_drift and len(stack) > 1:
        probe_idx = np.unique(np.linspace(0, len(stack) - 1, min(len(stack), 24)).astype(int))
        probe_shift = {int(i): _drift_shift(reference, stack[i]) for i in probe_idx}
        order = sorted(probe_shift)
        for axis in (0, 1):
            probes = np.asarray([probe_shift[i][axis] for i in order], dtype=np.float64)
            # A single probe can lock onto the cable instead of the board; a
            # 5-wide reflecting median over the probe series drops such an
            # outlier, including one on the first probe, while preserving a
            # genuine step -- which is what real staging drift looks like (the
            # board is nudged once, not continuously).
            probes = median_filter(probes, size=5, mode="reflect")
            measured_shift[:, axis] = np.round(
                np.interp(np.arange(len(stack)), np.asarray(order), probes)
            )
        # A one-pixel ZNCC result on a board whose pattern changes between a hot
        # and a cold frame is at the estimator's own resolution, and pasting it
        # into the crop would inject a step that looks like real dynamics.
        # Correct only a drift large enough to be motion rather than noise.
        peak = float(np.abs(measured_shift).max())
        apply = peak > _DRIFT_APPLY_PX
        if apply:
            shifts = measured_shift
        quality["drift"] = {
            "probe_frames": [int(i) for i in order],
            "measured_max_abs_shift_px": peak,
            "apply_threshold_px": _DRIFT_APPLY_PX,
            "applied": bool(apply),
            "note": (
                "measured drift is at the estimator resolution; left uncorrected so no "
                "sub-pixel step is injected into the series"
                if not apply
                else "drift exceeds one pixel and was corrected by bilinear shift"
            ),
        }
    else:
        quality["drift"] = {"applied": False, "measured_max_abs_shift_px": 0.0}

    corners = rotated_rect_corners(center, size, angle)
    cropped = np.empty((len(stack), *window.output_hw), dtype=np.float32)
    for i, frame in enumerate(stack):
        source = frame if not np.any(shifts[i]) else ndshift(
            frame.astype(np.float64), -shifts[i], order=1, mode="nearest"
        )
        cropped[i] = rectify_quad(source, corners, window.output_hw)

    pixel_size_mm = float(
        np.mean([
            measurements.board_size_mm[0] / window.output_hw[0],
            measurements.board_size_mm[1] / window.output_hw[1],
        ])
    )
    quality["pixel_size"] = {
        "value_mm": pixel_size_mm,
        "row_mm": measurements.board_size_mm[0] / window.output_hw[0],
        "col_mm": measurements.board_size_mm[1] / window.output_hw[1],
        "basis": f"{measurements.board_size_note} {measurements.board_size_mm} mm "
                 f"resampled onto a {window.output_hw} grid",
        "calibrated": False,
    }

    # --- emissivity repair, one mask for the whole sequence ------------------
    repaired, repair_mask, emissivity_map, trust, repair_report = _repair_sequence(
        cropped, ambient_c, spec, reference_index
    )
    quality["repair"] = repair_report

    valid = np.isfinite(repaired).all(axis=0)
    if spec.repair.enabled and spec.repair.max_extrapolation_px > 0:
        distance = em.distance_to_trusted(repair_mask)
        valid &= distance <= float(spec.repair.max_extrapolation_px)
    quality["valid_fraction"] = float(valid.mean())

    metadata = {
        "sequence": spec.name,
        "case_dir": str(case_dir),
        "device": spec.device,
        "surface": spec.surface,
        "load": spec.load,
        "time_unit": time_unit,
        "space_unit": "px" if pixel_size_mm is None else "px (+mm via board outline)",
        "quality": quality,
        "radiometry": {
            "assumed_emissivity": spec.radiometry.assumed_emissivity,
            "recorded_by_camera": spec.radiometry.recorded,
            "band_exponent": spec.radiometry.band_exponent,
            "reflected_temperature_c": spec.radiometry.reflected_temperature_c,
        },
        "notes": spec.notes,
    }
    return ThermalSequence(
        name=spec.name,
        temperature_c=repaired.astype(np.float32),
        raw_temperature_c=cropped.astype(np.float32),
        valid_mask=valid,
        trust_weight=trust.astype(np.float32),
        repair_mask=repair_mask,
        emissivity=emissivity_map.astype(np.float32),
        timestamps_s=timestamps_s,
        ambient_c=ambient_c,
        pixel_size_mm=pixel_size_mm,
        time_unit=time_unit,
        metadata=metadata,
    )


def _repair_sequence(
    cropped: np.ndarray,
    ambient_c: np.ndarray,
    spec: SequenceSpec,
    reference_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Detect the artifact mask once, then repair every frame through it."""
    height, width = cropped.shape[1:]
    if not spec.repair.enabled:
        empty = np.zeros((height, width), dtype=bool)
        emissivity = np.full((height, width), spec.radiometry.assumed_emissivity, np.float32)
        return cropped.copy(), empty, emissivity, np.ones((height, width), np.float32), {
            "enabled": False,
            "reason": "repair disabled in config (ablation: no emissivity correction)",
        }

    reflected = spec.radiometry.reflected_temperature_c
    reference = em.repair_frame(
        cropped[reference_index],
        ambient_c=float(ambient_c[reference_index]),
        fin_length_px=spec.repair.fin_length_px,
        assumed_emissivity=spec.radiometry.assumed_emissivity,
        reflected_c=reflected,
        band_exponent=spec.radiometry.band_exponent,
        method=spec.repair.method,
        infer_sources=spec.repair.infer_sources,
    )
    mask = reference.mask

    repaired = np.empty_like(cropped)
    for i, frame in enumerate(cropped):
        result = em.repair_frame(
            frame,
            ambient_c=float(ambient_c[i]),
            mask=mask,
            fin_length_px=spec.repair.fin_length_px,
            assumed_emissivity=spec.radiometry.assumed_emissivity,
            reflected_c=reflected,
            band_exponent=spec.radiometry.band_exponent,
            method=spec.repair.method,
            infer_sources=spec.repair.infer_sources,
        )
        repaired[i] = result.temperature_c

    emissivity = np.full((height, width), spec.radiometry.assumed_emissivity, np.float32)
    regions = []
    from scipy.ndimage import label as _label

    labels, _ = _label(mask, structure=em.CROSS)
    for report in reference.regions:
        if report.fitted_emissivity is not None:
            emissivity[labels == report.region_id] = report.fitted_emissivity
        regions.append({
            "region_id": report.region_id,
            "area_px": report.area_px,
            "centroid_rc": list(report.centroid_rc),
            "apparent_mean_c": report.apparent_mean_c,
            "repaired_mean_c": report.repaired_mean_c,
            "lift_c": report.lift_c,
            "fitted_emissivity": report.fitted_emissivity,
            "method": report.method,
            "touches_border": report.touches_border,
            "max_distance_to_trusted_px": report.max_distance_to_trusted_px,
        })

    # Confidence falls off with how far a pixel sits from real data: a pixel one
    # step from a trusted neighbour is pinned by it, one deep inside a connector
    # bank is an extrapolation.  This is the weight the region aggregation and
    # the losses use, so a reconstructed pixel never counts as a measurement.
    distance = em.distance_to_trusted(mask)
    trust = np.exp(-distance / float(spec.repair.trust_decay_px)).astype(np.float32)

    report = {
        "enabled": True,
        "method": spec.repair.method,
        "reference_frame_index": int(reference_index),
        "masked_fraction": float(mask.mean()),
        "n_regions": len(reference.regions),
        "max_distance_to_trusted_px": float(distance.max()),
        "trust_decay_px": float(spec.repair.trust_decay_px),
        "regions": regions,
        "caveat": (
            "this legacy dynamic-inverse input uses a boundary-only conduction fill and "
            "cannot place a maximum inside a powered package; production Stage 2 uses "
            "tdp.thermal.reconstruct with fitted sources and explicit uncertainty"
        ),
    }
    return repaired, mask, emissivity, trust, report


def _quality_report(stack: np.ndarray, timestamps_s: np.ndarray, time_unit: str) -> dict:
    """Audit the raw stack: gaps, NaN/Inf, implausible jumps."""
    flat = stack.reshape(len(stack), -1)
    diffs = np.diff(timestamps_s) if len(timestamps_s) > 1 else np.array([1.0])
    median_dt = float(np.median(diffs))
    gap_threshold = 1.8 * median_dt
    gaps = np.nonzero(diffs > gap_threshold)[0]

    frame_delta = np.abs(np.diff(flat.mean(1))) if len(stack) > 1 else np.array([0.0])
    jump_threshold = float(np.median(frame_delta) + 8.0 * (np.median(np.abs(frame_delta - np.median(frame_delta))) * 1.4826 + 1e-6))
    jumps = np.nonzero(frame_delta > max(jump_threshold, 1.0))[0]

    return {
        "n_frames": int(len(stack)),
        "frame_shape": list(stack.shape[1:]),
        "time_unit": time_unit,
        "duration": float(timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) > 1 else 0.0,
        "median_dt": median_dt,
        "dt_irregular": bool(np.ptp(diffs) > 0.5 * median_dt) if len(diffs) else False,
        "gap_indices": [int(i) for i in gaps],
        "gap_sizes": [float(diffs[i]) for i in gaps],
        "n_nan": int(np.isnan(flat).sum()),
        "n_inf": int(np.isinf(flat).sum()),
        "temperature_min_c": float(np.nanmin(flat)),
        "temperature_max_c": float(np.nanmax(flat)),
        "mean_jump_indices": [int(i) for i in jumps],
        "mean_jump_threshold_c": float(max(jump_threshold, 1.0)),
    }
