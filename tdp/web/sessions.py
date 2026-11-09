"""Turn arbitrary uploaded HIKMICRO CSVs or radiometric JPEGs into one board grid.

The desktop pipeline could assume a board outline, a crop window and a camera
resolution because every session came from one campaign.  An uploaded file
gives none of those: its board may be any size, its framing may differ, and the
sensor grid may not be 256 x 192.  This module therefore measures what it can
and records what it had to assume:

* the board is located by the same profile-sharpness and edge fit used by
  :mod:`tdp.dynamic_inverse.data`;
* when the physical outline is known the measured rectangle is forced onto that
  aspect; when it is unknown the measured aspect is kept, because forcing 85:56
  onto a different board would silently stretch one axis;
* every frame is rectified through one session window, so the grid is shared;
* the low-emissivity footprint comes from the existing image detector, never
  from a fitted emissivity.

Temperatures are kelvin above each frame's own ambient.  The ambient is the
median of the frame border ring, the same estimator the desktop loader uses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from tdp.data import emissivity as em
from tdp.data.crop import estimate_ambient, rectify_quad, rotated_rect_corners
from tdp.dynamic_inverse.data import estimate_board_window
from tdp.io import HMParseError, parse_thermal_upload

# The campaign's analysis scale, not an instrument reading: the 85 mm board
# resampled onto 157 rows.  It only sets how many pixels a new board occupies;
# every physical result is still reported with its unit and its caveat.
DEFAULT_PIXEL_SIZE_MM = 0.5425

MIN_GRID_PX = 32
MAX_GRID_PX = 192
DEFAULT_HALO_PX = 1
MAX_PAIRS = 600


class UploadError(ValueError):
    """Uploaded files could not be turned into a usable board session."""


@dataclass(frozen=True)
class UploadedFrame:
    """One parsed frame plus everything read from its filename and border."""

    name: str
    digest: str
    timestamp: datetime | None
    temps: np.ndarray  # (H, W) float32, raw camera grid, degrees Celsius
    ambient_c: float  # border-ring median, degrees Celsius

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.temps.shape[0]), int(self.temps.shape[1])


@dataclass
class RectifiedSession:
    """Frames on one board grid, in kelvin above each frame's own ambient."""

    names: tuple[str, ...]
    theta_k: np.ndarray  # (T, H, W) float64
    ambient_c: np.ndarray  # (T,)
    times_s: np.ndarray  # (T,)
    valid: np.ndarray  # (H, W) bool, pixels the rectifier could sample
    window: dict
    grid_hw: tuple[int, int]
    reference_index: int
    aspect_forced: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.theta_k.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.theta_k.shape[1]), int(self.theta_k.shape[2])


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _sort_key(frame: UploadedFrame) -> tuple:
    """Filename clock first; upload order is only the fallback."""
    if frame.timestamp is None:
        return (1, datetime.min, frame.name)
    return (0, frame.timestamp, frame.name)


def load_uploads(
    files: Iterable[tuple[str, bytes]],
) -> tuple[list[UploadedFrame], list[dict]]:
    """Parse uploads of any sensor resolution; keep per-file failures.

    A corrupt frame does not sink a whole session, exactly as in the desktop
    loader, but the caller receives every failure so it can be shown rather
    than hidden.  Duplicate uploads are ignored by content digest.
    """
    frames: list[UploadedFrame] = []
    failures: list[dict] = []
    seen: set[str] = set()
    for name, data in files:
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            parsed = parse_thermal_upload(data, name, expect_shape=None)
        except (HMParseError, ValueError, OSError) as exc:
            failures.append({"file": name, "error": str(exc)})
            continue
        frames.append(
            UploadedFrame(
                name=name,
                digest=digest,
                timestamp=parsed.timestamp,
                temps=np.asarray(parsed.temps, dtype=np.float32),
                ambient_c=estimate_ambient(parsed.temps),
            )
        )
    if not frames:
        detail = "; ".join(f"{item['file']}: {item['error']}" for item in failures[:3])
        raise UploadError("没有可用的温度 CSV 或 JPEG" + (f"（{detail}）" if detail else ""))
    frames.sort(key=_sort_key)
    return frames, failures


def frame_times(frames: Sequence[UploadedFrame]) -> np.ndarray:
    """Seconds from the first frame; frame index when no clock is available."""
    if frames and all(frame.timestamp is not None for frame in frames):
        base = frames[0].timestamp
        assert base is not None
        seconds = np.asarray(
            [(frame.timestamp - base).total_seconds() for frame in frames],  # type: ignore[operator]
            dtype=np.float64,
        )
        if np.all(np.diff(seconds) >= 0.0):
            return seconds
    return np.arange(len(frames), dtype=np.float64)


# --------------------------------------------------------------------------
# board window
# --------------------------------------------------------------------------
def reference_frame_index(frames: Sequence[UploadedFrame], quantile: float = 0.9) -> int:
    """The frame at the hot end of the session, where the board is clearest."""
    means = np.asarray([float(np.mean(frame.temps)) for frame in frames])
    order = np.argsort(means)
    return int(order[min(len(order) - 1, int(round(quantile * (len(order) - 1))))])


def grid_for_board(
    board_size_mm: tuple[float, float] | None,
    detected_size_hw: tuple[float, float],
    *,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
) -> tuple[int, int]:
    """Pick the shared output grid for one board.

    A known outline uses the campaign scale, so the same board always lands on
    the same grid.  An unknown outline keeps the camera's own sampling, with a
    cap that keeps the network's memory bounded.
    """
    if board_size_mm is not None:
        long_mm, short_mm = sorted((float(v) for v in board_size_mm), reverse=True)
        if short_mm <= 0:
            raise UploadError("板体尺寸必须为正数")
        measured = float(detected_size_hw[0]) / max(float(detected_size_hw[1]), 1e-9)
        long_is_height = abs(measured - long_mm / short_mm) <= abs(measured - short_mm / long_mm)
        rows = int(round((long_mm if long_is_height else short_mm) / pixel_size_mm))
        cols = int(round((short_mm if long_is_height else long_mm) / pixel_size_mm))
    else:
        rows = int(round(float(detected_size_hw[0])))
        cols = int(round(float(detected_size_hw[1])))
        scale = min(1.0, MAX_GRID_PX / max(rows, cols))
        rows, cols = int(round(rows * scale)), int(round(cols * scale))
    rows = max(MIN_GRID_PX, min(MAX_GRID_PX, rows))
    cols = max(MIN_GRID_PX, min(MAX_GRID_PX, cols))
    return rows, cols


def fit_window_to_frame(
    center_rc: Sequence[float],
    size_hw: Sequence[float],
    angle_deg: float,
    frame_shape: tuple[int, int],
    margin_px: float = 0.6,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Keep a measured rectangle inside the sensor without re-aiming it.

    An edge detector can put the board flush against the sensor border; the
    homography then asks for a sample at -0.4 px and refuses.  Scaling the
    rectangle down only when it cannot fit, and shifting it only by its own
    overflow, preserves the board's physical aspect and centre.
    """
    center = np.asarray(center_rc, dtype=np.float64).copy()
    size = np.asarray(size_hw, dtype=np.float64).copy()
    limit = np.asarray(frame_shape, dtype=np.float64) - 1.0 - 2.0 * margin_px
    if np.any(limit <= 0):
        raise UploadError(f"帧尺寸 {frame_shape} 太小，无法容纳板体")
    if np.any(size > limit):
        size *= float(np.min(limit / size))
    for _ in range(8):
        corners = rotated_rect_corners(center, size, angle_deg)
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        shift = np.where(
            low < margin_px,
            margin_px - low,
            np.where(high > np.asarray(frame_shape) - 1.0 - margin_px,
                     np.asarray(frame_shape) - 1.0 - margin_px - high, 0.0),
        )
        if not np.any(np.abs(shift) > 1e-9):
            break
        center += shift
    return (float(center[0]), float(center[1])), (float(size[0]), float(size[1]))


def detect_window(
    frames: Sequence[UploadedFrame],
    *,
    board_size_mm: tuple[float, float] | None,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
    search_deg: float = 20.0,
) -> dict:
    """Locate the board once per session and choose its output grid."""
    index = reference_frame_index(frames)
    frame = frames[index]
    aspect = None
    if board_size_mm is not None:
        long_mm, short_mm = sorted((float(v) for v in board_size_mm), reverse=True)
        if short_mm <= 0:
            raise UploadError("板体尺寸必须为正数")
        aspect = long_mm / short_mm
        measured = float(frame.shape_hw[0]) / max(float(frame.shape_hw[1]), 1e-9)
        # The detector reports the board's row and column extents; a camera may
        # hold the board either way round.  Pick the orientation that matches
        # the image instead of forcing a landscape assumption.
        if abs(measured - short_mm / long_mm) < abs(measured - long_mm / short_mm):
            aspect = short_mm / long_mm
    estimated = estimate_board_window(
        frame.temps,
        ambient_c=frame.ambient_c,
        angle_deg=None,
        aspect=aspect,
        search_deg=search_deg,
        edges_derotated=None,
    )
    center, size = fit_window_to_frame(
        estimated["center_rc"], estimated["size_hw"], estimated["angle_deg"],
        frame.shape_hw,
    )
    grid_hw = grid_for_board(
        board_size_mm, estimated["measured_size_hw"], pixel_size_mm=pixel_size_mm
    )
    return {
        "center_rc": [center[0], center[1]],
        "size_hw": [size[0], size[1]],
        "angle_deg": float(estimated["angle_deg"]),
        "measured_size_hw": list(estimated["measured_size_hw"]),
        "edge_source": estimated["edge_source"],
        "aspect_residual": (
            float(estimated["aspect_residual"])
            if np.isfinite(estimated["aspect_residual"]) else None
        ),
        "aspect_forced": bool(estimated["aspect_forced"]),
        "grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
        "reference_index": int(index),
        "source": "detected",
    }


def rectify_session(
    frames: Sequence[UploadedFrame],
    window: dict,
    grid_hw: tuple[int, int],
    *,
    warnings: Iterable[str] = (),
) -> RectifiedSession:
    """Rectify every frame through one window and remove its own ambient."""
    height, width = int(grid_hw[0]), int(grid_hw[1])
    center = window["center_rc"]
    size = window["size_hw"]
    angle = float(window["angle_deg"])
    corners = rotated_rect_corners(center, size, angle)
    stack = np.empty((len(frames), height, width), dtype=np.float64)
    for index, frame in enumerate(frames):
        crop = rectify_quad(frame.temps, corners, (height, width))
        stack[index] = crop.astype(np.float64) - frame.ambient_c
    valid = np.all(np.isfinite(stack), axis=0)
    if not valid.any():
        raise UploadError("配准后的板体没有任何有效像素")
    theta = np.where(valid[None, :, :], stack, np.nan)
    times = frame_times(frames)
    notes = list(warnings)
    if float(valid.mean()) < 0.98:
        notes.append(
            f"板体网格只有 {valid.mean() * 100:.1f}% 的有效像素；边缘可能超出传感器"
        )
    return RectifiedSession(
        names=tuple(frame.name for frame in frames),
        theta_k=theta,
        ambient_c=np.asarray([frame.ambient_c for frame in frames], dtype=np.float64),
        times_s=times,
        valid=valid,
        window=dict(window),
        grid_hw=(height, width),
        reference_index=int(window.get("reference_index", reference_frame_index(frames))),
        aspect_forced=bool(window.get("aspect_forced", True)),
        warnings=notes,
    )


def build_session(
    files: Sequence[tuple[str, bytes]],
    *,
    board_size_mm: tuple[float, float] | None = None,
    window: dict | None = None,
    grid_hw: tuple[int, int] | None = None,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
) -> tuple[RectifiedSession, list[dict], list[str]]:
    """Parse, locate, and rectify one uploaded session.

    ``window`` and ``grid_hw`` pin a session to an already-trained board.  When
    they are absent the board is detected from the upload itself.
    """
    frames, failures = load_uploads(files)
    warnings: list[str] = []
    if window is None or grid_hw is None:
        detected = detect_window(frames, board_size_mm=board_size_mm,
                                 pixel_size_mm=pixel_size_mm)
        if window is None:
            window = detected
        if grid_hw is None:
            grid_hw = (int(detected["grid_hw"][0]), int(detected["grid_hw"][1]))
    session = rectify_session(frames, window, grid_hw, warnings=warnings)
    return session, failures, session.warnings


# --------------------------------------------------------------------------
# masks and powered interval
# --------------------------------------------------------------------------
def detect_unknown_mask(theta_ref: np.ndarray, valid: np.ndarray, *, halo_px: int = DEFAULT_HALO_PX) -> np.ndarray:
    """Low-emissivity footprint from the bare reference frame."""
    field = np.where(valid, theta_ref, np.nan)
    if not np.all(np.isfinite(field)):
        fill = float(np.nanmedian(field))
        field = np.where(np.isfinite(field), field, fill)
    try:
        detection = em.detect_low_emissivity(field, halo_px=halo_px)
    except ValueError:
        return np.zeros(valid.shape, dtype=bool)
    return np.asarray(detection.mask, dtype=bool) & valid


def powered_prefix(
    theta: np.ndarray,
    valid: np.ndarray,
    times: np.ndarray,
    *,
    minimum_frames: int = 10,
) -> tuple[np.ndarray, dict]:
    """Frames before the load dropped, found from the images when possible.

    A training upload may be the powered stretch only or the whole cycle.  The
    board-mean warming rate steps down when the load ends, so the image itself
    marks the cut; no log is needed.  Three measured corrections matter here:

    * the largest negative step is not always the load stop.  case13 has a
      147 s recording gap whose restart looks like a bigger drop; a candidate
      that sits on a gap larger than three median frame intervals is rejected.
    * the board mean is noisy on a plateau, and small negative steps occur
      before the load ever stops.  A real cut must start near the session's
      peak level and must be followed by at least 0.4 K of sustained cooling.
    * a near-settled board warms at almost zero before the stop, so a positive
      pre-drop rate is not a usable test.  The post-drop rate must fall by a
      clear amount and stay negative.

    Among the strong candidates the largest rate step is taken.  If no candidate
    starts near the peak and is followed by sustained cooling, every frame is
    kept: a weak step during warm-up, or a step at a recording gap, is not
    evidence that the load stopped.
    """
    n_frames = len(theta)
    all_indices = np.arange(n_frames, dtype=np.int64)
    total = np.nansum(np.where(valid[None], theta, 0.0), axis=(1, 2))
    mean = total / max(float(valid.sum()), 1.0)
    report: dict = {"n_frames": int(n_frames), "cut": False}
    if n_frames < max(8, minimum_frames):
        report["reason"] = "too few frames for a load-drop test; every frame kept"
        return all_indices, report
    try:
        rate = np.gradient(mean, times)
    except (ValueError, FloatingPointError):
        report["reason"] = "load-rate estimator failed; every frame kept"
        return all_indices, report
    step = np.diff(rate)
    delta_t = np.diff(times)
    median_dt = float(np.median(delta_t))
    peak = float(np.nanmax(mean))
    candidates: list[int] = []
    for index in np.flatnonzero(step < -0.02):
        index = int(index)
        if index < 2 or index >= n_frames - 2:
            continue
        if delta_t[index] > max(3.0 * median_dt, median_dt + 10.0):
            continue  # a recording gap, not a load stop
        if not (rate[index + 1] < rate[index] - 0.005 and rate[index + 1] < 0.0):
            continue
        candidates.append(index)
    strong = [
        index for index in candidates
        if mean[index] >= 0.80 * peak
        and mean[min(index + 3, n_frames - 1)] < mean[index] - 0.4
    ]
    pool = strong
    if not pool:
        report["reason"] = "no clear load drop; every frame kept"
        report["n_candidates"] = 0
        return all_indices, report
    index = max(pool, key=lambda item: -float(step[item]))
    report.update({
        "drop_frame": index,
        "drop_time_s": float(times[index]),
        "kink_k_per_s": float(step[index]),
        "rate_before_k_per_s": float(rate[index]),
        "rate_after_k_per_s": float(rate[index + 1]),
        "n_candidates": len(candidates),
        "n_strong": len(strong),
        "cut": True,
    })
    return all_indices[: index + 1], report


# --------------------------------------------------------------------------
# state matching
# --------------------------------------------------------------------------
def level_series(theta: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Median trusted-pixel temperature per frame: the state key."""
    return np.nanmedian(theta[:, pixels], axis=1)


def state_pairs(
    bare: RectifiedSession,
    painted: RectifiedSession,
    *,
    max_pairs: int = MAX_PAIRS,
    unknown_mask: np.ndarray | None = None,
) -> dict:
    """Pair bare and painted frames by thermal state, without any log.

    Each painted frame becomes one label, matched to the bare frame whose
    trusted-pixel level is closest.  Matching by state rather than by elapsed
    time is what the existing case16 (no logs, stepped load) experience
    supports: the label is the field, and the field is what must agree.

    ``unknown_mask`` lets the caller detect the footprint once and report it
    before pairing.  Passing it in changes no pairing result.
    """
    if bare.grid_hw != painted.grid_hw:
        raise UploadError("裸板与黑化板被配准到了不同的网格")
    if bare.n_frames < 4 or painted.n_frames < 4:
        raise UploadError("每组数据至少需要 4 帧")
    board = bare.valid & painted.valid
    if not board.any():
        raise UploadError("裸板与黑化板没有共同的板体像素")
    unknown = (
        detect_unknown_mask(bare.theta_k[bare.reference_index], board)
        if unknown_mask is None else np.asarray(unknown_mask, dtype=bool)
    )
    if float(unknown[board].mean()) > 0.75:
        raise UploadError("不可信区域超过板体的 75%，无法建立可靠配对")
    trusted = board & ~unknown
    masked = board & unknown
    if int(trusted.sum()) < 40:
        raise UploadError("可信板体像素太少，无法建立配对")

    bare_index, bare_report = powered_prefix(bare.theta_k, board, bare.times_s)
    painted_index, painted_report = powered_prefix(painted.theta_k, board, painted.times_s)
    bare_theta = bare.theta_k[bare_index]
    painted_theta = painted.theta_k[painted_index]

    bare_level = level_series(bare_theta, trusted)
    painted_level = level_series(painted_theta, trusted)
    bare_match = np.empty(len(painted_level), dtype=np.int64)
    for j, target in enumerate(painted_level):
        bare_match[j] = int(np.argmin(np.abs(bare_level - target)))

    x = bare_theta[bare_match]
    y = painted_theta
    residual = y - x
    order = np.linspace(0, len(y) - 1, min(max_pairs, len(y))).round().astype(np.int64)
    order = np.unique(order)
    def masked_mean(values: np.ndarray, mask: np.ndarray, take_abs: bool) -> float:
        if not mask.any():
            return 0.0
        selected = values[:, mask]
        return float(np.abs(selected).mean() if take_abs else selected.mean())

    diagnostics = {
        "n_pairs": int(len(order)),
        "n_bare_frames": int(bare.n_frames),
        "n_painted_frames": int(painted.n_frames),
        "bare_powered": bare_report,
        "painted_powered": painted_report,
        "level_gap_k_p95": float(np.percentile(np.abs(bare_level[bare_match] - painted_level), 95)),
        "trusted_mae_k": masked_mean(residual, trusted, True),
        "trusted_bias_k": masked_mean(residual, trusted, False),
        "masked_mae_k": masked_mean(residual, masked, True),
        "masked_bias_k": masked_mean(residual, masked, False),
        "board_mask_fraction": float(board.mean()),
        "unknown_fraction_of_board": float(unknown[board].mean()),
    }
    return {
        "x_theta": np.where(board[None], x[order], 0.0).astype(np.float32),
        "y_theta": np.where(board[None], y[order], 0.0).astype(np.float32),
        "board_mask": board,
        "unknown_mask": unknown,
        "bare_index": bare_index[bare_match[order]],
        "painted_index": painted_index[order],
        "bare_ambient_c": bare.ambient_c[bare_index][bare_match[order]],
        "painted_ambient_c": painted.ambient_c[painted_index][order],
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------
# registration signature
# --------------------------------------------------------------------------
def gradient_signature(field: np.ndarray) -> np.ndarray:
    """Unit gradient-magnitude map: structure without the thermal level.

    A component edge looks the same on a cold and a hot frame, so this is the
    right quantity for asking "is this the same board?" even when the upload's
    thermal state is unknown.
    """
    values = np.asarray(field, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        finite = values[np.isfinite(values)]
        fill = float(finite.mean()) if finite.size else 0.0
        values = np.where(np.isfinite(values), values, fill)
    gy, gx = np.gradient(gaussian_filter(values, 1.0))
    magnitude = np.hypot(gy, gx).ravel()
    magnitude = magnitude - magnitude.mean()
    norm = float(np.linalg.norm(magnitude))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.zeros(values.shape, dtype=np.float32)
    return (magnitude / norm).reshape(values.shape).astype(np.float32)


def _window_score(
    source: UploadedFrame,
    center_rc: Sequence[float],
    size_hw: Sequence[float],
    angle_deg: float,
    grid_hw: tuple[int, int],
    reference_signature: np.ndarray,
) -> float:
    center, size = fit_window_to_frame(center_rc, size_hw, angle_deg, source.shape_hw)
    try:
        crop = rectify_quad(
            source.temps, rotated_rect_corners(center, size, angle_deg), grid_hw
        )
    except (ValueError, FloatingPointError):
        return -1.0
    signature = gradient_signature(crop.astype(np.float64) - source.ambient_c)
    return float(np.asarray(signature, dtype=np.float64).ravel() @ reference_signature.ravel())


def register_window(
    reference_theta: np.ndarray,
    reference_window: dict,
    source: UploadedFrame,
    grid_hw: tuple[int, int],
    *,
    initial_window: dict | None = None,
    coarse: bool = True,
    score_floor: float = 0.20,
) -> dict:
    """Find the window in ``source`` that shows the *same board* as the reference.

    Detecting the painted or taped session independently is not safe.  On the
    Orange Pi tape run the detector found the 174 mm tape sheet, 23 % larger
    and 20 px away from the 141 mm PCB underneath; the two sessions would then
    have been paired on different physical areas.  This routine instead keeps
    the board window's structure: it rectifies candidate windows from the source
    frame and maximises the gradient correlation against the bare reference,
    which is the same structural check the desktop project used to correct that
    session by hand.

    The search starts from the source detector and from the reference window
    (the two sessions are often staged similarly), then runs a coarse scale and
    translation grid followed by a fine local pass.  It reports the achieved
    score so a non-matching pair becomes visible instead of silent.
    """
    reference_signature = gradient_signature(reference_theta)
    base = initial_window or reference_window
    center0 = np.asarray(base["center_rc"], dtype=np.float64)
    size0 = np.asarray(base["size_hw"], dtype=np.float64)
    angle = float(base["angle_deg"])
    best_score = -1.0
    best = (center0.copy(), size0.copy(), angle)
    for candidate_center, candidate_size, candidate_angle in (
        (center0, size0, angle),
        (
            np.asarray(reference_window["center_rc"], dtype=np.float64),
            np.asarray(reference_window["size_hw"], dtype=np.float64),
            float(reference_window["angle_deg"]),
        ),
    ):
        score = _window_score(
            source, candidate_center, candidate_size, candidate_angle,
            grid_hw, reference_signature,
        )
        if score > best_score:
            best_score = score
            best = (candidate_center.copy(), candidate_size.copy(), candidate_angle)
    center0, size0, angle = best

    def search(scales: np.ndarray, shifts: np.ndarray, step: float) -> None:
        nonlocal best_score, best
        offsets = np.arange(-shifts, shifts + 1e-9, step)
        for scale in scales:
            size = size0 * float(scale)
            if np.any(size < 8.0):
                continue
            for dr in offsets:
                for dc in offsets:
                    center = center0 + np.asarray([dr, dc])
                    score = _window_score(
                        source, center, size, angle, grid_hw, reference_signature
                    )
                    if score > best_score:
                        best_score = score
                        best = (center.copy(), size.copy(), angle)

    if coarse:
        search(np.arange(0.55, 1.051, 0.05), 24.0, 6.0)
    # Fine pass around the coarse winner.  Shifts are relative to the
    # winner, so this refines rather than restarting the search.
    center0, size0 = best[0].copy(), best[1].copy()
    fine_step = 1.0
    for scale in np.arange(0.94, 1.061, 0.02):
        size = size0 * float(scale)
        for dr in np.arange(-3.0, 3.01, fine_step):
            for dc in np.arange(-3.0, 3.01, fine_step):
                center = center0 + np.asarray([dr, dc])
                score = _window_score(
                    source, center, size, angle, grid_hw, reference_signature
                )
                if score > best_score:
                    best_score = score
                    best = (center.copy(), size.copy(), angle)
    center, size, angle = best
    return {
        "center_rc": [float(center[0]), float(center[1])],
        "size_hw": [float(size[0]), float(size[1])],
        "angle_deg": float(angle),
        "score": float(best_score),
        "source": "registered",
        "score_floor": float(score_floor),
        "scale_from_initial": float(size[0] / max(float(base["size_hw"][0]), 1e-9)),
    }


def signature_score(reference: np.ndarray, field: np.ndarray) -> float:
    """Zero-normalised correlation between two gradient signatures."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = gradient_signature(field).ravel().astype(np.float64)
    ref = ref - ref.mean()
    cur = cur - cur.mean()
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cur))
    if denominator <= 0.0:
        return 0.0
    return float(ref @ cur / denominator)
