"""Find where a board makes heat, and how fast that heat spreads and leaks away.

The measurement everything else rests on
========================================

An infrared movie of a warming board obeys, at every pixel::

    how fast this pixel warms  =  heat arriving from its neighbours
                               -  heat leaking into the air
                               +  heat made right here

Rearranged, the last term -- the one we want -- is measurable directly::

    heating rate = (warming rate) + (cooling rate) x (temperature above ambient)
                                  - (spread rate) x (local curvature)

All three inputs come from the images. Nothing here needs a conductivity, a heat
capacity, or any other material constant, because the whole equation has already
been divided through by the areal heat capacity. That division is what makes the
remaining quantities measurable, and it is also what removes the absolute scale:
these are rates in kelvin per second, never watts.

Why not just look for bumps in one frame
----------------------------------------

The obvious detector -- find the peaks of the spatial second derivative in a single
image -- fails in both directions on this board, measured on case10:

* On raw frames every peak it reports lands *inside* a low-emissivity footprint.
  Those are the edges of metal lids, not sources: a lid produces a step in apparent
  temperature, and the second derivative of a step is enormous.
* On repaired frames it finds nothing under the lids at all, because the repair
  fills them by solving the same equation with the heating rate set to zero. The
  answer is built into the input.

The time derivative has neither problem. A lid biases the *level* it reports and
scales the *rate* by a static factor (see :func:`emissivity_gain`), but it does not
invent or move a peak, so a source under a lid stays visible and stays put.

Two estimators, deliberately
----------------------------

:func:`measure_heating_rate` offers two, and they are meant to be compared rather
than chosen between:

``rise``
    Least squares for one constant heating rate per pixel across the whole warming
    stretch. Measured on case10 this is stable in a way a short window is not --
    the ranked hot-spot list moves by 1-3 px between using 10 frames and 70, and
    the strongest rate drifts only 0.417 -> 0.352 K/s.

``onset``
    The first few frames only, where warming dominates both other terms. Sharper
    (peak-to-median contrast 10.9 against 2.2) and noisier. It leans much less on
    the assumed cooling rate, so agreement between the two is evidence that the hot
    spots are real rather than an artifact of that assumption.

On case10 the two agree on hot-spot position to 1-3 px.

What the cooling rate does and does not affect
----------------------------------------------

It enters as a smooth, near-uniform addition, so it barely moves a peak's position
but directly sets how the total splits between concentrated hot spots and the broad
background. Positions are robust; the split is not. Sweep it and report the spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.optimize import curve_fit

# The camera's own emissivity setting for this campaign. It is an assumption, not a
# recorded parameter -- the 2026-07-26/27 exports carry no radiometric block -- and
# it is the definition of "true" for well-behaved pixels rather than a fitted value.
ASSUMED_EMISSIVITY = 0.95

# Radiance goes as (T + 273.15) ** BAND_EXPONENT over this camera's band.
BAND_EXPONENT = 4.0

# A rectified board fills the whole grid, but the outermost pixels mix board with
# background through the rectification's bilinear sampling, and that mixing produces
# spurious sources: at a 3 px margin, five of the seven unconfirmed hot spots on
# case10 sat within 8 px of the border and none of them showed any answering signal
# in the independent estimator. Eight is where they stop.
BORDER_MARGIN_PX = 8


# --------------------------------------------------------------------------
# the powered stretch
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadDrop:
    """Where the load stopped, found from the images alone."""

    frame_index: int  # last frame that is still powered
    time_s: float
    kink_k_per_s: float  # size of the step in warming rate
    report: dict[str, Any] = field(default_factory=dict)


def find_load_drop(board_mean_theta: np.ndarray, times_s: np.ndarray) -> LoadDrop:
    """Locate the frame after which the board stops being driven.

    The board-mean warming rate steps sharply when the load stops and nowhere else,
    so the largest single-frame change in that rate locates it. On case10 this lands
    between frames 239 and 240 -- the true switch -- where the rate jumps from
    +0.009 to -0.089 K/s. The existing phase detector puts the transition at frame
    232 instead, which leaves seven still-powered frames inside what it calls the
    decay.

    Returns the last frame that is still powered, so ``theta[: drop.frame_index + 1]``
    is the powered stretch.
    """
    theta = np.asarray(board_mean_theta, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if theta.ndim != 1 or theta.shape != times.shape:
        raise ValueError(f"expected matching 1-D arrays, got {theta.shape} and {times.shape}")
    if theta.size < 8:
        raise ValueError(f"need at least 8 frames to locate a load drop, got {theta.size}")

    rate = np.gradient(theta, times)
    step = np.diff(rate)
    # A drop makes the rate fall, so search for the most negative step, not the
    # largest magnitude -- a warm-up transient at frame 0 can rival it in size.
    index = int(np.argmin(step))
    return LoadDrop(
        frame_index=index,
        time_s=float(times[index]),
        kink_k_per_s=float(step[index]),
        report={
            "rate_before_k_per_s": float(rate[index]),
            "rate_after_k_per_s": float(rate[index + 1]),
            "n_powered_frames": index + 1,
            "note": "largest fall in board-mean warming rate; no external log used",
        },
    )


# --------------------------------------------------------------------------
# cooling rate -- how fast heat leaks into the air
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CoolingFit:
    """How fast the board sheds heat, from the shape of its warm-up curve."""

    cooling_rate_per_s: float
    cooling_rate_stderr: float
    settled_theta_k: float  # where the board levels off, above ambient
    start_theta_k: float
    response_time_s: float  # 1 / cooling_rate
    residual_rmse_k: float
    n_frames: int
    report: dict[str, Any] = field(default_factory=dict)


def fit_cooling_rate(board_mean_theta: np.ndarray, times_s: np.ndarray) -> CoolingFit:
    """Fit the levelling-off curve of a warming board.

    A board driven at constant power approaches its settled level exponentially::

        theta(t) = settled + (start - settled) * exp(-cooling_rate * t)

    Both the settled level and the cooling rate come out of that one curve, so the
    cooling stretch after the load stops is not needed. Measured on case10's powered
    frames alone: cooling rate 0.00685 /s (response time 146 s) against 0.00762 /s
    from the discarded cooling data -- a 10 % difference. The fitted settled level
    lands at 26.63 K against a measured 26.70 K, and the fitted start at 13.70 K
    against the board's own measured idle level of 13.90 K, which the fit never saw.
    """
    theta = np.asarray(board_mean_theta, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if theta.ndim != 1 or theta.shape != times.shape:
        raise ValueError(f"expected matching 1-D arrays, got {theta.shape} and {times.shape}")
    if theta.size < 10:
        raise ValueError(f"need at least 10 frames to fit a warm-up curve, got {theta.size}")

    elapsed = times - times[0]

    def model(t, settled, rate, start):
        return settled + (start - settled) * np.exp(-rate * t)

    span = float(elapsed[-1]) or 1.0
    guess = [float(theta[-1]), 3.0 / span, float(theta[0])]
    # Rate is bounded away from zero because a flat line fits any rate equally well
    # and the optimiser will happily wander to 1e-12 and report a 30 000 s board.
    bounds = ([-np.inf, 1e-5, -np.inf], [np.inf, 1.0, np.inf])
    try:
        params, covariance = curve_fit(model, elapsed, theta, p0=guess, bounds=bounds, maxfev=20000)
    except RuntimeError as exc:  # pragma: no cover - depends on data pathology
        raise ValueError(f"warm-up curve did not converge: {exc}") from exc

    settled, rate, start = (float(v) for v in params)
    stderr = float(np.sqrt(np.diag(covariance))[1]) if np.all(np.isfinite(covariance)) else float("nan")
    residual = model(elapsed, *params) - theta
    return CoolingFit(
        cooling_rate_per_s=rate,
        cooling_rate_stderr=stderr,
        settled_theta_k=settled,
        start_theta_k=start,
        response_time_s=1.0 / rate,
        residual_rmse_k=float(np.sqrt(np.mean(residual**2))),
        n_frames=int(theta.size),
        report={
            "measured_final_theta_k": float(theta[-1]),
            "fraction_settled_at_last_frame": float(
                1.0 - np.exp(-rate * elapsed[-1]) if np.isfinite(rate) else np.nan
            ),
            "note": "fitted on powered frames only; no cooling-phase data used",
        },
    )


# --------------------------------------------------------------------------
# spread rate -- how fast heat creeps sideways
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SpreadFit:
    """How fast heat travels across the board surface, from a widening hot spot."""

    spread_rate_px2_per_s: float
    spread_rate_ci: tuple[float, float]
    intrinsic_width_px: float  # the hot spot's own size, before any spreading
    spread_distance_px: float  # sqrt(spread rate / cooling rate)
    spread_distance_mm: float | None
    valid_size_range_px: tuple[float, float]
    windows_s: tuple[float, ...]
    second_moments_px2: tuple[float, ...]
    report: dict[str, Any] = field(default_factory=dict)


def _warming_rate(theta: np.ndarray, times_s: np.ndarray, n_frames: int, smooth_px: float) -> np.ndarray:
    """Least-squares slope of each pixel's temperature over the first n frames."""
    span = times_s[:n_frames]
    values = theta[:n_frames]
    centred = span - span.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0:
        raise ValueError("cannot take a slope over zero elapsed time")
    slope = (centred[:, None, None] * (values - values.mean(axis=0))).sum(axis=0) / denominator
    return gaussian_filter(slope, smooth_px) if smooth_px > 0 else slope


def _gaussian_width_sq(
    rate: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    near: np.ndarray,
    row0: int,
    col0: int,
) -> float | None:
    """Squared width of a hot spot, by fitting a bump on a sloping background.

    A raw second moment was tried first and is not usable here: it depends on where
    the background baseline is put, and on case10 that dependence made the apparent
    width *shrink* with time for four of six candidate hot spots -- the opposite of
    diffusion, and physically impossible. Fitting the bump and the background
    together removes the baseline choice from the answer.

    The background is allowed a linear tilt because a hot spot sits on the flank of
    the board-wide warm region, and forcing it flat pushes that tilt into the width.
    """
    r = rows[near].astype(np.float64)
    c = cols[near].astype(np.float64)
    z = rate[near].astype(np.float64)

    def model(coords, amp, r_c, c_c, width, base, slope_r, slope_c):
        rr, cc = coords
        bump = amp * np.exp(-(((rr - r_c) ** 2 + (cc - c_c) ** 2) / (2.0 * width**2)))
        return bump + base + slope_r * (rr - row0) + slope_c * (cc - col0)

    span = float(z.max() - z.min())
    guess = [span if span > 0 else 1.0, float(row0), float(col0), 3.0, float(np.median(z)), 0.0, 0.0]
    bounds = (
        [0.0, row0 - 6.0, col0 - 6.0, 0.8, -np.inf, -np.inf, -np.inf],
        [np.inf, row0 + 6.0, col0 + 6.0, 25.0, np.inf, np.inf, np.inf],
    )
    try:
        params, _ = curve_fit(model, (r, c), z, p0=guess, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    amp, _, _, width, *_ = params
    if amp <= 0 or not np.isfinite(width):
        return None
    return float(width) ** 2


def fit_spread_rate(
    theta: np.ndarray,
    times_s: np.ndarray,
    hotspot_rc: tuple[int, int],
    *,
    n_frames_per_window: tuple[int, ...] = (3, 4, 5, 6, 8, 10, 12),
    radius_px: float = 14.0,
    smooth_px: float = 1.0,
    cooling_rate_per_s: float | None = None,
    pixel_size_mm: float | None = None,
) -> SpreadFit:
    """Measure how fast heat spreads by watching a hot spot widen.

    A patch of heat released into a conducting sheet grows so that its second moment
    climbs linearly::

        second moment = intrinsic width^2 + 2 x (spread rate) x time

    so the slope of that line *is* the spread rate. Nothing else is needed -- no
    conductivity, no heat capacity, no assumption about the source's shape.

    The intercept is not a nuisance: it is the hot spot's own physical size. On
    case10's strongest hot spot the intercept comes out at 5.9 px (3.2 mm), which is
    a sensible package footprint, and the second moment climbs only 34.4 -> 38.7 px^2
    between 6.5 s and 34.5 s. That slow growth is the whole finding -- heat barely
    travels on this surface.

    A cross-check that needs no fitting: case10's two strongest hot spots sit 28 px
    apart and stay separately resolvable over 13 s. Heat that spread fast enough to
    make the board near-isothermal would have merged them within that window.

    ``hotspot_rc`` should be the strongest hot spot from a first pass of
    :func:`measure_heating_rate`; the measurement is local to it.
    """
    theta = np.asarray(theta, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if theta.ndim != 3:
        raise ValueError(f"expected a (frames, rows, cols) stack, got shape {theta.shape}")
    windows = tuple(int(w) for w in n_frames_per_window if 3 <= int(w) <= theta.shape[0])
    if len(windows) < 3:
        raise ValueError(f"need at least 3 usable windows, got {len(windows)} from {n_frames_per_window}")

    row0, col0 = int(hotspot_rc[0]), int(hotspot_rc[1])
    rows, cols = np.mgrid[0 : theta.shape[1], 0 : theta.shape[2]]
    near = np.hypot(rows - row0, cols - col0) <= radius_px
    if near.sum() < 30:
        raise ValueError("hot-spot neighbourhood is too small to take a second moment")

    mid_times: list[float] = []
    moments: list[float] = []
    for window in windows:
        rate = _warming_rate(theta, times, window, smooth_px)
        width_sq = _gaussian_width_sq(rate, rows, cols, near, row0, col0)
        if width_sq is None:
            continue
        moments.append(width_sq)
        # The slope averages the source's signature over the window, so the width
        # corresponds to the middle of it, not its end.
        mid_times.append(float(times[window - 1] - times[0]) / 2.0)

    if len(moments) < 3:
        raise ValueError("hot spot never rose above its background; cannot measure spreading")

    t_eff = np.asarray(mid_times)
    second_moment = np.asarray(moments)
    design = np.vstack([t_eff, np.ones_like(t_eff)]).T
    (slope, intercept), residuals, *_ = np.linalg.lstsq(design, second_moment, rcond=None)
    spread_rate = float(slope) / 2.0

    # Standard error of the slope, propagated to the spread rate.
    predicted = design @ np.array([slope, intercept])
    dof = max(len(t_eff) - 2, 1)
    sigma_sq = float(((second_moment - predicted) ** 2).sum()) / dof
    slope_var = sigma_sq * float(np.linalg.inv(design.T @ design)[0, 0])
    slope_stderr = float(np.sqrt(max(slope_var, 0.0)))
    half = 1.96 * slope_stderr / 2.0
    # Spreading cannot be negative. A negative point estimate means the widening is
    # below what this measurement can resolve, so the honest report is zero with an
    # upper bound -- not a negative rate, and not a silently flipped interval.
    ci = (max(spread_rate - half, 0.0), max(spread_rate + half, 0.0))
    resolved = spread_rate > half
    spread_rate = max(spread_rate, 0.0)

    intrinsic = float(np.sqrt(max(intercept, 0.0)))
    distance_px = float("nan")
    if cooling_rate_per_s and cooling_rate_per_s > 0:
        distance_px = float(np.sqrt(max(spread_rate, 0.0) / cooling_rate_per_s))

    # The measurement only constrains spreading over the sizes it actually watched.
    sizes = np.sqrt(second_moment)
    return SpreadFit(
        spread_rate_px2_per_s=spread_rate,
        spread_rate_ci=ci,
        intrinsic_width_px=intrinsic,
        spread_distance_px=distance_px,
        spread_distance_mm=distance_px * pixel_size_mm if pixel_size_mm else None,
        valid_size_range_px=(float(sizes.min()), float(sizes.max())),
        windows_s=tuple(float(t) for t in t_eff),
        second_moments_px2=tuple(float(m) for m in second_moment),
        report={
            "hotspot_rc": [row0, col0],
            "slope_px2_per_s": float(slope),
            "slope_stderr_px2_per_s": slope_stderr,
            "intercept_px2": float(intercept),
            "resolved": bool(resolved),
            "note": (
                "spreading measured only over the size range listed; it is not "
                "established outside it, and it is a rate (px^2/s), not a diffusivity"
            ),
        },
    )


# --------------------------------------------------------------------------
# emissivity gain -- what a metal lid does to a measured rate
# --------------------------------------------------------------------------
def emissivity_gain(
    emissivity: np.ndarray,
    apparent_c: np.ndarray,
    true_c: np.ndarray,
    *,
    assumed_emissivity: float = ASSUMED_EMISSIVITY,
    band_exponent: float = BAND_EXPONENT,
) -> np.ndarray:
    """Factor converting an apparent warming rate into a true one.

    The camera inverts one fixed emissivity, so over a poorly-radiating patch the
    reported radiance is a blend of the surface and its surroundings::

        W(true) = k W(apparent) + (1 - k) W(reflected),   k = assumed / actual

    The reflected term is static, so differentiating in time drops it entirely and
    leaves a clean multiplicative factor::

        d(true)/dt = k ((apparent + 273.15) / (true + 273.15))^(n-1) d(apparent)/dt

    Two consequences, and the second is why this method works at all:

    * A lid understates warming badly. For a fitted emissivity near 0.3 the factor is
      about 2.3, so a rate measured over a lid is less than half the real one.
    * The factor is *static and multiplicative*. It rescales a hot spot's amplitude
      but cannot move it, widen it, or invent one -- so localisation under a lid is
      unaffected, and only the amplitude needs correcting.

    On well-behaved pixels the actual emissivity is the assumed one and the apparent
    temperature is the true one, so this returns exactly 1.0 and is a no-op.
    """
    emissivity = np.asarray(emissivity, dtype=np.float64)
    apparent = np.asarray(apparent_c, dtype=np.float64)
    true = np.asarray(true_c, dtype=np.float64)
    if not 0.0 < assumed_emissivity <= 1.0:
        raise ValueError(f"assumed_emissivity must be in (0, 1], got {assumed_emissivity}")

    safe = np.clip(emissivity, 1e-3, 1.0)
    ratio = assumed_emissivity / safe
    kelvin_apparent = np.clip(apparent + 273.15, 1.0, None)
    kelvin_true = np.clip(true + 273.15, 1.0, None)
    return ratio * (kelvin_apparent / kelvin_true) ** (band_exponent - 1.0)


# --------------------------------------------------------------------------
# heating rate -- where the board makes heat
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatingRateMap:
    """Per-pixel rate at which the board generates heat, in K/s."""

    rate_k_per_s: np.ndarray  # (H, W)
    method: str  # "rise" or "onset"
    n_frames: int
    cooling_rate_per_s: float
    spread_rate_px2_per_s: float
    gain_corrected: bool
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def contrast(self) -> float:
        """Peak over median -- how concentrated the map is."""
        median = float(np.median(self.rate_k_per_s))
        return float(np.max(self.rate_k_per_s) / median) if median > 0 else float("inf")


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Five-point curvature with mirrored edges, i.e. no heat crosses the border.

    A PCB loses heat from its faces, not sideways out of the frame, so mirroring is
    the physically right edge condition and it matches the repair solver's own.
    """
    padded = np.pad(field, 1, mode="reflect")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * field
    )


def measure_heating_rate(
    theta_apparent: np.ndarray,
    times_s: np.ndarray,
    *,
    cooling_rate_per_s: float,
    spread_rate_px2_per_s: float,
    theta_true: np.ndarray | None = None,
    gain: np.ndarray | None = None,
    method: str = "rise",
    n_frames: int = 70,
    smooth_px: float = 1.5,
) -> HeatingRateMap:
    """Measure where the board makes heat, from how its temperature evolves.

    Rearranging the governing equation puts the unknown alone on one side::

        heating rate = (warming rate) + (cooling rate) x (temperature above ambient)
                                      - (spread rate) x (curvature)

    ``method="rise"`` solves that for one constant heating rate per pixel across the
    whole warming stretch. Because the load is steady, a constant is the right model
    and every frame is another observation of it, which is what makes it stable: on
    case10 the ranked hot-spot list moves 1-3 px between 10 frames and 70, and the
    strongest rate drifts only 0.417 -> 0.352 K/s.

    ``method="onset"`` uses the first few frames only, where the warming term towers
    over the other two. It barely depends on the assumed cooling rate, which makes it
    the right cross-check rather than a second-best: agreement between the two is
    evidence about the hot spots, not about the assumption.

    **Use the raw field and leave ``theta_true`` and ``gain`` unset.** That is the
    supported path, and it is the one that needs no emissivity estimate anywhere. On
    well-behaved pixels the raw reading *is* the true temperature, so all three terms
    are correct as they stand. Under a metal lid the reading is wrong, but it is wrong
    by a static multiplicative factor, so a source underneath still produces a peak in
    the same place -- its strength comes out too low, and :func:`find_hotspots` marks
    it as a lower bound rather than pretending otherwise.

    The two optional arguments exist for a diagnostic comparison and both carry real
    hazards, measured on case10:

    * ``gain`` multiplies only the warming term (see :func:`emissivity_gain`), never
      the finished expression -- scaling the whole thing inflates the broad background
      under every lid by the same factor as the sources, which promoted all twenty
      leading hot spots to lid pixels and left none confirmed by the cross-check.
    * ``theta_true`` feeds the level and curvature terms an estimate of the real
      temperature. Supplying the repaired field is circular -- the repair was produced
      by assuming a heating rate -- and it adds ``cooling rate x 15 K`` of spurious
      source under every lid, roughly half the real signal.

    Both need an emissivity estimate to mean anything, and estimating emissivity is
    the step this design exists to avoid.
    """
    theta_apparent = np.asarray(theta_apparent, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if theta_apparent.ndim != 3:
        raise ValueError(f"expected a (frames, rows, cols) stack, got shape {theta_apparent.shape}")
    if method not in ("rise", "onset"):
        raise ValueError(f"method must be 'rise' or 'onset', got {method!r}")
    if cooling_rate_per_s < 0:
        raise ValueError(f"cooling rate must be non-negative, got {cooling_rate_per_s}")

    usable = min(int(n_frames), theta_apparent.shape[0])
    if usable < 3:
        raise ValueError(f"need at least 3 frames, got {usable}")

    level = theta_apparent if theta_true is None else np.asarray(theta_true, dtype=np.float64)
    if level.shape != theta_apparent.shape:
        raise ValueError(f"theta_true {level.shape} does not match theta_apparent {theta_apparent.shape}")

    corrected = False
    if gain is not None:
        gain = np.asarray(gain, dtype=np.float64)
        if gain.shape != theta_apparent.shape[1:]:
            raise ValueError(f"gain {gain.shape} does not match the board {theta_apparent.shape[1:]}")
        corrected = True

    if method == "onset":
        # Warming dominates here, so read it straight off without correcting for the
        # other two terms -- that independence is exactly what makes it a cross-check.
        rate = _warming_rate(theta_apparent, times, usable, smooth_px)
        if corrected:
            rate = rate * gain
    else:
        # One observation of the constant per interior frame; averaging them is the
        # least-squares solution and keeps the estimator transparent.
        contributions = []
        for index in range(1, usable - 1):
            step = float(times[index + 1] - times[index - 1])
            if step <= 0:
                continue
            measured = (theta_apparent[index + 1] - theta_apparent[index - 1]) / step
            if corrected:
                measured = measured * gain
            smoothed = gaussian_filter(level[index], 1.0)
            contributions.append(
                measured + cooling_rate_per_s * level[index] - spread_rate_px2_per_s * _laplacian(smoothed)
            )
        if not contributions:
            raise ValueError("no usable interior frames; check that times increase")
        rate = gaussian_filter(np.mean(contributions, axis=0), smooth_px)

    return HeatingRateMap(
        rate_k_per_s=rate,
        method=method,
        n_frames=usable,
        cooling_rate_per_s=float(cooling_rate_per_s),
        spread_rate_px2_per_s=float(spread_rate_px2_per_s),
        gain_corrected=corrected,
        report={
            "median_k_per_s": float(np.median(rate)),
            "max_k_per_s": float(np.max(rate)),
            "min_k_per_s": float(np.min(rate)),
            "negative_fraction": float((rate < 0).mean()),
            "note": "rate in K/s, i.e. power divided by an unknown heat capacity; not power",
        },
    )


# --------------------------------------------------------------------------
# hot spots
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Hotspot:
    """One place the board makes heat, at this method's spatial resolution."""

    rank: int
    row: int
    col: int
    row_mm: float | None
    col_mm: float | None
    heating_rate_k_per_s: float
    area_px: int
    width_px: float
    under_footprint: bool  # sits beneath a low-emissivity lid
    confidence_sd: float | None = None  # height above background in the second estimator
    stable: bool | None = None


def find_hotspots(
    heating_map: HeatingRateMap,
    *,
    footprint_mask: np.ndarray | None = None,
    relative_threshold: float = 0.45,
    min_area_px: int = 6,
    separation_px: int = 13,
    background_px: float = 12.0,
    pixel_size_mm: float | None = None,
) -> list[Hotspot]:
    """Rank the places that make heat, strongest first.

    A hot spot is a local maximum of the heating-rate map *above its broad
    background* that clears ``relative_threshold`` of the strongest such peak and
    owns at least ``min_area_px`` contiguous pixels.

    Subtracting a background on the scale of ``background_px`` is what separates
    "this spot makes heat" from "this region is warm". The board generates heat over
    most of its area, so ranking on raw height instead returns the broad warm zone
    and floods every half-height contour across the board.

    The count is threshold-dependent and should never be quoted bare: on case10 the
    same board yields 14 hot spots from a 10-frame map and 25 from a 70-frame one.
    What survives every window and both estimators -- about seven on case10 -- is the
    defensible statement. Use :func:`cross_check` to mark which ones those are.

    A hot spot is also not a component. Two packages closer together than
    ``separation_px`` merge into one.
    """
    rate = heating_map.rate_k_per_s
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError(f"relative_threshold must be in (0, 1), got {relative_threshold}")

    interior = np.zeros(rate.shape, dtype=bool)
    interior[BORDER_MARGIN_PX:-BORDER_MARGIN_PX, BORDER_MARGIN_PX:-BORDER_MARGIN_PX] = True
    if not interior.any():
        return []

    # Rank on height above the broad background, not on absolute rate. The board
    # makes heat almost everywhere -- on case10 the background is roughly half the
    # total -- so an absolute threshold ranks "warm region" rather than "source",
    # and an absolute half-height contour floods across most of the board.
    background = gaussian_filter(rate, background_px)
    excess = rate - background
    peak_value = float(np.max(excess[interior]))
    if peak_value <= 0:
        return []

    is_peak = (
        (excess == maximum_filter(excess, size=separation_px))
        & (excess > relative_threshold * peak_value)
        & interior
    )
    rows, cols = np.nonzero(is_peak)
    if rows.size == 0:
        return []

    # One labelling for all hot spots: the half-height contour is taken on the
    # background-subtracted map, so each blob is a source's own extent.
    blobs, _ = label(excess > 0.5 * peak_value * relative_threshold)
    order = np.argsort(-excess[rows, cols])
    spots: list[Hotspot] = []
    for rank, index in enumerate(order, start=1):
        row, col = int(rows[index]), int(cols[index])
        height = float(rate[row, col])
        area = int((blobs == blobs[row, col]).sum()) if blobs[row, col] > 0 else 0
        if area < min_area_px:
            continue
        spots.append(
            Hotspot(
                rank=rank,
                row=row,
                col=col,
                row_mm=row * pixel_size_mm if pixel_size_mm else None,
                col_mm=col * pixel_size_mm if pixel_size_mm else None,
                heating_rate_k_per_s=height,
                area_px=area,
                width_px=float(np.sqrt(area / np.pi) * 2.0),
                under_footprint=bool(footprint_mask[row, col]) if footprint_mask is not None else False,
            )
        )
    return spots


def cross_check(
    primary: list[Hotspot],
    confirming: HeatingRateMap,
    *,
    inner_px: float = 10.0,
    outer_px: float = 20.0,
    strong_sd: float = 3.0,
) -> list[Hotspot]:
    """Score each hot spot by how strongly a second, independent estimator sees it.

    The two estimators lean on different assumptions -- the rise map needs a cooling
    rate and a whole warming stretch, the onset map needs neither -- so a bump the
    second one also shows is evidence about the board rather than about the first
    one's assumptions.

    Scoring is deliberately *not* "did the other map also rank this in its top N".
    That question conflates absence with low rank and gave a badly pessimistic answer
    on case10: the onset map is spiky, so a threshold relative to its own maximum
    admits only four peaks and everything below them scores zero regardless of
    whether a real bump is there. Measuring each hot spot's height above its own
    surrounding annulus, in standard deviations of that annulus, asks the right
    question and separates cleanly -- on case10 into +3.8 and +3.1 for two hot spots,
    +0.5 to +1.1 for three marginal ones, and zero or negative for the rest.

    ``confidence_sd`` is the number to report and to rank on. ``stable`` is only a
    convenience threshold at ``strong_sd``; a hot spot at +1.1 is not a failure, it is
    a weaker claim, and the distinction should reach the reader.
    """
    rate = confirming.rate_k_per_s
    rows, cols = np.mgrid[0 : rate.shape[0], 0 : rate.shape[1]]
    scored: list[Hotspot] = []
    for spot in primary:
        distance = np.hypot(rows - spot.row, cols - spot.col)
        ring = (distance > inner_px) & (distance <= outer_px)
        if not ring.any():
            scored.append(Hotspot(**{**spot.__dict__, "confidence_sd": None, "stable": None}))
            continue
        background = float(np.median(rate[ring]))
        spread = float(np.std(rate[ring]))
        excess = (float(rate[spot.row, spot.col]) - background) / spread if spread > 0 else 0.0
        scored.append(
            Hotspot(**{**spot.__dict__, "confidence_sd": excess, "stable": bool(excess >= strong_sd)})
        )
    return scored


def cooling_rate_map(
    heating_map: HeatingRateMap,
    settled_theta: np.ndarray,
    *,
    min_theta_k: float = 5.0,
    max_curvature: float = 0.02,
) -> np.ndarray:
    """Cooling rate per pixel, with no fitting at all.

    Once the heating rate is known, the settled state says that what a pixel makes it
    must also shed::

        cooling rate = heating rate / temperature above ambient

    This holds wherever curvature is negligible, which on this board is most of it.
    Pixels that fail either guard come back as NaN rather than as a confident wrong
    number -- near a hot spot the sideways term is not negligible and this would
    silently absorb it into the cooling rate.
    """
    settled = np.asarray(settled_theta, dtype=np.float64)
    if settled.shape != heating_map.rate_k_per_s.shape:
        raise ValueError(f"settled field {settled.shape} does not match the map {heating_map.rate_k_per_s.shape}")

    curvature = np.abs(_laplacian(gaussian_filter(settled, 1.0)))
    usable = (settled > min_theta_k) & (curvature < max_curvature)
    result = np.full(settled.shape, np.nan)
    np.divide(heating_map.rate_k_per_s, settled, out=result, where=usable)
    return result
