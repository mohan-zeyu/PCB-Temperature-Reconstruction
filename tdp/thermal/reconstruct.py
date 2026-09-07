"""Build a full temperature map for every frame, driven by the heat sources.

The old fill solved the plate equation across a low-emissivity block with no source
underneath it.  That makes a peak in the middle of a block mathematically impossible
-- a source-free solution attains its extremes on its boundary -- so the main chip
always came out colder than its own edges.

The spatial method was measured on case17, the same board painted matte black, so
the components case10 has to guess at are *measured* there.  Imposing case10's
mask on case17 and reconstructing the hidden area gave

    edge-only fill, no source   3.70 K mean absolute error, bias -3.56 K
    source-driven, uniform      2.17 K mean absolute error, bias -0.51 K
    provisional centre-         1.79 K mean absolute error, bias -0.09 K
      weighted source
    broad-source plate lid      1.77 K mean absolute error, bias -0.09 K

case17 is also an input to the current case10 level calibration, so this is a
same-session shape diagnostic rather than a held-out validation score.  The bias
is still the point.  The old fill was not merely noisy under a lid, it was
systematically 3 K cold, every frame, in the same direction -- which is exactly the
error a missing source makes.  That bias is now gone.

What is solved, per frame
-------------------------
The governing relation, already divided by areal heat capacity, is

    (warming rate) = (spread rate) x curvature - (cooling rate) x temperature
                     + (heating rate)

Move the warming rate to the other side and it becomes a steady plate equation
whose source is ``heating rate - warming rate``:

    (spread rate) x curvature - (cooling rate) x temperature
        + [heating rate - warming rate] = 0

That rearrangement is exact, not a quasi-steady approximation: nothing is dropped,
the transient just moves into the source.  It matters because the board is never
cold -- frame 0 already sits 11.3 K above ambient on case10 -- so there is no known
initial condition to march from, and a per-frame solve needs none.

The bracketed quantity is what this module calls the block's **source rate**, in
K/s.  On trusted board both parts are measured (:func:`tdp.thermal.sources.
measure_heating_rate` and :func:`warming_rate`).  Under a block neither is: the
reading there is wrong by an unknown emissivity factor, and so is its time
derivative.  So one source rate is fitted per block per frame.

Why the two parts are not separated under a block
-------------------------------------------------
They cannot be, from one frame.  Only their difference enters the equation.  The
heating rate is recovered afterwards, once the whole sequence is reconstructed, by
adding back the warming rate of the *reconstructed* field -- see
:func:`heating_rate_from_reconstruction`.  Its being near-constant over a run at
constant load is then a check, not an assumption.

How one number per block is pinned down
---------------------------------------
A ring of trusted board around every block is held out of the solve.  Dirichlet
data is taken from beyond it, and the source rates are chosen to reproduce the
ring's measured temperature.  This is the same evidence
:func:`tdp.thermal.sources.judge_footprints` uses qualitatively -- a block that
makes heat pushes the board around it up -- turned into a number.

The ring has to be held out.  Feeding the ring its own measured heating rate while
using it as Dirichlet data is an identity that constrains nothing: the measured
heating map was *defined* by the plate equation applied to the measured field, so
putting it back in reproduces that field whatever the block does.

Everything is linear, so the whole sequence costs one solve per frame plus one per
block, not one per frame per block: with zero boundary data the solve is a fixed
influence field per block, and the frame only enters through its Dirichlet ring.
All 240 case10 frames reconstruct in about 5 s on a CPU.

How a heat-producing lid is shaped
----------------------------------
The distance-weighted source remains useful as a *provisional* solve: the exterior
ring can constrain the block's mean level, and centre weighting improved that level
over a uniform source.  It cannot determine the internal pattern.  On case17 the
imposed bell was several kelvin cold through the broad area around the SoC and hot
at the very centre -- a small die and a spreading lid had been collapsed into one
plate domain.

The production lid fill replaces the provisional shape with one physical plate
solution.  A positive source is distributed over the whole SoC footprint.  Without
an anchor its amplitude preserves the provisional fitted mean; with an anchor it
matches the supplied central surface temperature.  The solve includes one extra
rim pixel because the nominally
trusted first ring on case10 is still cold.  There is no pasted curve or flat core:
the broad plateau, its remaining curvature, and the transition into the board all
come from the same discrete conduction equation.  In the current case17
within-session diagnostic this changes the SoC error 2.93 -> 2.78 K and the whole
hidden-area error 1.79 -> 1.77 K without removing the source-driven plateau.  See
:func:`_broad_source_lids`.

For an unanchored lid, the old source-shaped result is kept as the alternative
model in the reported uncertainty range.  For an anchored lid, the deliberate
level correction is reported separately and the declared anchor uncertainty is
combined with the local ring/shape range.  The old result can also be requested
directly as an ablation.

What this cannot do, measured
-----------------------------
Fitting the source rates to case17's *hidden truth* instead of to the ring -- the
best any per-block constant could do -- gives 1.10 K, of which 1.09 K is shape and
0.15 K level.  So no extra *source* parameter can help the pattern.

The remaining level gap is a real information limit rather than evidence that the
ring fit is numerical truth.  Three estimators agreeing only shows that they all
extract the same exterior information; it does not identify the hidden lid level.
The case10 driver therefore supplies the independently calibrated surface anchor.

Reading structure *inside* a block was the obvious remaining source of information,
and it was tried four ways and loses (all scored on case17, gain and reflection
offset calibrated from the block's rim, never from truth):

* add the pattern that grew over the run, which is free of reflection because a
  fixed lid reflecting a fixed room is constant in time: 2.63 -> 2.44 K at the very
  best, and mean absolute error does not move at all while a level offset dominates.
* use only reflection-free time differences: 2.60 K.  97 % of the squared error is
  static, so differences can only reach the other 3 %.
* rebuild a block straight from its lid reading: 6.59 K, because the gain is
  recoverable only to 20 % (worst 98 %) -- the only reference for the true
  temperature under a block is the fill being corrected.
* take only the level from the lid reading: 4.83 K.

The last two fail for one structural reason: the lid's static reflection offset and
the fill's own static level error are the same unknown, and the lid reading gives
one static equation per block for the two of them.

Units
-----
Temperatures are kelvin above ambient, rates are per second, distances are pixels
unless a name says ``_mm``.  Nothing here is a power, and it cannot be turned into
one -- the areal heat capacity that was divided out is unknown and varies across
the board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label

from tdp.data.emissivity import CROSS, solve_plate

# A ring this wide is the widest that still leaves trusted board between the
# closer blocks on case10, and at 8 px the block's own influence has only decayed
# to exp(-8/13) = 0.54 across it, so the ring still carries the signal being fitted.
DEFAULT_RING_PX = 8

# Curvature is taken on a field smoothed by this much.  The CSV exports are
# quantised to 0.1 C, which the five-point stencil amplifies: unsmoothed, the
# plate equation fails to close on clean board by 0.256 K/s rms, three times the
# heating map's own 0.082 K/s.  At 1.5 px the residual falls to 0.041 K/s.
DEFAULT_SMOOTH_PX = 1.5

# Blocks smaller than this cannot be told apart from ring noise -- their influence
# field over the ring is too weak to fit against -- so they are filled from their
# edges with no source rather than given a fitted one.  Measured on case10: from a
# 12 px held-out ring down to a 3 px one the blocks above 50 mm2 hold their fitted
# rate to within 15 %, while the 8-16 mm2 ones swing from -0.13 to -0.93 K/s.
MIN_FITTABLE_PX = 12

# How sharply a heat-source block's source is pushed toward its own middle.  A die
# under a metal lid dissipates over a few mm and the lid spreads it, so the source
# is nowhere near uniform over the 269 mm2 the lid covers; weighting it by distance
# from the block's own rim matches that geometry and costs no extra parameter.
# Measured on case17 with case10's mask imposed, concentrating only the blocks the
# edge test calls heat sources: the SoC block's error falls 4.35 -> 2.50 K and the
# whole masked area 2.62 -> 2.36 K, with the worst change on any other block
# +0.03 K.  The optimum is flat -- 0.5 gives 2.43 K and 1.5 gives 2.40 K -- and
# 2.0 is already too sharp (2.46 K, SoC back up to 3.16 K).
CONCENTRATION_EXPONENT = 1.0


@dataclass(frozen=True)
class BlockSource:
    """One block's provisional ring fit and source used by the final solve.

    ``source_rate_*`` retains the exterior-ring fit because it is the measured
    diagnostic. ``reconstruction_source_rate_*`` records the rate that actually
    generated ``Reconstruction.theta_k``; the two differ when a hot lid changes
    from the provisional centre-weighted shape to the final broad source.
    """

    region_id: int
    area_px: int
    area_mm2: float | None
    centroid_rc: tuple[float, float]
    centroid_mm: tuple[float, float] | None
    source_rate_k_per_s: float  # median ring-fitted provisional (heating - warming)
    source_rate_by_frame: np.ndarray  # (T,) ring-fitted provisional rate
    source_rate_spread_k_per_s: float  # frame-to-frame scatter, 16th-84th percentile
    deepest_px: float  # furthest any pixel is from trusted board
    deepest_mm: float | None
    fitted: bool  # False => too small to fit, filled from its edges with no source
    touches_border: bool
    source_shape: str = "uniform"  # or "centre-weighted"; see _block_weight
    interior_fill: str = "source-shaped"
    reconstruction_source_rate_k_per_s: float | None = None  # rate used by final plate solve
    reconstruction_source_rate_by_frame: np.ndarray | None = None  # (T,)
    reconstruction_source_rate_spread_k_per_s: float | None = None


@dataclass(frozen=True)
class InteriorTemperatureAnchor:
    """Optional measured central temperature for one low-emissivity footprint.

    ``temperature_above_ambient_k_by_frame`` must already be aligned to the IR
    timestamps and have one value per reconstructed frame.  A caller using
    ``vcgencmd measure_temp`` must subtract ambient first.  That command reports
    the die rather than the visible lid surface, so its die-to-surface difference
    must also be removed before it is supplied here; if that difference is not
    known, include it in ``uncertainty_k`` rather than treating the die reading as
    an exact surface temperature.

    The anchor fixes the mean of the broad central area, not one isolated pixel.
    The plate solve still determines every individual pixel, avoiding both a
    nonphysical point singularity and a pasted flat core.
    """

    region_id: int
    temperature_above_ambient_k_by_frame: np.ndarray
    uncertainty_k: float | np.ndarray
    label: str = ""


@dataclass(frozen=True)
class Reconstruction:
    """Full temperature maps for every frame, with the sources that drive them."""

    theta_k: np.ndarray  # (T, H, W) kelvin above ambient; trusted pixels unchanged
    times_s: np.ndarray  # (T,)
    unknown_mask: np.ndarray  # (H, W) original low-emissivity footprint mask
    blocks: list[BlockSource]
    spread_rate_px2_per_s: float
    cooling_rate_per_s: float
    spread_distance_px: float
    ring_error_k: np.ndarray  # (T,) held-out ring error, per frame
    ring_error_no_source_k: np.ndarray  # (T,) the same ring with every source set to zero
    uncertainty_k: np.ndarray | None = None  # (T,H,W), model range; not a confidence interval
    lid_transition_mask: np.ndarray | None = None  # extra rim pixels replaced only around hot lids
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def spread_distance_mm(self) -> float | None:
        return self.report.get("spread_distance_mm")

    @property
    def rebuilt_mask(self) -> np.ndarray:
        """Every pixel replaced in ``theta_k``, including a hot-lid transition rim."""
        if self.lid_transition_mask is None:
            return self.unknown_mask
        return self.unknown_mask | self.lid_transition_mask


def warming_rate(
    theta_k: np.ndarray, times_s: np.ndarray, *, half_window: int = 2
) -> np.ndarray:
    """How fast each pixel is warming, K/s, by local straight-line fit.

    The window has to be centred and the intercept has to be fitted.  Regressing
    on uncentred time without an intercept -- which is only valid mid-sequence --
    reports 0.126 K/s as 1.40 K/s at frame 0, because there the window is
    one-sided and the intercept leaks into the slope.

    Two frames either side is the shortest window that beats the 0.1 C file
    quantisation: over 24 s the board moves ~3 K early on, against a 0.1 K step.
    """
    theta_k = np.asarray(theta_k, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)
    if theta_k.ndim != 3:
        raise ValueError(f"warming_rate expects (T, H, W), got {theta_k.shape}")
    if len(times_s) != len(theta_k):
        raise ValueError(f"{len(times_s)} times for {len(theta_k)} frames")
    if len(theta_k) < 2:
        raise ValueError("warming_rate needs at least two frames")

    out = np.zeros_like(theta_k)
    n = len(theta_k)
    for index in range(n):
        lo, hi = max(0, index - half_window), min(n, index + half_window + 1)
        x = times_s[lo:hi] - times_s[lo:hi].mean()
        denominator = float((x * x).sum())
        if denominator <= 0.0:
            continue
        block = theta_k[lo:hi].reshape(hi - lo, -1)
        block = block - block.mean(axis=0)
        out[index] = ((x[:, None] * block).sum(0) / denominator).reshape(theta_k.shape[1:])
    return out


def _curvature(field_2d: np.ndarray, smooth_px: float) -> np.ndarray:
    """Five-point curvature with mirrored edges: no heat crosses the frame border."""
    smoothed = gaussian_filter(field_2d, smooth_px) if smooth_px > 0 else field_2d
    padded = np.pad(smoothed, 1, mode="reflect")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * smoothed
    )


def plate_residual(
    theta_frame_k: np.ndarray,
    warming_frame_k_per_s: np.ndarray,
    heating_rate_k_per_s: np.ndarray,
    *,
    cooling_rate_per_s: float,
    spread_rate_px2_per_s: float,
    smooth_px: float = DEFAULT_SMOOTH_PX,
) -> np.ndarray:
    """By how much the measured field fails to satisfy the plate equation, K/s.

    Zero everywhere would mean the measured rates explain the measured field
    exactly.  On clean case10 board it is 0.041 K/s rms against a heating map of
    0.082 K/s rms, so the source map carries twice the residual -- worth using.
    It is the natural sanity check before trusting any fill, and it is what showed
    the curvature had to be smoothed.
    """
    curvature = _curvature(theta_frame_k, smooth_px)
    return (
        spread_rate_px2_per_s * curvature
        - cooling_rate_per_s * theta_frame_k
        + heating_rate_k_per_s
        - warming_frame_k_per_s
    )


def block_labels(footprint_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The one labelling of low-emissivity blocks the whole package uses.

    Public and shared on purpose.  ``judge_footprints`` and ``reconstruct_sequence``
    report per-block results that the driver joins on the region id, and they used
    to label independently -- 4-connectivity there, 8 here.  On case10 the two
    happen to coincide, but on a mask with a diagonal touch they do not, and the
    join then silently attaches one block's verdict to another's source rate.
    """
    labels, count = label(np.asarray(footprint_mask, dtype=bool), structure=np.ones((3, 3)))
    return labels, int(count)


_block_labels = block_labels  # kept so existing callers keep working


def _block_weight(region: np.ndarray, concentrate: bool) -> np.ndarray:
    """How one block's source is distributed inside it, scaled to average 1.

    Passive blocks use a uniform term: their apparent excess comes from the board
    around them, and one spatially constant degree of freedom is the honest fit.
    For heat-source blocks, the *provisional ring fit* peaks toward the middle to
    represent a small die under a much larger lid.  That improves the level inferred
    from the exterior ring.  Production later discards this assumed internal shape
    and solves the lid with a separate broad distributed source.

    Averaging to 1 keeps the fitted coefficient meaning the same thing either way
    -- the block's *mean* source rate in K/s -- so the numbers stay comparable
    between blocks and against the earlier uniform-only run.
    """
    inside = np.asarray(region, dtype=bool)
    if not concentrate:
        return np.where(inside, 1.0, 0.0)
    # Distance to this block's own rim, not to the nearest trusted pixel: a
    # neighbouring block must not count as interior.
    depth = distance_transform_edt(inside) ** CONCENTRATION_EXPONENT
    mean = float(depth[inside].mean())
    if mean <= 0.0:
        return np.where(inside, 1.0, 0.0)
    return np.where(inside, depth / mean, 0.0)


def _influence_fields(
    unknown: np.ndarray,
    labels: np.ndarray,
    ids: list[int],
    spread_rate_px2_per_s: float,
    spread_distance_px: float,
    concentrated: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Temperature each block raises, per unit of its source rate, with no boundary data.

    The solve is affine in (boundary data, source), so this is computed once and
    reused for every frame.  Without it the fit would cost one solve per block per
    frame -- 3120 sparse solves on case10 instead of 252.
    """
    zero = np.zeros(unknown.shape, dtype=np.float64)
    fields = np.empty((len(ids),) + unknown.shape, dtype=np.float64)
    for slot, region_id in enumerate(ids):
        weight = _block_weight(labels == region_id, region_id in concentrated)
        fields[slot] = solve_plate(
            zero, unknown, spread_distance_px, source=weight / spread_rate_px2_per_s
        )
    return fields


def _edge_profile_samples(
    region: np.ndarray, trusted: np.ndarray
) -> list[tuple[np.ndarray, ...]]:
    """Indices for two trusted and two interior pixels along each usable normal.

    The order along a sample is ``outside2, outside1, inside1, inside2``.  It
    supplies the two one-sided slopes the old Dirichlet-only fill ignored.
    Samples that hit another footprint, leave the board, or have no second
    interior pixel are omitted.
    """
    height, width = region.shape
    rows, cols = np.nonzero(region)
    samples: list[tuple[np.ndarray, ...]] = []
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        outside1_r, outside1_c = rows + d_row, cols + d_col
        outside2_r, outside2_c = rows + 2 * d_row, cols + 2 * d_col
        inside2_r, inside2_c = rows - d_row, cols - d_col
        in_frame = (
            (outside2_r >= 0)
            & (outside2_r < height)
            & (outside2_c >= 0)
            & (outside2_c < width)
            & (outside1_r >= 0)
            & (outside1_r < height)
            & (outside1_c >= 0)
            & (outside1_c < width)
            & (inside2_r >= 0)
            & (inside2_r < height)
            & (inside2_c >= 0)
            & (inside2_c < width)
        )
        arrays = [
            item[in_frame]
            for item in (
                rows,
                cols,
                inside2_r,
                inside2_c,
                outside1_r,
                outside1_c,
                outside2_r,
                outside2_c,
            )
        ]
        inside1_r, inside1_c, inside2_r, inside2_c, outside1_r, outside1_c, outside2_r, outside2_c = arrays
        usable = (
            region[inside2_r, inside2_c]
            & trusted[outside1_r, outside1_c]
            & trusted[outside2_r, outside2_c]
        )
        if usable.any():
            samples.append(tuple(item[usable] for item in arrays))
    return samples


def _anchor_arrays(
    anchors: Iterable[InteriorTemperatureAnchor], n_frames: int, valid_ids: set[int]
) -> dict[int, tuple[np.ndarray, np.ndarray, str]]:
    """Validate optional lid temperatures and return frame-aligned arrays."""
    out: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for item in anchors:
        region_id = int(item.region_id)
        if region_id not in valid_ids:
            raise ValueError(
                f"interior anchor names block {region_id}, but reconstructed lids are "
                f"{sorted(valid_ids)}"
            )
        if region_id in out:
            raise ValueError(f"more than one interior anchor supplied for block {region_id}")
        values = np.asarray(item.temperature_above_ambient_k_by_frame, dtype=np.float64)
        if values.shape != (n_frames,) or not np.isfinite(values).all():
            raise ValueError(
                f"anchor for block {region_id} needs {n_frames} finite temperatures, "
                f"got {values.shape}"
            )
        uncertainty = np.asarray(item.uncertainty_k, dtype=np.float64)
        if uncertainty.ndim == 0:
            uncertainty = np.full(n_frames, float(uncertainty))
        if uncertainty.shape != (n_frames,) or not np.isfinite(uncertainty).all():
            raise ValueError(
                f"anchor uncertainty for block {region_id} must be scalar or ({n_frames},)"
            )
        if (uncertainty < 0.0).any():
            raise ValueError(f"anchor uncertainty for block {region_id} must be non-negative")
        out[region_id] = (values, uncertainty, item.label)
    return out


def _broad_source_lids(
    preliminary_k: np.ndarray,
    measured_k: np.ndarray,
    footprint_mask: np.ndarray,
    board_mask: np.ndarray,
    labels: np.ndarray,
    region_ids: frozenset[int],
    spread_rate_px2_per_s: float,
    spread_distance_px: float,
    ring_error_k: np.ndarray,
    *,
    anchors: Iterable[InteriorTemperatureAnchor],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[int, np.ndarray],
]:
    """Rebuild heat-spreading lids with a smooth, source-driven plate solve.

    The exterior-ring fit supplies a provisional *mean level* for each lid.  Its
    assumed central-bump shape is discarded.  A positive source distributed over
    the whole original footprint is then calibrated to reproduce that mean, and
    the plate equation is solved over the footprint plus one residual-rim pixel.

    No temperature is pasted, flattened, or blended after the solve.  A broad
    plateau appears because a smooth distributed source in a conducting plate
    approaches a level interior while the plate equation makes its rim and all
    first differences change smoothly.  An optional sensor replaces the provisional
    mean with a measured central level but uses the same physical solve.
    """
    n_frames = len(preliminary_k)
    anchor_by_id = _anchor_arrays(anchors, n_frames, set(region_ids))
    out = preliminary_k.copy()
    depth_all = distance_transform_edt(footprint_mask)
    fit_range = ring_error_k[:, None, None] * (
        1.0 + depth_all[None, :, :] / max(float(spread_distance_px), 1e-6)
    )
    uncertainty = np.where(footprint_mask[None, :, :], fit_range, 0.0)

    claimed = np.zeros(footprint_mask.shape, dtype=bool)
    geometry: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    zero = np.zeros(footprint_mask.shape, dtype=np.float64)
    for region_id in sorted(region_ids):
        region = labels == region_id
        expanded = binary_dilation(region, CROSS) & board_mask
        expanded &= ~(footprint_mask & ~region) & ~claimed
        claimed |= expanded
        unit_source = np.where(region, 1.0 / spread_rate_px2_per_s, 0.0)
        influence = solve_plate(
            zero, expanded, fin_length_px=spread_distance_px, source=unit_source
        )
        depth = distance_transform_edt(region)
        central = region & (depth >= max(float(depth.max()) * 0.75, 1.0))
        if not central.any():
            central = region
        geometry[region_id] = (region, expanded, influence, central)

    transition_mask = claimed & ~footprint_mask
    rebuilt_mask = footprint_mask | transition_mask
    broad_rates: dict[int, list[float]] = {region_id: [] for region_id in region_ids}
    anchor_adjustments: dict[int, list[float]] = {
        region_id: [] for region_id in anchor_by_id
    }
    slope_mismatches: list[float] = []
    trusted = board_mask & ~rebuilt_mask
    edge_samples = {
        region_id: _edge_profile_samples(expanded, trusted)
        for region_id, (_, expanded, _, _) in geometry.items()
    }

    for frame_index in range(n_frames):
        for region_id, (region, expanded, influence, central) in geometry.items():
            boundary_solution = solve_plate(
                measured_k[frame_index], expanded, fin_length_px=spread_distance_px
            )
            anchor = anchor_by_id.get(region_id)
            calibration = region if anchor is None else central
            target = (
                float(preliminary_k[frame_index, region].mean())
                if anchor is None
                else float(anchor[0][frame_index])
            )
            denominator = float(influence[calibration].mean())
            if denominator <= 1e-12:
                raise ValueError(f"block {region_id} has no usable broad-source influence")
            required_rate = (
                target - float(boundary_solution[calibration].mean())
            ) / denominator
            if anchor is not None and required_rate < 0.0:
                raise ValueError(
                    f"anchor for block {region_id} is below its source-free boundary "
                    f"solution at frame {frame_index}; a non-negative heat source cannot "
                    "satisfy it"
                )
            broad_rate = max(0.0, required_rate)
            broad_rates[region_id].append(broad_rate)
            solved = boundary_solution[expanded] + broad_rate * influence[expanded]
            old = preliminary_k[frame_index, expanded]
            out[frame_index, expanded] = solved

            local_depth = distance_transform_edt(expanded)[expanded]
            local_fit = ring_error_k[frame_index] * (
                1.0 + local_depth / max(float(spread_distance_px), 1e-6)
            )
            model_difference = np.abs(solved - old)
            if anchor is None:
                uncertainty[frame_index, expanded] = local_fit + model_difference
            else:
                # The anchor deliberately replaces the ring-inferred lid level.
                # Counting that correction itself as uncertainty would say a
                # measured constraint becomes less informative the more it fixes.
                # Keep the discarded-model disagreement as a diagnostic, while
                # combining the ring/shape range and supplied anchor uncertainty.
                anchor_extra = float(anchor[1][frame_index])
                uncertainty[frame_index, expanded] = np.hypot(local_fit, anchor_extra)
                anchor_adjustments[region_id].append(float(np.mean(model_difference)))

            for sample in edge_samples[region_id]:
                (
                    inside1_r,
                    inside1_c,
                    inside2_r,
                    inside2_c,
                    outside1_r,
                    outside1_c,
                    outside2_r,
                    outside2_c,
                ) = sample
                outside_slope = (
                    measured_k[frame_index, outside1_r, outside1_c]
                    - measured_k[frame_index, outside2_r, outside2_c]
                )
                inside_slope = (
                    out[frame_index, inside2_r, inside2_c]
                    - out[frame_index, inside1_r, inside1_c]
                )
                slope_mismatches.extend(np.abs(inside_slope - outside_slope).tolist())

    metadata = {
        "n_broad_source_lids": len(region_ids),
        "n_interior_temperature_anchors": len(anchor_by_id),
        "n_lid_transition_px": int(transition_mask.sum()),
        "interior_temperature_anchor_labels": [value[2] for value in anchor_by_id.values()],
        "anchor_replacement_of_unanchored_model_median_k": {
            str(region_id): float(np.median(values))
            for region_id, values in anchor_adjustments.items()
            if values
        },
        "edge_slope_mismatch_median_k_per_px": (
            float(np.median(slope_mismatches)) if slope_mismatches else None
        ),
        "uncertainty_kind": "model range, not a statistical confidence interval",
        "uncertainty_median_k": float(np.median(uncertainty[:, rebuilt_mask])),
        "uncertainty_p95_k": float(np.percentile(uncertainty[:, rebuilt_mask], 95.0)),
        "lid_broad_source_rate_median_k_per_s": {
            str(region_id): float(np.median(values))
            for region_id, values in broad_rates.items()
            if values
        },
        "lid_broad_source_positive_fraction": {
            str(region_id): float(np.mean(np.asarray(values) > 0.0))
            for region_id, values in broad_rates.items()
            if values
        },
    }
    rate_series = {
        region_id: np.asarray(values, dtype=np.float64)
        for region_id, values in broad_rates.items()
    }
    return out, uncertainty, transition_mask, metadata, rate_series


def reconstruct_sequence(
    theta_k: np.ndarray,
    times_s: np.ndarray,
    footprint_mask: np.ndarray,
    heating_rate_k_per_s: np.ndarray,
    *,
    cooling_rate_per_s: float,
    spread_rate_px2_per_s: float,
    ring_px: int = DEFAULT_RING_PX,
    board_mask: np.ndarray | None = None,
    pixel_size_mm: float | None = None,
    min_fittable_px: int = MIN_FITTABLE_PX,
    heat_source_ids: Iterable[int] = (),
    lid_fill: str = "broad-source-plate",
    interior_temperature_anchors: Iterable[InteriorTemperatureAnchor] = (),
) -> Reconstruction:
    """Fill every low-emissivity block, in every frame, from a fitted source rate.

    Named ``reconstruct_sequence`` and not ``reconstruct`` so that re-exporting it
    from the package does not shadow this module: ``from tdp.thermal import
    reconstruct`` would otherwise hand back the function, and every
    ``reconstruct.warming_rate`` in a caller would fail with an attribute error.

    ``theta_k`` is the *raw* measured field in kelvin above ambient.  It must be
    raw: a repaired field already carries an assumed source under every block, so
    fitting a source to it would return that assumption back.

    ``heating_rate_k_per_s`` is used only on the held-out ring, where it is
    measured.  Its values inside the blocks are ignored -- they are the ones this
    routine exists to replace.

    ``heat_source_ids`` are the region ids the Stage 1 edge test judged to be
    generating heat.  Those blocks get a centre-weighted source instead of a
    uniform one; see ``_block_weight``.  Pass only confident calls: a block wrongly
    left uniform costs 0.03 K, while a passive block wrongly concentrated costs
    several kelvin, so the asymmetry runs strongly against false positives.  With
    the default empty set the result is identical to the uniform-only fit.

    ``lid_fill="broad-source-plate"`` keeps the ring-fitted mean level but
    replaces the assumed central bump with a source distributed over the lid.
    The final temperature is produced entirely by a plate solve, giving a broad
    smooth plateau without pasting or flattening pixels.  ``"source-shaped"``
    remains available as the ablation.

    ``interior_temperature_anchors`` optionally supplies a measured central
    surface temperature for a lid.  See :class:`InteriorTemperatureAnchor`; an
    uncorrected die reading is not a surface temperature.
    """
    theta_k = np.asarray(theta_k, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)
    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    heating_rate_k_per_s = np.asarray(heating_rate_k_per_s, dtype=np.float64)
    if theta_k.ndim != 3:
        raise ValueError(f"reconstruct expects (T, H, W), got {theta_k.shape}")
    if footprint_mask.shape != theta_k.shape[1:]:
        raise ValueError(f"mask {footprint_mask.shape} does not match {theta_k.shape[1:]}")
    if heating_rate_k_per_s.shape != theta_k.shape[1:]:
        raise ValueError("heating rate map does not match the frame shape")
    if spread_rate_px2_per_s <= 0.0 or cooling_rate_per_s <= 0.0:
        raise ValueError("spread rate and cooling rate must both be positive")
    if not footprint_mask.any():
        raise ValueError("nothing to reconstruct: the footprint mask is empty")
    if lid_fill not in {"source-shaped", "broad-source-plate"}:
        raise ValueError(
            "lid_fill must be 'source-shaped' or 'broad-source-plate', "
            f"got {lid_fill!r}"
        )

    board = np.ones(footprint_mask.shape, dtype=bool) if board_mask is None else np.asarray(
        board_mask, dtype=bool
    )
    spread_distance_px = float(np.sqrt(spread_rate_px2_per_s / cooling_rate_per_s))
    labels, count = _block_labels(footprint_mask)
    sizes = {region_id: int((labels == region_id).sum()) for region_id in range(1, count + 1)}
    fittable = [region_id for region_id, size in sizes.items() if size >= min_fittable_px]

    # The ring is trusted board only.  Blocks other than the one being fitted stay
    # unknown, so a neighbour's source is solved for rather than read off its
    # untrustworthy pixels.
    ring = binary_dilation(footprint_mask, CROSS, iterations=int(ring_px))
    ring &= ~footprint_mask & board
    if ring.sum() < 10 * max(len(fittable), 1):
        raise ValueError(f"only {int(ring.sum())} trusted ring pixels for {len(fittable)} blocks")
    unknown_fit = footprint_mask | ring

    warming = warming_rate(theta_k, times_s)
    asked = frozenset(int(i) for i in heat_source_ids)
    unknown_ids = asked - set(range(1, count + 1))
    if unknown_ids:
        raise ValueError(
            f"heat_source_ids {sorted(unknown_ids)} name no block; this mask has "
            f"{count}.  Label with tdp.thermal.block_labels so the ids agree."
        )
    concentrated = asked & set(fittable)
    anchors = tuple(interior_temperature_anchors)
    if anchors and lid_fill == "source-shaped":
        raise ValueError("interior temperature anchors require a reconstructed lid model")
    if anchors and not concentrated:
        raise ValueError("interior temperature anchors require a fitted heat-producing lid")
    fit_influence = _influence_fields(
        unknown_fit, labels, fittable, spread_rate_px2_per_s, spread_distance_px, concentrated
    )
    design = fit_influence[:, ring].T  # (n_ring, n_blocks)
    solver = np.linalg.pinv(design)

    n_frames = len(theta_k)
    source_by_frame = np.zeros((n_frames, len(fittable)))
    ring_error = np.zeros(n_frames)
    ring_error_no_source = np.zeros(n_frames)
    for index in range(n_frames):
        base = np.zeros(footprint_mask.shape, dtype=np.float64)
        base[ring] = (heating_rate_k_per_s[ring] - warming[index][ring]) / spread_rate_px2_per_s
        baseline = solve_plate(theta_k[index], unknown_fit, spread_distance_px, source=base)
        gap = theta_k[index][ring] - baseline[ring]
        rates = solver @ gap
        source_by_frame[index] = rates
        ring_error[index] = float(np.sqrt(np.mean((gap - design @ rates) ** 2)))
        ring_error_no_source[index] = float(np.sqrt(np.mean(gap**2)))

    # Production fill: the ring is *not* held out, so every trusted pixel is used
    # as boundary data.  Only the fitted sources carry over from the fit.
    production_influence = _influence_fields(
        footprint_mask, labels, fittable, spread_rate_px2_per_s, spread_distance_px, concentrated
    )
    out = theta_k.copy()
    for index in range(n_frames):
        filled = solve_plate(theta_k[index], footprint_mask, spread_distance_px)
        filled += np.tensordot(source_by_frame[index], production_influence, axes=(0, 0))
        out[index][footprint_mask] = filled[footprint_mask]

    depth = distance_transform_edt(footprint_mask)
    fit_range = ring_error[:, None, None] * (
        1.0 + depth[None, :, :] / max(spread_distance_px, 1e-6)
    )
    uncertainty = np.where(footprint_mask[None, :, :], fit_range, 0.0)
    lid_transition_mask = np.zeros(footprint_mask.shape, dtype=bool)
    lid_source_by_id: dict[int, np.ndarray] = {}
    lid_report: dict[str, Any] = {
        "n_broad_source_lids": 0,
        "n_interior_temperature_anchors": 0,
        "uncertainty_kind": "held-out-ring model range, not a statistical confidence interval",
        "uncertainty_median_k": float(np.median(uncertainty[:, footprint_mask])),
        "uncertainty_p95_k": float(np.percentile(uncertainty[:, footprint_mask], 95.0)),
    }
    if lid_fill == "broad-source-plate" and concentrated:
        out, uncertainty, lid_transition_mask, lid_report, lid_source_by_id = _broad_source_lids(
            out,
            theta_k,
            footprint_mask,
            board,
            labels,
            frozenset(concentrated),
            spread_rate_px2_per_s,
            spread_distance_px,
            ring_error,
            anchors=anchors,
        )
    rows, cols = np.mgrid[0 : footprint_mask.shape[0], 0 : footprint_mask.shape[1]]
    height, width = footprint_mask.shape
    blocks: list[BlockSource] = []
    for region_id in range(1, count + 1):
        region = labels == region_id
        centroid = (float(rows[region].mean()), float(cols[region].mean()))
        touches = bool(
            region[0].any() or region[height - 1].any() or region[:, 0].any()
            or region[:, width - 1].any()
        )
        if region_id in fittable:
            slot = fittable.index(region_id)
            series = source_by_frame[:, slot]
            low, high = np.percentile(series, [16.0, 84.0])
            rate, scatter, was_fitted = float(np.median(series)), float(high - low), True
        else:
            series = np.zeros(n_frames)
            rate, scatter, was_fitted = 0.0, 0.0, False
        reconstruction_series = lid_source_by_id.get(region_id, series)
        reconstruction_low, reconstruction_high = np.percentile(
            reconstruction_series, [16.0, 84.0]
        )
        blocks.append(
            BlockSource(
                region_id=region_id,
                area_px=sizes[region_id],
                area_mm2=None if pixel_size_mm is None else sizes[region_id] * pixel_size_mm**2,
                centroid_rc=centroid,
                centroid_mm=None
                if pixel_size_mm is None
                else (centroid[0] * pixel_size_mm, centroid[1] * pixel_size_mm),
                source_rate_k_per_s=rate,
                source_rate_by_frame=series,
                source_rate_spread_k_per_s=scatter,
                deepest_px=float(depth[region].max()),
                deepest_mm=None if pixel_size_mm is None else float(depth[region].max()) * pixel_size_mm,
                fitted=was_fitted,
                touches_border=touches,
                source_shape="centre-weighted" if region_id in concentrated else "uniform",
                interior_fill=(
                    "broad-source plate"
                    if lid_fill == "broad-source-plate" and region_id in concentrated
                    else "source-shaped"
                ),
                reconstruction_source_rate_k_per_s=float(np.median(reconstruction_series)),
                reconstruction_source_rate_by_frame=reconstruction_series,
                reconstruction_source_rate_spread_k_per_s=float(
                    reconstruction_high - reconstruction_low
                ),
            )
        )

    report = {
        "n_blocks": count,
        "n_fitted": len(fittable),
        "n_ring_px": int(ring.sum()),
        "ring_px": int(ring_px),
        "masked_fraction_of_board": float(footprint_mask[board].mean()),
        "deepest_px": float(depth.max()),
        "spread_distance_mm": None if pixel_size_mm is None else spread_distance_px * pixel_size_mm,
        "ring_error_median_k": float(np.median(ring_error)),
        "ring_error_no_source_median_k": float(np.median(ring_error_no_source)),
        "n_centre_weighted": len(concentrated),
        "concentration_exponent": CONCENTRATION_EXPONENT if concentrated else None,
        "lid_fill": lid_fill,
        **lid_report,
    }
    return Reconstruction(
        theta_k=out,
        times_s=times_s,
        unknown_mask=footprint_mask,
        blocks=blocks,
        spread_rate_px2_per_s=float(spread_rate_px2_per_s),
        cooling_rate_per_s=float(cooling_rate_per_s),
        spread_distance_px=spread_distance_px,
        ring_error_k=ring_error,
        ring_error_no_source_k=ring_error_no_source,
        uncertainty_k=uncertainty,
        lid_transition_mask=lid_transition_mask,
        report=report,
    )


def heating_rate_from_reconstruction(
    result: Reconstruction, region_id: int, *, half_window: int = 2
) -> np.ndarray:
    """Split the final plate solve's source rate into a heating rate, K/s per frame.

    The fit returns ``heating rate - warming rate``; the reconstruction supplies
    the warming rate, so the heating rate follows.  It is reported per frame and
    not averaged, because a heating rate that drifts while the load is constant is
    evidence the fill is wrong, and averaging hides exactly that.
    """
    block = next((b for b in result.blocks if b.region_id == region_id), None)
    if block is None:
        raise ValueError(f"no block {region_id} in this reconstruction")
    labels, _ = _block_labels(result.unknown_mask)
    region = labels == region_id
    interior = result.theta_k[:, region].mean(axis=1)[:, None, None]
    warming = warming_rate(interior, result.times_s, half_window=half_window)
    source_rate = (
        block.source_rate_by_frame
        if block.reconstruction_source_rate_by_frame is None
        else block.reconstruction_source_rate_by_frame
    )
    return source_rate + warming[:, 0, 0]


@dataclass(frozen=True)
class TruthCheck:
    """How far a reconstruction is from a field that was actually measured."""

    mean_absolute_k: float
    bias_k: float
    p95_absolute_k: float
    level_error_k: float  # per-block mean error: how wrong the block's level is
    shape_error_k: float  # what is left inside a block once its level is removed
    per_block: dict[int, tuple[float, float]]  # region id -> (level error, shape error)
    n_frames: int
    n_pixels: int


def validate_against_truth(
    result: Reconstruction, truth_k: np.ndarray, *, region_labels: np.ndarray | None = None
) -> TruthCheck:
    """Score a reconstruction where the hidden field happens to be known.

    The level/shape split is the useful part.  A level error means the block sits
    at the wrong temperature and a better source rate would fix it; a shape error
    means the block's internal pattern is wrong, and no per-block source can fix
    that however well it is fitted.  On case17 the split is 0.19 K level against
    1.53 K shape, which is why this module stops where it does.
    """
    truth_k = np.asarray(truth_k, dtype=np.float64)
    if truth_k.shape != result.theta_k.shape:
        raise ValueError(f"truth {truth_k.shape} does not match {result.theta_k.shape}")
    mask = result.unknown_mask
    labels = _block_labels(mask)[0] if region_labels is None else np.asarray(region_labels)

    error = result.theta_k[:, mask] - truth_k[:, mask]
    levels: list[tuple[float, int]] = []
    shapes: list[tuple[float, int]] = []
    per_block: dict[int, tuple[float, float]] = {}
    for region_id in np.unique(labels[mask]):
        region = labels == region_id
        block_error = result.theta_k[:, region] - truth_k[:, region]
        offset = block_error.mean(axis=1, keepdims=True)
        level = float(np.abs(offset).mean())
        shape = float(np.abs(block_error - offset).mean())
        per_block[int(region_id)] = (level, shape)
        levels.append((level, int(region.sum())))
        shapes.append((shape, int(region.sum())))

    weigh = lambda pairs: float(  # noqa: E731 -- one-line area weighting, used twice
        np.average([v for v, _ in pairs], weights=[n for _, n in pairs])
    )
    return TruthCheck(
        mean_absolute_k=float(np.abs(error).mean()),
        bias_k=float(error.mean()),
        p95_absolute_k=float(np.percentile(np.abs(error), 95.0)),
        level_error_k=weigh(levels),
        shape_error_k=weigh(shapes),
        per_block=per_block,
        n_frames=int(result.theta_k.shape[0]),
        n_pixels=int(mask.sum()),
    )
