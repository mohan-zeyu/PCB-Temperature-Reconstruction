"""Repair low-emissivity artifacts in bare-board IR frames using plate conduction.

An IR camera converts received radiance to temperature under one assumed
emissivity.  Wherever the board is *not* that emissive -- connector shields,
package lids, exposed metal -- the reported temperature is biased low, often
severely: the same shields at the same load read 31 C bare and 45 C taped.
Those pixels are not noisy, they are wrong, and they are wrong in a way no
amount of averaging removes.

The board itself, however, is a thin conducting plate, and the *trusted*
(high-emissivity) pixels around each artifact constrain what the field can be
underneath it.  In steady state the excess temperature theta = T - T_ambient
obeys

    k_eff d (d2theta/dx2 + d2theta/dy2) = h theta - q,

so away from sources theta is a screened-harmonic function whose interior
values are determined by its boundary.  Rewriting in pixel units with the
plate stiffness k_eff d absorbed into the source term gives the discrete
operator this module solves: the artifact footprints are treated as unknowns,
the surrounding trusted pixels as Dirichlet data, and the crop border as
adiabatic.  The plate stiffness cancels between the two sides, so no material
constants have to be known to fill a passive region.

Two facts set the scope of what that fill means, both measured rather than
assumed (see ``docs/emissivity_repair.md``):

* For *passive* metal -- shields, cages, hardware -- there is no source under
  the footprint, and the conduction fill is an estimate of the surface
  temperature.
* For a *powered* package such as the SoC lid, boundary data alone is not
  enough.  No solution of the homogeneous equation has an interior maximum, so
  a fill driven only by the ring around a package can never reconstruct the
  source under it: measured against ``vcgencmd`` on this board it lands 4.0 C
  low.  The fix is not a better interpolator, it is more data.  An on-die
  sensor reading is a *known interior value* in the very same boundary-value
  problem, and :func:`repair_frame` accepts it through ``anchors``: anchor
  pixels move from the unknown set into the Dirichlet set, and conduction then
  carries the source outward to meet the trusted board.  Without an anchor the
  repaired package footprint is a lower bound and is labelled as such.

Emissivity is not used to produce the field.  The default path is conduction
end to end; a per-region effective emissivity is *fitted and reported* as a
diagnostic, and is only substituted into pixels on explicit request
(``method="emissivity"``), where it stays guarded because the inversion is
ill-conditioned exactly where the artifacts are worst.

The public entry point is :func:`repair_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    label,
    percentile_filter,
)

# 4-neighbourhood: the conduction stencil, the growth neighbourhood and the
# halo dilation all have to agree on what "adjacent" means.
CROSS = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Screening length of the plate: sqrt(k_eff d / h).  For a 1.6 mm 6-layer PCB
# under natural convection this is a few centimetres, i.e. of the order of the
# board itself, so over an artifact a few millimetres wide the screening term
# is a small correction to a harmonic fill.  Repairs are insensitive to it
# (see the sweep in ``scripts/validate_emissivity.py``).
DEFAULT_FIN_LENGTH_PX = 100.0


def estimate_ambient(frame: np.ndarray, border_px: int = 8) -> float:
    """Ambient estimate = median of the frame's border ring.

    Mirrors :func:`tdp.data.crop.estimate_ambient` so a repair can be run on a
    raw frame before cropping.
    """
    b = int(border_px)
    ring = np.concatenate(
        [frame[:b].ravel(), frame[-b:].ravel(), frame[b:-b, :b].ravel(), frame[b:-b, -b:].ravel()]
    )
    return float(np.median(ring))


def solve_plate(
    theta: np.ndarray,
    unknown: np.ndarray,
    fin_length_px: float = DEFAULT_FIN_LENGTH_PX,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the steady plate equation on ``unknown``, Dirichlet elsewhere.

    Discretizing ``lap(theta) = kappa^2 theta - s`` with the 5-point stencil on
    a unit grid gives, for every unknown pixel,

        (4 + kappa^2) theta_ij - sum_neighbours theta = s_ij,

    with known neighbours moved to the right-hand side.  Neighbours outside the
    frame are mirrored, which imposes zero flux: the PCB edge loses heat to air
    from its faces, not sideways out of the crop, so an adiabatic border is the
    right in-plane condition and it keeps regions that touch the border (the
    connector bank) solvable.

    ``source`` is in the same units as the operator, i.e. power divided by the
    plate stiffness k_eff d, which is what :func:`outward_flux` returns.
    """
    if theta.ndim != 2:
        raise ValueError(f"solve_plate expects a 2-D field, got shape {theta.shape}")
    if unknown.shape != theta.shape:
        raise ValueError(f"unknown {unknown.shape} does not match field {theta.shape}")
    if not np.isfinite(theta).all():
        raise ValueError("solve_plate requires a finite field")
    if fin_length_px <= 0:
        raise ValueError(f"fin_length_px must be positive, got {fin_length_px}")

    height, width = theta.shape
    rows_idx, cols_idx = np.nonzero(unknown)
    n = len(rows_idx)
    out = theta.astype(np.float64, copy=True)
    if n == 0:
        return out
    if n == theta.size:
        raise ValueError("every pixel is unknown; the solve has no boundary data")

    kappa_sq = 1.0 / float(fin_length_px) ** 2
    order = -np.ones(theta.shape, dtype=np.int64)
    order[rows_idx, cols_idx] = np.arange(n)
    self_index = np.arange(n)

    rhs = np.zeros(n) if source is None else source[rows_idx, cols_idx].astype(np.float64).copy()
    rows: list[np.ndarray] = [self_index]
    cols: list[np.ndarray] = [self_index]
    vals: list[np.ndarray] = [np.full(n, 4.0 + kappa_sq)]

    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_row, n_col = rows_idx + d_row, cols_idx + d_col
        inside = (n_row >= 0) & (n_row < height) & (n_col >= 0) & (n_col < width)
        # Mirrored neighbour == the pixel itself, so an out-of-frame edge just
        # removes one unit from the diagonal.
        outside = ~inside
        if outside.any():
            rows.append(self_index[outside])
            cols.append(self_index[outside])
            vals.append(np.full(int(outside.sum()), -1.0))
        n_row = np.clip(n_row, 0, height - 1)
        n_col = np.clip(n_col, 0, width - 1)
        neighbour_unknown = inside & unknown[n_row, n_col]
        rows.append(self_index[neighbour_unknown])
        cols.append(order[n_row[neighbour_unknown], n_col[neighbour_unknown]])
        vals.append(np.full(int(neighbour_unknown.sum()), -1.0))
        neighbour_known = inside & ~unknown[n_row, n_col]
        rhs[neighbour_known] += theta[n_row[neighbour_known], n_col[neighbour_known]]

    matrix = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    out[rows_idx, cols_idx] = spla.spsolve(matrix.tocsc(), rhs)
    return out


def outward_flux(theta: np.ndarray, region: np.ndarray) -> float:
    """Net heat flow leaving ``region``, in units of the plate stiffness k_eff d.

    Discrete divergence theorem: every 4-neighbour edge that crosses the region
    boundary carries ``theta_inside - theta_outside``; interior edges cancel.
    Edges that leave the frame carry nothing, matching the adiabatic border of
    :func:`solve_plate`.  Because the same stiffness scales both this integral
    and the operator, the value can be fed straight back in as a source.
    """
    height, width = theta.shape
    total = 0.0
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        row_a = slice(max(0, -d_row), height - max(0, d_row))
        col_a = slice(max(0, -d_col), width - max(0, d_col))
        row_b = slice(max(0, d_row), height - max(0, -d_row))
        col_b = slice(max(0, d_col), width - max(0, -d_col))
        leaving = region[row_a, col_a] & ~region[row_b, col_b]
        total += float((theta[row_a, col_a][leaving] - theta[row_b, col_b][leaving]).sum())
    return total


def infer_region_source(
    theta: np.ndarray,
    region: np.ndarray,
    fin_length_px: float = DEFAULT_FIN_LENGTH_PX,
    gaps_px: tuple[int, ...] = (4, 6, 8, 10),
    mask: np.ndarray | None = None,
) -> float | None:
    """Power dissipated under a footprint, from the heat leaving it sideways.

    The divergence theorem turns a contour integral of the measured field into
    the power enclosed: everything generated inside a contour either crosses it
    or is lost to air on the way.  Both terms are in units of the plate
    stiffness k_eff d, which is also the unit :func:`solve_plate` wants for its
    source, so the number can be fed straight back in and no material constant
    is needed.

    ``theta`` must be a *repaired* field.  On raw data this integral is
    worthless: the radiometric step at a shield edge is a huge dipole in the
    Laplacian, and the same footprint measures 42, 107 and 81 depending on where
    the contour is drawn.  After the artifacts are filled the same contours give
    56, 56, 53, 55, 51 -- which is why this is a second pass and not a first.

    The contour integral measures the power enclosed by the *contour*, not by
    the footprint, so it is only attributable to the footprint when the annulus
    between them is clean board.  ``mask`` (when given) is used to refuse the
    estimate otherwise: a sprawling connector bank whose contour sweeps in its
    neighbours reports 1.75 W of "dissipation" that belongs to the board.

    Returns ``None`` when the contours disagree, when the annulus is crowded,
    or when the enclosed power is not positive -- i.e. when the footprint shows
    no trustworthy evidence of a source.
    """
    values: list[float] = []
    kappa_sq = 1.0 / float(fin_length_px) ** 2
    for gap in gaps_px:
        contour = binary_dilation(region, CROSS, iterations=int(gap))
        if mask is not None:
            annulus = contour & ~region
            if not annulus.any() or float((annulus & mask).mean() / annulus.mean()) > 0.10:
                return None
        values.append(outward_flux(theta, contour) + kappa_sq * float(theta[contour].sum()))
    estimate = float(np.median(values))
    if estimate <= 0.0:
        return None
    spread = float(np.max(values) - np.min(values))
    if spread > 0.5 * abs(estimate):  # contours disagree: not a trustworthy source
        return None
    return estimate


def distance_to_trusted(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from each masked pixel to the nearest trusted pixel.

    This is the natural confidence coordinate for a conduction fill.  A pixel
    one step from real data is pinned by it; a pixel in the middle of a wide
    footprint is an extrapolation, and the measured error grows with this
    distance (see ``scripts/validate_emissivity.py``).  Trusted pixels get 0.
    """
    return distance_transform_edt(np.asarray(mask, dtype=bool))


def _drop_below_local_envelope(theta: np.ndarray, window_px: int, percentile: float) -> np.ndarray:
    """How far each pixel sits below the upper envelope of its neighbourhood.

    The window is deliberately small (a few pixels).  A cool *board* -- the PCB
    perimeter, the region far from the SoC -- falls off over the plate's
    screening length, tens of pixels, so within a small window it barely drops
    below its own envelope.  A shield edge drops the same amount within two or
    three pixels.  Sharpness, not coldness, is what separates a radiometric
    artifact from a genuinely cool part of the board.
    """
    envelope = percentile_filter(theta, float(percentile), size=int(window_px), mode="nearest")
    return envelope - theta


@dataclass
class Detection:
    """Outcome of :func:`detect_low_emissivity`."""

    mask: np.ndarray  # bool (H, W): artifact footprints, halo included
    seed: np.ndarray  # bool (H, W): the sharp-drop seed the growth started from
    iterations: int
    history: list[dict] = field(default_factory=list)
    stopped_on: str = ""


def detect_low_emissivity(
    theta: np.ndarray,
    fin_length_px: float = DEFAULT_FIN_LENGTH_PX,
    seed_drop_c: float = 4.0,
    seed_rel_drop: float = 0.12,
    seed_window_px: int = 9,
    seed_percentile: float = 80.0,
    grow_drop_c: float = 4.0,
    probe_px: int = 6,
    min_region_px: int = 6,
    halo_px: int = 2,
    max_masked_fraction: float = 0.55,
    max_iterations: int = 40,
) -> Detection:
    """Locate low-emissivity footprints from one excess-temperature field.

    Three stages, all physical:

    1. *Seed* on drops too sharp for conduction -- a pixel far below the upper
       envelope of a small neighbourhood (:func:`_drop_below_local_envelope`).
       This deliberately under-detects: it finds shield edges, not shield
       middles, and it ignores the smooth cool perimeter of the board.
    2. *Grow* on the conduction residual, holding out a band ``probe_px`` wide
       around the current mask, solving the plate without it, and absorbing the
       pixels the solution says should be at least ``grow_drop_c`` hotter than
       they read.  The band has to be thick.  A one-pixel probe cannot find the
       middle of a 25 mm connector bank, because its immediate neighbours are
       equally wrong and anchor the prediction to the artifact itself; a
       six-pixel band reaches far enough that trusted board dominates the
       solve, and successive iterations march inward until the whole footprint
       is covered.
    3. *Absorb enclosed pockets.*  Where a seeded edge does close a loop, the
       interior is a connected pocket of not-yet-masked pixels cut off from the
       board.  Each such pocket is held out whole and tested the same way,
       which settles a compact component in one step.  Pockets touching the
       frame border are included, which a hole fill cannot do.

    Growth stops on its own: once every boundary sits on genuinely trusted
    board, prediction and measurement agree and nothing more qualifies.

    ``halo_px`` dilates the result. Camera MTF smears a shield edge over a
    couple of pixels, and those transition pixels are neither trusted data nor
    part of the component; including them keeps them out of the Dirichlet ring.

    ``grow_drop_c`` has to clear the fill's own error, or the mask grows on
    model error instead of on evidence.  The held-out measurement puts that
    error at p95 = 1.9 C for a surrounded patch but 4.2 C for a strip anchored
    to the frame edge, with a positive bias -- one-sided fills run warm -- which
    is what makes the cool board perimeter the place a low threshold overruns.
    The default of 4.0 C sits above the surrounded case with room to spare and
    at the edge of the one-sided case; lowering it to 3.0 C on this board
    swallows the whole perimeter and doubles the deepest extrapolation from
    18 px to 33 px.
    """
    if theta.ndim != 2:
        raise ValueError(f"detect_low_emissivity expects a 2-D field, got shape {theta.shape}")
    if not 0.0 < max_masked_fraction <= 1.0:
        raise ValueError(f"max_masked_fraction must be in (0, 1], got {max_masked_fraction}")

    drop = _drop_below_local_envelope(theta, seed_window_px, seed_percentile)
    envelope = theta + drop
    threshold = np.maximum(float(seed_drop_c), float(seed_rel_drop) * np.maximum(envelope, 0.0))
    seed = drop > threshold
    seed = binary_opening(seed, CROSS)
    seed = binary_fill_holes(seed)

    labels, n_labels = label(seed, structure=CROSS)
    if n_labels:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        seed = np.isin(labels, np.nonzero(sizes >= int(min_region_px))[0])

    mask = seed.copy()
    budget = int(max_masked_fraction * theta.size)
    history: list[dict] = []
    stopped_on = "converged"
    iterations = 0

    if not mask.any():
        return Detection(mask=mask, seed=seed, iterations=0, history=history, stopped_on="no seed")

    for iterations in range(1, int(max_iterations) + 1):
        unknown = binary_dilation(mask, CROSS, iterations=int(halo_px)) if halo_px else mask

        # Every pocket of trusted-looking pixels that the mask has cut off from
        # the main body is a candidate artifact interior.  The main body is the
        # largest pocket by area; it is the board and is never absorbed.
        pockets, n_pockets = label(~unknown, structure=CROSS)
        pocket_areas = np.bincount(pockets.ravel(), minlength=n_pockets + 1)
        pocket_areas[0] = 0
        body = int(np.argmax(pocket_areas)) if n_pockets else 0
        candidates = np.isin(pockets, [i for i in range(1, n_pockets + 1) if i != body])

        # The frontier has to be solved for, not just read back: a pixel left in
        # the Dirichlet set is reproduced exactly by the solve and would always
        # test as consistent with conduction.  Holding it out makes the test a
        # genuine prediction from the trusted pixels beyond it.
        frontier = (
            binary_dilation(unknown, CROSS, iterations=max(int(probe_px), 1))
            & ~unknown
            & ~candidates
        )
        predicted = solve_plate(
            theta, unknown | frontier | candidates, fin_length_px=fin_length_px
        )
        residual = predicted - theta  # positive: colder than conduction allows

        add = frontier & (residual > float(grow_drop_c))
        absorbed_px = 0
        for pocket_id in range(1, n_pockets + 1):
            if pocket_id == body or pocket_areas[pocket_id] == 0:
                continue
            pocket = pockets == pocket_id
            if float(np.median(residual[pocket])) > float(grow_drop_c):
                add = add | pocket
                absorbed_px += int(pocket.sum())

        history.append(
            {
                "iteration": iterations,
                "mask_px": int(mask.sum()),
                "unknown_px": int(unknown.sum()),
                "added_px": int(add.sum()),
                "absorbed_pocket_px": absorbed_px,
                "max_frontier_residual_c": (
                    float(residual[frontier].max()) if frontier.any() else 0.0
                ),
            }
        )
        if not add.any():
            break
        mask = mask | add
        if int(mask.sum()) > budget:
            stopped_on = "hit max_masked_fraction"
            break
    else:
        stopped_on = "hit max_iterations"

    mask = binary_fill_holes(mask)
    if halo_px:
        mask = binary_dilation(mask, CROSS, iterations=int(halo_px))
    return Detection(
        mask=mask, seed=seed, iterations=iterations, history=history, stopped_on=stopped_on
    )


def _radiance(temperature_c: np.ndarray | float, exponent: float) -> np.ndarray | float:
    return np.power(np.asarray(temperature_c, dtype=np.float64) + 273.15, exponent)


def correct_for_emissivity(
    apparent_c: np.ndarray,
    emissivity: float,
    assumed_emissivity: float,
    reflected_c: float,
    exponent: float = 4.0,
) -> np.ndarray:
    """Re-derive temperature from the same radiance under a different emissivity.

    The camera reported ``apparent_c`` by solving
    ``W_meas = eps_set W(T_app) + (1 - eps_set) W(T_refl)``.  The surface
    actually obeys ``W_meas = eps W(T_true) + (1 - eps) W(T_refl)``.  Equating
    the two measurements of the same radiance gives

        W(T_true) = (eps_set / eps) W(T_app) + (1 - eps_set / eps) W(T_refl),

    with ``W(T) = (T + 273.15)^n``.  The band exponent ``n`` is 4 for a
    grey body integrated over all wavelengths and close to 4 for LWIR near room
    temperature, so the default is 4.
    """
    if not 0.0 < emissivity <= 1.0:
        raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
    if not 0.0 < assumed_emissivity <= 1.0:
        raise ValueError(f"assumed_emissivity must be in (0, 1], got {assumed_emissivity}")
    ratio = assumed_emissivity / emissivity
    radiance = ratio * _radiance(apparent_c, exponent) + (1.0 - ratio) * _radiance(
        reflected_c, exponent
    )
    # A region whose fitted emissivity is too low drives the implied radiance
    # negative; the caller decides what to do, but returning NaN here beats
    # returning a complex root.
    radiance = np.where(radiance > 0.0, radiance, np.nan)
    return np.power(radiance, 1.0 / exponent) - 273.15


def fit_region_emissivity(
    apparent_c: np.ndarray,
    target_mean_c: float,
    assumed_emissivity: float,
    reflected_c: float,
    exponent: float = 4.0,
    bounds: tuple[float, float] = (0.03, 0.95),
) -> float | None:
    """Effective emissivity that lifts a region's mean onto ``target_mean_c``.

    Solving in radiance rather than in temperature makes this a closed form:
    averaging the correction over the region and demanding the result match the
    conduction estimate gives one linear equation in ``eps_set / eps``.

    Returns ``None`` when the solution leaves ``bounds`` -- a fitted emissivity
    above the assumed one would mean the region reads *cold* because it is
    cold, and one near zero means the region carries no usable signal.  Either
    way the caller should keep the plain conduction fill instead.
    """
    mean_apparent = float(np.mean(_radiance(apparent_c, exponent)))
    reflected = float(_radiance(reflected_c, exponent))
    target = float(_radiance(target_mean_c, exponent))
    denominator = mean_apparent - reflected
    if abs(denominator) < 1e-9:
        return None
    ratio = (target - reflected) / denominator  # eps_set / eps
    if ratio <= 0.0:
        return None
    emissivity = assumed_emissivity / ratio
    if not bounds[0] <= emissivity <= bounds[1]:
        return None
    return float(emissivity)


@dataclass
class Anchor:
    """A known temperature inside a footprint, from a sensor the camera cannot see.

    An on-die sensor (``vcgencmd measure_temp``) measures the one place the IR
    field has no information about: under a package lid.  Supplying it turns the
    fill from an extrapolation into an interpolation between two knowns -- the
    trusted board outside and the sensor inside -- which is the only way a
    conduction solve can put a maximum inside a footprint.

    ``radius_px`` spreads the value over a small disc rather than a single
    pixel.  A metal lid is a good conductor and close to isothermal, so a disc
    is the more faithful boundary condition and it keeps the solve away from
    the pinpoint singularity a one-pixel Dirichlet spot would create.

    ``fill_region`` sets the whole footprint the anchor lands in to the anchor
    value, ignoring ``radius_px``.  It leaves nothing of the corrupted patch
    behind, but the result is not a solution of anything: an isothermal plateau
    carries no internal flux and meets the trusted board at a step.  Prefer
    ``as_source``.

    ``as_source`` treats the reading as a *target* rather than a value to
    impose.  The footprint is given a heat source, and its strength is
    calibrated so the solved field reaches the sensor temperature at the sensor
    location.  Nothing is pasted in: the profile inside the footprint is
    generated by the plate equation, so it peaks over the die, falls off
    towards the edges, and joins the surrounding board smoothly.  This is the
    physically faithful way to use an on-die reading, and it also returns the
    package power that the reading implies.

    ``temperature_c`` is imposed as given.  A die sensor reads the junction,
    which is hotter than the lid and hotter than the board plane under it, so
    subtract the junction-to-board rise first if what you want out is the board
    plane; ``RegionReport.anchor_lift_c`` reports what the anchor added over the
    unanchored fill, which is the quantity to watch across load states.
    """

    row: float
    col: float
    temperature_c: float
    radius_px: float = 3.0
    label: str = ""
    fill_region: bool = False
    as_source: bool = False


def anchor_mask(shape: tuple[int, int], anchors: list[Anchor]) -> np.ndarray:
    """Discs covered by ``anchors``, as a boolean mask."""
    rows, cols = np.indices(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for item in anchors:
        if not np.isfinite([item.row, item.col, item.temperature_c, item.radius_px]).all():
            raise ValueError(f"anchor {item!r} has non-finite values")
        if item.radius_px < 0:
            raise ValueError(f"anchor radius must be non-negative, got {item.radius_px}")
        disc = (rows - item.row) ** 2 + (cols - item.col) ** 2 <= item.radius_px**2
        if not disc.any():  # radius smaller than a pixel: fall back to the nearest one
            row = int(round(item.row))
            col = int(round(item.col))
            if not (0 <= row < shape[0] and 0 <= col < shape[1]):
                raise ValueError(f"anchor {item!r} lies outside the frame {shape}")
            disc[row, col] = True
        mask |= disc
    return mask


@dataclass
class RegionReport:
    """Per-region record of what was done and why."""

    region_id: int
    area_px: int
    centroid_rc: tuple[float, float]
    bbox: tuple[int, int, int, int]  # row0, row1, col0, col1
    apparent_mean_c: float
    apparent_min_c: float
    repaired_mean_c: float
    repaired_max_c: float
    ring_mean_c: float
    lift_c: float
    fitted_emissivity: float | None
    method: str  # "emissivity" or "conduction"
    touches_border: bool
    net_outward_flux: float  # k_eff d units, measured on the repaired field
    max_distance_to_trusted_px: float  # deepest extrapolation inside this region
    anchor_labels: list[str]  # anchors pinned inside this region, if any
    anchor_lift_c: float | None  # region mean gained by anchoring, vs boundary data alone
    inferred_source: float | None  # power under the footprint, k_eff d units
    source_lift_c: float | None  # region mean gained by the source term


@dataclass
class RepairResult:
    """Outcome of :func:`repair_frame`."""

    temperature_c: np.ndarray  # (H, W) float32, repaired
    mask: np.ndarray  # bool (H, W), pixels that were replaced
    conduction_c: np.ndarray  # (H, W) float32, conduction-only fill
    ambient_c: float
    regions: list[RegionReport]
    detection: Detection
    anchors: list[Anchor]
    unanchored_c: np.ndarray | None = None  # fill from boundary data alone, when anchored

    @property
    def masked_fraction(self) -> float:
        return float(self.mask.mean())


def repair_frame(
    temperature_c: np.ndarray,
    *,
    ambient_c: float | None = None,
    mask: np.ndarray | None = None,
    fin_length_px: float = DEFAULT_FIN_LENGTH_PX,
    assumed_emissivity: float = 0.95,
    reflected_c: float | None = None,
    band_exponent: float = 4.0,
    method: str = "conduction",
    anchors: list[Anchor] | None = None,
    infer_sources: bool = False,
    core_erosion_px: int = 3,
    apply_emissivity_min: float = 0.5,
    flux_gap_px: int = 6,
    **detect_kwargs,
) -> RepairResult:
    """Detect low-emissivity footprints and repair them by plate conduction.

    ``method="conduction"`` (the default) replaces every masked pixel with the
    plate solution.  It is the conservative choice: the repaired value depends
    only on trusted neighbours and on geometry, never on the corrupted reading
    itself.

    ``infer_sources`` runs a second pass that treats each footprint as what it
    physically is -- a component that may be *generating* heat -- instead of a
    passive hole.  The power under a footprint is measured from the heat
    crossing a contour around it (:func:`infer_region_source`) and fed back as
    the source term, which is the only way a solve can raise the interior above
    its own boundary without an external sensor.  On the CPU footprint this
    recovers 1.97 W and +2.5 C, landing 1.5 C under the on-die reading.

    ``anchors`` adds known interior temperatures -- an on-die sensor reading,
    typically -- as extra Dirichlet data.  Anchor pixels leave the unknown set,
    so the same conduction solve now interpolates between the sensor inside a
    footprint and the trusted board outside it instead of extrapolating from
    the outside alone.  This is what lets the reconstruction carry a maximum
    inside a package footprint at all; the boundary-only fill is kept in
    ``RepairResult.unanchored_c`` so the two can be compared.

    ``method="emissivity"`` additionally uses the conduction level to identify
    one effective emissivity per region and re-derives each *core* pixel from
    the radiance the camera actually measured, which keeps whatever real
    structure the component has instead of pasting a smooth harmonic patch over
    it.  Only the core is treated this way.  Pixels within ``core_erosion_px``
    of the footprint edge are mixed -- part board, part shield -- and dividing a
    mixed pixel by a low emissivity turns a 55 C edge sample into a 76 C
    artifact, so the rim is bridged by a second conduction solve that takes the
    corrected core as additional boundary data.

    The inversion amplifies everything it touches, correct signal and error
    alike, by roughly ``assumed_emissivity / fitted_emissivity``.  Around
    eps = 0.3 -- what bare shields and package lids actually fit -- that is a
    threefold amplification of every stray pixel, which is why
    ``apply_emissivity_min`` (0.5 by default) restricts the substitution to
    regions where it is well conditioned.  Regions below the floor, regions with
    no core left, and regions whose fit is unphysical keep the conduction fill.
    The fitted emissivity is reported either way: a shield that fits eps = 0.3 is
    a useful, checkable statement about the surface even when it is not used to
    rebuild the pixels.

    Pass ``mask`` to reuse a mask agreed once for a board -- every frame of a
    stationary recording rectifies into the same coordinates, so the footprints
    do not move, and a fixed mask makes the repair reproducible across a
    session.
    """
    field_c = np.asarray(temperature_c, dtype=np.float64)
    if field_c.ndim != 2:
        raise ValueError(f"repair_frame expects a 2-D frame, got shape {field_c.shape}")
    if not np.isfinite(field_c).all():
        raise ValueError("repair_frame requires a finite frame")
    if method not in ("conduction", "emissivity"):
        raise ValueError(f"method must be 'conduction' or 'emissivity', got {method!r}")

    ambient = estimate_ambient(field_c) if ambient_c is None else float(ambient_c)
    reflected = ambient if reflected_c is None else float(reflected_c)
    theta = field_c - ambient

    if mask is None:
        detection = detect_low_emissivity(theta, fin_length_px=fin_length_px, **detect_kwargs)
        mask = detection.mask
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != field_c.shape:
            raise ValueError(f"mask {mask.shape} does not match frame {field_c.shape}")
        if detect_kwargs:
            raise ValueError("detection options are meaningless when an explicit mask is given")
        detection = Detection(mask=mask, seed=mask, iterations=0, stopped_on="mask supplied")

    labels, n_labels = label(mask, structure=CROSS)
    height, width = field_c.shape

    anchors = list(anchors or [])
    calibrated = [item for item in anchors if item.as_source]
    anchors = [item for item in anchors if not item.as_source]
    unanchored: np.ndarray | None = None
    pinned = np.zeros(field_c.shape, dtype=bool)
    anchor_region: dict[int, list[str]] = {}
    if anchors:
        # Keep the boundary-only solution so the anchor's contribution is visible.
        unanchored = solve_plate(theta, mask, fin_length_px=fin_length_px) + ambient
        for item in anchors:
            row, col = int(round(item.row)), int(round(item.col))
            name = item.label or f"anchor@{item.row:.0f},{item.col:.0f}"
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(f"anchor {name!r} at ({row}, {col}) lies outside the frame")
            region_id = int(labels[row, col])
            if region_id == 0:
                raise ValueError(
                    f"anchor {name!r} at ({row}, {col}) is not inside a repaired footprint; "
                    "an anchor only means something where the fill has to invent a value"
                )
            # Clip to the footprint the anchor sits in, not merely to the mask:
            # a generous radius must not leak a CPU reading into the neighbouring
            # connector shield, nor overwrite trusted board with it.
            footprint = labels == region_id
            disc = footprint if item.fill_region else anchor_mask(field_c.shape, [item]) & footprint
            theta[disc] = float(item.temperature_c) - ambient
            pinned |= disc
            anchor_region.setdefault(region_id, []).append(name)

    unknown = mask & ~pinned
    conduction = solve_plate(theta, unknown, fin_length_px=fin_length_px) + ambient

    # Calibrated sources: the sensor reading is a target, not a value to paste.
    # The solve is affine in the source, so one extra solve with unit power
    # gives the response and the required strength follows in closed form --
    # no search, and the profile inside the footprint is whatever the plate
    # equation produces for that power.
    sources: dict[int, float] = {}
    source_lift: dict[int, float] = {}
    # One accumulated source field: a calibrated anchor and an inferred source
    # must end up in the same solve, not in two that overwrite each other.
    source_map = np.zeros(field_c.shape, dtype=np.float64)
    if calibrated:
        for item in calibrated:
            row, col = int(round(item.row)), int(round(item.col))
            name = item.label or f"anchor@{item.row:.0f},{item.col:.0f}"
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(f"anchor {name!r} at ({row}, {col}) lies outside the frame")
            region_id = int(labels[row, col])
            if region_id == 0:
                raise ValueError(
                    f"anchor {name!r} at ({row}, {col}) is not inside a repaired footprint; "
                    "an anchor only means something where the fill has to invent a value"
                )
            region = labels == region_id
            unit = np.zeros(field_c.shape, dtype=np.float64)
            unit[region] = 1.0 / int(region.sum())
            response = (
                solve_plate(theta, unknown, fin_length_px=fin_length_px, source=unit)
                - solve_plate(theta, unknown, fin_length_px=fin_length_px)
            )
            if response[row, col] <= 1e-12:
                raise ValueError(f"anchor {name!r} sits where a source has no leverage")
            deficit = float(item.temperature_c) - float(conduction[row, col])
            power = deficit / float(response[row, col])
            if power <= 0.0:
                # The reading is at or below what conduction already gives: the
                # package is not driving its surroundings, so invent no source.
                anchor_region.setdefault(region_id, []).append(name)
                continue
            source_map[region] += power / int(region.sum())
            sources[region_id] = sources.get(region_id, 0.0) + power
            anchor_region.setdefault(region_id, []).append(name)
        if sources:
            with_source = (
                solve_plate(theta, unknown, fin_length_px=fin_length_px, source=source_map)
                + ambient
            )
            for region_id in sources:
                region = labels == region_id
                source_lift[region_id] = float(
                    with_source[region].mean() - conduction[region].mean()
                )
            if unanchored is None:
                unanchored = conduction
            conduction = with_source

    # Second pass: let each footprint carry the power its surroundings say is
    # under it.  A homogeneous fill has no interior maximum by construction, so
    # a powered package can only be reconstructed if the source term is put
    # back -- and the surrounding flux can only be measured once the first pass
    # has removed the radiometric steps that otherwise dominate it.
    if infer_sources:
        clean = conduction - ambient
        inferred: dict[int, float] = {}
        for region_id in range(1, n_labels + 1):
            if region_id in anchor_region or region_id in sources:
                continue  # already constrained by a sensor; a guess would fight it
            region = labels == region_id
            rows_r, cols_r = np.nonzero(region)
            if (
                rows_r.min() == 0
                or cols_r.min() == 0
                or rows_r.max() == height - 1
                or cols_r.max() == width - 1
            ):
                continue  # the adiabatic frame edge makes the balance unreliable
            power = infer_region_source(
                clean, region, fin_length_px=fin_length_px, mask=mask
            )
            if power is None:
                continue
            inferred[region_id] = power
            source_map[region] += power / int(region.sum())
        if inferred:
            with_source = (
                solve_plate(theta, unknown, fin_length_px=fin_length_px, source=source_map)
                + ambient
            )
            for region_id, power in inferred.items():
                region = labels == region_id
                source_lift[region_id] = float(
                    with_source[region].mean() - conduction[region].mean()
                )
                sources[region_id] = power
            conduction = with_source

    repaired = conduction.copy()
    fitted: dict[int, float] = {}
    applied: set[int] = set()
    corrected_core = np.zeros_like(mask)
    for region_id in range(1, n_labels + 1):
        region = labels == region_id
        core = binary_erosion(region, CROSS, iterations=int(core_erosion_px)) & ~pinned
        if not core.any():
            continue
        emissivity = fit_region_emissivity(
            field_c[core],
            float(conduction[core].mean()),
            assumed_emissivity,
            reflected,
            band_exponent,
        )
        if emissivity is None:
            continue
        fitted[region_id] = emissivity
        if method != "emissivity" or emissivity < float(apply_emissivity_min):
            continue
        values = correct_for_emissivity(
            field_c[core], emissivity, assumed_emissivity, reflected, band_exponent
        )
        if not np.isfinite(values).all():
            continue
        repaired[core] = values
        corrected_core |= core
        applied.add(region_id)
    if method == "emissivity":
        if corrected_core.any():
            # Re-solve the rim with the corrected cores pinned as boundary data,
            # so the emissivity patch joins the trusted board without a seam.
            rim = mask & ~corrected_core & ~pinned
            repaired = solve_plate(repaired - ambient, rim, fin_length_px=fin_length_px) + ambient


    regions: list[RegionReport] = []
    distance = distance_to_trusted(mask)
    for region_id in range(1, n_labels + 1):
        region = labels == region_id
        area = int(region.sum())
        rows_idx, cols_idx = np.nonzero(region)
        ring = binary_dilation(region, CROSS, iterations=2) & ~mask
        ring_mean = float(field_c[ring].mean()) if ring.any() else float("nan")
        emissivity = fitted.get(region_id)
        region_method = "emissivity" if region_id in applied else "conduction"
        region_anchors = anchor_region.get(region_id, [])
        anchor_lift = (
            float(repaired[region].mean() - unanchored[region].mean())
            if region_anchors and unanchored is not None
            else None
        )

        # Measured on the repaired field: with the artifact rings gone, the
        # divergence theorem on a contour a few pixels outside the footprint is
        # no longer dominated by the radiometric step, so this number is worth
        # reporting as evidence of whether the footprint hides a source.
        contour = binary_dilation(region, CROSS, iterations=int(flux_gap_px))
        flux = outward_flux(repaired - ambient, contour)

        regions.append(
            RegionReport(
                region_id=region_id,
                area_px=area,
                centroid_rc=(float(rows_idx.mean()), float(cols_idx.mean())),
                bbox=(
                    int(rows_idx.min()),
                    int(rows_idx.max()) + 1,
                    int(cols_idx.min()),
                    int(cols_idx.max()) + 1,
                ),
                apparent_mean_c=float(field_c[region].mean()),
                apparent_min_c=float(field_c[region].min()),
                repaired_mean_c=float(repaired[region].mean()),
                repaired_max_c=float(repaired[region].max()),
                ring_mean_c=ring_mean,
                lift_c=float(repaired[region].mean() - field_c[region].mean()),
                fitted_emissivity=emissivity,
                method=region_method,
                touches_border=bool(
                    rows_idx.min() == 0
                    or cols_idx.min() == 0
                    or rows_idx.max() == height - 1
                    or cols_idx.max() == width - 1
                ),
                net_outward_flux=flux,
                max_distance_to_trusted_px=float(distance[region].max()),
                anchor_labels=region_anchors,
                anchor_lift_c=anchor_lift,
                inferred_source=sources.get(region_id),
                source_lift_c=source_lift.get(region_id),
            )
        )

    return RepairResult(
        temperature_c=repaired.astype(np.float32),
        mask=mask,
        conduction_c=conduction.astype(np.float32),
        ambient_c=ambient,
        regions=regions,
        detection=detection,
        anchors=anchors,
        unanchored_c=unanchored.astype(np.float32) if unanchored is not None else None,
    )
