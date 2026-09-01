"""Grade the source-driven fill on a board that was generated, so truth is known.

Real infrared data cannot grade a fill under a metal lid -- nothing measured what
is under there.  case17 comes closest and is used for the honest error bar, but it
is one board.  These tests build a plate whose sources, spread rate and cooling
rate are chosen rather than fitted, hide part of it, and require the fill to bring
it back.

The negative control matters as much as the recovery test.  A method that invents a
source under a block will invent one everywhere, and on real data there is nothing
to catch it: the block is masked precisely because its own reading cannot be
trusted.  :func:`test_uniform_board_gets_no_spurious_source` is the guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

from tdp.data.emissivity import CROSS, solve_plate
from tdp.thermal import reconstruct as rc

GRID = (70, 50)
COOLING_RATE_PER_S = 0.007
SPREAD_RATE_PX2_PER_S = 1.0
FRAME_SECONDS = 6.0
N_FRAMES = 60


def _laplacian(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="reflect")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * field
    )


def _evolve(source_k_per_s: np.ndarray, start_k: float = 8.0, step_s: float = 0.05):
    """March the plate equation forward and sample it on the camera's cadence.

    The explicit step needs ``step_s < 1 / (4 * spread rate)`` = 0.125 s here; 0.05 s
    keeps it comfortably stable.  The board starts warm, not at ambient, because the
    real one does -- case10 frame 0 already sits 11.3 K up.
    """
    field = np.full(GRID, start_k, dtype=np.float64)
    per_frame = int(round(FRAME_SECONDS / step_s))
    frames = [field.copy()]
    for _ in range(N_FRAMES - 1):
        for _ in range(per_frame):
            field = field + step_s * (
                SPREAD_RATE_PX2_PER_S * _laplacian(field)
                - COOLING_RATE_PER_S * field
                + source_k_per_s
            )
        frames.append(field.copy())
    times = np.arange(N_FRAMES, dtype=np.float64) * FRAME_SECONDS
    return np.stack(frames), times


def _blob(centre, half) -> np.ndarray:
    rows, cols = np.mgrid[0 : GRID[0], 0 : GRID[1]]
    return (np.abs(rows - centre[0]) <= half[0]) & (np.abs(cols - centre[1]) <= half[1])


BACKGROUND_K_PER_S = 0.05
HOT_BLOCK_RATE = 0.40

# Sized against the real board, not for convenience.  What decides whether a block
# can be filled from its edges is its depth measured in spread distances: on case10
# the main chip is 15 px deep against a 17 px spread distance and the connector bank
# is 36 px deep.  Here the spread distance is 12 px, so a block 9-10 px deep puts the
# test in the same regime.  A smaller block makes the edge-only fill look far better
# than it is -- at half this size its bias is 0.8 K instead of 2 K.
HOT_BLOCK = _blob((22, 20), (9, 9))
COOL_BLOCK = _blob((50, 31), (7, 7))


@pytest.fixture(scope="module")
def board():
    """A board with one block that makes heat and one that only sits there."""
    source = np.full(GRID, BACKGROUND_K_PER_S)
    source[HOT_BLOCK] = HOT_BLOCK_RATE
    source[COOL_BLOCK] = 0.0
    theta, times = _evolve(source)
    mask = HOT_BLOCK | COOL_BLOCK
    return theta, times, mask, source


@pytest.fixture(scope="module")
def uniform_board():
    """The negative control: every pixel makes the same heat."""
    source = np.full(GRID, BACKGROUND_K_PER_S)
    theta, times = _evolve(source)
    return theta, times, HOT_BLOCK | COOL_BLOCK, source


def _fill(theta, times, mask, source):
    """Reconstruct, handing the ring its true heating rate -- it is trusted board."""
    return rc.reconstruct_sequence(
        theta,
        times,
        mask,
        source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6,
        pixel_size_mm=0.5425,
    )


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------
def test_warming_rate_is_exact_on_a_ramp_including_the_ends():
    """The one-sided windows at either end are where this went wrong before.

    Regressing on uncentred time without an intercept reported 0.126 K/s as
    1.40 K/s at frame 0, because there the window is one-sided.
    """
    times = np.arange(12, dtype=np.float64) * FRAME_SECONDS
    slope = 0.037
    theta = (slope * times)[:, None, None] + np.zeros((1,) + GRID)
    rate = rc.warming_rate(theta, times)
    assert np.allclose(rate, slope, atol=1e-9)


def test_warming_rate_rejects_a_single_frame():
    with pytest.raises(ValueError, match="at least two frames"):
        rc.warming_rate(np.zeros((1,) + GRID), np.zeros(1))


def test_plate_residual_vanishes_on_a_field_that_obeys_the_equation(board):
    """If the residual were not small, no fill built on this equation could be right."""
    theta, times, _, source = board
    warming = rc.warming_rate(theta, times)
    index = N_FRAMES - 1
    residual = rc.plate_residual(
        theta[index], warming[index], source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        smooth_px=0.0,
    )
    interior = np.zeros(GRID, dtype=bool)
    interior[6:-6, 6:-6] = True
    assert np.abs(residual[interior]).max() < 0.01 * HOT_BLOCK_RATE


# --------------------------------------------------------------------------
# what the fill is for
# --------------------------------------------------------------------------
def test_recovers_the_hidden_source_rates(board):
    """The fitted rate is heating minus warming, so it lands under the true heating."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    rates = {b.region_id: b.source_rate_k_per_s for b in result.blocks}
    hot = max(rates, key=lambda k: rates[k])
    cool = min(rates, key=lambda k: rates[k])
    assert rates[hot] == pytest.approx(HOT_BLOCK_RATE, rel=0.35)
    assert rates[cool] < 0.5 * rates[hot]
    assert len(result.blocks) == 2


def test_fill_beats_the_edge_only_fill_and_kills_its_cold_bias(board):
    """The claim the whole stage rests on: a source-free fill runs systematically cold."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    check = rc.validate_against_truth(result, theta)

    edge_only = np.stack(
        [solve_plate(theta[i], mask, result.spread_distance_px) for i in range(N_FRAMES)]
    )
    bias_edge_only = float((edge_only[:, mask] - theta[:, mask]).mean())
    error_edge_only = float(np.abs(edge_only[:, mask] - theta[:, mask]).mean())

    assert bias_edge_only < -1.0  # the old fill really is cold under a source
    assert abs(check.bias_k) < 0.3 * abs(bias_edge_only)
    assert check.mean_absolute_k < 0.6 * error_edge_only


def test_uniform_board_gets_no_spurious_source(uniform_board):
    """Negative control: nothing special is under either block, so say so."""
    theta, times, mask, source = uniform_board
    result = _fill(theta, times, mask, source)
    for block in result.blocks:
        assert abs(block.source_rate_k_per_s - BACKGROUND_K_PER_S) < 0.5 * BACKGROUND_K_PER_S
    check = rc.validate_against_truth(result, theta)
    assert check.mean_absolute_k < 0.3


def test_trusted_pixels_are_returned_untouched(board):
    """The fill writes only where the reading could not be trusted."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    assert np.array_equal(result.theta_k[:, ~mask], theta[:, ~mask])
    assert result.theta_k.shape == theta.shape


def test_holding_out_the_ring_is_what_pins_the_level(board):
    """With every source set to zero the same ring is much further out."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    assert np.median(result.ring_error_k) < 0.4 * np.median(result.ring_error_no_source_k)


def test_level_and_shape_split_is_zero_for_a_perfect_reconstruction(board):
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    perfect = rc.Reconstruction(
        theta_k=theta.copy(), times_s=times, unknown_mask=mask, blocks=result.blocks,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S, cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_distance_px=result.spread_distance_px,
        ring_error_k=result.ring_error_k, ring_error_no_source_k=result.ring_error_no_source_k,
    )
    check = rc.validate_against_truth(perfect, theta)
    assert check.mean_absolute_k == pytest.approx(0.0, abs=1e-12)
    assert check.level_error_k == pytest.approx(0.0, abs=1e-12)
    assert check.shape_error_k == pytest.approx(0.0, abs=1e-12)


def test_heating_rate_comes_back_out_of_the_reconstruction(board):
    """Fitted source is heating minus warming; adding the warming back gives heating."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    hot_id = max(result.blocks, key=lambda b: b.source_rate_k_per_s).region_id
    heating = rc.heating_rate_from_reconstruction(result, hot_id)
    assert heating.shape == (N_FRAMES,)
    # Early frames are still warming hard, so this is checked once the board settles.
    assert np.median(heating[N_FRAMES // 2 :]) == pytest.approx(HOT_BLOCK_RATE, rel=0.35)


# --------------------------------------------------------------------------
# the rules that must not be broken
# --------------------------------------------------------------------------
def test_nothing_reported_is_a_power(board):
    """Every rate here was divided by an unknown, spatially varying heat capacity."""
    theta, times, mask, source = board
    result = _fill(theta, times, mask, source)
    names = set(vars(result.blocks[0])) | set(result.report) | set(vars(result))
    forbidden = ("watt", "power", "conductivity", "heat_capacity", "joule")
    assert not [n for n in names if any(f in n.lower() for f in forbidden)]
    # "_w" has to be a suffix test, not a substring one: it is there to catch a
    # name like power_w, and as a substring it also flags n_centre_weighted.
    assert not [n for n in names if n.endswith("_w") or "_w_per_" in n or "_w_m" in n]
    # Every quantity carries its unit in its name, so a caller cannot mistake a
    # rate that was divided by heat capacity for one that was not.
    units = ("_k_per_s", "_per_s", "_px2_per_s", "_by_frame", "_px", "_mm", "_k")
    assert not [n for n in names if "rate" in n and not n.endswith(units)]


def test_refuses_an_empty_mask(board):
    theta, times, _, source = board
    with pytest.raises(ValueError, match="footprint mask is empty"):
        _fill(theta, times, np.zeros(GRID, dtype=bool), source)


def test_refuses_a_non_positive_spread_rate(board):
    theta, times, mask, source = board
    with pytest.raises(ValueError, match="must both be positive"):
        rc.reconstruct_sequence(theta, times, mask, source,
                       cooling_rate_per_s=COOLING_RATE_PER_S, spread_rate_px2_per_s=0.0)


# --------------------------------------------------------------------------
# the source's shape inside a block
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def die_board():
    """A block whose heat comes from a small die in the middle, not the whole area.

    This is the real geometry under an SoC lid: the die is a few mm across and the
    lid spreads its heat over ten times that area.  Spreading the fitted source
    uniformly over the whole block instead builds a flat-topped plateau, which
    reads about right at the block's edges and several kelvin cold in the middle.
    """
    source = np.full(GRID, BACKGROUND_K_PER_S)
    source[HOT_BLOCK] = 0.0
    # Concentrated by 3x, not into a point.  The test plate has one spread rate
    # everywhere and so has no lid to smooth a spike: a 25 px source carrying the
    # whole block's heat makes a 40 K peak that no single shape can represent, and
    # then both fills are equally hopeless (14.8 K against 18.8 K).  3x is the
    # regime the real board is in -- a dome, not a spike.
    die = _blob((22, 20), (5, 5))
    source[die] = HOT_BLOCK_RATE * HOT_BLOCK.sum() / die.sum()  # same total, concentrated
    source[COOL_BLOCK] = 0.0
    theta, times = _evolve(source)
    return theta, times, HOT_BLOCK | COOL_BLOCK, source


def _hot_id(mask):
    labels, _ = rc.block_labels(mask)
    return int(np.bincount(labels[HOT_BLOCK]).argmax())


def test_centre_weighted_source_beats_a_uniform_one_under_a_die(die_board):
    """The measured case: on case17 this takes the SoC block 4.38 K -> 2.51 K."""
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    flat = _fill(theta, times, mask, source)
    centred = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, pixel_size_mm=0.5425, heat_source_ids=[hot],
        lid_fill="source-shaped",
    )
    truth = theta
    flat_error = np.abs(flat.theta_k[:, HOT_BLOCK] - truth[:, HOT_BLOCK]).mean()
    centred_error = np.abs(centred.theta_k[:, HOT_BLOCK] - truth[:, HOT_BLOCK]).mean()
    assert centred_error < flat_error
    # and it is the middle of the block that gets fixed, not its rim: a uniform
    # source is roughly right at the edges by construction, since the edges are
    # pinned by measured board a pixel away.
    core = _blob((22, 20), (3, 3))
    rim = HOT_BLOCK & ~_blob((22, 20), (7, 7))
    core_gain = (np.abs(flat.theta_k[:, core] - truth[:, core]).mean()
                 - np.abs(centred.theta_k[:, core] - truth[:, core]).mean())
    rim_gain = (np.abs(flat.theta_k[:, rim] - truth[:, rim]).mean()
                - np.abs(centred.theta_k[:, rim] - truth[:, rim]).mean())
    assert core_gain > 0.0
    assert core_gain > rim_gain


def test_centre_weighting_is_recorded_on_the_block_it_was_applied_to(die_board):
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    result = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, pixel_size_mm=0.5425, heat_source_ids=[hot],
        lid_fill="source-shaped",
    )
    shapes = {b.region_id: b.source_shape for b in result.blocks}
    assert shapes[hot] == "centre-weighted"
    assert set(shapes.values()) - {"centre-weighted"} == {"uniform"}
    assert result.report["n_centre_weighted"] == 1


def test_broad_source_lid_keeps_the_fitted_level_and_is_not_flat(die_board):
    """The ring estimates the level; one plate solve supplies the smooth shape."""
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    bump = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, heat_source_ids=[hot], lid_fill="source-shaped",
    )
    broad = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, heat_source_ids=[hot], lid_fill="broad-source-plate",
    )
    labels, _ = rc.block_labels(mask)
    region = labels == hot
    assert np.allclose(
        broad.theta_k[:, region].mean(axis=1),
        bump.theta_k[:, region].mean(axis=1),
        atol=1e-9,
    )
    depth = distance_transform_edt(region)
    core = region & (depth >= 0.75 * depth.max())
    # A pasted plateau made every core pixel exactly identical.  A plate solution
    # is broad, but retains the small curvature required to carry heat outward.
    assert np.median(broad.theta_k[:, core].std(axis=1)) > 0.01
    assert next(b for b in broad.blocks if b.region_id == hot).interior_fill == (
        "broad-source plate"
    )
    assert broad.report["lid_broad_source_rate_median_k_per_s"][str(hot)] > 0.0
    assert broad.report["lid_broad_source_positive_fraction"][str(hot)] == 1.0
    block = next(b for b in broad.blocks if b.region_id == hot)
    assert block.reconstruction_source_rate_k_per_s == pytest.approx(
        broad.report["lid_broad_source_rate_median_k_per_s"][str(hot)]
    )


def test_broad_source_lid_rebuilds_the_edge_inside_one_plate_domain(die_board):
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    result = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, heat_source_ids=[hot], lid_fill="broad-source-plate",
    )
    # The extra rim means the old footprint border is no longer a Dirichlet seam:
    # pixels on both sides of it are unknowns in the same plate solve.
    labels, _ = rc.block_labels(mask)
    region = labels == hot
    transition = result.lid_transition_mask
    assert transition.any()
    assert np.any(transition & binary_dilation(region, CROSS))
    assert result.report["edge_slope_mismatch_median_k_per_px"] < 1.0

    # The final pixels themselves satisfy the plate equation: a spatially
    # constant positive source on the lid and zero source on its transition rim.
    field = result.theta_k[N_FRAMES // 2]
    operator = -_laplacian(field) + field / result.spread_distance_px**2
    assert np.std(operator[region]) < 1e-9
    assert np.mean(operator[region]) > 0.0
    assert np.max(np.abs(operator[transition])) < 1e-9
    block = next(b for b in result.blocks if b.region_id == hot)
    assert block.reconstruction_source_rate_by_frame[N_FRAMES // 2] == pytest.approx(
        float(operator[region].mean()) * SPREAD_RATE_PX2_PER_S
    )


def test_lid_uncertainty_is_explicit_and_only_applies_to_rebuilt_pixels(die_board):
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    result = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, heat_source_ids=[hot], lid_fill="broad-source-plate",
    )
    assert result.uncertainty_k is not None
    assert result.uncertainty_k.shape == theta.shape
    rebuilt = result.rebuilt_mask
    assert result.lid_transition_mask.any()
    assert np.array_equal(result.theta_k[:, ~rebuilt], theta[:, ~rebuilt])
    assert np.array_equal(
        result.uncertainty_k[:, ~rebuilt], np.zeros((N_FRAMES, (~rebuilt).sum()))
    )
    assert np.median(result.uncertainty_k[:, mask]) > 0.0
    assert "not a statistical confidence interval" in result.report["uncertainty_kind"]


def test_interior_temperature_anchor_sets_the_central_level_and_uncertainty(die_board):
    theta, times, mask, source = die_board
    hot = _hot_id(mask)
    target = np.linspace(31.0, 33.0, N_FRAMES)
    anchor = rc.InteriorTemperatureAnchor(
        region_id=hot,
        temperature_above_ambient_k_by_frame=target,
        uncertainty_k=0.4,
        label="corrected SoC sensor",
    )
    result = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, heat_source_ids=[hot], lid_fill="broad-source-plate",
        interior_temperature_anchors=[anchor],
    )
    labels, _ = rc.block_labels(mask)
    region = labels == hot
    depth = distance_transform_edt(region)
    core = region & (depth >= 0.75 * depth.max())
    assert np.allclose(result.theta_k[:, core].mean(axis=1), target, atol=1e-9)
    # The supplied sensor uncertainty is added to, not substituted for, the IR
    # model range because die-to-lid correction and spatial shape remain uncertain.
    assert np.all(result.uncertainty_k[:, core] >= 0.4)
    assert result.report["interior_temperature_anchor_labels"] == ["corrected SoC sensor"]


def test_no_heat_source_ids_reproduces_the_uniform_fit_exactly(board):
    """The ablation must be a true ablation, not a nearby variant."""
    theta, times, mask, source = board
    a = _fill(theta, times, mask, source)
    b = rc.reconstruct_sequence(
        theta, times, mask, source,
        cooling_rate_per_s=COOLING_RATE_PER_S,
        spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
        ring_px=6, pixel_size_mm=0.5425, heat_source_ids=(),
    )
    assert np.array_equal(a.theta_k, b.theta_k)
    assert a.report["n_centre_weighted"] == 0


def test_an_id_that_names_no_block_is_refused(board):
    """It used to do nothing at all.

    judge_footprints labelled with 4-connectivity and reconstruct_sequence with 8,
    so on a mask with a diagonal touch the ids disagreed and passing the edge
    test's verdict through silently selected the wrong block, or none.
    """
    theta, times, mask, source = board
    with pytest.raises(ValueError, match="name no block"):
        rc.reconstruct_sequence(
            theta, times, mask, source,
            cooling_rate_per_s=COOLING_RATE_PER_S,
            spread_rate_px2_per_s=SPREAD_RATE_PX2_PER_S,
            ring_px=6, heat_source_ids=[99],
        )


def test_the_edge_test_and_the_fill_label_blocks_the_same_way():
    """The driver joins their per-block tables on the region id."""
    from tdp.thermal import sources as src

    mask = np.zeros((30, 30), dtype=bool)
    mask[5:10, 5:10] = True
    mask[10:15, 10:15] = True  # touches the first block only at a corner
    theta = np.zeros((30, 30))
    verdicts = src.judge_footprints(theta, mask)
    labels, count = rc.block_labels(mask)
    assert count == 1, "8-connectivity must merge a diagonal touch"
    assert {v.region_id for v in verdicts} <= set(range(1, count + 1))


def test_the_figure_never_gives_contour_an_imshow_extent():
    """imshow reads (0, W, H, 0) as (left, right, bottom, top); contour reads the
    same tuple as (x0, x1, y0, y1) and maps row 0 to y = H.  Every block outline in
    every panel came out mirrored top-to-bottom until 2026-08-29.
    """
    import pathlib
    import re

    text = pathlib.Path("scripts/figure_reconstruct.py").read_text()
    for call in re.findall(r"contour\((?:[^()]|\([^()]*\))*\)", text, flags=re.S):
        assert "extent" not in call, f"contour with an extent is mirrored: {call[:70]}"


def test_the_figure_does_not_hide_reconstructed_pixels_with_the_analysis_mask():
    """Deep extrapolations are low-confidence, not outside the PCB."""
    import pathlib

    text = pathlib.Path("scripts/figure_reconstruct.py").read_text()
    assert "np.where(board, field" not in text
    assert "np.where(board, diff" not in text
    assert "np.where(board, fieldset" not in text
