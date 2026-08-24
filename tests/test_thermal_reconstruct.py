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

from tdp.data.emissivity import solve_plate
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
    forbidden = ("watt", "_w", "power", "conductivity", "heat_capacity", "joule")
    assert not [n for n in names if any(f in n.lower() for f in forbidden)]
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
