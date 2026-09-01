"""Physical pattern library and closed-form Stage 3 forecast core."""

from __future__ import annotations

import numpy as np
import pytest

from tdp.thermal.forecast import (
    ClosedFormPlateModel,
    ReconstructionArchive,
    load_reconstruction_archive,
)
from tdp.thermal.patterns import (
    build_pattern_library,
    gaussian_source_shape,
    load_pattern_library,
    steady_source_response,
)
from tdp.thermal.presentation import (
    model_range_fields,
    nearest_sample_indices,
    weighted_mae_by_frame,
)
from tdp.thermal.train_forecast import (
    fit_equilibrium_forecast,
    fit_source_histories,
    make_frozen_split,
    rollout_to_equilibrium,
)


SHAPE = (33, 25)
SPREAD = 1.2
COOLING = 0.02


@pytest.fixture(scope="module")
def source_case():
    board = np.ones(SHAPE, dtype=bool)
    source = gaussian_source_shape(SHAPE, (13.0, 9.0), width_px=1.5)
    library = build_pattern_library(
        board,
        [source],
        source_names=["chip"],
        n_smooth=12,
        trial_spread_rates=[0.5, SPREAD, 3.0],
        cooling_rate_per_time=COOLING,
    )
    return board, source, library


def test_library_is_physical_fixed_and_orthonormal(source_case):
    board, _source, library = source_case
    flat = library.basis[:, board]
    assert np.linalg.norm(flat @ flat.T - np.eye(library.n_patterns)) < 1e-10
    assert library.report["uses_measured_temperature_frames"] is False
    assert library.report["rank"] == library.n_patterns
    assert library.source_names == ("chip",)


def test_source_pattern_passes_the_hotspot_error_gate(source_case):
    """A source response in the library must survive compression below 0.10 K."""

    board, source, library = source_case
    truth = steady_source_response(
        board,
        source,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    rebuilt = library.reconstruct(library.project(truth))
    hot = source > 0.5 * source.max()
    assert np.max(np.abs(rebuilt[hot] - truth[hot])) < 0.10
    assert np.mean(np.abs(rebuilt[board] - truth[board])) < 0.01


def test_smooth_modes_alone_do_not_get_credit_for_a_missing_hotspot(source_case):
    board, source, _library = source_case
    truth = steady_source_response(
        board,
        source,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    smooth_only = build_pattern_library(
        board,
        n_smooth=12,
        cooling_rate_per_time=COOLING,
        trial_spread_rates=[SPREAD],
    )
    rebuilt = smooth_only.reconstruct(smooth_only.project(truth))
    peak = np.unravel_index(np.argmax(truth), truth.shape)
    assert truth[peak] - rebuilt[peak] > 0.10


def test_closed_form_settled_state_matches_the_fine_plate(source_case):
    board, source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    rate = np.array([0.35])
    expected = 0.35 * steady_source_response(
        board,
        source,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    actual = model.steady_field(rate)
    assert np.max(np.abs(actual[board] - expected[board])) < 0.10


def test_closed_form_advance_has_the_semigroup_property(source_case):
    _board, _source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    state = np.linspace(0.0, 1.0, library.n_patterns)
    rate = np.array([0.2])
    whole = model.advance_coordinates(state, 11.0, rate)
    split = model.advance_coordinates(model.advance_coordinates(state, 4.0, rate), 7.0, rate)
    assert np.allclose(whole, split, atol=1e-11, rtol=1e-11)
    assert model.time_to_settle(0.01) > 0.0


def test_rollout_accepts_arbitrary_forward_times(source_case):
    board, _source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    initial = np.zeros(SHAPE)
    result = model.rollout(initial, np.array([0.0, 1.0, 2.5, 9.0]), np.array([0.2]))
    assert result.shape == (4, *SHAPE)
    # Modal truncation can ring by a few millikelvin immediately after a sharp
    # source switch.  Do not clip it (that would destroy the exact semigroup),
    # but bound and report the approximation rather than calling it positivity.
    assert result[:, board].min() > -0.01
    assert result[-1, board].max() > 0.1
    with pytest.raises(ValueError, match="strictly increasing"):
        model.rollout(initial, np.array([0.0, 1.0, 1.0]), np.array([0.2]))
    with pytest.raises(ValueError, match="forward only"):
        model.advance_field(initial, -1.0, np.array([0.2]))


def test_arbitrary_starting_field_is_preserved_at_time_zero(source_case):
    _board, _source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    rows, cols = np.indices(SHAPE)
    initial = 4.0 + ((rows + cols) % 3) * 0.2
    result = model.rollout(initial, np.array([0.0, 1.0]), np.array([0.0]))
    assert np.allclose(result[0], initial, atol=1e-12, rtol=0.0)
    _state, residual, decay = model.split_initial_field(initial)
    assert np.linalg.norm(residual) > 0.0
    assert decay >= COOLING


def test_stage2_archive_contract_keeps_estimates_distinct_from_measurements(tmp_path):
    frames, height, width = 3, 5, 4
    theta = np.arange(frames * height * width, dtype=np.float32).reshape(frames, height, width) / 10
    measured = theta.copy()
    rebuilt = np.zeros((height, width), bool)
    rebuilt[1:3, 1:3] = True
    unknown = np.zeros_like(rebuilt)
    unknown[1:2, 1:3] = True
    theta[:, rebuilt] += 2.0
    uncertainty = np.zeros_like(theta)
    uncertainty[:, rebuilt] = 1.5
    path = tmp_path / "reconstructed_sequence.npz"
    np.savez_compressed(
        path,
        theta_k=theta,
        measured_theta_k=measured,
        times_s=np.array([0.0, 6.0, 13.0]),
        ambient_c=np.full(frames, 25.0),
        unknown_mask=unknown,
        rebuilt_mask=rebuilt,
        board_mask=np.ones((height, width), bool),
        heating_rate_k_per_s=np.ones((height, width), np.float32),
        source_rate_by_frame=np.ones((1, frames)),
        region_ids=np.array([4]),
        reconstruction_uncertainty_k=uncertainty,
        spread_rate_px2_per_s=1.0,
        cooling_rate_per_s=0.01,
        pixel_size_mm=0.5,
    )

    archive = load_reconstruction_archive(path)
    weight = archive.observation_weight()
    assert archive.n_frames == frames
    assert np.all(weight[:, ~rebuilt] == 1.0)
    assert np.all(weight[:, rebuilt] < 0.01)
    assert np.all(weight[:, rebuilt] > 0.0)


def test_stage2_archive_rejects_changes_outside_the_declared_rebuild(tmp_path):
    path = tmp_path / "bad.npz"
    shape = (2, 3, 3)
    np.savez_compressed(
        path,
        theta_k=np.ones(shape),
        measured_theta_k=np.zeros(shape),
        times_s=np.array([0.0, 1.0]),
        ambient_c=np.zeros(2),
        unknown_mask=np.zeros(shape[1:], bool),
        rebuilt_mask=np.zeros(shape[1:], bool),
        board_mask=np.ones(shape[1:], bool),
        heating_rate_k_per_s=np.zeros(shape[1:]),
        source_rate_by_frame=np.zeros((0, 2)),
        region_ids=np.zeros(0, int),
        reconstruction_uncertainty_k=np.zeros(shape),
        spread_rate_px2_per_s=1.0,
        cooling_rate_per_s=0.01,
        pixel_size_mm=0.5,
    )
    with pytest.raises(ValueError, match="changed directly measured pixels"):
        load_reconstruction_archive(path)


def test_exported_pattern_library_round_trips(source_case, tmp_path):
    _board, _source, library = source_case
    path = tmp_path / "patterns.npz"
    np.savez_compressed(
        path,
        basis=library.basis.astype(np.float32),
        board_mask=library.board_mask,
        candidate_names=np.asarray(library.candidate_names),
        candidate_coordinates=library.candidate_coordinates,
        source_names=np.asarray(library.source_names),
        source_shapes=library.source_shapes.astype(np.float32),
        rank_threshold=library.rank_threshold,
    )
    loaded = load_pattern_library(path)
    assert loaded.source_names == library.source_names
    assert loaded.n_patterns == library.n_patterns
    assert np.allclose(loaded.basis, library.basis, atol=1e-6)


def test_frozen_forecast_split_has_real_guard_gaps():
    split = make_frozen_split(100, guard_frames=3)
    assert {"train", "val", "test", "guard"} <= set(split.roles)
    for index in np.nonzero(split.roles[:-1] != split.roles[1:])[0]:
        left, right = split.roles[index], split.roles[index + 1]
        assert "guard" in {left, right}, "roles must never touch across a split seam"


def _synthetic_archive_from_model(model, frames=70):
    times = np.arange(frames, dtype=np.float64) * 5.0
    rate = np.array([0.20])
    settled = model.steady_field(rate)
    initial = 0.2 * settled
    theta = model.rollout(initial, times, rate)
    board = model.library.board_mask
    return ReconstructionArchive(
        theta_k=theta,
        measured_theta_k=theta.copy(),
        uncertainty_k=np.zeros_like(theta),
        times_s=times,
        ambient_c=np.full(frames, 25.0),
        board_mask=board,
        measured_board_mask=board.copy(),
        rebuilt_mask=np.zeros_like(board),
        unknown_mask=np.zeros_like(board),
        heating_rate_k_per_s=np.full(board.shape, 0.2),
        source_rate_by_frame=np.full((1, frames), 0.2),
        region_ids=np.array([1]),
        spread_rate_px2_per_s=model.spread_rate_px2_per_time,
        cooling_rate_per_s=model.cooling_rate_per_time,
        pixel_size_mm=0.5,
    )


def test_equilibrium_model_is_forward_only_and_recovers_a_synthetic_plate(source_case):
    _board, _source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    archive = _synthetic_archive_from_model(model)
    split = make_frozen_split(archive.n_frames)
    fit = fit_equilibrium_forecast(
        archive,
        library,
        split=split,
        spread_multipliers=(1.0,),
        cooling_multipliers=(1.0,),
        residual_decay_multipliers=(1.0,),
    )
    assert fit.metrics["test_forward"]["weighted_mae_k"] < 0.05
    forecast = rollout_to_equilibrium(
        fit, archive.theta_k[10], np.array([0.0, 10.0, 1000.0])
    )
    assert np.allclose(forecast[0], archive.theta_k[10], atol=1e-10)
    assert np.mean(np.abs(forecast[-1] - fit.settled_theta_k)) < 0.02
    with pytest.raises(ValueError, match="strictly increasing"):
        rollout_to_equilibrium(fit, archive.theta_k[10], np.array([0.0, 2.0, 1.0]))


def test_positive_source_history_fit_remains_a_bounded_ablation(source_case):
    _board, _source, library = source_case
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=SPREAD,
        cooling_rate_per_time=COOLING,
    )
    archive = _synthetic_archive_from_model(model)
    fit = fit_source_histories(
        archive,
        library,
        split=make_frozen_split(archive.n_frames),
        n_knots=5,
        smooth_strengths=(0.1,),
        spread_multipliers=(1.0,),
        cooling_multipliers=(1.0,),
        source_rate_upper_k_per_s=0.5,
    )
    assert np.all(fit.knot_source_rate_k_per_s >= 0.0)
    assert np.all(fit.knot_source_rate_k_per_s <= 0.5 + 1e-9)
    assert fit.metrics["acceptance"]["beats_no_source_on_weighted_test_mae"]


def test_presentation_weighted_error_is_one_value_per_frame():
    truth = np.zeros((2, 2, 2))
    predicted = np.array(
        [
            [[1.0, 3.0], [20.0, 20.0]],
            [[2.0, 4.0], [20.0, 20.0]],
        ]
    )
    weight = np.zeros_like(predicted)
    weight[:, 0] = np.array([1.0, 3.0])
    result = weighted_mae_by_frame(predicted, truth, weight)
    assert np.allclose(result, [2.5, 3.5])
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mae_by_frame(predicted, truth, np.zeros_like(weight))


def test_declared_model_range_has_exact_start_and_documented_limit():
    elapsed = np.array([0.0, 10.0, 10000.0])
    start = np.array([[2.0, 4.0]])
    settled = np.array([[1.0, 3.0]])
    result = model_range_fields(elapsed, start, settled, 0.25, 0.1)
    assert np.array_equal(result[0], start)
    assert np.allclose(result[-1], settled + 0.25)
    assert np.all(result >= 0.0)


def test_nearest_capture_indices_are_stable_at_ties_and_endpoints():
    captured = np.array([0.0, 6.0, 13.0])
    query = np.array([-2.0, 3.0, 4.0, 11.0, 20.0])
    # Ties choose the earlier captured frame, so a display cannot flicker
    # forward before that frame was actually observed.
    assert np.array_equal(nearest_sample_indices(captured, query), [0, 0, 1, 2, 2])
