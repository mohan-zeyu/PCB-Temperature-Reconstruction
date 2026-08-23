"""End-to-end recovery on a synthetic system whose answers are known.

Real data cannot grade an identified coupling -- nothing measured the true one.
These tests grade it on a board that was generated, and they deliberately keep
the truth out of the model's reach: :func:`make_case` hands back observations
and truth in separate objects, and only the assertions ever touch the truth.

They are the slowest tests in the suite (a few fits on a 48-node graph) and are
what stands behind every claim the report makes about the method working.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tdp.dynamic_inverse.config import AnalysisConfig
from tdp.dynamic_inverse.synthetic import make_case
from tdp.dynamic_inverse.train_source import fit_source, relative_contribution
from tdp.dynamic_inverse.train_thermal import NodeSeries, fit_thermal_operator


def _series(case) -> NodeSeries:
    n_frames = len(case.times)
    heating = np.zeros(n_frames, dtype=bool)
    heating[: (case.phase == "heating").sum()] = True
    phase = np.where(
        case.powered, np.where(heating, "heating", "near_steady"), "free_cooling_candidate"
    )
    return NodeSeries(
        theta=case.theta,
        times=case.times,
        weight=case.observed.astype(np.float64),
        trust=np.ones(case.graph.n_nodes),
        phase=phase,
        powered=case.powered,
    )


def _graph_with_geometry(case):
    """Geometry-only node features.  Nothing about the true parameters."""
    graph = case.graph
    features = np.stack(
        [graph.centroids_px[:, 0], graph.centroids_px[:, 1], graph.areas_px], axis=1
    ).astype(np.float32)
    object.__setattr__(graph, "node_features", features)
    return graph


def _config(**train) -> AnalysisConfig:
    base = AnalysisConfig()
    defaults = {
        "epochs_thermal": 400,
        "epochs_source": 400,
        "rollout_steps": 20,
        "batch_windows": 16,
        "patience": 250,
    }
    defaults.update(train)
    return dataclasses.replace(base, train=dataclasses.replace(base.train, **defaults))


@pytest.fixture(scope="module")
def fitted():
    """One fit shared by the assertions below; each is a separate claim about it."""
    case = make_case(seed=0, n_heating=30, n_steady=30, n_cooling=90, noise_k=0.05)
    graph = _graph_with_geometry(case)
    series = _series(case)
    config = _config()
    thermal = fit_thermal_operator(graph, series, config, seed=0)
    source = fit_source(graph, series, thermal, config, seed=0)
    return case, graph, series, thermal, source


# --------------------------------------------------------------------------
# stage A: the thermal operator
# --------------------------------------------------------------------------
def test_strong_coupling_edges_are_recovered(fitted):
    """The fast corridor must come out ranked above the slow substrate."""
    case, _, _, thermal, _ = fitted
    truth = case.truth.edge_rate
    fast = truth > truth.mean()
    # Stable across case/fit seeds: measured 0.69-0.82 over four combinations.
    assert np.corrcoef(thermal.edge_rate, truth)[0, 1] > 0.6

    n_fast = int(fast.sum())
    recovered = set(np.argsort(-thermal.edge_rate)[:n_fast].tolist())
    assert len(recovered & set(np.nonzero(fast)[0].tolist())) / n_fast >= 0.7
    assert thermal.edge_rate[fast].mean() > 1.5 * thermal.edge_rate[~fast].mean()


def test_cooling_rate_structure_is_recovered(fitted):
    """Perimeter regions must come out cooling faster than interior ones.

    The assertion is on the *ordering* and on a positive correlation, not on a
    tight correlation value.  Measured across four case/fit seed combinations
    the correlation ranges 0.35-0.61 while the perimeter/interior ratio stays
    1.34-1.54: the fit reliably recovers which regions lose heat fastest, and
    reliably *compresses* how much they differ.  Asserting a tight correlation
    would be asserting something the method does not deliver, and would make
    the test flaky rather than informative.  The compression is reported as a
    limitation rather than tested away.
    """
    case, _, _, thermal, _ = fitted
    truth = case.truth.beta
    assert np.corrcoef(thermal.beta, truth)[0, 1] > 0.3
    perimeter = truth > truth.mean()
    assert thermal.beta[perimeter].mean() > 1.2 * thermal.beta[~perimeter].mean()


def test_all_identified_rates_are_non_negative_and_stable(fitted):
    _, _, _, thermal, _ = fitted
    assert np.all(thermal.edge_rate >= 0.0)
    assert np.all(thermal.beta >= 0.0)
    assert thermal.metrics["spectral_bound_per_unit_time"] <= 0.0
    assert thermal.metrics["stable"]


def test_residual_idle_source_is_near_zero_when_the_system_truly_free_cools(fitted):
    """The synthetic board really is unpowered while cooling, so the fitted
    residual must come back at essentially zero -- the machinery must not
    invent a source that was not there."""
    case, _, _, thermal, _ = fitted
    scale = float(case.truth.source_rate.max())
    assert thermal.metrics["residual_idle_source_rate_mean"] < 0.02 * scale


def test_splits_are_contiguous_time_blocks(fitted):
    """No shuffling: adjacent frames are near-duplicates on this cadence."""
    _, _, _, thermal, source = fitted
    for windows in (thermal.windows, source.windows):
        blocks = windows.train + windows.val + windows.test
        for lo, hi in blocks:
            assert hi > lo
        assert windows.train[0][1] < windows.val[0][0]
        assert windows.val[0][1] < windows.test[0][0]


# --------------------------------------------------------------------------
# stage B: the source
# --------------------------------------------------------------------------
def test_source_is_localised_on_the_true_sources(fitted):
    case, graph, _, _, source = fitted
    mean_relative = source.relative_source[source.powered].mean(axis=0)
    ranking = np.argsort(-mean_relative)
    for node in case.truth.source_nodes:
        assert node in ranking[: len(case.truth.source_nodes)], (
            f"true source node {node} not in the top {len(case.truth.source_nodes)}"
        )

    truth_relative = relative_contribution(
        case.truth.source_rate[source.powered], graph.areas_px
    ).mean(axis=0)
    assert np.corrcoef(mean_relative, truth_relative)[0, 1] > 0.9


def test_source_is_not_smeared_over_the_whole_board(fitted):
    """The classic failure: a source that fits the mean and localises nothing."""
    _, _, _, _, source = fitted
    concentration = source.metrics["relative_source_concentration"]
    assert concentration["effective_regions"] < 0.2 * concentration["n_regions"]
    assert concentration["top_decile_share"] > 0.5


def test_source_is_non_negative_and_exactly_zero_when_unpowered(fitted):
    _, _, _, _, source = fitted
    assert source.metrics["source_non_negative"]
    assert source.metrics["source_zero_outside_powered"]
    assert np.all(source.source_rate[~source.powered] == 0.0)


# --------------------------------------------------------------------------
# what cannot be claimed
# --------------------------------------------------------------------------
def test_global_capacity_rescaling_is_not_identifiable():
    """Two boards differing only by a global C give identical observations.

    Generating with ``(a, beta, s)`` and with ``(a, beta, s)`` derived from a
    different absolute ``C`` produces the same trajectory, so no fit on that
    trajectory can distinguish them.  That is the reason the pipeline reports
    rates and shares and never W/(m K), W/K or W.
    """
    base = make_case(seed=3, noise_k=0.0, missing_fraction=0.0)
    scaled = make_case(
        seed=3,
        noise_k=0.0,
        missing_fraction=0.0,
        fast_rate=0.030,
        slow_rate=0.006,
        source_strength=(0.75, 0.30),
    )
    assert np.allclose(base.theta_clean, scaled.theta_clean, atol=1e-12)

    # The identifiable rates are what they are; the absolute conductance behind
    # them is a free multiplier that leaves the data untouched.
    for capacity in (0.5, 3.0):
        implied_conductance = capacity * base.truth.edge_rate
        assert np.allclose(implied_conductance / capacity, base.truth.edge_rate)


def test_uniform_baseline_cannot_represent_heterogeneous_coupling():
    """The single-alpha/beta baseline must fit worse than the graph model.

    If it does not, the extra freedom is not buying anything and the report
    must say so -- so this is checked rather than assumed.
    """
    case = make_case(seed=1, n_heating=30, n_steady=30, n_cooling=90, noise_k=0.02)
    graph = _graph_with_geometry(case)
    series = _series(case)
    config = _config(epochs_thermal=300)
    full = fit_thermal_operator(graph, series, config, seed=0)
    uniform = fit_thermal_operator(graph, series, config, seed=0, uniform_baseline=True)
    assert full.metrics["best_val_mae"] < uniform.metrics["best_val_mae"]
    assert np.ptp(uniform.beta) < 1e-9
    assert np.ptp(uniform.edge_rate) < 1e-9


def test_irregular_clock_is_handled_end_to_end():
    case = make_case(seed=2, n_cooling=90, irregular_dt=True, noise_k=0.02)
    assert np.ptp(np.diff(case.times)) > 1.0
    graph = _graph_with_geometry(case)
    thermal = fit_thermal_operator(graph, _series(case), _config(epochs_thermal=150), seed=0)
    assert np.all(np.isfinite(thermal.edge_rate))
    assert np.all(thermal.edge_rate >= 0.0)
    assert thermal.metrics["stable"]


# --------------------------------------------------------------------------
# the regime where the method must admit it cannot answer
# --------------------------------------------------------------------------
def test_audit_flags_the_regime_where_coupling_contrast_is_unrecoverable():
    """A near-isothermal board hides its own conduction, and the audit says so.

    When heat spreads much faster than it is lost, the profile flattens within a
    few frames and everything after that is the fundamental mode, in which every
    region reports the same rate whatever the coupling underneath is doing.  The
    fit still returns numbers; they are not measurements.  This test pins the
    two halves of that claim together: the audit calls the regime out *and* the
    fit demonstrably fails in it, so the warning is calibrated rather than
    decorative.
    """
    from tdp.dynamic_inverse.identifiability import analyse

    easy = make_case(seed=0, n_heating=30, n_steady=30, n_cooling=90, noise_k=0.05)
    hard = make_case(
        seed=0, n_heating=30, n_steady=30, n_cooling=90, noise_k=0.05,
        fast_rate=0.15, slow_rate=0.06,
    )
    cooling = slice(60, 150)
    easy_audit = analyse(easy.theta[cooling], easy.times[cooling])
    hard_audit = analyse(hard.theta[cooling], hard.times[cooling])

    assert easy_audit["coupling_contrast"]["coupling_contrast_recoverable"]
    assert not hard_audit["coupling_contrast"]["coupling_contrast_recoverable"]
    assert "alpha_eff" in " ".join(hard_audit["not_identifiable_from_this_decay"])

    # And the fit really does fail there, which is what makes the flag honest.
    graph = _graph_with_geometry(hard)
    thermal = fit_thermal_operator(graph, _series(hard), _config(epochs_thermal=250), seed=0)
    truth = hard.truth.edge_rate
    fast = truth > truth.mean()
    n_fast = int(fast.sum())
    recovered = set(np.argsort(-thermal.edge_rate)[:n_fast].tolist())
    recall = len(recovered & set(np.nonzero(fast)[0].tolist())) / n_fast
    assert recall < 0.5, "if the fit succeeded here the audit's threshold is miscalibrated"


def test_a_diverged_fit_is_raised_rather_than_returned():
    """A fit whose validation error never became finite must not be handed back.

    An unbounded global magnitude once produced exactly that on real data -- a
    coupling of 2.9 per second on a board whose cooling rate is 5e-3 -- and the
    run carried on and wrote a report.  The scale is bounded now, so this checks
    the guard directly.
    """
    import torch

    case = make_case(seed=0, n_cooling=60, noise_k=0.05)
    graph = _graph_with_geometry(case)
    series = _series(case)
    series.theta[:] = np.nan  # nothing finite to fit
    with pytest.raises((RuntimeError, ValueError)):
        fit_thermal_operator(graph, series, _config(epochs_thermal=5), seed=0)
    del torch
