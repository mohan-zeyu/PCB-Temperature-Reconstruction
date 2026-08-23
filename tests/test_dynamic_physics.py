"""Invariants of the graph heat ODE and its integrator.

These check properties that must hold for *any* parameters, so they are the
tests that catch a regression in the physics rather than in a particular fit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tdp.dynamic_inverse.losses import (
    edge_regularization,
    implied_steady_source,
    parameter_spatial_smoothness,
    steady_source_plausibility,
)
from tdp.dynamic_inverse.model import SourceDecoder, ThermalOperator
from tdp.dynamic_inverse.physics import (
    RegionGraph,
    dense_operator,
    graph_rhs,
    integrate,
    max_stable_dt,
    spectral_bound,
    steady_state,
)
from tdp.dynamic_inverse.synthetic import make_case


def _small_graph(seed: int = 0) -> tuple[RegionGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
    case = make_case(rows=5, cols=4, seed=seed)
    graph = case.graph
    rng = np.random.default_rng(seed)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    edge_rate = torch.as_tensor(rng.uniform(0.001, 0.05, graph.n_edges), dtype=torch.float64)
    beta = torch.as_tensor(rng.uniform(0.001, 0.02, graph.n_nodes), dtype=torch.float64)
    return graph, edge_index, edge_rate, beta


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def test_edge_index_is_canonical_and_unique():
    graph, _, _, _ = _small_graph()
    i, j = graph.edge_index
    assert np.all(i < j), "edges must be stored once, canonically ordered"
    assert len({tuple(pair) for pair in graph.edge_index.T}) == graph.n_edges


def test_duplicate_or_reversed_edges_are_rejected():
    with pytest.raises(ValueError, match="canonical"):
        RegionGraph(
            n_nodes=3,
            edge_index=np.array([[1, 0], [0, 2]]),
            centroids_px=np.zeros((3, 2)),
            areas_px=np.ones(3),
            edge_length_px=np.ones(2),
        )
    with pytest.raises(ValueError, match="duplicate"):
        RegionGraph(
            n_nodes=3,
            edge_index=np.array([[0, 0], [1, 1]]),
            centroids_px=np.zeros((3, 2)),
            areas_px=np.ones(3),
            edge_length_px=np.ones(2),
        )


def test_operator_matrix_is_symmetric_off_diagonal():
    """a_ij == a_ji exactly -- symmetry is structural, not learned."""
    graph, edge_index, edge_rate, beta = _small_graph()
    matrix = dense_operator(graph.n_nodes, edge_index, edge_rate, beta)
    off_diagonal = matrix - torch.diag(torch.diagonal(matrix))
    assert torch.allclose(off_diagonal, off_diagonal.T, atol=0.0)


def test_coupling_conserves_heat():
    """The coupling term moves heat between regions and never creates it."""
    graph, edge_index, edge_rate, _ = _small_graph()
    beta = torch.zeros(graph.n_nodes, dtype=torch.float64)
    theta = torch.as_tensor(np.random.default_rng(1).normal(size=graph.n_nodes))
    assert graph_rhs(theta, edge_index, edge_rate, beta).sum().abs() < 1e-10


def test_edge_head_is_symmetric_in_its_endpoints():
    """Swapping i and j cannot change the predicted rate, before storage helps."""
    torch.manual_seed(0)
    model = ThermalOperator(node_feature_dim=4, edge_feature_dim=3)
    features = torch.randn(6, 4)
    edge_features = torch.randn(2, 3)
    embedding = model.embed(features)
    forward = model.edge_rates(embedding, torch.tensor([[0, 2], [3, 4]]), edge_features)
    backward = model.edge_rates(embedding, torch.tensor([[3, 4], [0, 2]]), edge_features)
    assert torch.allclose(forward, backward, atol=1e-6)


# --------------------------------------------------------------------------
# non-negativity and stability
# --------------------------------------------------------------------------
def test_learned_rates_are_non_negative_at_random_initialisations():
    for seed in range(8):
        torch.manual_seed(seed)
        model = ThermalOperator(node_feature_dim=5, edge_feature_dim=4)
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.add_(torch.randn_like(parameter) * 3.0)
        edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]])
        rate, beta = model(torch.randn(3, 5), edge_index, torch.randn(3, 4))
        assert (rate >= 0).all() and (beta >= 0).all()


def test_unforced_system_cannot_warm_itself():
    """No eigenvalue with a positive real part, and no rising trajectory."""
    graph, edge_index, edge_rate, beta = _small_graph()
    assert spectral_bound(graph.n_nodes, edge_index, edge_rate, beta) <= 0.0

    theta0 = torch.as_tensor(np.random.default_rng(2).uniform(0.0, 20.0, graph.n_nodes))
    times = torch.arange(0.0, 600.0, 6.0, dtype=torch.float64)
    trajectory = integrate(theta0, times, edge_index, edge_rate, beta, None)
    energy = trajectory.abs().max(dim=1).values
    assert torch.all(energy[1:] <= energy[:-1] + 1e-9), "unforced maximum must not grow"
    assert trajectory[-1].max() < theta0.max()


def test_cooling_phase_source_is_exactly_zero():
    """The powered mask is a hard multiply, so load-off carries no source."""
    torch.manual_seed(0)
    decoder = SourceDecoder(node_feature_dim=4)
    times = torch.linspace(0.0, 300.0, 40)
    powered = torch.zeros(40, dtype=torch.bool)
    powered[:15] = True
    source = decoder(torch.randn(6, 4), times, powered, period=300.0)
    assert source.shape == (40, 6)
    assert torch.all(source[~powered] == 0.0)
    assert torch.all(source[powered] >= 0.0)
    assert source[powered].sum() > 0.0


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------
def test_rk4_matches_the_analytic_single_node_solution():
    """One isolated node decays as exp(-beta t); RK4 must track it closely."""
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_rate = torch.zeros(0, dtype=torch.float64)
    beta = torch.tensor([0.01], dtype=torch.float64)
    times = torch.arange(0.0, 300.0, 6.0, dtype=torch.float64)
    numeric = integrate(torch.tensor([20.0], dtype=torch.float64), times, edge_index, edge_rate, beta)
    analytic = 20.0 * torch.exp(-0.01 * times)
    assert torch.max((numeric.squeeze(-1) - analytic).abs()) < 1e-6


def test_irregular_timestamps_use_the_real_dt():
    """A jittery clock must be integrated on its own spacing, not a nominal one."""
    graph, edge_index, edge_rate, beta = _small_graph()
    # Gentle rates keep RK4 truncation error far below the effect being tested,
    # so a failure here means the spacing was ignored, not that the step was big.
    edge_rate, beta = edge_rate * 0.2, beta * 0.2
    theta0 = torch.ones(graph.n_nodes, dtype=torch.float64) * 10.0
    irregular = torch.tensor(
        np.cumsum([0.0, 4.0, 9.0, 5.0, 12.0, 6.0]), dtype=torch.float64
    )
    stepped = integrate(theta0, irregular, edge_index, edge_rate, beta)

    # Same endpoint reached by a fine uniform grid: agreement proves the
    # per-interval dt is honoured rather than an averaged one.
    fine = torch.linspace(0.0, float(irregular[-1]), 4001, dtype=torch.float64)
    reference = integrate(theta0, fine, edge_index, edge_rate, beta)
    truncation = float(torch.max((stepped[-1] - reference[-1]).abs()))
    assert truncation < 1e-4, "five RK4 steps should track a fine grid to well under a millikelvin"

    # A nominal, evenly-spaced clock lands somewhere else entirely at the
    # intermediate samples.  Comparing the two errors is what makes this a test
    # of the spacing rather than of the step size: honouring the real dt has to
    # beat assuming a uniform one by orders of magnitude, not by a hair.
    uniform = torch.arange(len(irregular), dtype=torch.float64) * float(irregular[-1]) / (
        len(irregular) - 1
    )
    drift = float(
        torch.max((integrate(theta0, uniform, edge_index, edge_rate, beta)[2] - stepped[2]).abs())
    )
    assert drift > 0.01
    assert drift / max(truncation, 1e-12) > 100.0

    # Sub-stepping is the escape hatch when a fast region pushes a*dt up.
    substepped = integrate(theta0, irregular, edge_index, edge_rate, beta, substeps=8)
    assert torch.max((substepped[-1] - reference[-1]).abs()) < truncation


def test_integrate_rejects_non_increasing_times():
    graph, edge_index, edge_rate, beta = _small_graph()
    theta0 = torch.ones(graph.n_nodes, dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate(theta0, torch.tensor([0.0, 6.0, 6.0]), edge_index, edge_rate, beta)


def test_prediction_shape_dtype_and_device():
    graph, edge_index, edge_rate, beta = _small_graph()
    for dtype in (torch.float32, torch.float64):
        theta0 = torch.ones(graph.n_nodes, dtype=dtype)
        times = torch.arange(0.0, 60.0, 6.0, dtype=dtype)
        out = integrate(theta0, times, edge_index, edge_rate.to(dtype), beta.to(dtype))
        assert out.shape == (times.numel(), graph.n_nodes)
        assert out.dtype == dtype
        assert out.device == theta0.device


def test_batched_states_integrate_independently():
    """Leading batch dimensions let an ensemble roll out in one call."""
    graph, edge_index, edge_rate, beta = _small_graph()
    batch = torch.as_tensor(np.random.default_rng(3).uniform(0, 10, (4, graph.n_nodes)))
    times = torch.arange(0.0, 60.0, 6.0, dtype=torch.float64)
    together = integrate(batch, times, edge_index, edge_rate, beta)
    for k in range(batch.shape[0]):
        alone = integrate(batch[k], times, edge_index, edge_rate, beta)
        assert torch.allclose(together[:, k], alone, atol=1e-10)


def test_max_stable_dt_is_reported_above_the_working_cadence():
    graph, edge_index, edge_rate, beta = _small_graph()
    assert max_stable_dt(graph.n_nodes, edge_index, edge_rate, beta) > 6.0


# --------------------------------------------------------------------------
# what the model must NOT claim
# --------------------------------------------------------------------------
def test_rescaling_capacity_leaves_the_trajectory_unchanged():
    """The global scale ambiguity that forbids reporting W/K or W.

    Multiplying C, G, h and q by one constant divides every identifiable rate
    by that constant *and* changes nothing observable once the rates are held
    fixed -- which is exactly why an absolute conductance cannot be recovered
    from temperatures alone.  Here: two systems whose (a, beta, s) agree give
    identical trajectories no matter what absolute C is imagined behind them.
    """
    graph, edge_index, edge_rate, beta = _small_graph()
    source = torch.as_tensor(np.random.default_rng(4).uniform(0, 0.05, graph.n_nodes))
    times = torch.arange(0.0, 600.0, 6.0, dtype=torch.float64)
    theta0 = torch.zeros(graph.n_nodes, dtype=torch.float64)

    baseline = integrate(theta0, times, edge_index, edge_rate, beta, lambda _t: source)
    for capacity in (0.5, 2.0, 17.0):
        # G' = capacity*G, h' = capacity*h, q' = capacity*q, C' = capacity*C
        scaled = integrate(
            theta0,
            times,
            edge_index,
            (capacity * edge_rate) / capacity,
            (capacity * beta) / capacity,
            lambda _t, s=source, c=capacity: (c * s) / c,
        )
        assert torch.allclose(scaled, baseline, atol=1e-12)


def test_steady_state_solves_the_operator():
    graph, edge_index, edge_rate, beta = _small_graph()
    source = torch.as_tensor(np.random.default_rng(5).uniform(0.0, 0.05, graph.n_nodes))
    equilibrium = steady_state(graph.n_nodes, edge_index, edge_rate, beta, source)
    assert torch.max(graph_rhs(equilibrium, edge_index, edge_rate, beta, source).abs()) < 1e-9

    # And it is what a long rollout converges to.  The step is chosen from the
    # operator's own stability limit rather than picked by hand.
    step = 0.25 * max_stable_dt(graph.n_nodes, edge_index, edge_rate, beta)
    times = torch.arange(0.0, 20000.0, step, dtype=torch.float64)
    rolled = integrate(
        torch.zeros(graph.n_nodes, dtype=torch.float64),
        times,
        edge_index,
        edge_rate,
        beta,
        lambda _t: source,
    )
    assert torch.max((rolled[-1] - equilibrium).abs()) < 1e-6


# --------------------------------------------------------------------------
# the penalties that break the conduction/convection degeneracy
# --------------------------------------------------------------------------
def test_implied_steady_source_inverts_the_operator():
    """``s = -A theta`` must reproduce the source that generated a steady state."""
    graph, edge_index, edge_rate, beta = _small_graph()
    source = torch.as_tensor(np.random.default_rng(7).uniform(0.0, 0.05, graph.n_nodes))
    equilibrium = steady_state(graph.n_nodes, edge_index, edge_rate, beta, source)
    recovered = implied_steady_source(equilibrium, edge_index, edge_rate, beta)
    assert torch.max((recovered - source).abs()) < 1e-9


def test_spread_metric_is_one_for_uniform_and_small_for_a_spike():
    """The spread penalty must actually measure concentration."""
    graph, edge_index, _, _ = _small_graph()
    n = graph.n_nodes

    # Choose parameters that make the implied source exactly what we want:
    # with zero coupling, s = beta * theta, so theta sets the shape directly.
    zero_rate = torch.zeros(graph.n_edges, dtype=torch.float64)
    beta = torch.full((n,), 0.01, dtype=torch.float64)

    uniform = torch.ones(n, dtype=torch.float64)
    _, spread_uniform = steady_source_plausibility(uniform, edge_index, zero_rate, beta)
    assert spread_uniform == pytest.approx(1.0, abs=1e-9)

    spike = torch.zeros(n, dtype=torch.float64)
    spike[0] = 1.0
    _, spread_spike = steady_source_plausibility(spike, edge_index, zero_rate, beta)
    assert spread_spike == pytest.approx(1.0 / np.sqrt(n), abs=1e-9)
    assert spread_spike < spread_uniform


def test_negativity_penalty_fires_only_on_an_impossible_source():
    graph, edge_index, edge_rate, beta = _small_graph()
    source = torch.as_tensor(np.random.default_rng(8).uniform(0.001, 0.05, graph.n_nodes))
    equilibrium = steady_state(graph.n_nodes, edge_index, edge_rate, beta, source)
    negativity, _ = steady_source_plausibility(equilibrium, edge_index, edge_rate, beta)
    assert float(negativity) < 1e-12

    # A profile with a cold spot in a hot board implies heat being removed there.
    cold_spot = equilibrium.clone()
    cold_spot[0] = -10.0 * equilibrium.abs().max()
    negativity_bad, _ = steady_source_plausibility(cold_spot, edge_index, edge_rate, beta)
    assert float(negativity_bad) > 1e-3


def test_regularisers_are_scale_free():
    """Switching the time unit must not silently rescale a penalty.

    An absolute squared penalty on a rate of order 1e-3 per second contributes
    about 1e-6 to a loss of order 1, so its configured weight does nothing --
    and re-expressing the same physics in frames multiplies it by the cadence
    squared.  Both regularisers are therefore relative to the parameter's own
    mean, which these assertions pin down.
    """
    graph, edge_index, edge_rate, beta = _small_graph()
    similarity = torch.ones(graph.n_edges, dtype=torch.float64)
    for factor in (6.3, 0.1, 1000.0):
        assert edge_regularization(edge_rate * factor) == pytest.approx(
            float(edge_regularization(edge_rate)), rel=1e-9
        )
        assert parameter_spatial_smoothness(beta * factor, edge_index, similarity) == pytest.approx(
            float(parameter_spatial_smoothness(beta, edge_index, similarity)), rel=1e-9
        )
