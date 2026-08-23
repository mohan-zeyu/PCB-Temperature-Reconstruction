"""A synthetic thermal system with known answers, for testing recovery.

Real data cannot tell you whether an identified coupling is right, because
nothing measured the true one.  A synthetic board can: heat is generated at
known places, conducted at known rates and lost at known rates, and the
recovered parameters are checked against those.

The generator deliberately reproduces the awkward parts of the real recording
rather than an easy problem: a heterogeneous plate (a fast "copper" corridor
crossing a slow substrate), non-uniform cooling, one to three sparse sources,
a heating/steady/cooling schedule, Gaussian noise and a missing-data mask.  The
one thing it does *not* do is hand any of that to the model -- :func:`make_case`
returns observations and truth in separate objects so a test cannot leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tdp.dynamic_inverse.physics import RegionGraph, integrate


@dataclass
class SyntheticTruth:
    """The parameters the generator used.  Never passed to a model."""

    edge_rate: np.ndarray  # (E,) 1/s
    beta: np.ndarray  # (N,) 1/s
    source_rate: np.ndarray  # (T, N) K/s
    source_nodes: np.ndarray  # (k,) indices of the true sources
    grid_shape: tuple[int, int]
    node_grid: np.ndarray  # (H, W) int, node id per cell


@dataclass
class SyntheticCase:
    """What a model is allowed to see."""

    graph: RegionGraph
    theta: np.ndarray  # (T, N) K, noisy observations
    theta_clean: np.ndarray  # (T, N) K, noise-free (for reporting only)
    times: np.ndarray  # (T,) s
    observed: np.ndarray  # (T, N) bool
    phase: np.ndarray  # (T,) of "heating" | "steady" | "cooling"
    powered: np.ndarray  # (T,) bool
    truth: SyntheticTruth


def _grid_graph(rows: int, cols: int) -> tuple[RegionGraph, np.ndarray]:
    """4-neighbour lattice of unit cells; node ids raster-ordered."""
    node_grid = np.arange(rows * cols).reshape(rows, cols)
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((node_grid[r, c], node_grid[r, c + 1]))
            if r + 1 < rows:
                edges.append((node_grid[r, c], node_grid[r + 1, c]))
    edge_index = np.asarray(edges, dtype=np.int64).T
    centroids = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"), -1)
    centroids = centroids.reshape(-1, 2).astype(np.float64)
    graph = RegionGraph(
        n_nodes=rows * cols,
        edge_index=edge_index,
        centroids_px=centroids,
        areas_px=np.ones(rows * cols),
        edge_length_px=np.ones(edge_index.shape[1]),
    )
    return graph, node_grid


def make_case(
    rows: int = 8,
    cols: int = 6,
    dt: float = 6.0,
    n_heating: int = 30,
    n_steady: int = 20,
    n_cooling: int = 40,
    fast_rate: float = 0.030,
    slow_rate: float = 0.006,
    beta_low: float = 0.0025,
    beta_high: float = 0.0090,
    source_nodes: tuple[int, ...] | None = None,
    source_strength: tuple[float, ...] = (0.75, 0.30),
    noise_k: float = 0.10,
    missing_fraction: float = 0.05,
    irregular_dt: bool = False,
    seed: int = 0,
) -> SyntheticCase:
    """Build one synthetic heating/steady/cooling recording.

    Rates are chosen to sit in the same decade as what the real board shows
    (a board time constant of minutes on a ~6 s frame cadence), so a test that
    passes here is testing the regime the pipeline actually runs in.
    """
    rng = np.random.default_rng(seed)
    graph, node_grid = _grid_graph(rows, cols)
    n_nodes = graph.n_nodes

    # A fast corridor down one column crossing a slow substrate: heterogeneous
    # coupling that a single global alpha cannot represent.
    corridor = cols // 2
    i, j = graph.edge_index
    rate = np.full(graph.n_edges, slow_rate)
    on_corridor = (graph.centroids_px[i][:, 1] == corridor) & (
        graph.centroids_px[j][:, 1] == corridor
    )
    rate[on_corridor] = fast_rate

    # Cooling is stronger at the perimeter, as it is on a real board edge.
    r_index, c_index = graph.centroids_px[:, 0], graph.centroids_px[:, 1]
    on_edge = (r_index == 0) | (r_index == rows - 1) | (c_index == 0) | (c_index == cols - 1)
    beta = np.where(on_edge, beta_high, beta_low)

    if source_nodes is None:
        source_nodes = (int(node_grid[rows // 3, corridor]), int(node_grid[2 * rows // 3, 1]))
    strengths = np.asarray(source_strength[: len(source_nodes)], dtype=np.float64)

    n_frames = n_heating + n_steady + n_cooling
    if irregular_dt:
        times = np.cumsum(rng.uniform(0.6 * dt, 1.6 * dt, n_frames))
        times = times - times[0]
    else:
        times = np.arange(n_frames, dtype=np.float64) * dt

    powered = np.zeros(n_frames, dtype=bool)
    powered[: n_heating + n_steady] = True
    phase = np.array(
        ["heating"] * n_heating + ["steady"] * n_steady + ["cooling"] * n_cooling
    )

    source = np.zeros((n_frames, n_nodes))
    for node, strength in zip(source_nodes, strengths, strict=False):
        source[powered, node] = strength

    edge_index_t = torch.as_tensor(graph.edge_index, dtype=torch.long)
    rate_t = torch.as_tensor(rate, dtype=torch.float64)
    beta_t = torch.as_tensor(beta, dtype=torch.float64)
    times_t = torch.as_tensor(times, dtype=torch.float64)
    source_t = torch.as_tensor(source, dtype=torch.float64)

    def source_fn(t: float) -> torch.Tensor:
        # Zero-order hold on the frame grid: the source is a schedule, not a
        # smooth function, and interpolating it would invent a ramp at load-off.
        index = int(np.searchsorted(times, t, side="right") - 1)
        return source_t[min(max(index, 0), n_frames - 1)]

    theta0 = torch.zeros(n_nodes, dtype=torch.float64)
    # Substeps keep the truth itself free of integration error, so a recovery
    # test measures the fit rather than the generator.
    clean = integrate(
        theta0, times_t, edge_index_t, rate_t, beta_t, source_fn, method="rk4", substeps=4
    ).numpy()

    noisy = clean + rng.normal(0.0, noise_k, clean.shape)
    observed = rng.random(clean.shape) >= missing_fraction

    truth = SyntheticTruth(
        edge_rate=rate,
        beta=beta,
        source_rate=source,
        source_nodes=np.asarray(source_nodes, dtype=np.int64),
        grid_shape=(rows, cols),
        node_grid=node_grid,
    )
    return SyntheticCase(
        graph=graph,
        theta=noisy,
        theta_clean=clean,
        times=times,
        observed=observed,
        phase=phase,
        powered=powered,
        truth=truth,
    )
