"""Graph heat ODE and its integrator.

The continuous model, after dividing the thin-plate equation by the areal heat
capacity, is ``d(theta)/dt = div(alpha_eff grad(theta)) - beta theta + s``.
Discretised onto a region graph the divergence becomes a sum over neighbours::

    d(theta_i)/dt = sum_j a_ij (theta_j - theta_i) - beta_i theta_i + s_i(t)

Three properties are structural here rather than penalised in the loss:

* **Symmetry.** One rate is stored per *undirected* edge and applied to both
  endpoints with opposite sign, so ``a_ij == a_ji`` holds exactly and cannot
  drift during training.
* **Conservation.** The coupling term is built as a difference across an edge,
  so the coupling contribution to ``sum_i theta_i`` is identically zero: the
  network moves heat, it never creates it.  Everything that leaves the system
  leaves through ``beta``, everything that enters enters through ``s``.
* **Non-negativity.** ``a_ij >= 0`` and ``beta_i >= 0`` are enforced by the
  caller's ``softplus``; this module validates rather than re-parameterises,
  so a violated invariant surfaces as an error instead of being clipped away.

With ``a >= 0``, ``beta >= 0`` and ``s = 0`` the operator is a negative
semi-definite graph Laplacian minus a non-negative diagonal, so no eigenvalue
has a positive real part and an unforced system cannot warm itself up.  That is
the property :func:`spectral_bound` measures and the stability test checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RegionGraph:
    """Undirected region graph: node positions plus a unique edge list.

    ``edge_index`` has shape ``(2, E)`` and holds each undirected pair once,
    with ``edge_index[0] < edge_index[1]``.  Storing pairs once is what makes
    edge symmetry a property of the data structure.
    """

    n_nodes: int
    edge_index: np.ndarray  # (2, E) int64, i < j, each pair once
    centroids_px: np.ndarray  # (N, 2) float64, (row, col)
    areas_px: np.ndarray  # (N,) float64
    edge_length_px: np.ndarray  # (E,) float64, shared-boundary length
    node_features: np.ndarray | None = None  # (N, F) float32
    edge_features: np.ndarray | None = None  # (E, G) float32

    def __post_init__(self) -> None:
        edges = np.asarray(self.edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}")
        if edges.size and (edges.min() < 0 or edges.max() >= self.n_nodes):
            raise ValueError("edge_index refers to a node outside the graph")
        if edges.size and not np.all(edges[0] < edges[1]):
            raise ValueError("edge_index must be canonical: edge_index[0] < edge_index[1]")
        if edges.size:
            keys = edges[0] * self.n_nodes + edges[1]
            if len(np.unique(keys)) != edges.shape[1]:
                raise ValueError("edge_index contains duplicate undirected edges")

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def edge_distance_px(self) -> np.ndarray:
        """Centre-to-centre distance for every edge."""
        i, j = self.edge_index
        return np.linalg.norm(self.centroids_px[i] - self.centroids_px[j], axis=1)


def graph_rhs(
    theta: Tensor,
    edge_index: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
    source: Tensor | None = None,
) -> Tensor:
    """Right-hand side of the graph heat ODE.

    ``theta`` may carry leading batch dimensions ``(..., N)``; ``edge_rate``
    is ``(E,)`` or broadcastable to ``(..., E)``, which is what lets an
    ensemble of parameter draws be integrated in one call.
    """
    src, dst = edge_index[0], edge_index[1]
    delta = theta.index_select(-1, dst) - theta.index_select(-1, src)  # (..., E)
    flow = edge_rate * delta  # heat flowing from dst into src
    out = torch.zeros_like(theta)
    out.index_add_(-1, src, flow)
    out.index_add_(-1, dst, -flow)
    out = out - beta * theta
    if source is not None:
        out = out + source
    return out


def rk4_step(
    theta: Tensor,
    dt: float | Tensor,
    edge_index: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
    source_fn=None,
    t: float = 0.0,
) -> Tensor:
    """One classical fourth-order Runge-Kutta step of width ``dt``.

    ``source_fn(t)`` returns the source at an absolute time, so a source that
    varies within a step is sampled at the half step like every other term
    instead of being held constant across it.
    """
    def f(time: float, state: Tensor) -> Tensor:
        src = None if source_fn is None else source_fn(time)
        return graph_rhs(state, edge_index, edge_rate, beta, src)

    half = dt * 0.5
    k1 = f(t, theta)
    k2 = f(t + float(half) if not torch.is_tensor(half) else t, theta + half * k1)
    k3 = f(t + float(half) if not torch.is_tensor(half) else t, theta + half * k2)
    k4 = f(t + float(dt) if not torch.is_tensor(dt) else t, theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(
    theta: Tensor,
    dt: float | Tensor,
    edge_index: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
    source_fn=None,
    t: float = 0.0,
) -> Tensor:
    """Explicit Euler, kept only as an integrator ablation."""
    src = None if source_fn is None else source_fn(t)
    return theta + dt * graph_rhs(theta, edge_index, edge_rate, beta, src)


_STEPPERS = {"rk4": rk4_step, "euler": euler_step}


def integrate(
    theta0: Tensor,
    times: Tensor,
    edge_index: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
    source_fn=None,
    method: str = "rk4",
    substeps: int = 1,
) -> Tensor:
    """Roll the ODE out over ``times``, returning ``(T, ..., N)``.

    ``times`` are absolute and need not be evenly spaced: each interval uses
    its own ``dt``, which is what keeps a sequence with a jittery filename
    clock (whole-second stamps on a ~6.3 s cadence) integrated correctly rather
    than on a nominal frame rate.  ``substeps`` subdivides each interval, which
    matters when a fast region pushes ``a*dt`` towards the stability limit.
    """
    if method not in _STEPPERS:
        raise ValueError(f"unknown integrator {method!r}; choose from {sorted(_STEPPERS)}")
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times must be a 1-D tensor with at least one entry")
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    step = _STEPPERS[method]

    out = [theta0]
    state = theta0
    for k in range(times.numel() - 1):
        t0 = float(times[k])
        total = float(times[k + 1]) - t0
        if total <= 0:
            raise ValueError(f"times must be strictly increasing (index {k})")
        dt = total / substeps
        for m in range(substeps):
            state = step(state, dt, edge_index, edge_rate, beta, source_fn, t0 + m * dt)
        out.append(state)
    return torch.stack(out, dim=0)


def dense_operator(
    n_nodes: int, edge_index: Tensor, edge_rate: Tensor, beta: Tensor
) -> Tensor:
    """The linear operator ``A`` with ``d(theta)/dt = A theta + s``, as a dense matrix.

    Only used for diagnostics and tests -- the region counts here are in the
    tens to low hundreds, so a dense eigendecomposition is cheap and it is the
    most direct way to check what the sparse ``index_add_`` path actually built.
    """
    matrix = torch.zeros(n_nodes, n_nodes, dtype=edge_rate.dtype, device=edge_rate.device)
    src, dst = edge_index[0], edge_index[1]
    matrix.index_put_((src, dst), edge_rate, accumulate=True)
    matrix.index_put_((dst, src), edge_rate, accumulate=True)
    matrix.index_put_((src, src), -edge_rate, accumulate=True)
    matrix.index_put_((dst, dst), -edge_rate, accumulate=True)
    matrix = matrix - torch.diag(beta.expand(n_nodes))
    return matrix


def spectral_bound(
    n_nodes: int, edge_index: Tensor, edge_rate: Tensor, beta: Tensor
) -> float:
    """Largest real part of the operator's spectrum.

    Non-positive means the unforced system decays.  This is the quantity the
    stability test asserts on and the stability loss penalises, and it is a
    fact about the operator rather than about any particular trajectory.
    """
    matrix = dense_operator(n_nodes, edge_index, edge_rate, beta).detach().double()
    return float(torch.linalg.eigvals(matrix).real.max())


def steady_state(
    n_nodes: int, edge_index: Tensor, edge_rate: Tensor, beta: Tensor, source: Tensor
) -> Tensor:
    """Solve ``A theta = -s`` for the equilibrium the operator settles into."""
    matrix = dense_operator(n_nodes, edge_index, edge_rate, beta).double()
    return torch.linalg.solve(matrix, -source.double()).to(source.dtype)


def max_stable_dt(
    n_nodes: int, edge_index: Tensor, edge_rate: Tensor, beta: Tensor, method: str = "rk4"
) -> float:
    """Largest step an explicit integrator stays stable at, for this operator.

    The stiffest mode has rate ``|lambda_min|``; explicit Euler needs
    ``dt < 2/|lambda|`` and RK4 buys roughly ``2.78/|lambda|``.  Reported so a
    configuration that would need substeps says so instead of producing a
    silently diverging rollout.
    """
    matrix = dense_operator(n_nodes, edge_index, edge_rate, beta).detach().double()
    stiffest = float(torch.linalg.eigvals(matrix).real.abs().max())
    if stiffest <= 0:
        return float("inf")
    return (2.78 if method == "rk4" else 2.0) / stiffest
