"""Loss terms for the physics-constrained fit.

The rollout term does the work; every other term exists to stop a specific
failure mode that was observed rather than imagined:

* ``source_l1`` -- without it the source spreads thinly over the whole board,
  because a diffuse source and a concentrated one both raise the mean.
* ``source_tv_time`` -- without it the source flickers frame to frame to chase
  sensor noise, and the "dynamic source map" becomes a noise movie.
* ``edge_reg`` -- an unbounded coupling rate makes neighbouring regions
  identical, which fits a smooth field perfectly and identifies nothing.
* ``spatial_smooth`` -- neighbouring regions of similar material should have
  similar parameters, but the weight is *gated by feature similarity* so the
  penalty does not smear a package boundary into the substrate.
* ``stability`` -- penalises any eigenvalue with a positive real part, i.e. an
  operator that would warm itself up with no source.

Weights are configuration, not constants: :mod:`tdp.dynamic_inverse.evaluate`
runs the ablation that switches each of them off.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Weighted Huber loss over observed entries.

    Huber rather than MSE because a repaired pixel deep inside a connector
    footprint can be several K off; squared error would let a handful of those
    dominate the whole fit.
    """
    residual = prediction - target
    absolute = residual.abs()
    quadratic = torch.clamp(absolute, max=delta)
    linear = absolute - quadratic
    per_element = 0.5 * quadratic**2 + delta * linear
    total = weight.sum()
    if total <= 0:
        return prediction.sum() * 0.0
    return (per_element * weight).sum() / total


def rollout_loss(
    predicted: Tensor,
    target: Tensor,
    weight: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Trajectory mismatch over a multi-step rollout, ``(T, N)`` shaped."""
    return masked_huber(predicted, target, weight, delta)


def observed_loss(
    predicted: Tensor,
    target: Tensor,
    trust: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Same mismatch, but weighted only by directly-measured confidence.

    ``trust`` is near 1 for high-emissivity pixels the camera actually measured
    and near 0 inside a reconstructed footprint.  Keeping this as a separate
    term means the fit is pulled hardest by real measurements even though the
    reconstructed regions still participate through ``rollout_loss``.
    """
    return masked_huber(predicted, target, trust, delta)


def source_l1(source: Tensor, area: Tensor | None = None) -> Tensor:
    """Encourage spatial sparsity: few regions generating, most passive."""
    if area is None:
        return source.abs().mean()
    weights = area / area.sum()
    return (source.abs() * weights).sum(dim=-1).mean()


def source_temporal_tv(source: Tensor, times: Tensor) -> Tensor:
    """Penalise frame-to-frame jumps in the source, per unit time.

    Dividing by ``dt`` keeps the penalty meaningful when the clock is irregular:
    a 1 K/s change across a 12 s gap is half the rate of change of the same
    jump across 6 s, and an undivided TV would punish them equally.
    """
    if source.shape[0] < 2:
        return source.sum() * 0.0
    dt = (times[1:] - times[:-1]).clamp_min(1e-6).unsqueeze(-1)
    return ((source[1:] - source[:-1]).abs() / dt).mean()


def edge_regularization(edge_rate: Tensor) -> Tensor:
    """Keep coupling finite, measured *relative* to the typical rate.

    Dividing by the mean is what makes the weight in the config mean anything.
    An absolute squared penalty on a rate of order 1e-3 per second contributes
    about 1e-6 to a loss of order 1, so no reasonable weight can move it -- and
    switching the time unit from seconds to frames would silently rescale it by
    the cadence squared.  The relative form is dimensionless, so one weight is
    valid for both time units and actually bites.
    """
    scale = edge_rate.detach().mean().clamp_min(1e-12)
    return ((edge_rate / scale) ** 2).mean()


def parameter_spatial_smoothness(
    values: Tensor,
    edge_index: Tensor,
    similarity: Tensor,
) -> Tensor:
    """Similarity-gated smoothness of a per-node parameter, relative to its scale.

    ``similarity`` is 1 for two regions that look like the same material and
    approaches 0 across a component boundary, so the penalty smooths the
    substrate without dragging a package towards its surroundings.  As with
    :func:`edge_regularization`, differences are measured against the mean so
    the weight is dimensionless and unit-independent.
    """
    scale = values.detach().mean().abs().clamp_min(1e-12)
    left = values.index_select(0, edge_index[0])
    right = values.index_select(0, edge_index[1])
    return (similarity * ((left - right) / scale) ** 2).mean()


def implied_steady_source(
    theta_steady: Tensor, edge_index: Tensor, edge_rate: Tensor, beta: Tensor
) -> Tensor:
    """The source the operator says must be running to hold ``theta_steady``.

    At a steady state ``0 = A theta + s``, so ``s = -A theta``.  With the
    measured steady profile in hand this is a direct, differentiable readout of
    the operator -- no extra unknowns.
    """
    from tdp.dynamic_inverse.physics import graph_rhs

    return -graph_rhs(theta_steady, edge_index, edge_rate, beta, None)


def steady_source_plausibility(
    theta_steady: Tensor,
    edge_index: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Is the implied steady source physically possible, and is it localised?

    This is what breaks the conduction/convection degeneracy.  A decay on its
    own cannot separate them: with no coupling at all, every region relaxes as a
    clean exponential at its own ``beta``, and coupled regions relaxing together
    look almost the same through sensor noise.  Measured on case10 the decay-only
    fit lands on a diffusion length of about 3 mm, which the *steady* profile
    flatly contradicts -- a 3 mm screening length would make each region roughly
    twenty times cooler than its neighbour, and the board is smooth.

    The steady phase supplies the missing constraint without any new
    measurement.  Set ``a = 0`` and the implied source becomes ``beta*theta``:
    positive everywhere, spread across bare laminate, generating heat where
    there are no components.  Real dissipation sits on packages.  So two
    scale-free penalties are applied to ``s = -A theta_steady``:

    * **negativity** -- the share of squared magnitude in negative entries; a
      source cannot be a sink.
    * **spread** -- ``||s||_1 / (sqrt(N) ||s||_2)``, which is 1 for a perfectly
      uniform source and ``1/sqrt(N)`` for a single spike.

    Both are dimensionless and bounded, so their weights transfer across time
    units and region counts.
    """
    source = implied_steady_source(theta_steady, edge_index, edge_rate, beta)
    total = (source**2).sum().clamp_min(1e-24)
    negativity = (torch.relu(-source) ** 2).sum() / total
    positive = torch.relu(source)
    n = torch.tensor(float(source.shape[-1]), dtype=source.dtype, device=source.device)
    spread = positive.sum() / (n.sqrt() * positive.norm(2).clamp_min(1e-12))
    return negativity, spread


def stability_penalty(
    n_nodes: int, edge_index: Tensor, edge_rate: Tensor, beta: Tensor
) -> Tensor:
    """Penalise any spectrum that would let an unforced system warm up.

    Uses a Gershgorin bound instead of an eigendecomposition: for row ``i`` the
    disc is centred at ``-(sum_j a_ij + beta_i)`` with radius ``sum_j a_ij``, so
    the largest possible real part is ``-beta_i``.  Non-negative ``beta``
    therefore already guarantees stability, and this term stays at zero unless
    the parameterisation is changed to allow a negative one -- which is exactly
    the regression it is here to catch.
    """
    del n_nodes, edge_index, edge_rate
    return torch.relu(-beta).pow(2).mean()


def total_loss(
    *,
    predicted: Tensor,
    target: Tensor,
    weight: Tensor,
    trust: Tensor,
    edge_rate: Tensor,
    beta: Tensor,
    edge_index: Tensor,
    n_nodes: int,
    similarity: Tensor,
    source: Tensor | None,
    times: Tensor | None,
    area: Tensor | None,
    weights,
    huber_delta: float = 1.0,
    theta_steady: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Assemble the weighted total and report every term for the run record."""
    terms: dict[str, Tensor] = {
        "rollout": rollout_loss(predicted, target, weight, huber_delta),
        "observed": observed_loss(predicted, target, trust, huber_delta),
        "edge_reg": edge_regularization(edge_rate),
        "spatial_smooth": (
            parameter_spatial_smoothness(beta, edge_index, similarity)
            if beta.numel() > 1
            else beta.sum() * 0.0
        ),
        "stability": stability_penalty(n_nodes, edge_index, edge_rate, beta),
    }
    if theta_steady is not None:
        negativity, spread = steady_source_plausibility(
            theta_steady, edge_index, edge_rate, beta
        )
        terms["steady_source_negativity"] = negativity
        terms["steady_source_spread"] = spread
    if source is not None and times is not None:
        terms["source_l1"] = source_l1(source, area)
        terms["source_tv_time"] = source_temporal_tv(source, times)

    total = sum(getattr(weights, name) * value for name, value in terms.items())
    report = {name: float(value.detach()) for name, value in terms.items()}
    report["total"] = float(total.detach())
    return total, report
