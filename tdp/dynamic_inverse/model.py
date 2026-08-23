"""Neural parameterisation of the graph heat operator and the source.

The network never predicts the next frame.  It predicts the *coefficients* of
the physical ODE -- one coupling rate per undirected edge, one cooling rate per
region, one source rate per region and time -- and the trajectory then comes
from integrating that ODE.  A black-box next-frame predictor would fit this
data at least as well and would say nothing about conduction, so it is used
only as a stated baseline, never as a result.

Two constraints are built into the architecture rather than added to the loss:

* the edge head is fed a *symmetric* function of its two endpoints
  (sum and absolute difference), so swapping ``i`` and ``j`` cannot change the
  predicted rate even before the single-rate-per-edge storage enforces it;
* every rate passes through ``softplus``, so non-negativity holds at
  initialisation, during training and at every intermediate optimiser step.

The source decoder is deliberately weak: a small MLP over a *fixed* Fourier
time basis rather than a recurrent net.  A flexible source can absorb any
mismatch in the thermal operator -- it is the one term that can explain
anything -- so it is given the least capacity of the three heads, is bounded
below by zero, and is fitted last, after the operator has been pinned down on
the cooling phase where the source is known to be constant.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

# softplus(UNIT_BIAS) == 1, so a head initialised at this bias emits exactly one
# `rate_scale`.  Passing the board's own measured decay rate as `rate_scale`
# then starts every head at the right order of magnitude instead of at an
# arbitrary constant -- which matters because softplus is nearly flat far below
# its knee, so a head that starts a decade too fast takes far longer to come
# down than the fit has epochs for.
UNIT_BIAS = math.log(math.e - 1.0)


def _inverse_softplus(value: float) -> float:
    """Bias whose softplus is ``value``."""
    return float(math.log(math.expm1(max(float(value), 1e-9))))


def _mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if k < len(sizes) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def fourier_time_basis(times: Tensor, n_features: int, period: float) -> Tensor:
    """Smooth, bounded time features: ``[t/period, sin/cos(2^k pi t/period)]``.

    A fixed basis rather than a learned encoder keeps the source's temporal
    freedom explicit and bounded -- the highest frequency present sets the
    fastest wiggle the source can have, which is what the temporal-TV penalty
    then has to work with rather than against.
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    scaled = (times / period).unsqueeze(-1)
    feats = [scaled]
    k = 0
    while len(feats) * 1 < n_features:
        arg = (2.0**k) * np.pi * scaled
        feats.append(torch.sin(arg))
        if len(feats) < n_features:
            feats.append(torch.cos(arg))
        k += 1
    return torch.cat(feats[:n_features], dim=-1)


class ThermalOperator(nn.Module):
    """Predicts ``a_ij >= 0`` per undirected edge and ``beta_i >= 0`` per region."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        rate_scale: float = 1e-2,
        edge_rate_scale: float | None = None,
        edge_init_fraction: float = 0.3,
    ):
        super().__init__()
        self.rate_scale = float(rate_scale)
        self.edge_rate_scale = float(
            rate_scale if edge_rate_scale is None else edge_rate_scale
        )
        # Coupling and cooling do not live at the same decade, so they get
        # separate scales, both read off the data by
        # `train_thermal.decay_rate_scale` and `coupling_rate_scale`.  They are
        # fixed, not learned.
        #
        # Learning the global magnitude as a free exponent was tried and
        # reverted.  It was meant to rescue the near-isothermal regime, where
        # every fixed scale lands the coupling five to fifteen times low -- but
        # that regime is not a scaling problem, it is unidentifiable: the
        # profile flattens within a handful of frames and after that every
        # region reports the same rate whatever the coupling is
        # (`identifiability.coupling_contrast_recoverable` detects and flags
        # exactly this).  So the free exponent fixed nothing there, while in the
        # regime that *is* recoverable it wandered and dropped the edge-rate
        # correlation from 0.82 to 0.34.  A simpler model that is honest about
        # what it cannot do beats a more flexible one that hides it.
        self.node_encoder = _mlp(
            [node_feature_dim, hidden_dim, hidden_dim, embedding_dim], dropout
        )
        # 2*embedding_dim: the symmetric pair (sum, |difference|) of endpoints.
        self.edge_head = _mlp([2 * embedding_dim + edge_feature_dim, hidden_dim, 1], dropout)
        self.cooling_head = _mlp([embedding_dim, hidden_dim, 1], dropout)
        # Both heads start at one unit of their (learnable) scale, so the fit
        # begins exactly at the estimated rates and moves outward from there.
        nn.init.zeros_(self.cooling_head[-1].weight)
        nn.init.constant_(self.cooling_head[-1].bias, UNIT_BIAS)
        # The edge head starts a little below its scale: of the two directions
        # of error, climbing is recoverable and sinking into softplus's flat
        # negative tail is not.
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.constant_(self.edge_head[-1].bias, _inverse_softplus(edge_init_fraction))

    def embed(self, node_features: Tensor) -> Tensor:
        return self.node_encoder(node_features)

    def edge_rates(
        self, embedding: Tensor, edge_index: Tensor, edge_features: Tensor
    ) -> Tensor:
        left = embedding.index_select(0, edge_index[0])
        right = embedding.index_select(0, edge_index[1])
        symmetric = torch.cat([left + right, (left - right).abs(), edge_features], dim=-1)
        return self.edge_rate_scale * nn.functional.softplus(
            self.edge_head(symmetric)
        ).squeeze(-1)

    def cooling_rates(self, embedding: Tensor) -> Tensor:
        return self.rate_scale * nn.functional.softplus(
            self.cooling_head(embedding)
        ).squeeze(-1)

    def forward(
        self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        embedding = self.embed(node_features)
        return self.edge_rates(embedding, edge_index, edge_features), self.cooling_rates(embedding)


class SourceDecoder(nn.Module):
    """Predicts ``s_i(t) >= 0`` in K/s over the powered phase."""

    def __init__(
        self,
        node_feature_dim: int,
        n_time_features: int = 8,
        embedding_dim: int = 32,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        source_scale: float = 1e-2,
    ):
        super().__init__()
        self.n_time_features = int(n_time_features)
        self.source_scale = float(source_scale)
        self.node_encoder = _mlp([node_feature_dim, hidden_dim, embedding_dim], dropout)
        self.head = _mlp([embedding_dim + n_time_features + 1, hidden_dim, 1], dropout)
        nn.init.zeros_(self.head[-1].weight)
        # Start well below one `source_scale`: the source is the term that can
        # explain anything, so it should have to earn its amplitude upward
        # rather than start large and be regularised back down.
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(
        self,
        node_features: Tensor,
        times: Tensor,
        powered: Tensor,
        period: float,
    ) -> Tensor:
        """Return ``(T, N)`` source rates, exactly zero wherever ``powered`` is False.

        The zeroing is a hard multiply, not a penalty: during the load-off
        interval the model must explain the decay with conduction and cooling
        alone, which is the whole reason that interval identifies the operator.
        """
        embedding = self.node_encoder(node_features)  # (N, D)
        time_feats = fourier_time_basis(times, self.n_time_features, period)  # (T, F)
        powered_f = powered.to(time_feats.dtype).unsqueeze(-1)  # (T, 1)
        context = torch.cat([time_feats, powered_f], dim=-1)  # (T, F+1)

        n_times, n_nodes = context.shape[0], embedding.shape[0]
        pair = torch.cat(
            [
                embedding.unsqueeze(0).expand(n_times, n_nodes, -1),
                context.unsqueeze(1).expand(n_times, n_nodes, -1),
            ],
            dim=-1,
        )
        raw = self.source_scale * nn.functional.softplus(self.head(pair)).squeeze(-1)
        return raw * powered.to(raw.dtype).unsqueeze(-1)


class UniformOperator(nn.Module):
    """Baseline: one global ``alpha`` and one global ``beta`` for the whole board.

    This is the model the harness requires as a comparison point -- it has the
    same physics and the same rollout fitting as the full model, and differs
    only in having no spatial freedom, so a win over it is a win for
    heterogeneity rather than for the training procedure.
    """

    def __init__(
        self,
        rate_scale: float = 1e-2,
        edge_rate_scale: float | None = None,
        edge_init_fraction: float = 0.3,
    ):
        super().__init__()
        self.rate_scale = float(rate_scale)
        self.edge_rate_scale = float(
            rate_scale if edge_rate_scale is None else edge_rate_scale
        )
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(edge_init_fraction)))
        self.raw_beta = nn.Parameter(torch.tensor(UNIT_BIAS))

    def forward(
        self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        alpha = self.edge_rate_scale * nn.functional.softplus(self.raw_alpha)
        beta = self.rate_scale * nn.functional.softplus(self.raw_beta)
        n_edges = edge_index.shape[1]
        n_nodes = node_features.shape[0]
        return alpha.expand(n_edges), beta.expand(n_nodes)


class FreeSource(nn.Module):
    """Baseline source: one free non-negative value per region and frame.

    No temporal structure and no sharing across regions, so it is the most
    permissive source the physics allows.  It exists to show what the
    structured decoder buys: if this fits no better, the structure costs
    nothing; if it fits far better, the decoder is too rigid.
    """

    def __init__(self, n_times: int, n_nodes: int, source_scale: float = 1e-2):
        super().__init__()
        self.source_scale = float(source_scale)
        self.raw = nn.Parameter(torch.full((n_times, n_nodes), -2.0))

    def forward(
        self, node_features: Tensor, times: Tensor, powered: Tensor, period: float
    ) -> Tensor:
        raw = self.source_scale * nn.functional.softplus(self.raw)
        return raw * powered.to(raw.dtype).unsqueeze(-1)
