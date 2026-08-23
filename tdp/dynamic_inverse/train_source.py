"""Stage B: invert the normalized heat source over the powered interval.

The thermal operator identified in stage A is *frozen* here.  That ordering is
the whole point.  A source is the one term in the model that can explain any
residual -- give it freedom while the operator is still moving and it will
absorb the operator's errors, producing a source map that is really a
mis-fit map.  Freezing conduction and cooling first means every kelvin the
source is asked to account for is a kelvin conduction could not.

What comes out is ``s_i(t) = q_i(t) / C_i`` in K/s: the rate at which a region
would heat if nothing carried the heat away.  It is *not* power.  Recovering
watts would need the areal heat capacity ``C``, and no measurement in this
campaign provides it -- which is why the outputs are named ``source_rate`` and
``relative_source`` and never ``power``.

The relative contribution *is* meaningful: ``s_i A_i / sum_j s_j A_j`` is the
share of the board's total normalized generation attributable to region ``i``,
and the unknown scale cancels in the ratio only under the stated assumption
that ``C`` is uniform across the board.  That assumption is false in detail (a
package lid and bare laminate differ), so the shares are reported as
``relative`` with the assumption attached.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tdp.dynamic_inverse.config import AnalysisConfig
from tdp.dynamic_inverse.losses import total_loss
from tdp.dynamic_inverse.model import FreeSource, SourceDecoder
from tdp.dynamic_inverse.physics import RegionGraph, integrate
from tdp.dynamic_inverse.train_thermal import (
    NodeSeries,
    ThermalFit,
    Windows,
    _sample_windows,
    contiguous_runs,
    contiguous_split,
)


@dataclass
class SourceFit:
    source_rate: np.ndarray  # (T, N) K/time-unit, zero outside the powered phase
    relative_source: np.ndarray  # (T, N) area-weighted share, sums to 1 per frame
    state_dict: dict
    history: list[dict]
    windows: Windows
    metrics: dict[str, Any]
    time_unit: str
    powered: np.ndarray


def relative_contribution(source_rate: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Area-weighted share of the board's total normalized generation, per frame."""
    weighted = source_rate * areas[None, :]
    total = weighted.sum(axis=1, keepdims=True)
    return np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)


def fit_source(
    graph: RegionGraph,
    series: NodeSeries,
    thermal: ThermalFit,
    config: AnalysisConfig,
    *,
    seed: int = 0,
    device: str = "cpu",
    free_source_baseline: bool = False,
    powered_phases: tuple[str, ...] = ("heating", "near_steady"),
    joint_finetune: bool = True,
    max_epochs: int | None = None,
    progress: bool = False,
) -> SourceFit:
    """Fit ``s_i(t) >= 0`` over the powered interval with the operator frozen."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    powered = np.isin(series.phase, list(powered_phases))
    if powered.sum() < 8:
        raise ValueError(
            f"only {int(powered.sum())} powered frames; the source cannot be inverted"
        )
    powered_index = np.nonzero(powered)[0]
    # The powered label can select more than one interval -- on case10 the board
    # is "near steady" both under load and again while idling after the decay --
    # so each contiguous run is split on its own.  Splitting the union would let
    # a rollout window start before the gap and integrate straight through the
    # load-off interval.
    runs = [run for run in contiguous_runs(powered_index) if run.size >= 12]
    if not runs:
        raise ValueError("no contiguous powered interval long enough to fit a source")
    windows = Windows()
    for run in runs:
        part = contiguous_split(
            run,
            config.split.train_fraction,
            config.split.val_fraction,
            mode=config.split.mode,
            n_blocks=config.split.n_blocks,
            guard=config.split.guard_frames,
        )
        windows.train.extend(part.train)
        windows.val.extend(part.val)
        windows.test.extend(part.test)

    t_node = torch.as_tensor(thermal.node_features, device=dev)
    t_edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=dev)
    t_sim = torch.as_tensor(thermal.similarity, device=dev)
    t_area = torch.as_tensor(graph.areas_px, dtype=torch.float32, device=dev)
    t_theta = torch.as_tensor(series.theta, dtype=torch.float32, device=dev)
    t_times = torch.as_tensor(series.times, dtype=torch.float32, device=dev)
    t_weight = torch.as_tensor(series.weight, dtype=torch.float32, device=dev)
    t_trust = torch.as_tensor(
        np.broadcast_to(series.trust[None, :], series.theta.shape).copy(),
        dtype=torch.float32,
        device=dev,
    )
    t_powered = torch.as_tensor(powered, dtype=torch.bool, device=dev)
    period = float(max(series.times[-1] - series.times[0], 1.0))

    # The frozen operator.  Held as plain tensors, not a module, so there is no
    # way for an optimiser step to touch them by accident.
    edge_rate = torch.as_tensor(thermal.edge_rate, dtype=torch.float32, device=dev)
    beta = torch.as_tensor(thermal.beta, dtype=torch.float32, device=dev)

    # The fastest observed warming rate bounds what a source can plausibly be:
    # d(theta)/dt = (conduction) - beta theta + s, so s cannot exceed the peak
    # rise by much.  Using it as the head's scale starts the decoder in the
    # right decade for this board rather than at an arbitrary constant.
    with np.errstate(invalid="ignore"):
        rise = np.gradient(np.nan_to_num(series.theta), series.times, axis=0)
    source_scale = float(max(np.nanpercentile(rise[powered], 99.0), 1e-6))

    decoder: torch.nn.Module = (
        FreeSource(n_times=len(series.times), n_nodes=graph.n_nodes, source_scale=source_scale)
        if free_source_baseline
        else SourceDecoder(
            node_feature_dim=thermal.node_features.shape[1],
            n_time_features=config.model.source_time_features,
            embedding_dim=config.model.node_embedding_dim,
            hidden_dim=config.model.source_hidden_dim,
            dropout=config.model.dropout,
            source_scale=source_scale,
        )
    ).to(dev)

    optimiser = torch.optim.Adam(decoder.parameters(), lr=config.train.lr_source)
    epochs = int(max_epochs if max_epochs is not None else config.train.epochs_source)
    length = int(config.train.rollout_steps)
    history: list[dict] = []
    best = {"val": np.inf, "state": copy.deepcopy(decoder.state_dict()), "epoch": -1}
    patience = 0

    def rollout(spans: list[tuple[int, int]]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        source = decoder(t_node, t_times, t_powered, period)  # (T, N)

        def source_fn(t: float) -> Tensor:
            # Zero-order hold on the frame grid.  The source is a schedule, and
            # interpolating between frames would invent a ramp at load-off.
            index = int(torch.searchsorted(t_times, torch.tensor(t, device=dev), right=True)) - 1
            return source[min(max(index, 0), source.shape[0] - 1)]

        preds, targets, weights, trusts = [], [], [], []
        for start, stop in spans:
            predicted = integrate(
                t_theta[start],
                t_times[start : stop + 1],
                t_edge_index,
                edge_rate,
                beta,
                source_fn,
                method=config.train.integrator,
            )
            preds.append(predicted)
            targets.append(t_theta[start : stop + 1])
            weights.append(t_weight[start : stop + 1])
            trusts.append(t_trust[start : stop + 1])
        return (
            torch.cat(preds),
            torch.cat(targets),
            torch.cat(weights),
            torch.cat(trusts),
            source,
        )

    for epoch in range(epochs):
        decoder.train()
        spans = _sample_windows(
            windows.train, length, config.train.rollout_stride, rng, config.train.batch_windows
        )
        if not spans:
            raise ValueError(
                f"powered training block {windows.train} is too short for a rollout window"
            )
        optimiser.zero_grad(set_to_none=True)
        predicted, target, weight, trust, source = rollout(spans)
        loss, report = total_loss(
            predicted=predicted,
            target=target,
            weight=weight,
            trust=trust,
            edge_rate=edge_rate,
            beta=beta,
            edge_index=t_edge_index,
            n_nodes=graph.n_nodes,
            similarity=t_sim,
            source=source[t_powered],
            times=t_times[t_powered],
            area=t_area,
            weights=config.loss,
            huber_delta=config.train.huber_delta,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.train.grad_clip)
        optimiser.step()

        decoder.eval()
        with torch.no_grad():
            val_spans = _sample_windows(windows.val, length, 1, rng, 64)
            if not val_spans:
                raise ValueError(
                    f"powered validation block {windows.val} is too short for a rollout window"
                )
            predicted, target, weight, _, _ = rollout(val_spans)
            val = float(
                ((predicted - target).abs() * weight).sum() / weight.sum().clamp_min(1e-9)
            )
        report["val_mae"] = val
        report["epoch"] = epoch
        history.append(report)
        if progress and epoch % 50 == 0:
            print(f"  epoch {epoch:4d} loss {report['total']:.5f} val_mae {val:.4f}")
        if val < best["val"] - 1e-6:
            best = {"val": val, "state": copy.deepcopy(decoder.state_dict()), "epoch": epoch}
            patience = 0
        else:
            patience += 1
            if patience >= config.train.patience:
                break

    decoder.load_state_dict(best["state"])
    decoder.eval()
    with torch.no_grad():
        source = decoder(t_node, t_times, t_powered, period).cpu().numpy().astype(np.float64)

    areas = graph.areas_px
    relative = relative_contribution(source, areas)
    powered_source = source[powered]
    spread = _concentration(relative[powered])

    metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val"]),
        "epochs_run": len(history),
        "powered_runs": [[int(run[0]), int(run[-1])] for run in runs],
        "n_powered_frames": int(powered.sum()),
        "source_kind": "free_per_frame" if free_source_baseline else "structured_decoder",
        "source_rate_units": f"K/{series.time_unit}",
        "source_scale_per_unit_time": source_scale,
        "source_rate_max": float(powered_source.max()),
        "source_rate_mean": float(powered_source.mean()),
        "source_zero_outside_powered": bool(np.all(source[~powered] == 0.0)),
        "source_non_negative": bool(np.all(source >= 0.0)),
        "top_regions_by_relative_source": _top_regions(relative[powered].mean(axis=0)),
        "relative_source_concentration": spread,
        "operator_frozen": True,
        "units_note": (
            "s = q/C in K/{unit}, not power.  Absolute watts need the areal heat "
            "capacity C, which this campaign does not measure."
        ).format(unit=series.time_unit),
    }
    del joint_finetune
    return SourceFit(
        source_rate=source,
        relative_source=relative,
        state_dict={k: v.cpu() for k, v in best["state"].items()},
        history=history,
        windows=windows,
        metrics=metrics,
        time_unit=series.time_unit,
        powered=powered,
    )


def _top_regions(mean_relative: np.ndarray, count: int = 8) -> list[dict[str, float]]:
    order = np.argsort(-mean_relative)[:count]
    return [
        {"region": int(i), "mean_relative_source": float(mean_relative[i])} for i in order
    ]


def _concentration(relative: np.ndarray) -> dict[str, float]:
    """How concentrated the inverted source is, as a guard against smearing.

    A source spread evenly over the board is the classic failure of this
    inversion: it fits the mean and localises nothing.  These two numbers say
    plainly whether that happened -- the share held by the top 10% of regions,
    and the effective number of contributing regions (the inverse participation
    ratio), which equals N for a perfectly uniform source and 1 for a point
    source.
    """
    mean_relative = relative.mean(axis=0)
    n = len(mean_relative)
    top = max(1, int(round(0.1 * n)))
    order = np.argsort(-mean_relative)
    squared = float((mean_relative**2).sum())
    return {
        "top_decile_share": float(mean_relative[order[:top]].sum()),
        "effective_regions": float(1.0 / squared) if squared > 0 else float(n),
        "n_regions": int(n),
        "uniform_reference_effective_regions": float(n),
    }
