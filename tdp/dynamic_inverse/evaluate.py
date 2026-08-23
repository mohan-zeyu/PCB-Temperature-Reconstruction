"""Metrics, baselines and ablations.

Every claim in the report has to be beaten out of a baseline here.  Five models
are compared on the same held-out contiguous blocks:

1. **steady_plate** -- the existing screened-harmonic fill from
   :mod:`tdp.data.emissivity`, applied per frame.  It has no dynamics at all,
   so it sets the bar a *time-series* model has to clear.
2. **uniform_dynamic** -- one global ``alpha`` and one global ``beta``.  Same
   physics, same fitting, no spatial freedom.
3. **laplacian_source** -- the naive source map: take the measured field's
   Laplacian and call the residual a source.  No fitting.
4. **region_no_modality** -- the graph model with emissivity and geometry
   removed from the node features, leaving only the trajectory.
5. **graph_neural_ode** -- the full model.

If the full model does not win, the numbers say so.  ``compare_models`` returns
whatever it measured; nothing here suppresses an unfavourable row.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tdp.dynamic_inverse.config import AnalysisConfig
from tdp.dynamic_inverse.physics import RegionGraph, integrate
from tdp.dynamic_inverse.train_source import fit_source
from tdp.dynamic_inverse.train_thermal import NodeSeries, ThermalFit, fit_thermal_operator


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rollout_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    times: np.ndarray,
) -> dict[str, float]:
    """Trajectory-quality metrics, all confidence-weighted."""
    residual = predicted - target
    total = weight.sum()
    if total <= 0:
        raise ValueError("no weighted observations to score")
    mae = float((np.abs(residual) * weight).sum() / total)
    rmse = float(np.sqrt((residual**2 * weight).sum() / total))
    max_abs = float(np.abs(residual)[weight > 0].max())
    # Drift is the error at the far end of the rollout: a model can have a
    # small average error and still walk away from the truth.
    drift = float((np.abs(residual[-1]) * weight[-1]).sum() / max(weight[-1].sum(), 1e-9))
    mean_trace = (predicted * weight).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-9)
    rises = int(np.sum(np.diff(mean_trace) > 0))
    return {
        "temperature_mae_k": mae,
        "temperature_rmse_k": rmse,
        "max_abs_error_k": max_abs,
        "long_rollout_drift_k": drift,
        "cooling_monotonicity_violations": rises,
        "n_rollout_steps": int(len(times) - 1),
    }


def free_rollout(
    graph: RegionGraph,
    series: NodeSeries,
    thermal: ThermalFit,
    block: tuple[int, int],
    source_rate: np.ndarray | None = None,
    residual_source: np.ndarray | None = "fit",
) -> np.ndarray:
    """Integrate from the first frame of ``block`` to its last, without correction.

    No re-initialisation along the way.  A model that is only ever asked for one
    step ahead can hide a systematic error that this exposes.

    ``residual_source`` defaults to the constant idle source the fit identified
    over the load-off interval, because that is the model that was fitted.
    Rolling the operator out *without* it would score a different model than the
    one identified -- and would silently reward a fit that had gone wrong in the
    direction of a near-zero cooling rate, since such a fit barely moves and so
    matches a flat tail by accident.  Pass ``None`` to score the strict
    ``q = 0`` free-cooling model instead, which this experiment cannot justify.
    """
    lo, hi = block
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    edge_rate = torch.as_tensor(thermal.edge_rate, dtype=torch.float32)
    beta = torch.as_tensor(thermal.beta, dtype=torch.float32)
    times = torch.as_tensor(series.times[lo : hi + 1], dtype=torch.float32)
    theta0 = torch.as_tensor(series.theta[lo], dtype=torch.float32)

    source_fn = None
    if source_rate is not None:
        source = torch.as_tensor(source_rate[lo : hi + 1], dtype=torch.float32)
        window_times = series.times[lo : hi + 1]

        def source_fn(t: float):  # noqa: F811
            # Zero-order hold on the frame grid, matching how it was fitted.
            index = int(np.searchsorted(window_times, t, side="right") - 1)
            return source[min(max(index, 0), source.shape[0] - 1)]
    else:
        if isinstance(residual_source, str) and residual_source == "fit":
            residual_source = thermal.idle_source_rate
        if residual_source is not None and np.any(residual_source):
            constant = torch.as_tensor(residual_source, dtype=torch.float32)

            def source_fn(t: float):  # noqa: F811
                del t
                return constant

    with torch.no_grad():
        return integrate(theta0, times, edge_index, edge_rate, beta, source_fn).numpy()


def evaluate_fit(
    graph: RegionGraph,
    series: NodeSeries,
    thermal: ThermalFit,
    source_rate: np.ndarray | None = None,
    blocks: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Score a fit on each split, plus a per-region error breakdown."""
    blocks = blocks or {
        "train": thermal.windows.train,
        "val": thermal.windows.val,
        "test": thermal.windows.test,
    }
    out: dict[str, Any] = {}
    for name, spans in blocks.items():
        scores: list[dict[str, float]] = []
        for block in spans:
            predicted = free_rollout(graph, series, thermal, block, source_rate)
            lo, hi = block
            scores.append(
                rollout_metrics(
                    predicted, series.theta[lo : hi + 1], series.weight[lo : hi + 1], series.times[lo : hi + 1]
                )
            )
        out[name] = {
            key: float(np.mean([s[key] for s in scores])) for key in scores[0]
        } if scores else {}
    # Per-region test error: a board-average hides a region the model cannot fit.
    if blocks.get("test"):
        lo, hi = blocks["test"][0]
        predicted = free_rollout(graph, series, thermal, (lo, hi), source_rate)
        residual = np.abs(predicted - series.theta[lo : hi + 1])
        per_region = residual.mean(axis=0)
        out["per_region_test_mae_k"] = [float(v) for v in per_region]
        out["worst_regions"] = [
            {"region": int(i), "mae_k": float(per_region[i])}
            for i in np.argsort(-per_region)[:8]
        ]
    return out


def _identifiability_summary(graph: RegionGraph, fit: ThermalFit) -> dict[str, Any]:
    """The numbers that say whether the conduction/convection split held up."""
    from tdp.dynamic_inverse.derived import characteristic_length, diffusivity_proxy

    alpha = diffusivity_proxy(graph, fit.edge_rate)
    length, _ = characteristic_length(alpha, fit.beta)
    steady = fit.metrics.get("implied_steady_source", {})
    return {
        "alpha_proxy_median_px2_per_unit_time": float(np.median(alpha)),
        "length_median_px": float(np.median(length)),
        "tau_median_unit_time": float(1.0 / max(np.median(fit.beta), 1e-12)),
        "steady_source_top_decile_share": steady.get("top_decile_share"),
        "steady_source_spread_uniform_is_one": steady.get("spread_uniform_is_one"),
        "steady_source_negativity_fraction": steady.get("negativity_fraction"),
    }


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def steady_plate_baseline(
    sequence, labels: np.ndarray, series: NodeSeries, block: tuple[int, int]
) -> dict[str, float]:
    """Baseline 1: the existing steady screened-harmonic fill, per frame.

    Reuses :func:`tdp.data.emissivity.solve_plate` rather than reimplementing
    it.  Given the *board perimeter* as Dirichlet data it solves for the
    interior, which is exactly what the single-frame pipeline does; it carries
    no time dependence, so its prediction for every frame of a rollout is that
    frame's own steady solution.  It is scored the same way as everything else.
    """
    from tdp.data.emissivity import solve_plate

    lo, hi = block
    predictions = []
    for index in range(lo, hi + 1):
        theta = sequence.theta[index].astype(np.float64)
        unknown = sequence.valid_mask.copy()
        # Keep a one-pixel Dirichlet ring so the solve has boundary data.
        unknown[0, :] = unknown[-1, :] = unknown[:, 0] = unknown[:, -1] = False
        filled = solve_plate(np.nan_to_num(theta), unknown)
        predictions.append(filled)
    stacked = np.stack(predictions)
    node_pred = np.zeros((len(stacked), series.n_nodes))
    inside = labels > 0
    index_map = labels[inside] - 1
    for t, frame in enumerate(stacked):
        values = frame[inside]
        counts = np.bincount(index_map, minlength=series.n_nodes)
        sums = np.bincount(index_map, weights=values, minlength=series.n_nodes)
        node_pred[t] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return rollout_metrics(
        np.nan_to_num(node_pred),
        series.theta[lo : hi + 1],
        series.weight[lo : hi + 1],
        series.times[lo : hi + 1],
    )


def laplacian_source_baseline(
    graph: RegionGraph, series: NodeSeries, block: tuple[int, int]
) -> dict[str, Any]:
    """Baseline 3: call the graph Laplacian of the measured field a source.

    ``s_naive = d(theta)/dt - L theta`` with one global coupling.  It needs no
    fitting and produces a source map instantly -- and, because it differentiates
    noisy data twice, a map dominated by noise.  Reported as the concentration
    of what it produces, so the comparison with the fitted source is like for
    like.
    """
    lo, hi = block
    theta = series.theta[lo : hi + 1]
    times = series.times[lo : hi + 1]
    derivative = np.gradient(theta, times, axis=0)
    i, j = graph.edge_index
    laplacian = np.zeros_like(theta)
    for edge in range(graph.n_edges):
        delta = theta[:, j[edge]] - theta[:, i[edge]]
        laplacian[:, i[edge]] += delta
        laplacian[:, j[edge]] -= delta
    scale = float(np.median(np.abs(derivative)) / max(np.median(np.abs(laplacian)), 1e-9))
    naive = derivative - scale * laplacian
    positive = np.clip(naive, 0.0, None)
    weighted = positive * graph.areas_px[None, :]
    total = weighted.sum(axis=1, keepdims=True)
    relative = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
    mean_relative = relative.mean(axis=0)
    squared = float((mean_relative**2).sum())
    return {
        "implied_coupling_scale": scale,
        "negative_fraction": float((naive < 0).mean()),
        "effective_regions": float(1.0 / squared) if squared > 0 else float(graph.n_nodes),
        "n_regions": int(graph.n_nodes),
        "top_regions": [int(k) for k in np.argsort(-mean_relative)[:8]],
        "note": (
            "differentiating a noisy field twice; reported for comparison only, "
            "and its negative values have no physical meaning"
        ),
    }


# --------------------------------------------------------------------------
# comparison and ablations
# --------------------------------------------------------------------------
@dataclass
class ComparisonRow:
    name: str
    kind: str
    metrics: dict[str, Any]
    note: str = ""


def compare_models(
    sequence,
    partition,
    series: NodeSeries,
    config: AnalysisConfig,
    *,
    device: str = "cpu",
    seed: int = 0,
    include_source: bool = True,
) -> dict[str, Any]:
    """Run the baseline ladder and the ablations on one sequence."""
    graph = partition.graph
    rows: list[ComparisonRow] = []

    full = fit_thermal_operator(graph, series, config, seed=seed, device=device)
    full_scores = evaluate_fit(graph, series, full)
    rows.append(ComparisonRow("graph_neural_ode", "full", full_scores))

    uniform = fit_thermal_operator(
        graph, series, config, seed=seed, device=device, uniform_baseline=True
    )
    rows.append(
        ComparisonRow(
            "uniform_dynamic",
            "baseline",
            evaluate_fit(graph, series, uniform),
            "single global alpha and beta; same physics and fitting, no spatial freedom",
        )
    )

    no_modality = fit_thermal_operator(
        graph, series, config, seed=seed, device=device, use_static_features=False
    )
    rows.append(
        ComparisonRow(
            "region_no_modality",
            "ablation",
            evaluate_fit(graph, series, no_modality),
            "emissivity and geometry removed from the node features",
        )
    )

    # The decay alone cannot separate conduction from convection; this ablation
    # is what shows it.  Without the steady-profile prior the fit still predicts
    # the decay well but collapses to a diffusion length far shorter than the
    # measured steady profile allows.
    no_steady = dataclasses.replace(
        config,
        loss=dataclasses.replace(
            config.loss, steady_source_negativity=0.0, steady_source_spread=0.0
        ),
    )
    no_steady_fit = fit_thermal_operator(graph, series, no_steady, seed=seed, device=device)
    no_steady_scores = evaluate_fit(graph, series, no_steady_fit)
    no_steady_scores["identifiability"] = _identifiability_summary(graph, no_steady_fit)
    rows.append(
        ComparisonRow(
            "no_steady_source_prior",
            "ablation",
            no_steady_scores,
            "steady-source negativity and spread penalties switched off",
        )
    )
    full_scores["identifiability"] = _identifiability_summary(graph, full)

    test_block = full.windows.test[0]
    rows.append(
        ComparisonRow(
            "steady_plate",
            "baseline",
            {"test": steady_plate_baseline(sequence, partition.labels, series, test_block)},
            "existing screened-harmonic fill, no dynamics",
        )
    )
    rows.append(
        ComparisonRow(
            "laplacian_source",
            "baseline",
            {"test": laplacian_source_baseline(graph, series, test_block)},
            "naive source from the measured field's Laplacian, no fitting",
        )
    )

    source_rows: list[ComparisonRow] = []
    if include_source:
        structured = fit_source(graph, series, full, config, seed=seed, device=device)
        source_rows.append(
            ComparisonRow(
                "structured_decoder",
                "full",
                {
                    "val_mae": structured.metrics["best_val_mae"],
                    "concentration": structured.metrics["relative_source_concentration"],
                    "top_regions": structured.metrics["top_regions_by_relative_source"],
                },
            )
        )
        no_sparsity = dataclasses.replace(
            config, loss=dataclasses.replace(config.loss, source_l1=0.0, source_tv_time=0.0)
        )
        without = fit_source(graph, series, full, no_sparsity, seed=seed, device=device)
        source_rows.append(
            ComparisonRow(
                "no_source_sparsity",
                "ablation",
                {
                    "val_mae": without.metrics["best_val_mae"],
                    "concentration": without.metrics["relative_source_concentration"],
                    "top_regions": without.metrics["top_regions_by_relative_source"],
                },
                "source_l1 and source_tv_time weights set to zero",
            )
        )
        free = fit_source(
            graph, series, full, config, seed=seed, device=device, free_source_baseline=True
        )
        source_rows.append(
            ComparisonRow(
                "free_per_frame_source",
                "baseline",
                {
                    "val_mae": free.metrics["best_val_mae"],
                    "concentration": free.metrics["relative_source_concentration"],
                },
                "one unconstrained non-negative value per region and frame",
            )
        )

    def as_rows(items: list[ComparisonRow]) -> list[dict[str, Any]]:
        return [
            {"name": r.name, "kind": r.kind, "note": r.note, "metrics": r.metrics} for r in items
        ]

    thermal_test = {
        r.name: r.metrics.get("test", {}).get("temperature_mae_k")
        for r in rows
        if "test" in r.metrics
    }
    best = min(
        (k for k, v in thermal_test.items() if v is not None),
        key=lambda k: thermal_test[k],
        default=None,
    )
    return {
        "thermal_operator": as_rows(rows),
        "source": as_rows(source_rows),
        "test_mae_by_model_k": thermal_test,
        "best_model_on_test": best,
        "full_model_wins": bool(best == "graph_neural_ode"),
        "verdict_note": (
            "test MAE on a held-out contiguous block, free rollout from the block's first "
            "frame. If the full model is not best, that is reported as measured."
        ),
    }
