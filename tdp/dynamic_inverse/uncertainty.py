"""Low-cost uncertainty: how much does a result depend on choices we made?

Three perturbations, each answering a different question:

* **seeds** -- initialisation and window sampling.  Answers "is this the fit or
  the optimiser?"
* **time-window bootstrap** -- which contiguous stretch of the decay was used.
  Answers "is this the board or that particular minute?"  Blocks stay
  contiguous; resampling individual frames would resample near-duplicates.
* **region resolution and partition jitter** -- how the board was cut up.
  Answers "is this a component or an artefact of the tessellation?"  Results
  from different partitions are compared on the pixel grid, since region ``7``
  in one partition is not region ``7`` in another.

A region whose coefficient of variation crosses ``high_cv_threshold`` is listed
as not stably identifiable.  Reporting that is the point: the harness forbids
presenting an unstable number as a measurement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from tdp.dynamic_inverse.config import AnalysisConfig
from tdp.dynamic_inverse.physics import RegionGraph
from tdp.dynamic_inverse.regions import expand_to_pixels
from tdp.dynamic_inverse.train_thermal import NodeSeries, ThermalFit, fit_thermal_operator


@dataclass
class UncertaintyResult:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    cv: dict[str, np.ndarray]
    low_confidence: dict[str, np.ndarray]
    members: list[dict[str, Any]]
    report: dict[str, Any]


def _summarise(stack: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Pixels outside the board are NaN in every member, and reducing an all-NaN
    # column is a warning, not an error: suppress it and let the NaN through.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.inf)
    return mean, std, cv, cv > threshold


def seed_ensemble(
    graph: RegionGraph,
    series: NodeSeries,
    config: AnalysisConfig,
    seeds: tuple[int, ...] | None = None,
    device: str = "cpu",
    fit_fn: Callable[..., ThermalFit] = fit_thermal_operator,
) -> UncertaintyResult:
    """Refit the thermal operator from several initialisations."""
    seeds = tuple(seeds if seeds is not None else config.uncertainty.seeds)
    betas, rates, members = [], [], []
    for seed in seeds:
        fit = fit_fn(graph, series, config, seed=seed, device=device)
        betas.append(fit.beta)
        rates.append(fit.edge_rate)
        members.append({
            "seed": int(seed),
            "val_mae": fit.metrics["best_val_mae"],
            "spectral_bound": fit.metrics["spectral_bound_per_unit_time"],
            "residual_idle_source_rate_mean": fit.metrics["residual_idle_source_rate_mean"],
        })
    threshold = config.uncertainty.high_cv_threshold
    beta_mean, beta_std, beta_cv, beta_low = _summarise(np.stack(betas), threshold)
    rate_mean, rate_std, rate_cv, rate_low = _summarise(np.stack(rates), threshold)
    return UncertaintyResult(
        mean={"beta": beta_mean, "edge_rate": rate_mean},
        std={"beta": beta_std, "edge_rate": rate_std},
        cv={"beta": beta_cv, "edge_rate": rate_cv},
        low_confidence={"beta": beta_low, "edge_rate": rate_low},
        members=members,
        report={
            "kind": "seed_ensemble",
            "seeds": [int(s) for s in seeds],
            "high_cv_threshold": float(threshold),
            "n_low_confidence_regions": int(beta_low.sum()),
            "n_low_confidence_edges": int(rate_low.sum()),
        },
    )


def window_bootstrap(
    graph: RegionGraph,
    series: NodeSeries,
    config: AnalysisConfig,
    n_windows: int | None = None,
    device: str = "cpu",
    cooling_phase: str = "free_cooling_candidate",
) -> UncertaintyResult:
    """Refit on several contiguous sub-windows of the load-off interval.

    Frames are never resampled individually.  On a 6 s cadence neighbouring
    frames are almost the same measurement, so a frame-level bootstrap would
    mostly resample duplicates and report a confidence interval far too tight.
    """
    n_windows = int(n_windows if n_windows is not None else config.uncertainty.bootstrap_windows)
    index = np.nonzero(series.phase == cooling_phase)[0]
    if len(index) < 20:
        raise ValueError("load-off interval too short to bootstrap over sub-windows")

    width = max(12, int(0.7 * len(index)))
    starts = np.linspace(0, len(index) - width, n_windows).astype(int)
    betas, rates, members = [], [], []
    for k, start in enumerate(starts):
        keep = index[start : start + width]
        phase = np.full(len(series.times), "invalid_transition", dtype=object)
        phase[keep] = cooling_phase
        powered = series.phase == "heating"
        sub = NodeSeries(
            theta=series.theta,
            times=series.times,
            weight=series.weight,
            trust=series.trust,
            phase=phase.astype(str),
            powered=powered,
            time_unit=series.time_unit,
        )
        fit = fit_thermal_operator(graph, sub, config, seed=config.seed, device=device)
        betas.append(fit.beta)
        rates.append(fit.edge_rate)
        members.append({
            "window": k,
            "frames": [int(keep[0]), int(keep[-1])],
            "val_mae": fit.metrics["best_val_mae"],
        })
    threshold = config.uncertainty.high_cv_threshold
    beta_mean, beta_std, beta_cv, beta_low = _summarise(np.stack(betas), threshold)
    rate_mean, rate_std, rate_cv, rate_low = _summarise(np.stack(rates), threshold)
    return UncertaintyResult(
        mean={"beta": beta_mean, "edge_rate": rate_mean},
        std={"beta": beta_std, "edge_rate": rate_std},
        cv={"beta": beta_cv, "edge_rate": rate_cv},
        low_confidence={"beta": beta_low, "edge_rate": rate_low},
        members=members,
        report={
            "kind": "contiguous_window_bootstrap",
            "n_windows": n_windows,
            "window_frames": width,
            "high_cv_threshold": float(threshold),
            "n_low_confidence_regions": int(beta_low.sum()),
        },
    )


def partition_ensemble(
    sequence,
    config: AnalysisConfig,
    build_fn: Callable,
    series_fn: Callable,
    resolutions: tuple[int, ...] | None = None,
    jitter: float = 0.5,
    device: str = "cpu",
) -> UncertaintyResult:
    """Refit across region counts and jittered partitions, compared on pixels.

    Region ids are not comparable between partitions, so each member's per-region
    parameters are painted back onto the board grid and the spread is taken
    there.  That also makes the answer directly readable as a map.
    """
    resolutions = tuple(
        resolutions if resolutions is not None else config.uncertainty.region_resolutions
    )
    from tdp.dynamic_inverse.derived import characteristic_length, diffusivity_proxy

    beta_maps, alpha_maps, members = [], [], []
    for k, n_regions in enumerate(resolutions):
        partition = build_fn(sequence, n_regions=n_regions, seed=k, jitter=jitter if k else 0.0)
        series = series_fn(sequence, partition)
        fit = fit_thermal_operator(partition.graph, series, config, seed=config.seed, device=device)
        alpha = diffusivity_proxy(partition.graph, fit.edge_rate)
        length, _ = characteristic_length(alpha, fit.beta)
        beta_maps.append(expand_to_pixels(fit.beta, partition.labels))
        alpha_maps.append(expand_to_pixels(alpha, partition.labels))
        members.append({
            "n_regions_requested": int(n_regions),
            "n_regions_found": int(partition.n_regions),
            "jitter": float(jitter if k else 0.0),
            "val_mae": fit.metrics["best_val_mae"],
            "beta_median": float(np.median(fit.beta)),
            # alpha_proxy is built from a graph Laplacian on an irregular
            # tessellation, so unlike beta it has no reason to be invariant to
            # the region count.  Tracking it here is the only honest way to say
            # how much of its value is the board and how much is the mesh.
            "alpha_proxy_median": float(np.median(alpha)),
            "length_median_px": float(np.median(length)),
        })
    threshold = config.uncertainty.high_cv_threshold
    beta_mean, beta_std, beta_cv, beta_low = _summarise(np.stack(beta_maps), threshold)
    alpha_mean, alpha_std, alpha_cv, alpha_low = _summarise(np.stack(alpha_maps), threshold)
    beta_medians = [m["beta_median"] for m in members]
    alpha_medians = [m["alpha_proxy_median"] for m in members]
    return UncertaintyResult(
        mean={"beta_map": beta_mean, "alpha_proxy_map": alpha_mean},
        std={"beta_map": beta_std, "alpha_proxy_map": alpha_std},
        cv={"beta_map": beta_cv, "alpha_proxy_map": alpha_cv},
        low_confidence={"beta_map": beta_low, "alpha_proxy_map": alpha_low},
        members=members,
        report={
            "kind": "partition_ensemble",
            "resolutions": [int(r) for r in resolutions],
            "jitter": float(jitter),
            "high_cv_threshold": float(threshold),
            "low_confidence_pixel_fraction": float(np.nanmean(beta_low.astype(float))),
            "beta_median_across_resolutions": beta_medians,
            "beta_resolution_spread": float(max(beta_medians) / max(min(beta_medians), 1e-12)),
            "alpha_proxy_median_across_resolutions": alpha_medians,
            "alpha_resolution_spread": float(max(alpha_medians) / max(min(alpha_medians), 1e-12)),
            "resolution_note": (
                "beta is a per-region property and should be nearly resolution-invariant; "
                "alpha_proxy is a graph-Laplacian construction and is expected to move with "
                "the tessellation, which is why it is reported as a proxy"
            ),
        },
    )
