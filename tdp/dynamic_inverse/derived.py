"""Turn identified graph rates into the continuum-facing quantities.

Each function here carries the naming discipline the harness demands, because
this is where a rate stops being an obviously-graph-shaped number and starts
looking like a material property.  It is not one.  In particular
:func:`diffusivity_proxy` returns ``px^2`` per unit time, is named a *proxy*
everywhere it appears, and is the closest this experiment gets to a
diffusivity -- it is not a conductivity and not a conductance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tdp.dynamic_inverse.physics import RegionGraph


def diffusivity_proxy(graph: RegionGraph, edge_rate: np.ndarray) -> np.ndarray:
    """Node-level effective diffusivity proxy, ``px^2`` per unit time.

    A coupling rate ``a_ij`` [1/t] times the squared centre distance
    ``d_ij^2`` [px^2] has the units of a diffusivity, and for a uniform lattice
    with spacing ``d`` the graph Laplacian ``a * d^2`` reduces exactly to the
    continuum ``alpha``.  The node value is the boundary-length-weighted mean
    over that node's edges, so a region touching a neighbour along a long
    shared edge counts that neighbour more than one touching at a corner.
    """
    distance = graph.edge_distance_px()
    weight = graph.edge_length_px
    contribution = edge_rate * distance**2
    numerator = np.zeros(graph.n_nodes)
    denominator = np.zeros(graph.n_nodes)
    for node_index in (0, 1):
        np.add.at(numerator, graph.edge_index[node_index], contribution * weight)
        np.add.at(denominator, graph.edge_index[node_index], weight)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def time_constants(beta: np.ndarray, floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``tau = 1/beta`` with a floor, plus a low-confidence flag.

    A cooling rate identified as near zero means "the record does not constrain
    this region's losses", not "this region never cools".  Reporting an infinite
    time constant would dress that ignorance up as a measurement, so the rate is
    floored and the affected regions are flagged instead.
    """
    beta = np.asarray(beta, dtype=np.float64)
    positive = beta[beta > 0]
    if floor is None:
        floor = float(np.percentile(positive, 5)) if positive.size else 1e-12
        floor = max(floor, 1e-12)
    low_confidence = beta < floor
    return 1.0 / np.maximum(beta, floor), low_confidence


def characteristic_length(
    alpha_proxy: np.ndarray, beta: np.ndarray, floor: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``L = sqrt(alpha_proxy / beta)`` in px, with the same low-beta guard.

    This is the distance over which the plate spreads heat before losing it --
    the fin length.  It inherits every caveat of its inputs: it is built from a
    diffusivity *proxy*, so it is a length scale of the identified operator, not
    a material constant.
    """
    _, low_confidence = time_constants(beta, floor)
    beta = np.asarray(beta, dtype=np.float64)
    positive = beta[beta > 0]
    if floor is None:
        floor = float(np.percentile(positive, 5)) if positive.size else 1e-12
        floor = max(floor, 1e-12)
    return np.sqrt(np.maximum(alpha_proxy, 0.0) / np.maximum(beta, floor)), low_confidence


def summarise(
    graph: RegionGraph,
    edge_rate: np.ndarray,
    beta: np.ndarray,
    time_unit: str,
    pixel_size_mm: float | None,
) -> dict[str, Any]:
    """All derived quantities plus a units block, ready for metrics.json."""
    alpha = diffusivity_proxy(graph, edge_rate)
    tau, tau_low = time_constants(beta)
    length, length_low = characteristic_length(alpha, beta)

    out: dict[str, Any] = {
        "edge_rate_median": float(np.median(edge_rate)),
        "edge_rate_max": float(edge_rate.max()),
        "edge_rate_min": float(edge_rate.min()),
        "beta_median": float(np.median(beta)),
        "beta_max": float(beta.max()),
        "beta_min": float(beta.min()),
        "tau_median": float(np.median(tau)),
        "alpha_proxy_median": float(np.median(alpha)),
        "alpha_proxy_max": float(alpha.max()),
        "length_median_px": float(np.median(length)),
        "length_max_px": float(length.max()),
        "n_low_confidence_tau": int(tau_low.sum()),
        "n_low_confidence_length": int(length_low.sum()),
        "units": {
            "edge_rate": f"1/{time_unit}",
            "beta": f"1/{time_unit}",
            "tau": time_unit,
            "alpha_proxy": f"px^2/{time_unit}",
            "length": "px",
        },
        "naming_note": (
            "alpha_proxy is an effective diffusivity PROXY built from graph coupling rates "
            "and centre distances.  It is not a material thermal conductivity and not an "
            "areal conductance; neither is identifiable from temperatures alone."
        ),
    }
    if pixel_size_mm:
        out["alpha_proxy_median_mm2"] = float(np.median(alpha) * pixel_size_mm**2)
        out["length_median_mm"] = float(np.median(length) * pixel_size_mm)
        out["length_max_mm"] = float(length.max() * pixel_size_mm)
        out["units"]["alpha_proxy_mm"] = f"mm^2/{time_unit}"
        out["units"]["length_mm"] = "mm"
        out["pixel_size_mm"] = float(pixel_size_mm)
        out["pixel_size_note"] = (
            "derived from the PCB outline resampled onto the canonical board grid; "
            "a geometric estimate, not an optical calibration"
        )
    return out


def arrays(
    graph: RegionGraph, edge_rate: np.ndarray, beta: np.ndarray
) -> dict[str, np.ndarray]:
    """The per-region arrays themselves, for maps and the .npz dump."""
    alpha = diffusivity_proxy(graph, edge_rate)
    tau, tau_low = time_constants(beta)
    length, length_low = characteristic_length(alpha, beta)
    return {
        "edge_rate": edge_rate,
        "beta": beta,
        "tau": tau,
        "tau_low_confidence": tau_low,
        "alpha_proxy": alpha,
        "length_px": length,
        "length_low_confidence": length_low,
    }
