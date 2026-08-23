"""Measure what this recording can and cannot separate, before believing a fit.

A fit always returns numbers.  Whether those numbers are determined by the data
or by the prior is a separate question, and on this experiment the answer is
different for different parameters.  Everything here is computed from the
measurements alone -- no model, no fit -- so it can be read before and
independently of any result.

The central question is whether conduction and convection can be told apart.
They enter each region's relaxation only through their sum::

    d(psi_i)/dt = sum_j a_ij (psi_j - psi_i) - beta_i psi_i
                = -(beta_i + sum_j a_ij) psi_i + sum_j a_ij psi_j

If the board relaxes *self-similarly* -- every region's excess falling in the
same proportion -- the second term is a fixed multiple of the first and only
the total rate is observable.  Conduction then leaves no signature at all, and
a fit is free to set ``a = 0`` and absorb everything into a heterogeneous
``beta``.  Measured on case10 that is very nearly what happens: the shape of
the profile relaxes over the first three minutes and is self-similar for the
remaining fifteen.

So this module reports three things:

* how fast the *normalised* profile stops changing (``profile_collapse``);
* what each region's total relaxation rate is, fitted directly (``region_rates``);
* whether the observed differential decay can be explained by heterogeneous
  cooling alone (``beta_only_sufficient``).

When the answer to the last is yes, ``alpha_eff``, ``L`` and the individual
``a_ij`` are *not identifiable from the decay* and must be reported as
prior-dependent, whatever a fit produces for them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _decaying_part(theta: np.ndarray, tail_frames: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Split the observed excess into its asymptote and the part that decays."""
    tail = max(3, min(int(tail_frames), len(theta) // 3))
    asymptote = theta[-tail:].mean(axis=0)
    return asymptote, theta - asymptote


def profile_collapse(
    theta: np.ndarray, times: np.ndarray, min_amplitude_k: float = 1.0
) -> dict[str, Any]:
    """How much the *shape* of the excess-temperature field changes as it decays.

    The normalised profile ``psi_i(t) / mean(psi(t))`` is constant exactly when
    the decay is self-similar.  Its spatial standard deviation is therefore a
    direct, model-free readout of how much conduction information the interval
    carries, and the time at which it stops falling marks the end of that
    information.
    """
    asymptote, psi = _decaying_part(theta)
    mean_psi = psi.mean(axis=1)
    usable = mean_psi > float(min_amplitude_k)
    if usable.sum() < 4:
        return {"available": False, "reason": "decay amplitude never exceeds the threshold"}

    index = np.nonzero(usable)[0]
    elapsed = times[index] - times[index[0]]
    spread = np.asarray([float((psi[k] / mean_psi[k]).std()) for k in index])

    initial = float(spread[0])
    floor = float(np.median(spread[len(spread) // 2 :]))

    # The simple picture -- a profile that starts uneven and flattens -- is not
    # the only thing a board does.  When one component is far hotter than the
    # rest it can cool *faster* than the mean, which makes the normalised
    # profile more uneven before it settles, and the spread rises before it
    # falls.  Reporting a settle time of zero for that case would read as "the
    # shape never changed", which is the opposite of what happened.  Say so
    # instead, and stay conservative about what the interval can identify.
    monotonic = floor < initial
    if not monotonic:
        return {
            "available": True,
            "monotonic_collapse": False,
            "initial_profile_spread": initial,
            "settled_profile_spread": floor,
            "spread_reduction_factor": 1.0,
            "peak_profile_spread": float(spread.max()),
            "shape_settle_time": float("nan"),
            "informative_frames": 1,
            "informative_fraction": 1.0 / len(theta),
            "usable_frames": int(usable.sum()),
            "profile_correlation_first_last": float(
                np.corrcoef(psi[index[0]] / mean_psi[index[0]], psi[index[-1]] / mean_psi[index[-1]])[0, 1]
            ),
            "asymptote_mean_k": float(asymptote.mean()),
            "note": (
                "the normalised profile does not collapse monotonically -- its spread rises "
                "before it settles, which happens when a hot component cools faster than the "
                "board it sits on.  The collapse-rate diagnostic does not apply, so no "
                "conduction-information estimate is made and the coupling is treated as "
                "not separately identifiable"
            ),
        }

    # Where the shape has finished most of its change.
    target = floor + 0.25 * (initial - floor)
    reached = np.nonzero(spread <= target)[0]
    settle_time = float(elapsed[reached[0]]) if reached.size else float(elapsed[-1])

    # The *rate* of that relaxation, not just when it finishes.  These differ by
    # more than a constant: the settle time above is ~1.4 e-folds, so using its
    # reciprocal as a rate overestimates by that factor -- enough, when it is
    # used to set an initialisation scale, to push a fit past the answer.
    # Fitted above the floor, which is where the profile is signal rather than
    # sensor noise.
    excess_spread = spread - floor
    fittable = excess_spread > 0.1 * max(initial - floor, 1e-9)
    if fittable.sum() >= 4:
        slope = np.polyfit(elapsed[fittable], np.log(excess_spread[fittable]), 1)[0]
        shape_rate = float(-slope) if slope < 0 else float(1.0 / max(settle_time, 1e-9))
    else:
        shape_rate = float(1.0 / max(settle_time, 1e-9))

    first, last = psi[index[0]] / mean_psi[index[0]], psi[index[-1]] / mean_psi[index[-1]]
    return {
        "available": True,
        "monotonic_collapse": True,
        "initial_profile_spread": initial,
        "settled_profile_spread": floor,
        "spread_reduction_factor": initial / max(floor, 1e-9),
        "shape_settle_time": settle_time,
        "shape_relaxation_rate": shape_rate,
        "usable_frames": int(usable.sum()),
        "informative_frames": int(np.searchsorted(elapsed, settle_time) + 1),
        "informative_fraction": float((np.searchsorted(elapsed, settle_time) + 1) / len(theta)),
        "profile_correlation_first_last": float(np.corrcoef(first, last)[0, 1]),
        "asymptote_mean_k": float(asymptote.mean()),
        "note": (
            "conduction is only visible while the normalised profile is still changing; "
            "after it settles the board relaxes in its fundamental mode and only the "
            "total per-region rate is observable"
        ),
    }


def region_rates(
    theta: np.ndarray, times: np.ndarray, min_fraction: float = 0.3
) -> dict[str, Any]:
    """Per-region total relaxation rate, by direct log-linear fit.

    This is the quantity the decay actually determines: ``beta_i + sum_j a_ij``,
    not either term on its own.  Fitted without any model so it can be compared
    against what the graph fit reports.
    """
    _, psi = _decaying_part(theta)
    rates = np.full(theta.shape[1], np.nan)
    for node in range(theta.shape[1]):
        curve = psi[:, node]
        peak = float(curve[0])
        if peak <= 0:
            continue
        keep = curve > float(min_fraction) * peak
        if keep.sum() < 8:
            continue
        rates[node] = -float(np.polyfit(times[keep] - times[keep][0], np.log(curve[keep]), 1)[0])
    finite = rates[np.isfinite(rates) & (rates > 0)]
    if finite.size < 4:
        return {"available": False, "reason": "too few regions with a fittable decay"}
    return {
        "available": True,
        "median_rate": float(np.median(finite)),
        "median_time_constant": float(1.0 / np.median(finite)),
        "p10_rate": float(np.percentile(finite, 10)),
        "p90_rate": float(np.percentile(finite, 90)),
        "p90_over_p10": float(np.percentile(finite, 90) / max(np.percentile(finite, 10), 1e-12)),
        "n_regions_fitted": int(finite.size),
        "rates": rates.tolist(),
        "note": (
            "this is the TOTAL relaxation rate beta_i + sum_j a_ij, which is what the "
            "decay determines; it is not beta_i"
        ),
    }


def beta_only_sufficient(collapse: dict[str, Any], rates: dict[str, Any]) -> dict[str, Any]:
    """Can heterogeneous cooling alone explain the observed shape change?

    If the profile spread falls by a factor ``F`` over a time ``T``, a model
    with no conduction at all needs its per-region rates to differ by about
    ``ln(F)/T`` to do the same job.  When the rates measured directly already
    differ by at least that much, conduction is not *needed* to explain
    anything the decay shows -- and a fit that nevertheless reports a coupling
    is reporting its prior.
    """
    if not collapse.get("available") or not rates.get("available"):
        return {"available": False}
    factor = float(collapse["spread_reduction_factor"])
    settle = max(float(collapse["shape_settle_time"]), 1e-9)
    required_spread = float(np.log(max(factor, 1.0 + 1e-9)) / settle)
    observed_spread = float(rates["p90_rate"] - rates["p10_rate"])
    sufficient = observed_spread >= required_spread
    return {
        "available": True,
        "required_rate_spread_for_beta_only": required_spread,
        "observed_rate_spread_p10_p90": observed_spread,
        "beta_heterogeneity_alone_suffices": bool(sufficient),
        "verdict": (
            "conduction and per-region cooling are NOT separable from this decay: the "
            "observed shape change is fully reproducible by heterogeneous cooling with "
            "zero conduction, so alpha_eff, L and individual a_ij are prior-dependent"
            if sufficient
            else "the observed shape change exceeds what heterogeneous cooling alone can "
            "produce, so the decay does carry conduction information"
        ),
    }


# Minimum number of frames that must resolve the shape mode before the spatial
# *contrast* of the coupling can be recovered.  Calibrated against the synthetic
# generator, where the answer is known: a case whose shape mode is resolved by
# 14 frames recovers strong edges with recall 0.71-0.86, while one resolved by
# only 7 recovers nothing (recall 0.29, i.e. chance, and the magnitude 5-15x
# low).  The threshold is on the frame count, not on the fraction of the record:
# a longer tail adds frames that carry no shape information either way.
MIN_INFORMATIVE_FRAMES = 12


def coupling_contrast_recoverable(collapse: dict[str, Any]) -> dict[str, Any]:
    """Whether the decay resolves the shape mode well enough to rank edges.

    Conduction shows up only while the normalised profile is still changing.
    Once it settles the board relaxes in its fundamental mode, and every region
    reports the same rate no matter what the coupling underneath is doing.  A
    handful of frames before that happens is enough to see *that* heat moved;
    it is not enough to see *where it moved fastest*.
    """
    if not collapse.get("available"):
        return {"available": False}
    if not collapse.get("monotonic_collapse", True):
        return {
            "available": True,
            "informative_frames": None,
            "minimum_required": MIN_INFORMATIVE_FRAMES,
            "coupling_contrast_recoverable": False,
            "verdict": (
                "the normalised profile does not collapse monotonically, so how much "
                "conduction information the decay carries cannot be read off it; the "
                "coupling map is reported but must be treated as prior-dependent"
            ),
        }
    frames = int(collapse["informative_frames"])
    ok = frames >= MIN_INFORMATIVE_FRAMES
    return {
        "available": True,
        "informative_frames": frames,
        "minimum_required": MIN_INFORMATIVE_FRAMES,
        "coupling_contrast_recoverable": bool(ok),
        "verdict": (
            f"{frames} frames resolve the profile's shape change, at or above the "
            f"{MIN_INFORMATIVE_FRAMES} calibrated as sufficient, so the *ranking* of "
            "coupling strengths is meaningful; its absolute magnitude remains the least "
            "well determined quantity here"
            if ok
            else f"only {frames} frames resolve the profile's shape change, below the "
            f"{MIN_INFORMATIVE_FRAMES} calibrated as sufficient on synthetic data: the "
            "spatial contrast of a_ij is NOT recoverable from this decay and any map of "
            "it reports the prior, not the board"
        ),
    }


def analyse(theta: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Run the whole model-free identifiability audit on one decay interval."""
    collapse = profile_collapse(theta, times)
    rates = region_rates(theta, times)
    verdict = beta_only_sufficient(collapse, rates)
    contrast = coupling_contrast_recoverable(collapse)

    identifiable = [
        "total per-region relaxation rate",
        "time constant tau",
        "idle equilibrium profile",
    ]
    not_identifiable = ["absolute conductance G", "absolute power q", "material conductivity k"]
    if verdict.get("beta_heterogeneity_alone_suffices"):
        not_identifiable = [
            "conduction/convection split (a_ij vs beta_i individually)",
            "effective diffusivity alpha_eff",
            "diffusion length L",
        ] + not_identifiable
    elif contrast.get("coupling_contrast_recoverable"):
        identifiable = identifiable + ["relative ranking of effective coupling a_ij"]
        not_identifiable = [
            "absolute magnitude of alpha_eff and L (weakly determined; report a range)"
        ] + not_identifiable
    else:
        not_identifiable = [
            "spatial contrast of a_ij",
            "effective diffusivity alpha_eff",
            "diffusion length L",
        ] + not_identifiable
    return {
        "profile_collapse": collapse,
        "region_rates": {k: v for k, v in rates.items() if k != "rates"},
        "separability": verdict,
        "coupling_contrast": contrast,
        "identifiable_from_this_decay": identifiable,
        "not_identifiable_from_this_decay": not_identifiable,
    }
