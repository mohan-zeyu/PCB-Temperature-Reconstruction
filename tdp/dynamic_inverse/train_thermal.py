"""Stage A: identify the thermal operator on the load-off interval.

Why this stage exists separately: during the powered phase the source and the
operator trade off against each other -- a hotter region can be explained by
more power or by less cooling -- so fitting both at once has a flat direction.
During load-off the source is *constant*, which removes the trade-off, and the
decay identifies conduction and cooling on their own.

The board here is never actually powered down (see the case10 notes in
``configs/measurements_v3.yaml``: the CPU load stops but the SoC never returns
to ambient).  Asserting ``q = 0`` would therefore bias every cooling rate.  The
fit instead runs on the deviation from the *idle steady state*::

    theta   = T - T_ambient                        excess over ambient
    psi     = theta - theta_idle                   deviation from idle equilibrium
    0       = div(alpha grad theta_idle) - beta theta_idle + s_idle
    d psi/dt = div(alpha grad psi) - beta psi      s_idle cancels exactly

The constant idle source drops out of the difference for a linear operator, so
the same ``a_ij`` and ``beta_i`` are identified without ever claiming the board
was unplugged.  The interval keeps the label ``free_cooling_candidate``
everywhere it is reported.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import Tensor

from tdp.dynamic_inverse.config import AnalysisConfig
from tdp.dynamic_inverse.identifiability import analyse as identifiability_analyse
from tdp.dynamic_inverse.losses import total_loss
from tdp.dynamic_inverse.model import ThermalOperator, UniformOperator
from tdp.dynamic_inverse.physics import (
    RegionGraph,
    integrate,
    spectral_bound,
    steady_state,
)


@dataclass
class NodeSeries:
    """Region-aggregated observations ready for fitting."""

    theta: np.ndarray  # (T, N) K, excess over ambient
    times: np.ndarray  # (T,) s (or frames)
    weight: np.ndarray  # (T, N) aggregation confidence
    trust: np.ndarray  # (N,) directly-measured share of each region
    phase: np.ndarray  # (T,) phase label per frame
    powered: np.ndarray  # (T,) bool
    time_unit: str = "s"

    @property
    def n_nodes(self) -> int:
        return int(self.theta.shape[1])


@dataclass
class Windows:
    """Contiguous time blocks.  Never shuffled frames.

    Adjacent frames on a 6 s cadence are almost the same measurement, so a
    random split would put near-duplicates on both sides and report a test
    error that measures interpolation, not generalisation.
    """

    train: list[tuple[int, int]] = field(default_factory=list)
    val: list[tuple[int, int]] = field(default_factory=list)
    test: list[tuple[int, int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "train": [list(w) for w in self.train],
            "val": [list(w) for w in self.val],
            "test": [list(w) for w in self.test],
        }


def contiguous_runs(index: np.ndarray) -> list[np.ndarray]:
    """Split a possibly-gapped index array into maximal contiguous runs.

    A phase label need not select one interval.  case10 is labelled
    ``near_steady`` both while the load is running and again while the board
    idles after the decay, and those are separated by the whole load-off
    interval.  Treating the union as one block would let a rollout window start
    before the gap and integrate straight through frames that are not in the
    phase at all.
    """
    index = np.asarray(index)
    if index.size == 0:
        return []
    breaks = np.nonzero(np.diff(index) > 1)[0]
    return [run for run in np.split(index, breaks + 1) if run.size]


# Which of five contiguous chunks goes where.  Train sees the fast head, the
# middle and part of the tail; validation and test each sit between train
# chunks, so neither is a pure extrapolation.
_INTERLEAVED_PATTERN = ("train", "train", "val", "train", "test")


def contiguous_split(
    index: np.ndarray,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    mode: str = "interleaved",
    n_blocks: int = 5,
    guard: int = 3,
) -> Windows:
    """Split one contiguous index range into train / val / test blocks.

    Frames are never shuffled in either mode: on a ~6 s cadence neighbouring
    frames are almost the same measurement, and a random split would report a
    test error that measures interpolation between near-duplicates.

    ``mode="sequential"`` takes the first 60% / next 20% / last 20% in time.
    That is the strictest test -- pure forward extrapolation -- but on a single
    decay it also hands the fit only the *fast* head of the curve and asks it to
    predict the slow tail, which is a different regime.  Measured on case10 the
    result is a fit whose validation error starts rising after a handful of
    epochs while the training error keeps falling.

    ``mode="interleaved"`` (the default) cuts the interval into ``n_blocks``
    contiguous chunks and assigns whole chunks, so training covers the whole
    decay regime while validation and test remain blocks the fit never saw.
    ``guard`` frames are dropped either side of every boundary so no training
    frame is adjacent to an evaluation frame -- without that the split leaks
    through correlation across the seam.  The harness's stricter forward
    rollout is still reported, as ``long_rollout_drift_k`` from a sequential
    split, so both readings are available.
    """
    n = len(index)
    if n < 6:
        raise ValueError(f"need at least 6 frames to split, got {n}")
    if mode == "sequential":
        n_train = max(2, int(round(train_fraction * n)))
        n_val = max(1, int(round(val_fraction * n)))
        n_train = min(n_train, n - 3)
        n_val = min(n_val, n - n_train - 2)
        return Windows(
            train=[(int(index[0]), int(index[n_train - 1]))],
            val=[(int(index[n_train]), int(index[n_train + n_val - 1]))],
            test=[(int(index[n_train + n_val]), int(index[-1]))],
        )
    if mode != "interleaved":
        raise ValueError(f"mode must be 'interleaved' or 'sequential', got {mode!r}")

    n_blocks = int(n_blocks)
    if n < 4 * n_blocks:
        # Too short to interleave without shredding it; fall back and say so
        # through the shape of the result rather than silently interleaving.
        return contiguous_split(index, train_fraction, val_fraction, mode="sequential")
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    windows = Windows()
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        role = _INTERLEAVED_PATTERN[k % len(_INTERLEAVED_PATTERN)]
        start = lo + (guard if k > 0 else 0)
        stop = hi - 1 - (guard if k < n_blocks - 1 else 0)
        if stop - start < 3:
            continue
        getattr(windows, role).append((int(index[start]), int(index[stop])))
    if not windows.train or not windows.val or not windows.test:
        return contiguous_split(index, train_fraction, val_fraction, mode="sequential")
    return windows


def _flat_window(series: NodeSeries, index: np.ndarray, window_seconds: float) -> tuple[np.ndarray, float]:
    """Indices of the flattest sub-window of ``index``, and its mean slope."""
    times = series.times[index]
    span = float(times[-1] - times[0])
    width = min(float(window_seconds), max(span * 0.35, 3.0))
    mean_theta = np.nanmean(series.theta[index], axis=1)
    best = (0, min(len(index) - 1, 3), np.inf)
    for begin in range(len(index)):
        stop = begin
        while stop + 1 < len(index) and times[stop + 1] - times[begin] < width:
            stop += 1
        if stop - begin < 2:
            continue
        block = slice(begin, stop + 1)
        slope = float(np.polyfit(times[block] - times[begin], mean_theta[block], 1)[0])
        if abs(slope) < best[2]:
            best = (begin, stop, abs(slope))
    begin, stop, slope = best
    return index[begin : stop + 1], slope


def idle_steady_state(
    series: NodeSeries,
    window_seconds: float = 300.0,
    phase_label: str = "free_cooling_candidate",
    method: str = "flat_window",
) -> tuple[np.ndarray, dict]:
    """Per-region idle equilibrium: the level the decay is heading for.

    This has to be a genuine *equilibrium* of the operator, not merely a late
    reading.  Subtracting a level the board has not reached yet is not a change
    of variable, it is an extra constant forcing: if ``theta_idle`` is not a
    steady state then ``psi = theta - theta_idle`` obeys
    ``d(psi)/dt = A psi + A theta_idle``, and the leftover ``A theta_idle``
    term corrupts every rate fitted from it.  Averaging the flattest tail --
    the obvious thing to do -- is exactly that mistake whenever the tail is
    still drifting, and it destroys parameter recovery on synthetic data where
    the true asymptote is known.

    So each region's tail is extrapolated instead: ``theta(t) = c + A exp(-t/tau)``
    is fitted and ``c`` is taken as the equilibrium.  Regions whose fit is
    degenerate (no curvature left, or a time constant longer than the record)
    fall back to the flat-window mean, and the count is reported.
    """
    mask = series.phase == phase_label
    if not mask.any():
        raise ValueError(f"no frames labelled {phase_label!r}; cannot reference an idle state")
    index = np.nonzero(mask)[0]
    flat_index, flat_slope = _flat_window(series, index, window_seconds)
    weights = series.weight[flat_index]
    total = weights.sum(axis=0)
    flat_level = np.where(
        total > 0,
        (series.theta[flat_index] * weights).sum(axis=0) / np.maximum(total, 1e-9),
        np.nanmean(series.theta[flat_index], axis=0),
    )
    if method not in ("asymptote", "flat_window"):
        raise ValueError(f"method must be 'asymptote' or 'flat_window', got {method!r}")
    if method == "flat_window":
        return flat_level, {
            "method": "flat_window",
            "frames": [int(flat_index[0]), int(flat_index[-1])],
            "residual_mean_slope_per_unit_time": float(flat_slope),
            "mean_theta_idle": float(np.nanmean(flat_level)),
        }

    times = series.times[index] - series.times[index[0]]
    span = float(times[-1]) if times[-1] > 0 else 1.0
    theta = series.theta[index]
    level = flat_level.copy()
    tau_used = np.full(series.n_nodes, np.nan)
    fallbacks = 0
    for node in range(series.n_nodes):
        y = theta[:, node]
        good = np.isfinite(y)
        if good.sum() < 6:
            fallbacks += 1
            continue
        t_fit, y_fit = times[good], y[good]
        amplitude0 = float(y_fit[0] - y_fit[-1])
        if abs(amplitude0) < 1e-6:
            fallbacks += 1
            continue
        try:
            params, _ = curve_fit(
                lambda t, c, a, rate: c + a * np.exp(-rate * t),
                t_fit,
                y_fit,
                p0=[float(y_fit[-1]), amplitude0, 3.0 / span],
                bounds=(
                    [-np.inf, -np.inf, 1.0 / (20.0 * span)],
                    [np.inf, np.inf, 50.0 / span],
                ),
                maxfev=8000,
            )
        except (RuntimeError, ValueError):
            fallbacks += 1
            continue
        constant, _, rate = params
        # A time constant longer than the record cannot be told from a linear
        # drift, so its extrapolated asymptote is unconstrained -- keep the
        # measured level instead of a confident-looking extrapolation.
        if rate * span < 0.5 or not np.isfinite(constant):
            fallbacks += 1
            continue
        level[node] = float(constant)
        tau_used[node] = float(1.0 / rate)

    report = {
        "method": "asymptote",
        "frames": [int(index[0]), int(index[-1])],
        "flat_window_frames": [int(flat_index[0]), int(flat_index[-1])],
        "flat_window_mean_theta": float(np.nanmean(flat_level)),
        "residual_mean_slope_per_unit_time": float(flat_slope),
        "mean_theta_idle": float(np.nanmean(level)),
        "asymptote_minus_flat_window_k": float(np.nanmean(level - flat_level)),
        "n_fallback_regions": int(fallbacks),
        "median_tau_per_unit_time": float(np.nanmedian(tau_used)) if np.isfinite(tau_used).any() else float("nan"),
        "definition": (
            "per-region excess temperature at the idle equilibrium, extrapolated from the "
            "decay tail; the fit runs on psi = theta - theta_idle so a constant residual "
            "idle source cancels instead of being asserted to be zero"
        ),
    }
    return level, report


def _powered_steady_profile(
    series: NodeSeries, cooling_index: np.ndarray, phase_label: str = "near_steady"
) -> np.ndarray | None:
    """Confidence-weighted mean excess profile over the powered plateau.

    Only the plateau *before* the decay counts: a board labelled near-steady
    again while idling afterwards is a different operating point with a
    different source, and averaging the two would blur both.
    """
    mask = (series.phase == phase_label) & (np.arange(len(series.phase)) < int(cooling_index[0]))
    if mask.sum() < 3:
        return None
    weights = series.weight[mask]
    total = weights.sum(axis=0)
    profile = np.where(
        total > 0,
        (series.theta[mask] * weights).sum(axis=0) / np.maximum(total, 1e-9),
        np.nanmean(series.theta[mask], axis=0),
    )
    return np.nan_to_num(profile)


def graph_fiedler_value(graph: RegionGraph) -> float:
    """Smallest non-zero eigenvalue of the *unweighted* region-graph Laplacian.

    For unit coupling this is the rate of the slowest mode that redistributes
    heat *within* the board, as opposed to the uniform mode that only changes
    its level.  It depends on how the board was tessellated, which is exactly
    what makes it the right divisor when converting an observed shape-relaxation
    rate into a per-edge coupling.
    """
    if graph.n_edges == 0 or graph.n_nodes < 2:
        return 1.0
    laplacian = np.zeros((graph.n_nodes, graph.n_nodes))
    i, j = graph.edge_index
    np.add.at(laplacian, (i, j), -1.0)
    np.add.at(laplacian, (j, i), -1.0)
    np.add.at(laplacian, (i, i), 1.0)
    np.add.at(laplacian, (j, j), 1.0)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    positive = eigenvalues[eigenvalues > 1e-9]
    return float(positive[0]) if positive.size else 1.0


def coupling_rate_scale(
    graph: RegionGraph,
    series: NodeSeries,
    cooling_index: np.ndarray,
    rate_scale: float,
) -> float:
    """Order-of-magnitude coupling rate, from how fast the profile *shape* relaxes.

    A board relaxing after load-off does two separable things.  Its level falls
    at the fundamental rate, essentially ``beta``, which
    :func:`decay_rate_scale` measures.  Its *shape* flattens faster, at a rate
    :func:`~tdp.dynamic_inverse.identifiability.profile_collapse` measures,
    because conduction acts on differences and not on the mean.  The gap
    between the two rates is what conduction contributes, and converting that
    gap into a per-edge rate needs the eigenvalue of the mode it belongs to::

        a_upper ~ (lambda_shape - beta) / lambda_2(L_graph)
        a       ~ sqrt(a_upper * beta)

    where ``lambda_2`` is the graph's Fiedler value.  Using the coordination
    number instead -- the obvious guess -- is wrong by the ratio between them,
    which on a 64-region board is a factor of forty: a shape mode spanning the
    whole board is far slower than a single edge's exchange rate, so dividing by
    the local degree badly *underestimates* the coupling.  Measured against the
    radial profile that error showed up plainly, as a fitted diffusion length of
    3 mm for a board whose steady excess temperature is flat to within 10% over
    40 mm.

    Getting this scale right matters because ``softplus`` saturates in both
    directions: a head started far too low never climbs, and one started far too
    high is pushed into the flat negative tail and collapses (an earlier scale
    taken from ``theta / delta_theta`` overshot and produced ``a = 1e-10``).

    Like :func:`decay_rate_scale` this is a *scale*, not a result.
    """
    if graph.n_edges == 0:
        return float(rate_scale)
    from tdp.dynamic_inverse.identifiability import profile_collapse

    collapse = profile_collapse(series.theta[cooling_index], series.times[cooling_index])
    if not collapse.get("available"):
        return float(rate_scale)
    shape_rate = float(collapse.get("shape_relaxation_rate", 0.0))
    if shape_rate <= 0:
        return float(rate_scale)
    excess = max(shape_rate - float(rate_scale), 0.1 * float(rate_scale))
    from_shape = excess / max(graph_fiedler_value(graph), 1e-9)
    # Hedge between the two estimates that bracket the answer.  The
    # shape-mode inversion above is an upper bracket: the measured relaxation
    # of the profile also contains sensor noise and the source turning off, so
    # attributing all of it to conduction overstates the coupling.  Simply
    # assuming coupling is of order the cooling rate is a lower bracket -- it
    # is what the board would look like if heat barely moved.  Their geometric
    # mean is the natural compromise between two multiplicative bounds, and it
    # is what keeps both the synthetic recovery (where the answer is known) and
    # the real fit (where the near-isothermal steady profile is the check) in
    # a sensible range.
    scale = float(np.sqrt(max(from_shape, 1e-12) * float(rate_scale)))
    # Never let the scale collapse or run away: the fit has to start somewhere
    # it can move in both directions, and it must not start past the
    # integrator's stability limit.
    scale = float(np.clip(scale, 0.02 * rate_scale, 2000.0 * rate_scale))
    return _stable_rate(scale, series.times[cooling_index])


def decay_rate_scale(series: NodeSeries, index: np.ndarray) -> float:
    """Order-of-magnitude rate of the board's own decay, ``1/tau``.

    Read straight off the board-mean curve: the time to fall to ``1/e`` of the
    interval's total drop.  This is not a result, it is a *scale* -- it tells
    the model's rate heads what decade to start in.  Getting that wrong is not
    cosmetic: softplus is nearly flat far below its knee, so a head initialised
    a decade too fast spends the whole fit crawling back down and early
    stopping fires on its initialisation.
    """
    times = series.times[index] - series.times[index[0]]
    curve = np.nanmean(series.theta[index], axis=1)
    drop = float(curve[0] - curve[-1])
    if drop <= 0 or times[-1] <= 0:
        return 1.0 / max(float(times[-1]), 1.0)
    target = curve[0] - (1.0 - np.exp(-1.0)) * drop
    crossed = np.nonzero(curve <= target)[0]
    tau = float(times[crossed[0]]) if crossed.size else float(times[-1])
    rate = 1.0 / max(tau, float(np.median(np.diff(times))), 1e-9)
    return _stable_rate(rate, times)


def _stable_rate(rate: float, times: np.ndarray) -> float:
    """Clamp a rate so an explicit step at this cadence cannot blow up.

    RK4 is stable to about ``dt * lambda < 2.78``.  A short interval -- a
    bootstrap sub-window, say -- can produce a 1/e time of only a couple of
    frames, and initialising the rate heads there sends the very first rollout
    to NaN.  The run then has no finite validation error at any epoch, which the
    guard at the end of the fit correctly refuses to return, aborting an
    otherwise healthy analysis.  Clamping well inside the limit keeps the
    initialisation usable; the fit is free to move above it if the data say so,
    and `max_stable_dt` reports where the limit actually is.
    """
    if len(times) < 2:
        return float(rate)
    step = float(np.median(np.diff(times)))
    if step <= 0:
        return float(rate)
    return float(min(rate, 0.5 / step))


def _node_features(graph: RegionGraph, series: NodeSeries, use_static: bool = True) -> np.ndarray:
    """Static + trajectory features per region, standardised.

    ``use_static=False`` is the ablation that drops emissivity and geometry and
    leaves only the temperature trajectory, which answers whether the extra
    modalities carry information or just capacity.
    """
    theta = series.theta
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(theta, axis=0)
        final = theta[-1]
        mean = np.nanmean(theta, axis=0)
        rise = np.nanmax(np.gradient(np.nan_to_num(theta), axis=0), axis=0)
        fall = np.nanmin(np.gradient(np.nan_to_num(theta), axis=0), axis=0)
    trajectory = np.stack([peak, final, mean, rise, fall], axis=1)

    if use_static and graph.node_features is not None:
        features = np.concatenate([graph.node_features, trajectory], axis=1)
    else:
        features = trajectory
    features = np.nan_to_num(features.astype(np.float64))
    centre = features.mean(0, keepdims=True)
    scale = features.std(0, keepdims=True)
    return ((features - centre) / np.maximum(scale, 1e-6)).astype(np.float32)


def _edge_features(graph: RegionGraph, node_features: np.ndarray) -> np.ndarray:
    """Geometry per edge, plus a symmetric summary of the endpoints' features."""
    i, j = graph.edge_index
    distance = graph.edge_distance_px()
    shared = graph.edge_length_px
    area_pair = np.minimum(graph.areas_px[i], graph.areas_px[j])
    difference = np.abs(node_features[i] - node_features[j]).mean(axis=1)
    raw = np.stack(
        [distance, np.log1p(shared), np.log1p(area_pair), difference, 1.0 / np.maximum(distance, 1e-3)],
        axis=1,
    )
    centre = raw.mean(0, keepdims=True)
    scale = raw.std(0, keepdims=True)
    return ((raw - centre) / np.maximum(scale, 1e-6)).astype(np.float32)


def _feature_similarity(graph: RegionGraph, node_features: np.ndarray) -> np.ndarray:
    """Per-edge similarity in [0, 1]; gates the spatial-smoothness penalty."""
    i, j = graph.edge_index
    distance = np.linalg.norm(node_features[i] - node_features[j], axis=1)
    scale = np.median(distance) + 1e-6
    return np.exp(-distance / scale).astype(np.float32)


def _sample_windows(
    blocks: list[tuple[int, int]],
    length: int,
    stride: int,
    rng: np.random.Generator,
    count: int,
    min_length: int = 3,
) -> list[tuple[int, int]]:
    """Rollout windows ``(start, stop)`` inside the allowed blocks.

    Windows are truncated at a block's end rather than dropped.  Dropping them
    is what makes a block shorter than one full window yield nothing at all,
    and a validation set that yields nothing silently freezes early stopping at
    the initial parameters -- a fit that looks like it ran and did not.
    """
    spans: list[tuple[int, int]] = []
    for lo, hi in blocks:
        if hi - lo < min_length:
            continue
        for start in range(lo, hi - min_length + 1, max(1, stride)):
            spans.append((start, min(start + length, hi)))
    if not spans:
        return []
    if len(spans) <= count:
        return spans
    picked = rng.choice(len(spans), size=count, replace=False)
    return [spans[i] for i in picked]


@dataclass
class ThermalFit:
    """Everything stage A produced, ready to be frozen for stage B."""

    edge_rate: np.ndarray  # (E,) 1/time-unit
    beta: np.ndarray  # (N,) 1/time-unit
    theta_idle: np.ndarray  # (N,) K, equilibrium implied by the fitted residual source
    idle_source_rate: np.ndarray  # (N,) K/time-unit, residual source over the load-off interval
    state_dict: dict
    history: list[dict]
    windows: Windows
    metrics: dict[str, Any]
    node_features: np.ndarray
    edge_features: np.ndarray
    similarity: np.ndarray
    model_kind: str
    time_unit: str


def fit_thermal_operator(
    graph: RegionGraph,
    series: NodeSeries,
    config: AnalysisConfig,
    *,
    seed: int = 0,
    device: str = "cpu",
    uniform_baseline: bool = False,
    use_static_features: bool = True,
    cooling_phase: str = "free_cooling_candidate",
    theta_idle: np.ndarray | None = None,
    max_epochs: int | None = None,
    progress: bool = False,
) -> ThermalFit:
    """Fit ``a_ij`` and ``beta_i`` on the load-off interval with the source held at zero."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    cooling_index = np.nonzero(series.phase == cooling_phase)[0]
    if len(cooling_index) < 8:
        raise ValueError(
            f"only {len(cooling_index)} frames labelled {cooling_phase!r}; "
            "the thermal operator cannot be identified without a decay interval"
        )
    theta_idle_init, idle_report = idle_steady_state(
        series, config.phases.idle_window_seconds, cooling_phase
    )
    learn_idle = theta_idle is None
    if not learn_idle:
        theta_idle_init = np.asarray(theta_idle, dtype=np.float64)
        idle_report = {"method": "supplied", "mean_theta_idle": float(np.nanmean(theta_idle_init))}

    # The decay label can also select more than one interval; split each run on
    # its own so no window integrates across a stretch that is not in the phase.
    cooling_runs = [run for run in contiguous_runs(cooling_index) if run.size >= 12]
    if not cooling_runs:
        raise ValueError(
            f"no contiguous {cooling_phase!r} interval of at least 12 frames; "
            "the thermal operator cannot be identified"
        )
    windows = Windows()
    for run in cooling_runs:
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

    node_features = _node_features(graph, series, use_static_features)
    edge_features = _edge_features(graph, node_features)
    similarity = _feature_similarity(graph, node_features)

    # The measured steady profile, from the powered near-steady stretch that
    # precedes the decay.  It carries the information the decay does not: how
    # far heat actually spreads before it is lost.
    theta_steady_np = _powered_steady_profile(series, cooling_index)
    # Model-free audit of what this decay can separate, computed before any fit
    # so it can be read independently of the result it is about to qualify.
    identifiability_audit = identifiability_analyse(
        series.theta[cooling_index], series.times[cooling_index]
    )
    t_theta_steady = (
        None
        if theta_steady_np is None
        else torch.as_tensor(theta_steady_np, dtype=torch.float32, device=dev)
    )

    rate_scale = decay_rate_scale(series, cooling_index)
    edge_scale = coupling_rate_scale(graph, series, cooling_index, rate_scale)
    model: torch.nn.Module = (
        UniformOperator(rate_scale=rate_scale, edge_rate_scale=edge_scale)
        if uniform_baseline
        else ThermalOperator(
            node_feature_dim=node_features.shape[1],
            edge_feature_dim=edge_features.shape[1],
            embedding_dim=config.model.node_embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            rate_scale=rate_scale,
            edge_rate_scale=edge_scale,
        )
    ).to(dev)

    t_node = torch.as_tensor(node_features, device=dev)
    t_edge_feat = torch.as_tensor(edge_features, device=dev)
    t_edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=dev)
    t_sim = torch.as_tensor(similarity, device=dev)
    t_area = torch.as_tensor(graph.areas_px, dtype=torch.float32, device=dev)
    t_theta = torch.as_tensor(series.theta, dtype=torch.float32, device=dev)
    # The load-off interval is a free-cooling *candidate*, not a power-down: on
    # this board the CPU load stops but the SoC never returns to ambient, so a
    # constant residual source is still running.  Two ways to handle that are
    # algebraically equivalent and numerically are not.
    #
    # Subtracting a pre-estimated idle level is the fragile one.  It is only a
    # change of variable if that level is a true steady state; otherwise
    # d(psi)/dt = A psi + A theta_idle and the leftover term corrupts every
    # rate.  On synthetic data with a known answer that mistake drops the
    # edge-rate correlation from 0.89 to below zero, and estimating the level
    # jointly does not rescue it -- the offset enters through the initial
    # condition, which is the worst-conditioned place to put an unknown.
    #
    # So the residual is carried as what it physically is: a constant,
    # non-negative source fitted alongside the operator.  Rollouts then start
    # from the measured theta, the unknown enters linearly, and s_idle = 0
    # (true free cooling) is the initialisation rather than an assertion.
    # Parameterised as `rate_scale * softplus(raw)`, so `softplus(raw)` carries
    # the units of an excess temperature and the product is K per unit time.
    # It is initialised at `rate_scale * theta_idle_init`, i.e. exactly the
    # residual that would hold the board at its *observed* idle plateau.
    #
    # Starting it near zero instead is a trap.  Adam moves a raw parameter by
    # about one learning rate per step regardless of gradient magnitude, so a
    # residual initialised two decades low needs thousands of steps to climb --
    # far more than early stopping allows.  On case10 that produced a fit with
    # tau = 1868 s and an implied idle plateau of 0.36 K, against a measured
    # decay of about 300 s to a plateau near 10 K: the fit had settled into
    # explaining only the flat tail.  The plateau is measured, so the
    # initialisation uses it; the fit is still free to move away from it, and
    # zero remains reachable when the board really was powered down.
    idle_source_scale = float(decay_rate_scale(series, cooling_index))
    inverse_softplus = np.log(np.expm1(np.clip(theta_idle_init, 1e-3, None)))
    raw_idle_source = torch.nn.Parameter(
        torch.as_tensor(inverse_softplus, dtype=torch.float32, device=dev),
        requires_grad=learn_idle,
    )

    def idle_source() -> Tensor:
        if not learn_idle:
            return torch.zeros(graph.n_nodes, dtype=torch.float32, device=dev)
        return idle_source_scale * torch.nn.functional.softplus(raw_idle_source)

    t_times = torch.as_tensor(series.times, dtype=torch.float32, device=dev)
    t_weight = torch.as_tensor(series.weight, dtype=torch.float32, device=dev)
    t_trust = torch.as_tensor(
        np.broadcast_to(series.trust[None, :], series.theta.shape).copy(),
        dtype=torch.float32,
        device=dev,
    )

    # The global log-magnitudes get a much larger learning rate than the network
    # that shapes structure around them: they are two numbers that may need to
    # travel a decade, and in log space that is fifty steps rather than
    # thousands.
    scale_params = list(getattr(model, "scale_parameters", list)())
    scale_ids = {id(p) for p in scale_params}
    network_params = [p for p in model.parameters() if id(p) not in scale_ids]
    parameters = [{"params": network_params, "lr": config.train.lr_thermal}]
    if scale_params:
        parameters.append({"params": scale_params, "lr": config.train.lr_scale})
    if learn_idle:
        parameters.append({"params": [raw_idle_source], "lr": config.train.lr_thermal})
    optimiser = torch.optim.Adam(parameters)
    epochs = int(max_epochs if max_epochs is not None else config.train.epochs_thermal)
    length = int(config.train.rollout_steps)
    history: list[dict] = []
    def snapshot() -> dict:
        return {
            "model": copy.deepcopy(model.state_dict()),
            "raw_idle_source": raw_idle_source.detach().clone(),
        }

    best = {"val": np.inf, "state": snapshot(), "epoch": -1}
    patience = 0

    def rollout(spans: list[tuple[int, int]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_rate, beta = model(t_node, t_edge_index, t_edge_feat)
        residual = idle_source()
        source_fn = None if not learn_idle else (lambda _t: residual)
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
        )

    for epoch in range(epochs):
        model.train()
        spans = _sample_windows(
            windows.train, length, config.train.rollout_stride, rng, config.train.batch_windows
        )
        if not spans:
            raise ValueError(
                f"training block {windows.train} is too short for a rollout window; "
                "shorten train.rollout_steps or widen the cooling interval"
            )
        optimiser.zero_grad(set_to_none=True)
        predicted, target, weight, trust = rollout(spans)
        edge_rate, beta = model(t_node, t_edge_index, t_edge_feat)
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
            source=None,
            times=None,
            area=t_area,
            weights=config.loss,
            huber_delta=config.train.huber_delta,
            theta_steady=t_theta_steady,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimiser.step()

        model.eval()
        with torch.no_grad():
            val_spans = _sample_windows(windows.val, length, 1, rng, 64)
            if not val_spans:
                raise ValueError(
                    f"validation block {windows.val} is too short for any rollout window; "
                    "early stopping would silently keep the initial parameters"
                )
            predicted, target, weight, _ = rollout(val_spans)
            val = float(
                ((predicted - target).abs() * weight).sum() / weight.sum().clamp_min(1e-9)
            )
        report["val_mae"] = val
        report["epoch"] = epoch
        history.append(report)
        if progress and epoch % 50 == 0:
            print(f"  epoch {epoch:4d} loss {report['total']:.5f} val_mae {val:.4f}")

        if np.isfinite(val) and val < best["val"] - 1e-6:
            best = {"val": val, "state": snapshot(), "epoch": epoch}
            patience = 0
        else:
            patience += 1
            if patience >= config.train.patience:
                break

    if best["epoch"] < 0 or not np.isfinite(best["val"]):
        raise RuntimeError(
            "the thermal fit never produced a finite validation error; the operator "
            "diverged rather than converged.  Returning it would report an "
            "initialisation as a result."
        )
    model.load_state_dict(best["state"]["model"])
    with torch.no_grad():
        raw_idle_source.copy_(best["state"]["raw_idle_source"])
    model.eval()
    with torch.no_grad():
        edge_rate, beta = model(t_node, t_edge_index, t_edge_feat)
        bound = spectral_bound(graph.n_nodes, t_edge_index.cpu(), edge_rate.cpu(), beta.cpu())
        residual = idle_source().cpu()
        steady_report: dict[str, Any] = {"available": t_theta_steady is not None}
        if t_theta_steady is not None:
            from tdp.dynamic_inverse.losses import (
                implied_steady_source,
                steady_source_plausibility,
            )

            implied = implied_steady_source(
                t_theta_steady.cpu(), t_edge_index.cpu(), edge_rate.cpu(), beta.cpu()
            ).numpy()
            negativity, spread = steady_source_plausibility(
                t_theta_steady.cpu(), t_edge_index.cpu(), edge_rate.cpu(), beta.cpu()
            )
            positive = np.clip(implied, 0.0, None)
            weighted = positive * graph.areas_px
            share = weighted / max(weighted.sum(), 1e-12)
            order = np.argsort(-share)
            top = max(1, int(round(0.1 * graph.n_nodes)))
            steady_report.update({
                "implied_steady_source_rate_mean": float(implied.mean()),
                "implied_steady_source_rate_max": float(implied.max()),
                "negativity_fraction": float(negativity),
                "spread_uniform_is_one": float(spread),
                "top_decile_share": float(share[order[:top]].sum()),
                "top_regions": [int(i) for i in order[:8]],
                "units": f"K/{series.time_unit}",
                "note": (
                    "s = -A theta_steady, the source the identified operator says must be "
                    "running to hold the measured steady profile.  Reported in K per unit "
                    "time (q/C), never as power."
                ),
            })
        # The equilibrium the fitted residual implies: 0 = A theta_idle + s_idle.
        # Derived, not assumed -- a clearly positive residual is direct evidence
        # the board was still dissipating through the "cooling" interval.
        theta_idle_final = (
            steady_state(graph.n_nodes, t_edge_index.cpu(), edge_rate.cpu(), beta.cpu(), residual)
            .numpy()
            .astype(np.float64)
        )
        idle_source_np = residual.numpy().astype(np.float64)

    metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val"]),
        "epochs_run": len(history),
        "spectral_bound_per_unit_time": bound,
        "stable": bool(bound <= 1e-9),
        "cooling_runs": [[int(run[0]), int(run[-1])] for run in cooling_runs],
        "idle_reference": idle_report,
        "idle_source_fitted": bool(learn_idle),
        "idle_theta_mean_k": float(np.mean(theta_idle_final)),
        "idle_theta_vs_flat_window_k": float(np.mean(theta_idle_final - theta_idle_init)),
        "residual_idle_source_rate_mean": float(np.mean(idle_source_np)),
        "residual_idle_source_rate_init_mean": float(np.mean(idle_source_scale * np.clip(theta_idle_init, 1e-3, None))),
        "residual_idle_source_rate_max": float(np.max(idle_source_np)),
        "residual_idle_source_units": f"K/{series.time_unit}",
        "residual_idle_source_note": (
            "constant non-negative source fitted over the load-off interval, initialised at "
            "zero.  A positive value means the board was still dissipating there, which is "
            "why the interval is labelled a free-cooling candidate and q=0 is never asserted."
        ),
        "rollout_steps": length,
        "rate_scale_per_unit_time": float(rate_scale),
        "coupling_rate_scale_per_unit_time": float(edge_scale),
        "graph_fiedler_value": float(graph_fiedler_value(graph)),
        "rate_scale_note": "board-mean 1/e decay rate; sets the heads' initialisation decade",
        "identifiability_audit": identifiability_audit,
        "steady_profile_used": bool(t_theta_steady is not None),
        "implied_steady_source": steady_report,
        "model_kind": "uniform" if uniform_baseline else "graph_neural_ode",
        "static_features_used": bool(use_static_features),
        "time_unit": series.time_unit,
    }
    return ThermalFit(
        edge_rate=edge_rate.detach().cpu().numpy(),
        beta=beta.detach().cpu().numpy(),
        theta_idle=theta_idle_final,
        idle_source_rate=idle_source_np,
        state_dict={
            "model": {k: v.cpu() for k, v in best["state"]["model"].items()},
            "raw_idle_source": best["state"]["raw_idle_source"].cpu(),
            "idle_source_scale": idle_source_scale,
            "idle_source_fitted": bool(learn_idle),
        },
        history=history,
        windows=windows,
        metrics=metrics,
        node_features=node_features,
        edge_features=edge_features,
        similarity=similarity,
        model_kind=metrics["model_kind"],
        time_unit=series.time_unit,
    )
