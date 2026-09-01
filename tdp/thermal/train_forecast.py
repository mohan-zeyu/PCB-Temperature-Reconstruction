"""Fit positive, smooth source histories on the fixed Stage 3 plate model.

Only source histories and a small predeclared choice around the measured spread
and cooling rates are selected here.  The spatial patterns remain frozen.  The
fit uses contiguous training blocks; validation chooses the rate pair and
smoothness; test blocks are evaluated once after selection.

The reconstructed footprint pixels remain estimates.  Their Stage 2 model
range enters the fit through a fixed observation metric, while directly
measured pixels retain weight one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from tdp.dynamic_inverse.train_thermal import Windows, contiguous_split
from tdp.thermal.forecast import ClosedFormPlateModel, ReconstructionArchive
from tdp.thermal.patterns import PatternLibrary


@dataclass(frozen=True)
class FrozenForecastSplit:
    """Contiguous train/validation/test blocks plus unassigned guard frames."""

    roles: np.ndarray  # (T,), U8: train | val | test | guard
    windows: Windows
    report: dict[str, Any]


@dataclass(frozen=True)
class StaticTextureCorrection:
    """One bounded training-only texture omitted by the fixed physical basis."""

    pattern_k: np.ndarray  # (H, W), orthogonal to the fixed basis
    scale_intercept: float
    scale_per_mean_temperature: float
    minimum_scale: float
    maximum_scale: float
    report: dict[str, Any]

    def scale(self, board_mean_temperature: np.ndarray | float) -> np.ndarray:
        raw = self.scale_intercept + self.scale_per_mean_temperature * np.asarray(
            board_mean_temperature, dtype=np.float64
        )
        return np.clip(raw, self.minimum_scale, self.maximum_scale)


@dataclass(frozen=True)
class SourceHistoryFit:
    """Selected physical rates and non-negative source histories."""

    model: ClosedFormPlateModel
    split: FrozenForecastSplit
    knot_times_s: np.ndarray
    knot_source_rate_k_per_s: np.ndarray  # (knots, sources)
    frame_source_rate_k_per_s: np.ndarray  # (T, sources)
    interval_source_rate_k_per_s: np.ndarray  # (T-1, sources)
    source_names: tuple[str, ...]
    correction: StaticTextureCorrection
    smooth_strength: float
    source_rate_upper_k_per_s: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class EquilibriumForecastFit:
    """Production forecast of deviations from a training-only settled field."""

    model: ClosedFormPlateModel
    split: FrozenForecastSplit
    settled_theta_k: np.ndarray
    settled_coordinates: np.ndarray
    settled_model_range_k: np.ndarray
    residual_decay_multiplier: float
    metrics: dict[str, Any]


def make_frozen_split(
    n_frames: int,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    n_blocks: int = 5,
    guard_frames: int = 3,
) -> FrozenForecastSplit:
    """Reuse the repository's existing contiguous, guarded split unchanged."""

    windows = contiguous_split(
        np.arange(int(n_frames)),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        mode="interleaved",
        n_blocks=n_blocks,
        guard=guard_frames,
    )
    roles = np.full(int(n_frames), "guard", dtype="U8")
    for role in ("train", "val", "test"):
        for start, stop in getattr(windows, role):
            roles[start : stop + 1] = role
    counts = {role: int(np.count_nonzero(roles == role)) for role in np.unique(roles)}
    return FrozenForecastSplit(
        roles=roles,
        windows=windows,
        report={
            "method": "five contiguous blocks with gaps; no shuffled frames",
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "n_blocks": int(n_blocks),
            "guard_frames": int(guard_frames),
            "windows": windows.as_dict(),
            "frame_counts": counts,
        },
    )


def linear_knot_basis(times: np.ndarray, knot_times: np.ndarray) -> np.ndarray:
    """Piecewise-linear non-negative interpolation weights."""

    values = np.asarray(times, dtype=np.float64)
    knots = np.asarray(knot_times, dtype=np.float64)
    if values.ndim != 1 or knots.ndim != 1 or len(knots) < 1:
        raise ValueError("times and knot_times must be 1-D and knots non-empty")
    if len(knots) > 1 and not np.all(np.diff(knots) > 0.0):
        raise ValueError("knot_times must be strictly increasing")
    if len(knots) == 1:
        return np.ones((len(values), 1), dtype=np.float64)
    clipped = np.clip(values, knots[0], knots[-1])
    right = np.searchsorted(knots, clipped, side="right")
    right = np.clip(right, 1, len(knots) - 1)
    left = right - 1
    fraction = (clipped - knots[left]) / (knots[right] - knots[left])
    out = np.zeros((len(values), len(knots)), dtype=np.float64)
    row = np.arange(len(values))
    out[row, left] = 1.0 - fraction
    out[row, right] += fraction
    return out


def _observation_metric(
    archive: ReconstructionArchive,
    library: PatternLibrary,
    measurement_resolution_k: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Square root of the fixed confidence-weighted pattern inner product."""

    pixel_weight = np.median(
        archive.observation_weight(measurement_resolution_k), axis=0
    )[library.board_mask]
    basis = library.basis[:, library.board_mask].T
    gram = basis.T @ (pixel_weight[:, None] * basis)
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    values = np.clip(values, 1e-10, None)
    square_root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
    return square_root, {
        "measurement_resolution_k": float(measurement_resolution_k),
        "median_pixel_weight": float(np.median(pixel_weight)),
        "minimum_pattern_metric_eigenvalue": float(values.min()),
        "maximum_pattern_metric_eigenvalue": float(values.max()),
        "time_handling": "median Stage 2 model-range weight over the powered sequence",
    }


def _source_parameter_map(
    knot_weights: np.ndarray, n_sources: int
) -> np.ndarray:
    """Map source-major knot parameters to one interval's source vector."""

    row = np.asarray(knot_weights, dtype=np.float64)
    out = np.zeros((n_sources, n_sources * len(row)), dtype=np.float64)
    for source in range(n_sources):
        out[source, source * len(row) : (source + 1) * len(row)] = row
    return out


def _smooth_difference(n_sources: int, n_knots: int) -> np.ndarray:
    if n_knots <= 1:
        return np.empty((0, n_sources * n_knots), dtype=np.float64)
    out = np.zeros((n_sources * (n_knots - 1), n_sources * n_knots), dtype=np.float64)
    row = 0
    for source in range(n_sources):
        offset = source * n_knots
        for knot in range(n_knots - 1):
            out[row, offset + knot] = -1.0
            out[row, offset + knot + 1] = 1.0
            row += 1
    return out


def _design_matrix(
    archive: ReconstructionArchive,
    library: PatternLibrary,
    model: ClosedFormPlateModel,
    split: FrozenForecastSplit,
    knot_times: np.ndarray,
    metric_root: np.ndarray,
    rollout_steps: int = 12,
    rollout_stride: int = 6,
    mean_constraint_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coordinates = library.project(archive.theta_k)
    interval_midpoint = 0.5 * (archive.times_s[:-1] + archive.times_s[1:])
    knot_basis = linear_knot_basis(interval_midpoint, knot_times)
    matrix_blocks = []
    target_blocks = []
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    n_parameters = model.n_sources * len(knot_times)
    mean_coordinate = library.basis[:, library.board_mask].mean(axis=1)
    mean_weight = float(mean_constraint_weight) * np.sqrt(library.board_mask.sum())
    starts: list[int] = []
    targets: list[int] = []
    for window_start, window_stop in split.windows.train:
        latest_start = max(window_start, window_stop - 1)
        for start in range(window_start, latest_start + 1, int(rollout_stride)):
            stop = min(window_stop, start + int(rollout_steps))
            if stop <= start:
                continue
            starts.append(start)
            free_state = coordinates[start].copy()
            sensitivity = np.zeros(
                (library.n_patterns, n_parameters), dtype=np.float64
            )
            for index in range(start, stop):
                elapsed = float(archive.times_s[index + 1] - archive.times_s[index])
                if elapsed not in cache:
                    cache[elapsed] = model.interval_matrices(elapsed)
                transition, forcing = cache[elapsed]
                parameter_map = _source_parameter_map(
                    knot_basis[index], model.n_sources
                )
                free_state = transition @ free_state
                sensitivity = transition @ sensitivity + forcing @ parameter_map
                matrix_blocks.append(metric_root @ sensitivity)
                target_blocks.append(
                    metric_root @ (coordinates[index + 1] - free_state)
                )
                # Diffusion conserves the board mean exactly.  Without this
                # separate row, that invariant is only one coordinate among
                # dozens and the fit can trade a visible long-horizon drift for
                # small improvements across many spatial modes.
                matrix_blocks.append(
                    mean_weight
                    * (mean_coordinate @ sensitivity)[None, :]
                )
                target_blocks.append(np.array([
                    mean_weight
                    * float(mean_coordinate @ (coordinates[index + 1] - free_state))
                ]))
                targets.append(index + 1)
    if len(matrix_blocks) < 4:
        raise ValueError("frozen split has too few training rollout targets")
    matrix = np.vstack(matrix_blocks)
    target = np.concatenate(target_blocks)
    return matrix, target, knot_basis, coordinates, {
        "training_objective": "multi-step forward rollouts from uniformly spaced starts",
        "rollout_steps": int(rollout_steps),
        "rollout_stride": int(rollout_stride),
        "board_mean_conservation_weight_relative_to_one_full_field_kelvin": float(
            mean_constraint_weight
        ),
        "board_mean_conservation_matrix_weight": mean_weight,
        "n_training_starts": len(starts),
        "training_start_indices": starts,
        "n_training_targets_with_overlap": len(targets),
        "training_target_indices_with_overlap": targets,
        "distinct_frame_gaps_s": sorted(cache),
    }


def _solve_knots(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_sources: int,
    n_knots: int,
    smooth_strength: float,
    source_rate_upper: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_parameters = n_sources * n_knots
    difference = _smooth_difference(n_sources, n_knots)
    scale = float(np.linalg.norm(matrix, ord="fro") / max(np.sqrt(n_parameters), 1.0))
    blocks = [matrix]
    values = [target]
    if len(difference):
        blocks.append(float(smooth_strength) * scale * difference)
        values.append(np.zeros(len(difference)))
    # A tiny ridge resolves a knot constrained only through a guard/test tail;
    # it is deliberately far weaker than every candidate smoothness penalty.
    blocks.append(1e-6 * scale * np.eye(n_parameters))
    values.append(np.zeros(n_parameters))
    augmented = np.vstack(blocks)
    augmented_target = np.concatenate(values)
    result = lsq_linear(
        augmented,
        augmented_target,
        bounds=(0.0, float(source_rate_upper)),
        method="trf",
        lsmr_tol="auto",
        max_iter=300,
    )
    if not result.success:
        raise RuntimeError(f"source history solve failed: {result.message}")
    knot_rates = result.x.reshape(n_sources, n_knots).T
    return knot_rates, {
        "solver": "bounded linear least squares",
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "active_lower_bounds": int(np.count_nonzero(result.x <= 1e-9)),
        "active_upper_bounds": int(
            np.count_nonzero(result.x >= float(source_rate_upper) - 1e-9)
        ),
        "regularisation_scale": scale,
    }


def _coordinate_rollout(
    model: ClosedFormPlateModel,
    initial: np.ndarray,
    times: np.ndarray,
    interval_rates: np.ndarray,
) -> np.ndarray:
    states = [np.asarray(initial, dtype=np.float64)]
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    state = states[0]
    for elapsed, rates in zip(np.diff(times), interval_rates, strict=True):
        key = float(elapsed)
        if key not in cache:
            cache[key] = model.interval_matrices(key)
        transition, forcing = cache[key]
        state = transition @ state + forcing @ rates
        states.append(state)
    return np.stack(states)


def _coordinate_validation_error(
    archive: ReconstructionArchive,
    model: ClosedFormPlateModel,
    coordinates: np.ndarray,
    interval_rates: np.ndarray,
    windows: Sequence[tuple[int, int]],
    metric_root: np.ndarray,
) -> float:
    squared = 0.0
    count = 0
    for start, stop in windows:
        predicted = _coordinate_rollout(
            model,
            coordinates[start],
            archive.times_s[start : stop + 1],
            interval_rates[start:stop],
        )
        error = (predicted[1:] - coordinates[start + 1 : stop + 1]) @ metric_root.T
        squared += float(np.sum(error**2))
        count += int(error.size)
    return float(np.sqrt(squared / max(count, 1)))


def fit_static_texture_correction(
    archive: ReconstructionArchive,
    library: PatternLibrary,
    split: FrozenForecastSplit,
) -> StaticTextureCorrection:
    """Fit one fixed residual pattern using training frames only.

    The fixed plate basis intentionally does not encode every package/surface
    edge.  On case10 that omitted detail is nearly static after the first few
    frames rather than an unforced hot wrinkle.  One pattern is therefore
    allowed, with its amplitude determined only by predicted board temperature
    and hard-bounded to the training range.
    """

    coordinates = library.project(archive.theta_k)
    projected = library.reconstruct(coordinates)
    residual = np.where(
        library.board_mask[None, :, :], archive.theta_k - projected, 0.0
    )
    train = np.nonzero(split.roles == "train")[0]
    if len(train) < 4:
        raise ValueError("too few training frames for the static texture correction")
    pattern = residual[train].mean(axis=0)
    values = pattern[library.board_mask]
    denominator = float(values @ values)
    if denominator <= 1e-12:
        return StaticTextureCorrection(
            pattern_k=np.zeros(library.shape_hw, dtype=np.float64),
            scale_intercept=0.0,
            scale_per_mean_temperature=0.0,
            minimum_scale=0.0,
            maximum_scale=0.0,
            report={
                "construction": "none: fixed basis already spans every training field",
                "degrees_of_freedom": 0,
                "pattern_rms_fraction_of_training_field": 0.0,
            },
        )
    observed_scale = np.array([
        float(residual[index, library.board_mask] @ values) / denominator
        for index in range(archive.n_frames)
    ])
    board_mean = archive.theta_k[:, library.board_mask].mean(axis=1)
    design = np.column_stack([np.ones(len(train)), board_mean[train]])
    regression = lsq_linear(
        design,
        observed_scale[train],
        bounds=(np.array([0.0, 0.0]), np.array([2.0, 0.1])),
    )
    if not regression.success:
        raise RuntimeError(f"static texture scale fit failed: {regression.message}")
    predicted_train = design @ regression.x
    maximum_scale = float(max(1.0, 1.1 * np.max(observed_scale[train])))
    pattern_rms = float(np.sqrt(np.mean(values**2)))
    field_rms = float(
        np.sqrt(np.mean(archive.theta_k[train][:, library.board_mask] ** 2))
    )
    return StaticTextureCorrection(
        pattern_k=pattern,
        scale_intercept=float(regression.x[0]),
        scale_per_mean_temperature=float(regression.x[1]),
        minimum_scale=0.0,
        maximum_scale=maximum_scale,
        report={
            "construction": "mean fixed-basis residual over training frames only",
            "amplitude_driver": "affine function of predicted board-mean excess temperature",
            "scale_intercept": float(regression.x[0]),
            "scale_per_mean_temperature": float(regression.x[1]),
            "hard_scale_bounds": [0.0, maximum_scale],
            "training_scale_rmse": float(
                np.sqrt(np.mean((predicted_train - observed_scale[train]) ** 2))
            ),
            "pattern_rms_k": pattern_rms,
            "pattern_rms_fraction_of_training_field": pattern_rms / max(field_rms, 1e-12),
            "training_residual_correlation_median": float(np.median([
                np.corrcoef(residual[index, library.board_mask], values)[0, 1]
                for index in train
            ])),
            "degrees_of_freedom": 3,
            "note": (
                "one spatial pattern plus intercept/slope; source histories cannot change its shape"
            ),
        },
    )


def _rollout_with_correction(
    fit: SourceHistoryFit,
    initial_field: np.ndarray,
    elapsed_times: np.ndarray,
    interval_rates: np.ndarray,
) -> np.ndarray:
    initial_coordinates = fit.model.library.project(initial_field)
    coordinates = _coordinate_rollout(
        fit.model, initial_coordinates, elapsed_times, interval_rates
    )
    fields = fit.model.library.reconstruct(coordinates)
    board = fit.model.library.board_mask
    board_mean = fields[:, board].mean(axis=1)
    texture_scale = fit.correction.scale(board_mean)
    texture = texture_scale[:, None, None] * fit.correction.pattern_k[None, :, :]
    starting_texture = texture[0]
    starting_remainder = np.where(
        board,
        np.asarray(initial_field, dtype=np.float64) - fields[0] - starting_texture,
        0.0,
    )
    decay_rate = fit.model.residual_decay_rate(starting_remainder)
    remainder = (
        np.exp(-decay_rate * np.asarray(elapsed_times))[:, None, None]
        * starting_remainder[None, :, :]
    )
    return fields + texture + remainder


def _field_metrics(
    predicted: list[np.ndarray],
    actual: list[np.ndarray],
    weights: list[np.ndarray],
    rebuilt_mask: np.ndarray,
    source_names: tuple[str, ...],
    source_shapes: np.ndarray,
) -> dict[str, Any]:
    pred = np.concatenate(predicted, axis=0)
    truth = np.concatenate(actual, axis=0)
    weight = np.concatenate(weights, axis=0)
    error = np.abs(pred - truth)
    board = weight > 0.0
    measured = (~rebuilt_mask)[None, :, :] & board
    rebuilt = rebuilt_mask[None, :, :] & board
    report: dict[str, Any] = {
        "weighted_mae_k": float(np.sum(error * weight) / max(float(weight.sum()), 1e-12)),
        "directly_measured_mae_k": (
            float(error[measured].mean()) if measured.any() else float("nan")
        ),
        "rebuilt_estimate_agreement_mae_k": (
            float(error[rebuilt].mean()) if rebuilt.any() else float("nan")
        ),
        "unweighted_board_mae_k": float(error[board].mean()),
        "n_target_frames": int(len(pred)),
        "note": "rebuilt-area error is agreement with an estimate, not hidden ground truth",
    }
    source_region = {}
    for name, shape in zip(source_names, source_shapes, strict=True):
        active = shape > 0.5 * float(shape.max())
        source_region[name] = float(error[:, active].mean())
    report["source_region_mae_k"] = source_region
    return report


def evaluate_forward_windows(
    archive: ReconstructionArchive,
    fit: SourceHistoryFit,
    windows: Sequence[tuple[int, int]],
    *,
    source_kind: str = "fitted",
    constant_rates: np.ndarray | None = None,
    measurement_resolution_k: float = 0.1,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Forward-only window scores and predictions, excluding each known start."""

    predicted_blocks: list[np.ndarray] = []
    actual_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    saved: dict[int, np.ndarray] = {}
    all_weight = archive.observation_weight(measurement_resolution_k)
    for start, stop in windows:
        elapsed = archive.times_s[start : stop + 1] - archive.times_s[start]
        if source_kind == "fitted":
            rates = fit.interval_source_rate_k_per_s[start:stop]
            predicted = _rollout_with_correction(
                fit, archive.theta_k[start], elapsed, rates
            )
        elif source_kind == "constant":
            if constant_rates is None:
                raise ValueError("constant source evaluation requires constant_rates")
            rates = np.broadcast_to(
                np.asarray(constant_rates, dtype=np.float64),
                (stop - start, fit.model.n_sources),
            )
            predicted = _rollout_with_correction(
                fit, archive.theta_k[start], elapsed, rates
            )
        elif source_kind == "none":
            predicted = _rollout_with_correction(
                fit,
                archive.theta_k[start],
                elapsed,
                np.zeros((stop - start, fit.model.n_sources)),
            )
        elif source_kind == "persistence":
            predicted = np.broadcast_to(
                archive.theta_k[start][None, :, :],
                (stop - start + 1, *archive.shape_hw),
            ).copy()
        else:
            raise ValueError(f"unknown source_kind {source_kind!r}")
        # The starting frame is given, not predicted.
        predicted = predicted[1:]
        truth = archive.theta_k[start + 1 : stop + 1]
        predicted_blocks.append(predicted)
        actual_blocks.append(truth)
        weight_blocks.append(all_weight[start + 1 : stop + 1])
        saved[int(start)] = predicted.astype(np.float32)
    return _field_metrics(
        predicted_blocks,
        actual_blocks,
        weight_blocks,
        archive.rebuilt_mask,
        fit.source_names,
        fit.model.library.source_shapes,
    ), saved


def rollout_to_equilibrium(
    fit: EquilibriumForecastFit,
    initial_theta_k: np.ndarray,
    elapsed_times_s: np.ndarray,
) -> np.ndarray:
    """Closed-form forward trajectory from an exact start to the settled field."""

    times = np.asarray(elapsed_times_s, dtype=np.float64)
    if times.ndim != 1 or len(times) < 1 or times[0] < 0.0:
        raise ValueError("elapsed_times_s must be a non-empty non-negative 1-D array")
    if len(times) > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("elapsed_times_s must be strictly increasing")
    initial_coordinates = fit.model.library.project(initial_theta_k)
    delta = initial_coordinates - fit.settled_coordinates
    cache: dict[float, np.ndarray] = {}

    def transition_for(elapsed: float) -> np.ndarray:
        key = float(elapsed)
        if key not in cache:
            cache[key] = fit.model.interval_matrices(key)[0]
        return cache[key]

    state = delta
    if times[0] > 0.0:
        state = transition_for(float(times[0])) @ state
    states = [state]
    for elapsed in np.diff(times):
        state = transition_for(float(elapsed)) @ state
        states.append(state)
    base = fit.settled_theta_k[None, :, :] + fit.model.library.reconstruct(
        np.stack(states)
    )
    start_remainder = np.where(
        fit.model.library.board_mask,
        np.asarray(initial_theta_k, dtype=np.float64) - base[0],
        0.0,
    )
    residual_rate = (
        fit.residual_decay_multiplier * fit.model.cooling_rate_per_time
    )
    return base + np.exp(-residual_rate * times)[:, None, None] * start_remainder


def evaluate_equilibrium_windows(
    archive: ReconstructionArchive,
    fit: EquilibriumForecastFit,
    windows: Sequence[tuple[int, int]],
    *,
    measurement_resolution_k: float = 0.1,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    predicted_blocks: list[np.ndarray] = []
    actual_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    saved: dict[int, np.ndarray] = {}
    all_weight = archive.observation_weight(measurement_resolution_k)
    for start, stop in windows:
        elapsed = archive.times_s[start : stop + 1] - archive.times_s[start]
        predicted = rollout_to_equilibrium(
            fit, archive.theta_k[start], elapsed
        )[1:]
        predicted_blocks.append(predicted)
        actual_blocks.append(archive.theta_k[start + 1 : stop + 1])
        weight_blocks.append(all_weight[start + 1 : stop + 1])
        saved[int(start)] = predicted.astype(np.float32)
    return _field_metrics(
        predicted_blocks,
        actual_blocks,
        weight_blocks,
        archive.rebuilt_mask,
        fit.model.library.source_names,
        fit.model.library.source_shapes,
    ), saved


def fit_equilibrium_forecast(
    archive: ReconstructionArchive,
    library: PatternLibrary,
    *,
    split: FrozenForecastSplit | None = None,
    spread_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    cooling_multipliers: Sequence[float] = (0.9, 1.0, 1.1),
    residual_decay_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    measurement_resolution_k: float = 0.1,
    selection_tolerance_k: float = 0.01,
) -> EquilibriumForecastFit:
    """Select a stable plate decay toward a training-only settled field."""

    if archive.shape_hw != library.shape_hw or not np.array_equal(
        archive.board_mask, library.board_mask
    ):
        raise ValueError("reconstruction archive and pattern library use different boards")
    split = split or make_frozen_split(archive.n_frames)
    if not split.windows.train:
        raise ValueError("frozen split has no training window")
    settled_start, settled_stop = split.windows.train[-1]
    settled_frames = archive.theta_k[settled_start : settled_stop + 1]
    settled = settled_frames.mean(axis=0)
    settled_coordinates = library.project(settled)
    # This remains a model range, not a confidence interval.  It combines the
    # supplied Stage 2 range with the observed variation across training-only
    # settled frames without dividing by sqrt(n).
    supplied_range = np.sqrt(np.mean(
        archive.uncertainty_k[settled_start : settled_stop + 1] ** 2, axis=0
    ))
    settled_range = supplied_range + settled_frames.std(axis=0)

    candidates = []
    fitted_candidates: list[dict[str, Any]] = []
    for spread_multiplier in spread_multipliers:
        for cooling_multiplier in cooling_multipliers:
            model = ClosedFormPlateModel.build(
                library,
                spread_rate_px2_per_time=(
                    archive.spread_rate_px2_per_s * float(spread_multiplier)
                ),
                cooling_rate_per_time=(
                    archive.cooling_rate_per_s * float(cooling_multiplier)
                ),
            )
            for residual_multiplier in residual_decay_multipliers:
                provisional = EquilibriumForecastFit(
                    model=model,
                    split=split,
                    settled_theta_k=settled,
                    settled_coordinates=settled_coordinates,
                    settled_model_range_k=settled_range,
                    residual_decay_multiplier=float(residual_multiplier),
                    metrics={},
                )
                validation, _ = evaluate_equilibrium_windows(
                    archive,
                    provisional,
                    split.windows.val,
                    measurement_resolution_k=measurement_resolution_k,
                )
                item = {
                    "spread_multiplier": float(spread_multiplier),
                    "cooling_multiplier": float(cooling_multiplier),
                    "residual_decay_multiplier": float(residual_multiplier),
                    "spread_rate_px2_per_s": model.spread_rate_px2_per_time,
                    "cooling_rate_per_s": model.cooling_rate_per_time,
                    "validation_weighted_forward_mae_k": validation["weighted_mae_k"],
                    "model": model,
                }
                fitted_candidates.append(item)
                candidates.append({key: value for key, value in item.items() if key != "model"})
    if not fitted_candidates:
        raise RuntimeError("no equilibrium forecast candidate was evaluated")
    raw_best = min(
        fitted_candidates, key=lambda item: item["validation_weighted_forward_mae_k"]
    )
    eligible = [
        item
        for item in fitted_candidates
        if item["validation_weighted_forward_mae_k"]
        <= raw_best["validation_weighted_forward_mae_k"] + float(selection_tolerance_k)
    ]
    # Differences below 0.01 K are not resolved by these source files.  Inside
    # that equivalence band, prefer the rates measured before forecasting and
    # the nominal residual rule rather than selecting a grid boundary.
    best = min(
        eligible,
        key=lambda item: (
            (item["spread_multiplier"] - 1.0) ** 2
            + (item["cooling_multiplier"] - 1.0) ** 2
            + (item["residual_decay_multiplier"] - 1.0) ** 2
        ),
    )
    selected = EquilibriumForecastFit(
        model=best["model"],
        split=split,
        settled_theta_k=settled,
        settled_coordinates=settled_coordinates,
        settled_model_range_k=settled_range,
        residual_decay_multiplier=best["residual_decay_multiplier"],
        metrics={},
    )
    validation, validation_predictions = evaluate_equilibrium_windows(
        archive,
        selected,
        split.windows.val,
        measurement_resolution_k=measurement_resolution_k,
    )
    test, test_predictions = evaluate_equilibrium_windows(
        archive,
        selected,
        split.windows.test,
        measurement_resolution_k=measurement_resolution_k,
    )

    # Persistence is evaluated through the already existing baseline path.  A
    # minimal SourceHistoryFit shell is sufficient because persistence uses no
    # fitted source or correction.
    correction = fit_static_texture_correction(archive, library, split)
    shell = SourceHistoryFit(
        model=selected.model,
        split=split,
        knot_times_s=np.array([archive.times_s[0], archive.times_s[-1]]),
        knot_source_rate_k_per_s=np.zeros((2, selected.model.n_sources)),
        frame_source_rate_k_per_s=np.zeros((archive.n_frames, selected.model.n_sources)),
        interval_source_rate_k_per_s=np.zeros((archive.n_frames - 1, selected.model.n_sources)),
        source_names=library.source_names,
        correction=correction,
        smooth_strength=0.0,
        source_rate_upper_k_per_s=0.0,
        metrics={},
    )
    persistence, persistence_predictions = evaluate_forward_windows(
        archive,
        shell,
        split.windows.test,
        source_kind="persistence",
        measurement_resolution_k=measurement_resolution_k,
    )

    whole_phase = rollout_to_equilibrium(
        selected,
        archive.theta_k[0],
        archive.times_s - archive.times_s[0],
    )
    whole_error = np.abs(whole_phase[1:] - archive.theta_k[1:])
    whole_weight = archive.observation_weight(measurement_resolution_k)[1:]
    whole_weighted_mae = float(
        np.sum(whole_error * whole_weight) / np.sum(whole_weight)
    )
    metrics = {
        "selection": {
            "criterion": "full-field forward validation MAE under Stage 2 uncertainty weights",
            "parsimony_rule": (
                "within selection_tolerance_k of the numerical minimum, prefer measured "
                "spread/cooling rates and nominal residual decay"
            ),
            "selection_tolerance_k": float(selection_tolerance_k),
            "raw_minimum_validation_weighted_forward_mae_k": raw_best[
                "validation_weighted_forward_mae_k"
            ],
            "selected_validation_weighted_forward_mae_k": best[
                "validation_weighted_forward_mae_k"
            ],
            "selected_spread_multiplier": best["spread_multiplier"],
            "selected_cooling_multiplier": best["cooling_multiplier"],
            "selected_residual_decay_multiplier": best["residual_decay_multiplier"],
            "selected_spread_rate_px2_per_s": best["spread_rate_px2_per_s"],
            "selected_cooling_rate_per_s": best["cooling_rate_per_s"],
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "settled_field": {
            "source": "mean of the final training block only",
            "training_frame_start": settled_start,
            "training_frame_stop": settled_stop,
            "training_time_start_s": float(archive.times_s[settled_start]),
            "training_time_stop_s": float(archive.times_s[settled_stop]),
            "model_range_kind": "Stage 2 range plus temporal settled-frame variation; not a confidence interval",
            "median_model_range_k": float(np.median(settled_range[archive.board_mask])),
            "p95_model_range_k": float(np.percentile(settled_range[archive.board_mask], 95.0)),
        },
        "validation_forward": validation,
        "test_forward": test,
        "test_persistence": persistence,
        "frame0_whole_powered_phase_weighted_mae_k": whole_weighted_mae,
        "acceptance": {
            "beats_persistence_on_weighted_test_mae": bool(
                test["weighted_mae_k"] < persistence["weighted_mae_k"]
            ),
            "forward_only": True,
            "settled_field_uses_validation_or_test": False,
        },
        "interpretation": (
            "the fixed source-shaped basis controls how deviations relax; the training-only "
            "equilibrium absorbs unmeasured material/cooling heterogeneity that three sparse "
            "source channels cannot identify"
        ),
        "_validation_predictions": validation_predictions,
        "_test_predictions": test_predictions,
        "_persistence_predictions": persistence_predictions,
        "_whole_phase_prediction": whole_phase.astype(np.float32),
    }
    return EquilibriumForecastFit(
        model=selected.model,
        split=split,
        settled_theta_k=settled,
        settled_coordinates=settled_coordinates,
        settled_model_range_k=settled_range,
        residual_decay_multiplier=selected.residual_decay_multiplier,
        metrics=metrics,
    )


def fit_source_histories(
    archive: ReconstructionArchive,
    library: PatternLibrary,
    *,
    split: FrozenForecastSplit | None = None,
    n_knots: int = 12,
    smooth_strengths: Sequence[float] = (0.01, 0.03, 0.1, 0.3),
    spread_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    cooling_multipliers: Sequence[float] = (0.9, 1.0, 1.1),
    source_rate_upper_k_per_s: float | None = None,
    measurement_resolution_k: float = 0.1,
) -> SourceHistoryFit:
    """Select rates on validation blocks, then score forward test rollouts."""

    if archive.shape_hw != library.shape_hw or not np.array_equal(
        archive.board_mask, library.board_mask
    ):
        raise ValueError("reconstruction archive and pattern library use different boards")
    if n_knots < 2:
        raise ValueError("n_knots must be at least two for a time-varying source fit")
    split = split or make_frozen_split(archive.n_frames)
    metric_root, metric_report = _observation_metric(
        archive, library, measurement_resolution_k
    )
    correction = fit_static_texture_correction(archive, library, split)
    knot_times = np.linspace(archive.times_s[0], archive.times_s[-1], int(n_knots))
    positive_stage2 = archive.source_rate_by_frame[archive.source_rate_by_frame > 0.0]
    evidence = (
        float(np.percentile(positive_stage2, 99.0)) if len(positive_stage2) else 0.25
    )
    upper = float(source_rate_upper_k_per_s or max(0.25, 2.0 * evidence))
    if upper <= 0.0:
        raise ValueError("source_rate_upper_k_per_s must be positive")

    candidates = []
    best: dict[str, Any] | None = None
    for spread_multiplier in spread_multipliers:
        for cooling_multiplier in cooling_multipliers:
            spread = archive.spread_rate_px2_per_s * float(spread_multiplier)
            cooling = archive.cooling_rate_per_s * float(cooling_multiplier)
            model = ClosedFormPlateModel.build(
                library,
                spread_rate_px2_per_time=spread,
                cooling_rate_per_time=cooling,
            )
            matrix, target, interval_basis, coordinates, design_report = _design_matrix(
                archive, library, model, split, knot_times, metric_root
            )
            for smooth in smooth_strengths:
                knots, solver_report = _solve_knots(
                    matrix,
                    target,
                    n_sources=model.n_sources,
                    n_knots=len(knot_times),
                    smooth_strength=float(smooth),
                    source_rate_upper=upper,
                )
                interval_rates = interval_basis @ knots
                validation = _coordinate_validation_error(
                    archive,
                    model,
                    coordinates,
                    interval_rates,
                    split.windows.val,
                    metric_root,
                )
                frame_rates_candidate = linear_knot_basis(
                    archive.times_s, knot_times
                ) @ knots
                candidate_fit = SourceHistoryFit(
                    model=model,
                    split=split,
                    knot_times_s=knot_times,
                    knot_source_rate_k_per_s=knots,
                    frame_source_rate_k_per_s=frame_rates_candidate,
                    interval_source_rate_k_per_s=interval_rates,
                    source_names=library.source_names,
                    correction=correction,
                    smooth_strength=float(smooth),
                    source_rate_upper_k_per_s=upper,
                    metrics={},
                )
                validation_field, _ = evaluate_forward_windows(
                    archive,
                    candidate_fit,
                    split.windows.val,
                    measurement_resolution_k=measurement_resolution_k,
                )
                item = {
                    "spread_multiplier": float(spread_multiplier),
                    "cooling_multiplier": float(cooling_multiplier),
                    "spread_rate_px2_per_s": spread,
                    "cooling_rate_per_s": cooling,
                    "smooth_strength": float(smooth),
                    "validation_coordinate_rmse_k": validation,
                    "validation_weighted_forward_mae_k": validation_field[
                        "weighted_mae_k"
                    ],
                    "knot_rates": knots,
                    "interval_rates": interval_rates,
                    "model": model,
                    "solver": solver_report,
                    "design": design_report,
                }
                candidates.append({
                    key: value
                    for key, value in item.items()
                    if key not in {"knot_rates", "interval_rates", "model"}
                })
                if best is None or item["validation_weighted_forward_mae_k"] < best[
                    "validation_weighted_forward_mae_k"
                ]:
                    best = item
    if best is None:
        raise RuntimeError("no Stage 3b model candidate was fitted")

    frame_basis = linear_knot_basis(archive.times_s, knot_times)
    frame_rates = frame_basis @ best["knot_rates"]
    provisional = SourceHistoryFit(
        model=best["model"],
        split=split,
        knot_times_s=knot_times,
        knot_source_rate_k_per_s=best["knot_rates"],
        frame_source_rate_k_per_s=frame_rates,
        interval_source_rate_k_per_s=best["interval_rates"],
        source_names=library.source_names,
        correction=correction,
        smooth_strength=best["smooth_strength"],
        source_rate_upper_k_per_s=upper,
        metrics={},
    )

    # Constant-source physics is fitted on exactly the same training transitions
    # and selected operator.  It is a real baseline, not the fitted spline averaged
    # after seeing validation/test data.
    constant_knots = np.array([0.5 * (archive.times_s[0] + archive.times_s[-1])])
    constant_matrix, constant_target, _basis, _coordinates, _report = _design_matrix(
        archive, library, provisional.model, split, constant_knots, metric_root
    )
    constant_rate, constant_solver = _solve_knots(
        constant_matrix,
        constant_target,
        n_sources=provisional.model.n_sources,
        n_knots=1,
        smooth_strength=0.0,
        source_rate_upper=upper,
    )
    constant_rate = constant_rate[0]

    validation_metrics, _ = evaluate_forward_windows(
        archive,
        provisional,
        split.windows.val,
        measurement_resolution_k=measurement_resolution_k,
    )
    test_by_model = {}
    test_predictions: dict[str, dict[int, np.ndarray]] = {}
    for name, kind, rates in (
        ("fitted_smooth_source", "fitted", None),
        ("constant_source", "constant", constant_rate),
        ("no_source", "none", None),
        ("persistence", "persistence", None),
    ):
        metrics, prediction = evaluate_forward_windows(
            archive,
            provisional,
            split.windows.test,
            source_kind=kind,
            constant_rates=rates,
            measurement_resolution_k=measurement_resolution_k,
        )
        test_by_model[name] = metrics
        test_predictions[name] = prediction

    knot_difference = np.diff(best["knot_rates"], axis=0)
    metrics = {
        "selection": {
            "criterion": "full-field forward validation MAE under the Stage 2 uncertainty weights",
            "selected_spread_multiplier": best["spread_multiplier"],
            "selected_cooling_multiplier": best["cooling_multiplier"],
            "selected_spread_rate_px2_per_s": best["spread_rate_px2_per_s"],
            "selected_cooling_rate_per_s": best["cooling_rate_per_s"],
            "selected_smooth_strength": best["smooth_strength"],
            "validation_coordinate_rmse_k": best["validation_coordinate_rmse_k"],
            "validation_weighted_forward_mae_k": best[
                "validation_weighted_forward_mae_k"
            ],
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "split": split.report,
        "observation_metric": metric_report,
        "static_texture_correction": correction.report,
        "source_constraints": {
            "non_negative": bool((best["knot_rates"] >= -1e-12).all()),
            "upper_k_per_s": upper,
            "maximum_fitted_k_per_s": float(best["knot_rates"].max()),
            "mean_absolute_knot_change_k_per_s": float(np.mean(np.abs(knot_difference))),
            "solver": best["solver"],
        },
        "validation_forward": validation_metrics,
        "test_forward_by_model": test_by_model,
        "constant_source_baseline": {
            "source_rate_k_per_s": constant_rate.tolist(),
            "solver": constant_solver,
        },
        "acceptance": {
            "beats_no_source_on_weighted_test_mae": bool(
                test_by_model["fitted_smooth_source"]["weighted_mae_k"]
                < test_by_model["no_source"]["weighted_mae_k"]
            ),
            "beats_persistence_on_weighted_test_mae": bool(
                test_by_model["fitted_smooth_source"]["weighted_mae_k"]
                < test_by_model["persistence"]["weighted_mae_k"]
            ),
            "note": (
                "the late test block is near steady, so persistence is a deliberately hard baseline; "
                "both comparisons are reported even if only one passes"
            ),
        },
        # Kept out of JSON by the driver; returned here so it can write exact
        # held-out arrays without recomputing or touching the fit.
        "_test_predictions": test_predictions,
    }
    return SourceHistoryFit(
        model=provisional.model,
        split=split,
        knot_times_s=knot_times,
        knot_source_rate_k_per_s=best["knot_rates"],
        frame_source_rate_k_per_s=frame_rates,
        interval_source_rate_k_per_s=best["interval_rates"],
        source_names=library.source_names,
        correction=correction,
        smooth_strength=best["smooth_strength"],
        source_rate_upper_k_per_s=upper,
        metrics=metrics,
    )
