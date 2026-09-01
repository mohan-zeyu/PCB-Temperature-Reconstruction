"""Deterministic presentation metrics for saved thermal forecasts.

This module contains no fitting.  It reconstructs the declared model range and
computes frame-aligned errors from already frozen predictions, keeping plotting
scripts small and making the numbers in figures independently testable.
"""

from __future__ import annotations

import numpy as np


def weighted_mae_by_frame(
    predicted: np.ndarray,
    actual: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Return one weighted mean absolute error for every leading frame."""

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if prediction.ndim != 3 or truth.shape != prediction.shape:
        raise ValueError("predicted and actual must share shape (T, H, W)")
    if weights.shape != prediction.shape:
        raise ValueError("weight must share the prediction shape")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(truth).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("presentation arrays must be finite")
    if (weights < 0.0).any():
        raise ValueError("weights cannot be negative")
    denominator = weights.sum(axis=(1, 2))
    if (denominator <= 0.0).any():
        raise ValueError("every frame must contain positive total weight")
    error = np.abs(prediction - truth)
    return (error * weights).sum(axis=(1, 2)) / denominator


def model_range_fields(
    elapsed_s: np.ndarray,
    start_range_k: np.ndarray,
    settled_range_k: np.ndarray,
    heldout_forward_error_k: float,
    decay_rate_per_s: float,
) -> np.ndarray:
    """Reconstruct the documented per-pixel model range at requested times.

    The range moves from Stage 2's starting reconstruction range to the
    training-only settled range.  The measured held-out discrepancy is added as
    the forecast takes over.  This is a model range, not a confidence interval.
    """

    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    start = np.asarray(start_range_k, dtype=np.float64)
    settled = np.asarray(settled_range_k, dtype=np.float64)
    empirical = float(heldout_forward_error_k)
    rate = float(decay_rate_per_s)
    if elapsed.ndim != 1 or (elapsed < 0.0).any() or not np.isfinite(elapsed).all():
        raise ValueError("elapsed_s must be a finite non-negative 1-D array")
    if start.ndim != 2 or settled.shape != start.shape:
        raise ValueError("start and settled ranges must share shape (H, W)")
    if not np.isfinite(start).all() or not np.isfinite(settled).all():
        raise ValueError("model ranges must be finite")
    if (start < 0.0).any() or (settled < 0.0).any():
        raise ValueError("model ranges cannot be negative")
    if not np.isfinite(empirical) or empirical < 0.0:
        raise ValueError("heldout_forward_error_k must be finite and non-negative")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("decay_rate_per_s must be finite and positive")
    decay = np.exp(-rate * elapsed)[:, None, None]
    return (
        decay * start[None, :, :]
        + (1.0 - decay) * settled[None, :, :]
        + (1.0 - decay) * empirical
    )


def nearest_sample_indices(sample_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Locate the nearest strictly ordered captured time for every query time."""

    samples = np.asarray(sample_times, dtype=np.float64)
    queries = np.asarray(query_times, dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("sample_times must be a non-empty 1-D array")
    if queries.ndim != 1:
        raise ValueError("query_times must be a 1-D array")
    if not np.isfinite(samples).all() or not np.isfinite(queries).all():
        raise ValueError("times must be finite")
    if len(samples) > 1 and not np.all(np.diff(samples) > 0.0):
        raise ValueError("sample_times must increase strictly")
    right = np.searchsorted(samples, queries, side="left")
    right = np.clip(right, 0, len(samples) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(queries - samples[left]) <= np.abs(samples[right] - queries)
    return np.where(choose_left, left, right).astype(np.int64)
