"""Closed-form forecasts in a fixed physical pattern library.

This is the fixed physical core of Stage 3.  It accepts an excess-temperature
field from any reconstructed frame, projects it onto the physics-built pattern
library, and advances the linear plate equation exactly over arbitrary time
gaps.  A settled state is obtained by one linear solve rather than by marching
thousands of tiny steps.

Stage 3b tested positive source histories as an ablation and selected decay
toward a training-only settled field for production.  Neither route can change
the spatial basis or bypass this physical state equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg

from tdp.thermal.patterns import PatternLibrary, plate_laplacian


@dataclass(frozen=True)
class ReconstructionArchive:
    """Validated Stage 2 output consumed by forecasting."""

    theta_k: np.ndarray
    measured_theta_k: np.ndarray
    uncertainty_k: np.ndarray
    times_s: np.ndarray
    ambient_c: np.ndarray
    board_mask: np.ndarray  # model domain: measured board union rebuilt pixels
    measured_board_mask: np.ndarray  # Stage 2 input mask before the rebuild
    rebuilt_mask: np.ndarray
    unknown_mask: np.ndarray
    heating_rate_k_per_s: np.ndarray
    source_rate_by_frame: np.ndarray
    region_ids: np.ndarray
    spread_rate_px2_per_s: float
    cooling_rate_per_s: float
    pixel_size_mm: float

    @property
    def n_frames(self) -> int:
        return int(self.theta_k.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.theta_k.shape[1]), int(self.theta_k.shape[2])

    def observation_weight(self, measurement_resolution_k: float = 0.1) -> np.ndarray:
        """Inverse-variance-style weight; estimates never become measurements.

        Directly observed board pixels retain weight one.  Rebuilt pixels are
        downweighted by the explicit Stage 2 model range.  The range is not a
        statistical confidence interval, so this is a transparent modelling
        weight, not a claim of calibrated probability.
        """

        resolution = float(measurement_resolution_k)
        if resolution <= 0.0:
            raise ValueError("measurement_resolution_k must be positive")
        effective_range = np.maximum(self.uncertainty_k, resolution)
        weight = np.where(
            self.rebuilt_mask[None, :, :],
            resolution**2 / (resolution**2 + effective_range**2),
            1.0,
        )
        return weight * self.board_mask[None, :, :]


def load_reconstruction_archive(path: Path | str) -> ReconstructionArchive:
    """Load Stage 2's NPZ with strict shape and provenance checks."""

    source = Path(path)
    with np.load(source) as data:
        required = {
            "theta_k",
            "measured_theta_k",
            "times_s",
            "ambient_c",
            "unknown_mask",
            "rebuilt_mask",
            "board_mask",
            "heating_rate_k_per_s",
            "source_rate_by_frame",
            "region_ids",
            "reconstruction_uncertainty_k",
            "spread_rate_px2_per_s",
            "cooling_rate_per_s",
            "pixel_size_mm",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"reconstruction archive missing {sorted(missing)}")
        payload = {name: np.array(data[name]) for name in required}

    theta = payload["theta_k"].astype(np.float64)
    measured = payload["measured_theta_k"].astype(np.float64)
    uncertainty = payload["reconstruction_uncertainty_k"].astype(np.float64)
    if theta.ndim != 3 or measured.shape != theta.shape or uncertainty.shape != theta.shape:
        raise ValueError("theta, measured theta and uncertainty must share shape (T, H, W)")
    times = payload["times_s"].astype(np.float64)
    ambient = payload["ambient_c"].astype(np.float64)
    if times.shape != (len(theta),) or ambient.shape != (len(theta),):
        raise ValueError("timestamps and ambient must have one value per reconstructed frame")
    if len(times) > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("reconstruction timestamps must be strictly increasing")
    shape = theta.shape[1:]
    masks = {
        name: payload[name].astype(bool)
        for name in ("unknown_mask", "rebuilt_mask", "board_mask")
    }
    if any(mask.shape != shape for mask in masks.values()):
        raise ValueError("all reconstruction masks must match the frame shape")
    if not np.all(masks["unknown_mask"] <= masks["rebuilt_mask"]):
        raise ValueError("unknown_mask must be contained in rebuilt_mask")
    # Stage 2 inherited ``board_mask`` from the legacy adapter's validity mask.
    # That mask deliberately excludes the deepest low-emissivity pixels, which
    # are exactly the pixels Stage 2 reconstructs.  Forecasting must retain the
    # union as its model domain while keeping the original mask for provenance.
    measured_board = masks["board_mask"]
    model_board = measured_board | masks["rebuilt_mask"]
    if not np.isfinite(theta).all() or not np.isfinite(uncertainty).all():
        raise ValueError("reconstruction contains non-finite values")
    if (uncertainty < 0.0).any():
        raise ValueError("reconstruction uncertainty cannot be negative")
    outside = ~masks["rebuilt_mask"]
    if not np.allclose(theta[:, outside], measured[:, outside], atol=1e-6, rtol=0.0):
        raise ValueError("Stage 2 changed directly measured pixels outside rebuilt_mask")
    if not np.allclose(uncertainty[:, outside], 0.0, atol=1e-7, rtol=0.0):
        raise ValueError("uncertainty must be zero outside rebuilt_mask")

    rates = payload["source_rate_by_frame"].astype(np.float64)
    region_ids = payload["region_ids"].astype(np.int64)
    if rates.ndim != 2 or rates.shape != (len(region_ids), len(theta)):
        raise ValueError("source_rate_by_frame must have shape (n_regions, n_frames)")
    heating = payload["heating_rate_k_per_s"].astype(np.float64)
    if heating.shape != shape:
        raise ValueError("heating_rate_k_per_s must match the board shape")

    spread = float(payload["spread_rate_px2_per_s"])
    cooling = float(payload["cooling_rate_per_s"])
    pixel_size = float(payload["pixel_size_mm"])
    if spread < 0.0 or cooling <= 0.0 or pixel_size <= 0.0:
        raise ValueError("archive rates and pixel size are outside their physical ranges")

    return ReconstructionArchive(
        theta_k=theta,
        measured_theta_k=measured,
        uncertainty_k=uncertainty,
        times_s=times,
        ambient_c=ambient,
        board_mask=model_board,
        measured_board_mask=measured_board,
        rebuilt_mask=masks["rebuilt_mask"],
        unknown_mask=masks["unknown_mask"],
        heating_rate_k_per_s=heating,
        source_rate_by_frame=rates,
        region_ids=region_ids,
        spread_rate_px2_per_s=spread,
        cooling_rate_per_s=cooling,
        pixel_size_mm=pixel_size,
    )


@dataclass(frozen=True)
class ClosedFormPlateModel:
    """Linear plate dynamics projected into an orthonormal pattern basis."""

    library: PatternLibrary
    spread_rate_px2_per_time: float
    cooling_rate_per_time: float
    operator: np.ndarray  # (K, K)
    source_drive: np.ndarray  # (K, S)

    @classmethod
    def build(
        cls,
        library: PatternLibrary,
        *,
        spread_rate_px2_per_time: float,
        cooling_rate_per_time: float,
    ) -> "ClosedFormPlateModel":
        spread = float(spread_rate_px2_per_time)
        cooling = float(cooling_rate_per_time)
        if spread < 0.0 or cooling <= 0.0:
            raise ValueError("spread rate must be non-negative and cooling rate positive")
        laplacian, _ = plate_laplacian(library.board_mask)
        basis = library.basis[:, library.board_mask].T  # (P, K)
        full_action = -spread * (laplacian @ basis) - cooling * basis
        operator = basis.T @ full_action
        sources = library.source_shapes[:, library.board_mask].T
        drive = basis.T @ sources
        return cls(library, spread, cooling, operator, drive)

    @property
    def n_sources(self) -> int:
        return int(self.source_drive.shape[1])

    def _validated_source_rates(self, source_rates: np.ndarray) -> np.ndarray:
        rates = np.asarray(source_rates, dtype=np.float64)
        if rates.shape != (self.n_sources,):
            raise ValueError(f"source rates have {rates.shape}, expected ({self.n_sources},)")
        if not np.isfinite(rates).all() or (rates < 0.0).any():
            raise ValueError("source rates must be finite and non-negative")
        return rates

    def _source_vector(self, source_rates: np.ndarray) -> np.ndarray:
        return self.source_drive @ self._validated_source_rates(source_rates)

    def interval_matrices(self, elapsed_time: float) -> tuple[np.ndarray, np.ndarray]:
        """State transition and source response for one interval width."""

        elapsed = float(elapsed_time)
        if elapsed < 0.0:
            raise ValueError("forecasting is forward only; elapsed_time cannot be negative")
        size = self.library.n_patterns
        if elapsed == 0.0:
            return np.eye(size), np.zeros((size, self.n_sources), dtype=np.float64)
        transition = linalg.expm(self.operator * elapsed)
        forcing = linalg.solve(
            self.operator,
            (transition - np.eye(size)) @ self.source_drive,
        )
        return transition, forcing

    def advance_coordinates(
        self, coordinates: np.ndarray, elapsed_time: float, source_rates: np.ndarray
    ) -> np.ndarray:
        """Exact advance for a source held constant over one time interval."""

        state = np.asarray(coordinates, dtype=np.float64)
        if state.shape != (self.library.n_patterns,):
            raise ValueError(
                f"coordinates have {state.shape}, expected ({self.library.n_patterns},)"
            )
        transition, forcing = self.interval_matrices(elapsed_time)
        return transition @ state + forcing @ self._validated_source_rates(source_rates)

    def advance_field(
        self,
        theta_k: np.ndarray,
        elapsed_time: float,
        source_rates: np.ndarray,
        *,
        preserve_initial_residual: bool = True,
    ) -> np.ndarray:
        """Advance one field, optionally retaining its unresolved initial detail.

        The fixed library cannot exactly represent every component edge in an
        arbitrary starting image.  Silently projecting the image would create
        an immediate jump at elapsed time zero.  By default that small residual
        is therefore carried as one start-specific correction and decayed with
        its plate-energy rate.  The reduced state remains the exact matrix-
        exponential solution; the correction is explicit and reportable rather
        than hidden in the source.
        """

        state, residual, residual_rate = self.split_initial_field(theta_k)
        field = self.library.reconstruct(
            self.advance_coordinates(state, elapsed_time, source_rates)
        )
        if preserve_initial_residual:
            field = field + np.exp(-residual_rate * float(elapsed_time)) * residual
        return field

    def split_initial_field(
        self, theta_k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Reduced coordinates, unresolved residual, and its energy-decay rate."""

        field = np.asarray(theta_k, dtype=np.float64)
        if field.shape != self.library.shape_hw:
            raise ValueError(f"initial field has {field.shape}, expected {self.library.shape_hw}")
        coordinates = self.library.project(field)
        projected = self.library.reconstruct(coordinates)
        residual = np.where(self.library.board_mask, field - projected, 0.0)
        return coordinates, residual, self.residual_decay_rate(residual)

    def residual_decay_rate(self, residual_k: np.ndarray) -> float:
        """Plate-energy decay rate of a stated unresolved residual field."""

        residual = np.asarray(residual_k, dtype=np.float64)
        if residual.shape != self.library.shape_hw:
            raise ValueError(
                f"residual field has {residual.shape}, expected {self.library.shape_hw}"
            )
        values = residual[self.library.board_mask]
        energy = float(values @ values)
        if energy <= 1e-20:
            return self.cooling_rate_per_time
        laplacian, _ = plate_laplacian(self.library.board_mask)
        roughness = float(values @ (laplacian @ values)) / energy
        return self.cooling_rate_per_time + self.spread_rate_px2_per_time * max(
            roughness, 0.0
        )

    def steady_coordinates(self, source_rates: np.ndarray) -> np.ndarray:
        """Settled state from one solve: ``operator @ state + source = 0``."""

        return linalg.solve(-self.operator, self._source_vector(source_rates))

    def steady_field(self, source_rates: np.ndarray) -> np.ndarray:
        return self.library.reconstruct(self.steady_coordinates(source_rates))

    def rollout(
        self,
        theta0_k: np.ndarray,
        output_times: np.ndarray,
        source_rates: np.ndarray | Callable[[float, float], np.ndarray],
    ) -> np.ndarray:
        """Forecast from any state at arbitrary increasing output times.

        An array source may be constant ``(S,)`` or piecewise constant
        ``(T-1, S)``.  A callable receives each interval's start and end time.
        """

        times = np.asarray(output_times, dtype=np.float64)
        if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all():
            raise ValueError("output_times must be a finite non-empty 1-D array")
        if len(times) > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("output_times must be strictly increasing")
        if times[0] < 0.0:
            raise ValueError("output times are elapsed times and cannot be negative")

        if callable(source_rates):
            source_for = source_rates
        else:
            values = np.asarray(source_rates, dtype=np.float64)
            if values.shape == (self.n_sources,):
                source_for = lambda _start, _stop: values
            elif values.shape == (max(len(times) - 1, 0), self.n_sources):
                if times[0] != 0.0:
                    raise ValueError(
                        "piecewise source arrays require output_times to start at zero"
                    )
                counter = iter(values)
                source_for = lambda _start, _stop: next(counter)
            else:
                raise ValueError(
                    "source_rates must be (n_sources,) or (n_intervals, n_sources)"
                )

        interval_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        def advance_cached(current: np.ndarray, elapsed: float, rates: np.ndarray) -> np.ndarray:
            key = float(elapsed)
            if key not in interval_cache:
                interval_cache[key] = self.interval_matrices(key)
            transition, forcing = interval_cache[key]
            return transition @ current + forcing @ self._validated_source_rates(rates)

        state, initial_residual, residual_rate = self.split_initial_field(theta0_k)
        if times[0] > 0.0:
            state = advance_cached(state, times[0], source_for(0.0, times[0]))
        states = [state]
        for start, stop in zip(times[:-1], times[1:], strict=True):
            state = advance_cached(
                state, stop - start, source_for(float(start), float(stop))
            )
            states.append(state)
        fields = self.library.reconstruct(np.stack(states))
        fields += (
            np.exp(-residual_rate * times)[:, None, None] * initial_residual[None, :, :]
        )
        return fields

    def time_to_settle(self, remaining_fraction: float = 0.01) -> float:
        """Conservative duration for every unforced mode to shrink by a fraction."""

        fraction = float(remaining_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("remaining_fraction must lie strictly between zero and one")
        slowest = float(np.max(linalg.eigvals(self.operator).real))
        if slowest >= 0.0:
            raise ValueError("reduced plate operator is not stable")
        return float(np.log(fraction) / slowest)
