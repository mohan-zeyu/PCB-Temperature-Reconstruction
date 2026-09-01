"""Physics-built spatial patterns for compact PCB temperature forecasts.

The reconstructed board contains roughly sixteen thousand pixels, but a plate
does not have sixteen thousand independently useful thermal degrees of freedom.
This module builds a fixed library from the plate itself:

* the slow, smooth modes of the masked board Laplacian; and
* steady source-response patterns for every detected source at several trial
  spread rates.

The source-response patterns are essential.  A small hot spot projects badly
onto even a large collection of smooth modes, so omitting them would recreate
the central-temperature loss that Stage 2 was built to remove.  Candidates are
orthonormalised with pivoted QR and a stated numerical rank cut.  No measured
temperature frame is used to construct the library.

All rates are already divided by areal heat capacity.  ``spread_rate`` is in
pixel squared per time unit, ``cooling_rate`` is in inverse time, and a source
is a temperature-rise rate.  Nothing in this module is an absolute material
conductivity or power in watts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import eigsh, spsolve


def plate_laplacian(board_mask: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Return the positive graph Laplacian and the flat indices it represents.

    Four-neighbour links are used inside ``board_mask``.  A missing neighbour
    contributes no edge, which is the discrete no-flux boundary condition: heat
    cannot diffuse into pixels outside the board.
    """

    mask = np.asarray(board_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("board_mask must be a non-empty 2-D boolean array")

    flat_inside = np.flatnonzero(mask.ravel())
    index = np.full(mask.size, -1, dtype=np.int64)
    index[flat_inside] = np.arange(len(flat_inside), dtype=np.int64)
    index = index.reshape(mask.shape)

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for first, second in (
        (index[:-1, :], index[1:, :]),
        (index[:, :-1], index[:, 1:]),
    ):
        keep = (first >= 0) & (second >= 0)
        left.append(first[keep])
        right.append(second[keep])

    if left:
        src = np.concatenate(left)
        dst = np.concatenate(right)
    else:  # pragma: no cover - a one-pixel board has no links
        src = dst = np.empty(0, dtype=np.int64)
    rows = np.concatenate([src, dst, src, dst])
    cols = np.concatenate([dst, src, src, dst])
    values = np.concatenate([
        -np.ones(len(src)),
        -np.ones(len(src)),
        np.ones(len(src)),
        np.ones(len(src)),
    ])
    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(flat_inside), len(flat_inside))
    ).tocsr()
    return matrix, flat_inside


def _normalise_sources(
    source_shapes: Sequence[np.ndarray], board_mask: np.ndarray
) -> np.ndarray:
    """Flatten source maps and give each unit mean over its active support."""

    if not source_shapes:
        return np.empty((int(np.asarray(board_mask, bool).sum()), 0), dtype=np.float64)
    mask = np.asarray(board_mask, dtype=bool)
    columns = []
    for number, shape in enumerate(source_shapes):
        source = np.asarray(shape, dtype=np.float64)
        if source.shape != mask.shape:
            raise ValueError(
                f"source shape {number} has {source.shape}, expected {mask.shape}"
            )
        values = np.where(mask, source, 0.0)[mask]
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"source shape {number} must be finite and non-negative")
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError(f"source shape {number} is empty inside board_mask")
        # A coefficient of one now means a mean rise rate of one over the
        # source's effective area, independent of how many pixels it covers.
        active_area = total * total / max(float(values @ values), 1e-12)
        columns.append(values * (active_area / total))
    return np.column_stack(columns)


def steady_source_response(
    board_mask: np.ndarray,
    source_shape: np.ndarray,
    *,
    spread_rate_px2_per_time: float,
    cooling_rate_per_time: float,
) -> np.ndarray:
    """Fine-grid settled excess temperature from one non-negative source."""

    spread = float(spread_rate_px2_per_time)
    cooling = float(cooling_rate_per_time)
    if spread < 0.0 or cooling <= 0.0:
        raise ValueError("spread rate must be non-negative and cooling rate positive")
    laplacian, flat_inside = plate_laplacian(board_mask)
    source = _normalise_sources([source_shape], board_mask)[:, 0]
    system = spread * laplacian + cooling * eye(laplacian.shape[0], format="csr")
    values = spsolve(system, source)
    out = np.zeros(np.asarray(board_mask).shape, dtype=np.float64)
    out.ravel()[flat_inside] = values
    return out


@dataclass(frozen=True)
class PatternLibrary:
    """Orthonormal board patterns and the candidates they preserve."""

    basis: np.ndarray  # (K, H, W), zero outside board_mask
    board_mask: np.ndarray  # (H, W)
    candidate_names: tuple[str, ...]
    candidate_coordinates: np.ndarray  # (K, C), candidate = basis.T @ coordinates
    source_names: tuple[str, ...]
    source_shapes: np.ndarray  # (S, H, W), unit-mean-over-support convention
    rank_threshold: float
    report: dict

    @property
    def n_patterns(self) -> int:
        return int(self.basis.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.basis.shape[1]), int(self.basis.shape[2])

    def project(self, fields: np.ndarray) -> np.ndarray:
        """Project ``(..., H, W)`` fields onto the orthonormal patterns."""

        values = np.asarray(fields, dtype=np.float64)
        if values.shape[-2:] != self.shape_hw:
            raise ValueError(f"field ends in {values.shape[-2:]}, expected {self.shape_hw}")
        flat = values[..., self.board_mask]
        basis = self.basis[:, self.board_mask]
        return flat @ basis.T

    def reconstruct(self, coordinates: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
        """Expand ``(..., K)`` pattern strengths back onto the board grid."""

        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape[-1] != self.n_patterns:
            raise ValueError(
                f"coordinate axis has {values.shape[-1]}, expected {self.n_patterns}"
            )
        flat = values @ self.basis[:, self.board_mask]
        out = np.full((*values.shape[:-1], *self.shape_hw), fill, dtype=np.float64)
        out[..., self.board_mask] = flat
        return out


def load_pattern_library(path: Path | str) -> PatternLibrary:
    """Load the frozen library written by ``scripts/run_patterns.py``."""

    source = Path(path)
    with np.load(source) as data:
        required = {
            "basis",
            "board_mask",
            "candidate_names",
            "candidate_coordinates",
            "source_names",
            "source_shapes",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"pattern library missing {sorted(missing)}")
        basis = np.asarray(data["basis"], dtype=np.float64)
        board_mask = np.asarray(data["board_mask"], dtype=bool)
        candidate_names = tuple(str(value) for value in data["candidate_names"])
        candidate_coordinates = np.asarray(data["candidate_coordinates"], dtype=np.float64)
        source_names = tuple(str(value) for value in data["source_names"])
        source_shapes = np.asarray(data["source_shapes"], dtype=np.float64)
        rank_threshold = (
            float(data["rank_threshold"])
            if "rank_threshold" in data.files
            else float("nan")
        )
    if basis.ndim != 3 or board_mask.shape != basis.shape[1:]:
        raise ValueError("basis must have shape (patterns, H, W) matching board_mask")
    if source_shapes.shape != (len(source_names), *board_mask.shape):
        raise ValueError("source_shapes do not match source_names and board geometry")
    if candidate_coordinates.shape != (basis.shape[0], len(candidate_names)):
        raise ValueError("candidate_coordinates have inconsistent dimensions")
    gram = basis[:, board_mask] @ basis[:, board_mask].T
    orthogonality_error = float(np.linalg.norm(gram - np.eye(len(basis))))
    if orthogonality_error > 1e-4:
        raise ValueError(f"loaded patterns are not orthonormal (error {orthogonality_error:g})")
    return PatternLibrary(
        basis=basis,
        board_mask=board_mask,
        candidate_names=candidate_names,
        candidate_coordinates=candidate_coordinates,
        source_names=source_names,
        source_shapes=source_shapes,
        rank_threshold=rank_threshold,
        report={
            "loaded_from": str(source),
            "rank": int(len(basis)),
            "n_candidates": len(candidate_names),
            "source_names": list(source_names),
            "orthogonality_error_after_float32_export": orthogonality_error,
        },
    )


def build_pattern_library(
    board_mask: np.ndarray,
    source_shapes: Sequence[np.ndarray] = (),
    *,
    source_names: Sequence[str] | None = None,
    n_smooth: int = 64,
    trial_spread_rates: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    cooling_rate_per_time: float,
    rank_relative_tolerance: float = 1e-8,
) -> PatternLibrary:
    """Build smooth plate modes plus source-response patterns.

    ``trial_spread_rates`` deliberately brackets rather than reuses one fitted
    number.  The later fit can therefore move the physical rate without making
    the spatial library itself depend on that fitted value.
    """

    mask = np.asarray(board_mask, dtype=bool)
    laplacian, flat_inside = plate_laplacian(mask)
    n_pixels = len(flat_inside)
    if n_smooth < 1 or n_smooth >= n_pixels:
        raise ValueError(f"n_smooth must lie in [1, {n_pixels - 1}], got {n_smooth}")
    rates = tuple(float(value) for value in trial_spread_rates)
    if any(value < 0.0 for value in rates) or not rates:
        raise ValueError("trial spread rates must be a non-empty non-negative sequence")
    cooling = float(cooling_rate_per_time)
    if cooling <= 0.0:
        raise ValueError("cooling_rate_per_time must be positive")
    if not 0.0 < rank_relative_tolerance < 1.0:
        raise ValueError("rank_relative_tolerance must lie strictly between zero and one")

    source_matrix = _normalise_sources(source_shapes, mask)
    names = tuple(source_names or (f"source_{i}" for i in range(source_matrix.shape[1])))
    if len(names) != source_matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("source_names must be unique and match source_shapes")

    # The smallest graph-Laplacian eigenvectors are the board's slow natural
    # heat-distribution shapes.  A fixed starting vector and deterministic sign
    # remove the arbitrary choices made by an iterative eigensolver in ordinary
    # (non-degenerate) modes.
    eigenvalues, smooth = eigsh(
        laplacian,
        k=n_smooth,
        which="SM",
        v0=np.linspace(1.0, 2.0, n_pixels, dtype=np.float64),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    smooth = smooth[:, order]
    for column in range(smooth.shape[1]):
        pivot = int(np.argmax(np.abs(smooth[:, column])))
        if smooth[pivot, column] < 0.0:
            smooth[:, column] *= -1.0

    candidates = [smooth]
    candidate_names = [f"smooth_{i:03d}" for i in range(n_smooth)]
    identity = eye(n_pixels, format="csr")
    for source_index, source_name in enumerate(names):
        for spread in rates:
            system = spread * laplacian + cooling * identity
            response = spsolve(system, source_matrix[:, source_index])
            norm = float(np.linalg.norm(response))
            if norm <= 0.0 or not np.isfinite(norm):
                raise ValueError(f"source response {source_name!r} is not finite")
            candidates.append((response / norm)[:, None])
            candidate_names.append(f"source:{source_name}:spread={spread:g}")

    candidate_matrix = np.column_stack(candidates)
    basis, upper, pivots = linalg.qr(candidate_matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    threshold = float(diagonal[0] * rank_relative_tolerance)
    rank = int(np.count_nonzero(diagonal > threshold))
    basis = basis[:, :rank]
    coordinates = basis.T @ candidate_matrix

    basis_images = np.zeros((rank, *mask.shape), dtype=np.float64)
    basis_images[:, mask] = basis.T
    source_images = np.zeros((source_matrix.shape[1], *mask.shape), dtype=np.float64)
    if source_matrix.shape[1]:
        source_images[:, mask] = source_matrix.T

    return PatternLibrary(
        basis=basis_images,
        board_mask=mask,
        candidate_names=tuple(candidate_names),
        candidate_coordinates=coordinates,
        source_names=names,
        source_shapes=source_images,
        rank_threshold=threshold,
        report={
            "construction": "masked plate modes + source responses; pivoted QR rank cut",
            "n_board_pixels": n_pixels,
            "n_smooth_requested": int(n_smooth),
            "smooth_eigenvalues": eigenvalues.tolist(),
            "trial_spread_rates_px2_per_time": list(rates),
            "cooling_rate_per_time": cooling,
            "n_sources": len(names),
            "source_names": list(names),
            "n_candidates": int(candidate_matrix.shape[1]),
            "rank": rank,
            "rank_relative_tolerance": float(rank_relative_tolerance),
            "rank_absolute_threshold": threshold,
            "qr_pivots": pivots.tolist(),
            "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(rank))),
            "uses_measured_temperature_frames": False,
        },
    )


def gaussian_source_shape(
    shape_hw: tuple[int, int],
    centre_rc: tuple[float, float],
    *,
    width_px: float = 2.0,
    board_mask: np.ndarray | None = None,
) -> np.ndarray:
    """A compact non-negative source shape for a point-like detected hot spot."""

    if width_px <= 0.0:
        raise ValueError("width_px must be positive")
    rows, cols = np.indices(shape_hw)
    distance2 = (rows - float(centre_rc[0])) ** 2 + (cols - float(centre_rc[1])) ** 2
    out = np.exp(-0.5 * distance2 / float(width_px) ** 2)
    if board_mask is not None:
        mask = np.asarray(board_mask, dtype=bool)
        if mask.shape != shape_hw:
            raise ValueError(f"board_mask has {mask.shape}, expected {shape_hw}")
        out = np.where(mask, out, 0.0)
    return out
