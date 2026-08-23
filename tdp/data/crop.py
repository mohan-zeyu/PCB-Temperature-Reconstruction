"""Locate and extract the fixed-size board window from raw thermal frames.

The board window (157 rows x 103 cols ≈ the Pi 4B's 85x56 mm PCB) is found by
maximizing the enclosed excess-temperature energy with an integral image — a
threshold-free criterion that is stable against the warm halo around the board.

For re-staged scenes (case00 was shot 2 h later, moved/rotated) the window is
instead registered against a reference crop by zero-normalized cross-correlation
of gradient-magnitude maps over the four 90° rotation hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

DEFAULT_SIZE = (157, 103)  # (rows, cols)


def estimate_ambient(frame: np.ndarray, border_px: int = 8) -> float:
    """Ambient estimate = median of the frame's border ring (background around the board)."""
    b = border_px
    ring = np.concatenate(
        [frame[:b].ravel(), frame[-b:].ravel(), frame[b:-b, :b].ravel(), frame[b:-b, -b:].ravel()]
    )
    return float(np.median(ring))


def find_crop(
    frame: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    t_amb: float | None = None,
) -> tuple[int, int, float]:
    """Top-left (row0, col0) of the `size` window maximizing enclosed excess energy.

    Returns (row0, col0, captured_fraction) where captured_fraction is the share
    of the frame's total excess-temperature energy inside the window.
    """
    h, w = size
    if frame.shape[0] < h or frame.shape[1] < w:
        raise ValueError(f"frame {frame.shape} smaller than window {size}")
    if t_amb is None:
        t_amb = estimate_ambient(frame)
    excess = np.clip(frame.astype(np.float64) - t_amb, 0.0, None)
    ii = np.pad(excess, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window_sums = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    r, c = np.unravel_index(int(np.argmax(window_sums)), window_sums.shape)
    total = excess.sum()
    frac = float(window_sums[r, c] / total) if total > 0 else 0.0
    return int(r), int(c), frac


def apply_crop(
    frames: np.ndarray, row0: int, col0: int, size: tuple[int, int] = DEFAULT_SIZE
) -> np.ndarray:
    """Crop (H, W) or (N, H, W) frames to the fixed window."""
    h, w = size
    if frames.ndim == 2:
        return frames[row0 : row0 + h, col0 : col0 + w]
    return frames[:, row0 : row0 + h, col0 : col0 + w]


def _check_quad_order(quad_xy: np.ndarray, name: str) -> None:
    """Require four corners to run TL, TR, BR, BL around a convex quadrilateral.

    Each corner's turn direction is the cross product of its two adjacent
    edges.  For the expected order every turn is positive (rows increase
    downward, so that traversal is clockwise on screen).  Mixed signs mean the
    outline crosses itself; all-negative means the same convex shape traversed
    backwards, which rectifies to a mirror image of the board.
    """
    edges = np.roll(quad_xy, -1, axis=0) - quad_xy
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(np.abs(turns) < 1e-6):
        raise ValueError(f"{name} define a degenerate quadrilateral")
    if np.any(turns > 0) and np.any(turns < 0):
        raise ValueError(f"{name} outline crosses itself; order them TL, TR, BR, BL")
    if turns[0] < 0:
        raise ValueError(f"{name} wind the wrong way; order them TL, TR, BR, BL")


def quad_skew(corners_rc: np.ndarray | list[list[float]]) -> dict[str, float]:
    """Measure how far a board quadrilateral departs from a square-on view.

    A board photographed along its normal images as a rectangle: both pairs of
    opposite edges are equal and every corner is 90 degrees.  Out-of-plane tilt
    keystones it — the near edge grows, the far edge shrinks — so the two edge
    ratios and the corner angles say how far off-angle the shot was, and
    therefore how much work the homography in :func:`rectify_quad` is doing.

    Returns ``keystone_top_bottom`` and ``keystone_left_right`` (both 1.0 when
    square-on) and ``max_corner_deviation_deg`` (0.0 when square-on).  In-plane
    rotation alone leaves all three at their square-on values.
    """
    quad = np.asarray(corners_rc, dtype=np.float64)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("corners_rc must contain four finite (row, col) pairs")
    top, right, bottom, left = (
        float(np.linalg.norm(quad[b] - quad[a])) for a, b in ((0, 1), (1, 2), (3, 2), (0, 3))
    )
    if min(top, right, bottom, left) < 1e-9:
        raise ValueError("corners_rc define a degenerate quadrilateral")

    edges = np.roll(quad, -1, axis=0) - quad
    incoming = -np.roll(edges, 1, axis=0)
    cosines = np.sum(incoming * edges, axis=1) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(edges, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return {
        "keystone_top_bottom": max(top, bottom) / min(top, bottom),
        "keystone_left_right": max(left, right) / min(left, right),
        "max_corner_deviation_deg": float(np.abs(angles - 90.0).max()),
    }


def rectify_quad(
    frame: np.ndarray,
    corners_rc: np.ndarray | list[list[float]] | tuple[tuple[float, float], ...],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Rectify a tilted/perspective board quadrilateral to a rectangle.

    ``corners_rc`` contains the board corners in the source frame, ordered as
    top-left, top-right, bottom-right, bottom-left.  Coordinates are expressed
    as ``(row, col)`` after any configured 90-degree rotation.  The returned
    image has ``output_size == (height, width)`` and is sampled bilinearly.

    A full projective homography is used rather than a rotated bounding box, so
    the same operation corrects both in-plane rotation and trapezoidal
    perspective distortion.  Applying one fixed homography to every frame in a
    stationary recording keeps all thermal pixels in a common coordinate
    system.
    """
    if frame.ndim != 2:
        raise ValueError(f"rectify_quad expects a 2-D frame, got shape {frame.shape}")

    height, width = (int(output_size[0]), int(output_size[1]))
    if height < 2 or width < 2:
        raise ValueError(f"output_size must be at least 2x2, got {output_size}")

    corners = np.asarray(corners_rc, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("corners_rc must contain four finite (row, col) pairs")

    # Work in conventional (x=col, y=row) coordinates, then check the corner
    # order before solving.  Total signed area is not enough: a reversed
    # winding and a self-intersecting ("bow-tie") order both still solve to a
    # valid-looking homography, and silently deliver a mirrored or shredded
    # board instead of failing.
    source_xy = corners[:, ::-1]
    _check_quad_order(source_xy, "corners_rc")

    destination_xy = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0],
         [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )

    # Solve the inverse homography directly (destination -> source).  Inverse
    # mapping avoids holes and gives every output pixel one bilinear sample.
    rows: list[list[float]] = []
    values: list[float] = []
    for (dx, dy), (sx, sy) in zip(destination_xy, source_xy, strict=True):
        rows.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sx * dy])
        values.append(sx)
        rows.append([0.0, 0.0, 0.0, dx, dy, 1.0, -sy * dx, -sy * dy])
        values.append(sy)
    try:
        coeff = np.linalg.solve(np.asarray(rows), np.asarray(values))
    except np.linalg.LinAlgError as exc:
        raise ValueError("corners_rc do not define a stable homography") from exc
    homography = np.asarray(
        [[coeff[0], coeff[1], coeff[2]],
         [coeff[3], coeff[4], coeff[5]],
         [coeff[6], coeff[7], 1.0]],
        dtype=np.float64,
    )

    dest_y, dest_x = np.indices((height, width), dtype=np.float64)
    homogeneous = homography @ np.stack(
        [dest_x.ravel(), dest_y.ravel(), np.ones(height * width)], axis=0
    )
    denominator = homogeneous[2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("homography maps part of the output to infinity")
    source_x = (homogeneous[0] / denominator).reshape(height, width)
    source_y = (homogeneous[1] / denominator).reshape(height, width)
    if (
        source_x.min() < -0.5
        or source_y.min() < -0.5
        or source_x.max() > frame.shape[1] - 0.5
        or source_y.max() > frame.shape[0] - 0.5
    ):
        raise ValueError("corners_rc extend outside the source frame")

    rectified = map_coordinates(
        frame.astype(np.float64, copy=False),
        [source_y, source_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return rectified.astype(np.float32)


def rotated_rect_corners(
    center_rc: tuple[float, float] | list[float],
    size_hw: tuple[float, float] | list[float],
    angle_deg: float,
) -> np.ndarray:
    """Return TL, TR, BR, BL corners of one rigid rotated rectangle.

    ``angle_deg`` is the image-coordinate angle of the top edge: positive angles
    slope down toward the right.  Unlike an arbitrary quadrilateral, opposite
    edges are guaranteed parallel and adjacent edges perpendicular.
    """
    center = np.asarray(center_rc, dtype=np.float64)
    size = np.asarray(size_hw, dtype=np.float64)
    if center.shape != (2,) or size.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_rc and size_hw must each contain two finite values")
    if not np.isfinite(size).all() or np.any(size <= 0) or not np.isfinite(angle_deg):
        raise ValueError("size_hw must be positive and angle_deg must be finite")
    angle = np.deg2rad(float(angle_deg))
    horizontal_rc = np.asarray([np.sin(angle), np.cos(angle)])
    vertical_rc = np.asarray([np.cos(angle), -np.sin(angle)])
    half_h, half_w = size / 2.0
    return np.asarray(
        [center - half_h * vertical_rc - half_w * horizontal_rc,
         center - half_h * vertical_rc + half_w * horizontal_rc,
         center + half_h * vertical_rc + half_w * horizontal_rc,
         center + half_h * vertical_rc - half_w * horizontal_rc]
    )


def crop_rotated_rect(
    frame: np.ndarray,
    center_rc: tuple[float, float] | list[float],
    size_hw: tuple[float, float] | list[float],
    angle_deg: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    """De-rotate and crop one rectangle using an affine transform only.

    This intentionally cannot correct a trapezoid or move four corners
    independently.  It applies one rigid rotation plus rectangular resampling,
    which is preferable when perspective distortion is negligible.
    """
    corners = rotated_rect_corners(center_rc, size_hw, angle_deg)
    return rectify_quad(frame, corners, output_size)


def _fit_edge_line(
    derivative: np.ndarray,
    regions: list[list[int]] | tuple[tuple[int, int, int, int], ...],
    horizontal: bool,
) -> tuple[float, float]:
    """Fit one PCB edge as y=ax+b (horizontal) or x=ay+b (vertical)."""
    independent: list[np.ndarray] = []
    dependent: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    for region in regions:
        if len(region) != 4:
            raise ValueError("each edge-fit region must be [row0, row1, col0, col1]")
        row0, row1, col0, col1 = (int(value) for value in region)
        if not (0 <= row0 < row1 <= derivative.shape[0]
                and 0 <= col0 < col1 <= derivative.shape[1]):
            raise ValueError(f"edge-fit region {region} lies outside {derivative.shape}")
        patch = np.abs(derivative[row0:row1, col0:col1])
        if horizontal:
            peak = np.argmax(patch, axis=0)
            indices = np.arange(col1 - col0)
            independent.append(np.arange(col0, col1, dtype=np.float64))
            dependent.append((row0 + peak).astype(np.float64))
            strengths.append(patch[peak, indices])
        else:
            peak = np.argmax(patch, axis=1)
            indices = np.arange(row1 - row0)
            independent.append(np.arange(row0, row1, dtype=np.float64))
            dependent.append((col0 + peak).astype(np.float64))
            strengths.append(patch[indices, peak])

    x = np.concatenate(independent)
    y = np.concatenate(dependent)
    weight = np.concatenate(strengths).astype(np.float64)
    if len(x) < 4 or float(weight.max(initial=0.0)) <= 0:
        raise ValueError("edge-fit regions contain too little thermal-edge information")

    keep = np.ones(len(x), dtype=bool)
    slope = intercept = 0.0
    for _ in range(8):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        sqrt_weight = np.sqrt(np.clip(weight[keep], 1e-9, None))
        weighted_design = design * sqrt_weight[:, None]
        weighted_y = y[keep] * sqrt_weight
        slope, intercept = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        residual = y - (slope * x + intercept)
        center = float(np.median(residual[keep]))
        mad = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        updated = np.abs(residual - center) <= max(1.25, 2.5 * mad)
        if updated.sum() < 4 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def fit_board_quad(
    frame: np.ndarray,
    edge_bands: dict[str, list[list[int]]],
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Fit the four physical PCB edges inside user-supplied reliable bands.

    This deliberately does *not* segment the hottest connected component.
    Dark connectors and warm cable halos can dominate such a segmentation.
    Instead, each PCB edge is fitted only from one or more narrow, reliable
    regions.  Multiple disjoint regions are useful for skipping ports/cables.

    Returns four floating-point ``(row, col)`` intersections ordered TL, TR,
    BR, BL, ready for :func:`rectify_quad`.
    """
    if frame.ndim != 2:
        raise ValueError(f"fit_board_quad expects a 2-D frame, got shape {frame.shape}")
    missing = {"top", "right", "bottom", "left"} - set(edge_bands)
    if missing:
        raise ValueError(f"edge_bands missing edges: {sorted(missing)}")

    smooth = gaussian_filter(frame.astype(np.float64), float(smooth_sigma))
    grad_row, grad_col = np.gradient(smooth)
    top = _fit_edge_line(grad_row, edge_bands["top"], horizontal=True)
    bottom = _fit_edge_line(grad_row, edge_bands["bottom"], horizontal=True)
    left = _fit_edge_line(grad_col, edge_bands["left"], horizontal=False)
    right = _fit_edge_line(grad_col, edge_bands["right"], horizontal=False)

    def intersection(horizontal_line: tuple[float, float],
                     vertical_line: tuple[float, float]) -> tuple[float, float]:
        # row = ah*col+bh; col = av*row+bv
        ah, bh = horizontal_line
        av, bv = vertical_line
        denominator = 1.0 - av * ah
        if abs(denominator) < 1e-9:
            raise ValueError("fitted PCB edges do not form a finite quadrilateral")
        col = (av * bh + bv) / denominator
        row = ah * col + bh
        return row, col

    corners = np.asarray(
        [intersection(top, left), intersection(top, right),
         intersection(bottom, right), intersection(bottom, left)],
        dtype=np.float64,
    )
    # Reuse rectification validation without doing a resample.  An edge that
    # drifts out of its band latches onto the wrong gradient, which shows up
    # here as a crossed or backwards outline rather than as a plausible crop.
    if not np.isfinite(corners).all():
        raise ValueError("fitted PCB corner coordinates are not finite")
    _check_quad_order(corners[:, ::-1], "fitted PCB corners")
    return corners


def _gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(frame.astype(np.float64))
    return np.hypot(gy, gx)


def _zncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-normalized cross-correlation of `template` slid over `image` ('valid')."""
    t = template - template.mean()
    t_norm = float(np.sqrt((t * t).sum()))
    out_shape = (image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1)
    if t_norm < 1e-12 or min(out_shape) < 1:
        return np.zeros(out_shape)
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")
    ones = np.ones_like(t)
    win_sum = fftconvolve(image, ones, mode="valid")
    win_sumsq = fftconvolve(image * image, ones, mode="valid")
    win_var = np.clip(win_sumsq - win_sum**2 / t.size, 1e-12, None)
    return num / (np.sqrt(win_var) * t_norm)


@dataclass
class Registration:
    rot90: int  # number of CCW 90° rotations applied to the raw frame before cropping
    row0: int
    col0: int
    zncc: float
    candidates: dict[int, tuple[int, int, float]]  # rot90 -> (row0, col0, zncc)


def register_crop(
    frame: np.ndarray,
    ref_patch: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> Registration:
    """Register a re-staged frame against a reference crop.

    Gradient-magnitude ZNCC template matching over the four rotation hypotheses;
    gradients are level-invariant, so a cold (near-isothermal) board can still be
    matched against a hot reference through its edge/component pattern.
    """
    template = _gradient_magnitude(ref_patch)
    candidates: dict[int, tuple[int, int, float]] = {}
    for k in range(4):
        rotated = np.rot90(frame, k)
        if rotated.shape[0] < size[0] or rotated.shape[1] < size[1]:
            continue
        zmap = _zncc_map(_gradient_magnitude(rotated), template)
        r, c = np.unravel_index(int(np.argmax(zmap)), zmap.shape)
        candidates[k] = (int(r), int(c), float(zmap[r, c]))
    best_k = max(candidates, key=lambda k: candidates[k][2])
    r, c, score = candidates[best_k]
    return Registration(rot90=best_k, row0=r, col0=c, zncc=score, candidates=candidates)
