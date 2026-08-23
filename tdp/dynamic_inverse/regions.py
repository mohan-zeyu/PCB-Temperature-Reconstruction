"""Region partition and region graph.

Per-pixel thermal parameters are not identifiable from this data and the
appearance of the board does not reveal what is under it -- the copper pour and
the vias that actually carry the heat are buried, and the surface only shows
where components sit.  So the field is reduced to regions and one parameter set
is fitted per region.  The region count is a stated resolution, not a hidden
detail: :func:`build_regions` is run at several counts and the report shows how
much the conclusions move.

Segmentation uses whatever modalities the recording actually has.  There is no
visible-light image in this campaign -- the ``photo/*.jpeg`` files are the
camera's own colourised renderings of the same thermal data, not an RGB view --
so the feature stack is the *dynamics* (steady level, peak, decay rate) plus
the fitted emissivity map, which is the one channel that distinguishes a metal
lid from the board it sits on.

Aggregation is confidence-weighted throughout.  A pixel reconstructed in the
middle of a connector footprint must not count as much as one the camera
measured directly, and ``ThermalSequence.trust_weight`` carries exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from tdp.dynamic_inverse.data import ThermalSequence
from tdp.dynamic_inverse.physics import RegionGraph


@dataclass
class RegionPartition:
    labels: np.ndarray  # (H, W) int, 0 = outside the partition
    graph: RegionGraph
    feature_names: list[str]
    report: dict[str, Any]

    @property
    def n_regions(self) -> int:
        return self.graph.n_nodes


def _feature_stack(sequence: ThermalSequence) -> tuple[np.ndarray, list[str]]:
    """Channels a segmentation can legitimately use, each standardised."""
    theta = sequence.theta
    peak = np.nanmax(theta, axis=0)
    final = theta[-1]
    span = peak - final
    with np.errstate(invalid="ignore", divide="ignore"):
        gradient = np.gradient(theta, sequence.timestamps_s, axis=0)
    fastest_fall = np.nanmin(gradient, axis=0)
    channels = [peak, final, span, fastest_fall, sequence.emissivity.astype(np.float64)]
    names = ["peak_theta", "final_theta", "theta_span", "fastest_fall", "emissivity"]
    stacked = np.stack(channels, axis=-1)
    valid = sequence.valid_mask
    for k in range(stacked.shape[-1]):
        channel = stacked[..., k]
        inside = channel[valid]
        centre = float(np.nanmean(inside)) if inside.size else 0.0
        scale = float(np.nanstd(inside)) if inside.size else 1.0
        stacked[..., k] = (channel - centre) / max(scale, 1e-6)
    return np.nan_to_num(stacked), names


def _grid_labels(mask: np.ndarray, n_regions: int) -> np.ndarray:
    """Regular-grid fallback partition, sized to hit ``n_regions`` inside the mask."""
    height, width = mask.shape
    aspect = height / width
    cols = max(1, int(round(np.sqrt(n_regions / aspect))))
    rows = max(1, int(round(n_regions / cols)))
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    col_edges = np.linspace(0, width, cols + 1).astype(int)
    labels = np.zeros((height, width), dtype=np.int64)
    next_id = 1
    for r0, r1 in zip(row_edges[:-1], row_edges[1:], strict=True):
        for c0, c1 in zip(col_edges[:-1], col_edges[1:], strict=True):
            block = np.zeros_like(mask)
            block[r0:r1, c0:c1] = True
            block &= mask
            if block.any():
                labels[block] = next_id
                next_id += 1
    return labels


def _relabel_consecutive(labels: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop specks, split disconnected labels, renumber 1..N."""
    out = np.zeros_like(labels)
    next_id = 1
    for value in np.unique(labels):
        if value == 0:
            continue
        pieces, count = ndi.label(labels == value)
        for piece in range(1, count + 1):
            block = pieces == piece
            if block.sum() < min_pixels:
                continue
            out[block] = next_id
            next_id += 1
    return out


def _adjacency(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges and shared-boundary lengths from a label image."""
    pairs: dict[tuple[int, int], int] = {}
    for axis in (0, 1):
        left = labels if axis == 0 else labels
        a = np.take(left, np.arange(labels.shape[axis] - 1), axis=axis)
        b = np.take(left, np.arange(1, labels.shape[axis]), axis=axis)
        differ = (a != b) & (a > 0) & (b > 0)
        for lo, hi in zip(a[differ], b[differ], strict=True):
            key = (int(min(lo, hi)), int(max(lo, hi)))
            pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0)
    keys = sorted(pairs)
    edge_index = np.asarray(keys, dtype=np.int64).T - 1  # labels are 1-based
    lengths = np.asarray([pairs[k] for k in keys], dtype=np.float64)
    return edge_index, lengths


def build_regions(
    sequence: ThermalSequence,
    n_regions: int = 64,
    method: str = "slic",
    compactness: float = 4.0,
    use_emissivity: bool = True,
    min_valid_fraction: float = 0.25,
    seed: int = 0,
    jitter: float = 0.0,
) -> RegionPartition:
    """Partition the board and build the region graph.

    SLIC is deterministic given its inputs, so ``seed`` alone would produce the
    same partition every time.  ``jitter`` perturbs the (standardised) feature
    stack by that fraction of a standard deviation before segmenting, which
    moves boundaries without changing what the channels mean.  That is what the
    uncertainty estimate uses to ask whether a result survives being cut up
    slightly differently.
    """
    valid = sequence.valid_mask
    if not valid.any():
        raise ValueError("sequence has no valid pixels to partition")

    features, names = _feature_stack(sequence)
    if not use_emissivity:
        keep = [k for k, name in enumerate(names) if name != "emissivity"]
        features, names = features[..., keep], [names[k] for k in keep]
    if jitter > 0:
        rng = np.random.default_rng(seed)
        features = features + rng.normal(0.0, float(jitter), features.shape)

    if method == "slic":
        from skimage.segmentation import slic

        labels = slic(
            features,
            n_segments=int(n_regions),
            compactness=float(compactness),
            channel_axis=-1,
            mask=valid,
            start_label=1,
            enforce_connectivity=True,
        ).astype(np.int64)
    elif method == "grid":
        labels = _grid_labels(valid, int(n_regions))
    else:
        raise ValueError(f"method must be 'slic' or 'grid', got {method!r}")

    total_pixels = int(valid.sum())
    min_pixels = max(4, int(0.15 * total_pixels / max(n_regions, 1)))
    labels = _relabel_consecutive(labels * valid, min_pixels)
    n_found = int(labels.max())
    if n_found < 2:
        raise ValueError(f"partition produced {n_found} usable regions; lower n_regions")

    edge_index, edge_length = _adjacency(labels)

    trust = np.where(valid, sequence.trust_weight, 0.0)
    centroids = np.zeros((n_found, 2))
    areas = np.zeros(n_found)
    emissivity = np.zeros(n_found)
    valid_fraction = np.zeros(n_found)
    mean_trust = np.zeros(n_found)
    index = np.arange(1, n_found + 1)
    rows, cols = np.indices(labels.shape)
    for k, value in enumerate(index):
        block = labels == value
        weight = trust[block]
        total = weight.sum()
        areas[k] = float(block.sum())
        valid_fraction[k] = float(valid[block].mean())
        mean_trust[k] = float(weight.mean())
        if total > 0:
            centroids[k] = [
                float((rows[block] * weight).sum() / total),
                float((cols[block] * weight).sum() / total),
            ]
            emissivity[k] = float((sequence.emissivity[block] * weight).sum() / total)
        else:
            centroids[k] = [float(rows[block].mean()), float(cols[block].mean())]
            emissivity[k] = float(sequence.emissivity[block].mean())

    theta = sequence.theta
    node_theta = aggregate_series(sequence, labels)[0]
    node_features = np.stack(
        [
            np.log1p(areas),
            centroids[:, 0] / labels.shape[0],
            centroids[:, 1] / labels.shape[1],
            emissivity,
            valid_fraction,
            mean_trust,
            np.nanmax(node_theta, axis=0),
            node_theta[-1],
        ],
        axis=1,
    ).astype(np.float32)

    graph = RegionGraph(
        n_nodes=n_found,
        edge_index=edge_index,
        centroids_px=centroids,
        areas_px=areas,
        edge_length_px=edge_length,
        node_features=np.nan_to_num(node_features),
    )
    low_confidence = valid_fraction < float(min_valid_fraction)
    report = {
        "method": method,
        "n_regions_requested": int(n_regions),
        "n_regions_found": n_found,
        "n_edges": int(graph.n_edges),
        "compactness": float(compactness),
        "min_pixels_per_region": int(min_pixels),
        "median_area_px": float(np.median(areas)),
        "smallest_area_px": float(areas.min()),
        "feature_channels": names,
        "used_emissivity_channel": bool(use_emissivity),
        "feature_jitter": float(jitter),
        "seed": int(seed),
        "mean_trust_weight": float(mean_trust.mean()),
        "low_confidence_regions": [int(i) for i in np.nonzero(low_confidence)[0]],
        "rgb_available": False,
        "rgb_note": (
            "photo/*.jpeg are the camera's colourised renderings of the same thermal "
            "frames, not a visible-light view, so no RGB modality is available"
        ),
    }
    del theta
    return RegionPartition(labels=labels, graph=graph, feature_names=names, report=report)


def aggregate_series(
    sequence: ThermalSequence, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-weighted region means.

    Returns ``(theta (T, N), weight (T, N), trust (N,))``.  ``weight`` is the
    total confidence behind each region-mean, so a region built mostly from
    reconstructed pixels carries less of the loss than one the camera measured;
    ``trust`` is the same quantity normalised per region, used by the loss term
    that pulls hardest on directly-measured data.
    """
    valid = sequence.valid_mask
    confidence = np.where(valid, sequence.trust_weight, 0.0)
    n_regions = int(labels.max())
    theta = sequence.theta
    flat_labels = labels.ravel()
    flat_conf = confidence.ravel()
    keep = flat_labels > 0
    index = flat_labels[keep] - 1
    weights = flat_conf[keep]

    totals = np.bincount(index, weights=weights, minlength=n_regions)
    counts = np.bincount(index, minlength=n_regions).astype(np.float64)
    out = np.zeros((sequence.n_frames, n_regions), dtype=np.float64)
    for t in range(sequence.n_frames):
        values = theta[t].ravel()[keep]
        accumulated = np.bincount(index, weights=values * weights, minlength=n_regions)
        out[t] = np.where(totals > 0, accumulated / np.maximum(totals, 1e-9), np.nan)

    weight = np.broadcast_to(totals[None, :], out.shape).copy()
    trust = np.where(counts > 0, totals / np.maximum(counts, 1e-9), 0.0)
    return out, weight, trust


def expand_to_pixels(values: np.ndarray, labels: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Paint a per-region value back onto the board grid.

    The result is a *region-resolution* map drawn at pixel resolution; the
    spatial resolution of every downstream map is the partition, not the
    camera, and every figure that uses this says so.
    """
    values = np.asarray(values)
    out = np.full(labels.shape, fill, dtype=np.float64)
    inside = labels > 0
    out[inside] = values[labels[inside] - 1]
    return out
