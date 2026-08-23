"""Figures and animations.

Every title carries the qualifier that belongs to the quantity -- *effective*,
*normalized*, *relative*, *candidate* -- and every colourbar carries units in
the sequence's own time unit.  That is not decoration: a diffusivity proxy
plotted without the word "proxy" is the exact mislabelling the harness forbids,
and a figure outlives the report it was made for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tdp.dynamic_inverse.regions import expand_to_pixels  # noqa: E402

PHASE_COLOURS = {
    "heating": "#e8a33d",
    "near_steady": "#7fb069",
    "free_cooling_candidate": "#4c8fc0",
    "invalid_transition": "#c05a5a",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _length_label(unit: str, pixel_size_mm: float | None) -> str:
    return "px" if pixel_size_mm is None else "px  (1 px = %.4f mm)" % pixel_size_mm


def plot_phase_detection(phases, times, out: Path, unit: str = "s", extra: dict | None = None) -> Path:
    """Mean excess temperature with the detected phases shaded behind it."""
    scale, axis_label = (60.0, "minutes") if unit == "s" else (1.0, "frames")
    t = times / scale
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    axes[0].plot(t, phases.mean_theta, color="black", lw=1.4, label="mean excess temperature")
    for segment in phases.segments:
        axes[0].axvspan(
            t[segment["start_index"]],
            t[segment["stop_index"]],
            color=PHASE_COLOURS.get(segment["label"], "#999999"),
            alpha=0.30,
        )
    for name, series in (extra or {}).items():
        axes[0].plot(t, series, lw=0.9, alpha=0.75, label=name)
    axes[0].set_ylabel("mean theta  [K]")
    axes[0].set_title("Detected thermal phases (decay labelled free-cooling CANDIDATE)")
    axes[0].legend(
        handles=[Patch(color=c, alpha=0.4, label=k) for k, c in PHASE_COLOURS.items()]
        + list(axes[0].get_legend_handles_labels()[0]),
        fontsize=7,
        loc="upper right",
        ncol=2,
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, phases.slope, color="#4c8fc0", lw=1.0, label=f"d(mean theta)/dt  [K/{unit}]")
    axes[1].axhline(0.0, color="black", lw=0.6)
    axes[1].plot(t, phases.positive_fraction, color="#c05a5a", lw=0.9, label="fraction still warming")
    axes[1].set_xlabel(axis_label)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    return _save(fig, out)


def plot_region_partition(sequence, partition, out: Path, frame_index: int | None = None) -> Path:
    from skimage.segmentation import mark_boundaries

    if frame_index is None:
        frame_index = int(np.argmax(sequence.board_mean_theta()))
    base = sequence.temperature_c[frame_index].astype(np.float64)
    norm = (base - np.nanmin(base)) / max(np.ptp(base), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    axes[0].imshow(mark_boundaries(np.stack([norm] * 3, -1), partition.labels, color=(0, 1, 1), mode="thick"))
    axes[0].set_title(f"Region partition ({partition.n_regions} regions, {partition.graph.n_edges} edges)")
    im = axes[1].imshow(sequence.emissivity, cmap="magma", vmin=0.2, vmax=1.0)
    axes[1].set_title("Fitted effective emissivity (diagnostic)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    im = axes[2].imshow(np.where(sequence.valid_mask, sequence.trust_weight, np.nan), cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Observation confidence\n(1 = measured, low = reconstructed)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return _save(fig, out)


def plot_coupling_graph(partition, edge_rate: np.ndarray, out: Path, unit: str = "s") -> Path:
    """The effective thermal coupling network, drawn on the board."""
    graph = partition.graph
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    ax.imshow(expand_to_pixels(np.zeros(graph.n_nodes), partition.labels, fill=np.nan), cmap="gray")
    strength = edge_rate / max(edge_rate.max(), 1e-12)
    order = np.argsort(strength)
    cmap = plt.get_cmap("inferno")
    for e in order:
        i, j = graph.edge_index[:, e]
        y = [graph.centroids_px[i, 0], graph.centroids_px[j, 0]]
        x = [graph.centroids_px[i, 1], graph.centroids_px[j, 1]]
        ax.plot(x, y, color=cmap(strength[e]), lw=0.4 + 3.4 * strength[e], alpha=0.85, solid_capstyle="round")
    ax.scatter(graph.centroids_px[:, 1], graph.centroids_px[:, 0], s=7, color="#3aa0ff", zorder=3)
    ax.set_title(f"Effective thermal coupling network  a_ij  [1/{unit}]\n(region-resolution; not a conductance)")
    ax.set_xticks([])
    ax.set_yticks([])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, float(edge_rate.max())))
    plt.colorbar(mappable, ax=ax, fraction=0.046, label=f"a_ij  [1/{unit}]")
    return _save(fig, out)


def plot_parameter_map(
    values: np.ndarray,
    partition,
    out: Path,
    title: str,
    label: str,
    cmap: str = "viridis",
    log: bool = False,
    pixel_size_mm: float | None = None,
) -> Path:
    grid = expand_to_pixels(values, partition.labels)
    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    norm = matplotlib.colors.LogNorm() if log and np.nanmin(grid) > 0 else None
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(_length_label("px", pixel_size_mm), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label=label)
    return _save(fig, out)


def plot_source_peak(
    source_rate: np.ndarray, partition, out: Path, unit: str = "s", powered: np.ndarray | None = None
) -> Path:
    """Peak normalized source rate over the powered interval."""
    block = source_rate if powered is None else source_rate[powered]
    peak = block.max(axis=0)
    return plot_parameter_map(
        peak,
        partition,
        out,
        f"Peak normalized source rate  s = q/C  [K/{unit}]\n(region-resolution; NOT power)",
        f"s  [K/{unit}]",
        cmap="inferno",
    )


def plot_uncertainty(results: dict[str, Any], partition, out: Path, unit: str = "s") -> Path:
    """Mean, spread and coefficient of variation for the cooling rate."""
    panels = []
    for key, result in results.items():
        if "beta" in result.mean:
            panels.append((key, result.mean["beta"], result.std["beta"], result.cv["beta"], False))
        elif "beta_map" in result.mean:
            panels.append((key, result.mean["beta_map"], result.std["beta_map"], result.cv["beta_map"], True))
    if not panels:
        raise ValueError("no uncertainty results to plot")
    fig, axes = plt.subplots(len(panels), 3, figsize=(12, 4.0 * len(panels)), squeeze=False)
    for row, (name, mean, std, cv, is_map) in enumerate(panels):
        grids = [
            (mean, f"mean beta  [1/{unit}]", "viridis"),
            (std, f"std beta  [1/{unit}]", "magma"),
            (np.clip(cv, 0, 2), "coefficient of variation", "coolwarm"),
        ]
        for col, (values, label, cmap) in enumerate(grids):
            grid = values if is_map else expand_to_pixels(values, partition.labels)
            im = axes[row, col].imshow(grid, cmap=cmap)
            axes[row, col].set_title(f"{name}\n{label}", fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle("Effective cooling-rate uncertainty (high CV = not stably identifiable)", fontsize=11)
    return _save(fig, out)


def plot_rollout_check(
    series, thermal, predicted: np.ndarray, block: tuple[int, int], out: Path, unit: str = "s"
) -> Path:
    """Held-out free rollout against the measurement, board mean and worst regions."""
    lo, hi = block
    scale, axis_label = (60.0, "minutes") if unit == "s" else (1.0, "frames")
    t = series.times[lo : hi + 1] / scale
    target = series.theta[lo : hi + 1]
    weight = series.weight[lo : hi + 1]
    mean_target = (target * weight).sum(1) / np.maximum(weight.sum(1), 1e-9)
    mean_pred = (predicted * weight).sum(1) / np.maximum(weight.sum(1), 1e-9)
    residual = np.abs(predicted - target).mean(axis=0)
    worst = np.argsort(-residual)[:3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t, mean_target, "k-", lw=1.6, label="measured")
    axes[0].plot(t, mean_pred, "--", color="#4c8fc0", lw=1.6, label="free rollout")
    axes[0].set_title("Held-out free rollout, board mean")
    axes[0].set_xlabel(axis_label)
    axes[0].set_ylabel("theta  [K]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for k in worst:
        axes[1].plot(t, target[:, k], lw=1.2, label=f"region {k} measured")
        axes[1].plot(t, predicted[:, k], "--", lw=1.2, label=f"region {k} rollout")
    axes[1].set_title("Worst three regions")
    axes[1].set_xlabel(axis_label)
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(alpha=0.3)
    return _save(fig, out)


def plot_emissivity_repair(sequence, out: Path) -> Path:
    """What the low-emissivity repair changed, and where it is an extrapolation."""
    hot = int(np.argmax(sequence.board_mean_theta()))
    raw = sequence.raw_temperature_c[hot]
    repaired = sequence.temperature_c[hot]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    vmin, vmax = float(np.nanmin(raw)), float(np.nanmax(repaired))
    for ax, (img, title, cmap, kw) in zip(
        axes,
        [
            (raw, "As measured (camera emissivity assumed)", "inferno", dict(vmin=vmin, vmax=vmax)),
            (repaired, "After low-emissivity repair", "inferno", dict(vmin=vmin, vmax=vmax)),
            (repaired - raw, "Repair lift  [K]", "coolwarm", dict(vmin=-25, vmax=25)),
            (sequence.repair_mask.astype(float), "Repaired footprints", "gray", {}),
        ],
        strict=True,
    ):
        im = ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def animate_source(
    source_rate: np.ndarray,
    partition,
    out: Path,
    times: np.ndarray,
    unit: str = "s",
    montage_fallback: Path | None = None,
) -> dict[str, str]:
    """Animated normalized source map, with a keyframe montage as fallback.

    The animation depends on a writer that may not be installed.  Rather than
    failing the run, the montage and the ``.npz`` are always written, so the
    numbers survive even when the movie does not.
    """
    written: dict[str, str] = {}
    peak = float(np.nanmax(source_rate)) if source_rate.size else 1.0
    frames = np.linspace(0, len(source_rate) - 1, min(12, len(source_rate))).astype(int)
    scale = 60.0 if unit == "s" else 1.0
    suffix = "min" if unit == "s" else "frame"

    montage = montage_fallback or out.with_name(out.stem + "_montage.png")
    cols = 6
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 3.0 * rows), squeeze=False)
    for ax, index in zip(axes.ravel(), frames, strict=False):
        ax.imshow(expand_to_pixels(source_rate[index], partition.labels), cmap="inferno", vmin=0, vmax=peak)
        ax.set_title(f"t={times[index] / scale:.1f} {suffix}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.suptitle(f"Normalized source rate  s = q/C  [K/{unit}]  (region-resolution; NOT power)", fontsize=10)
    written["montage"] = str(_save(fig, montage))

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(4.4, 6.2))
        image = ax.imshow(expand_to_pixels(source_rate[0], partition.labels), cmap="inferno", vmin=0, vmax=peak)
        title = ax.set_title("", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(image, ax=ax, fraction=0.046, label=f"s  [K/{unit}]")

        step = max(1, len(source_rate) // 120)
        indices = range(0, len(source_rate), step)

        def update(index):
            image.set_data(expand_to_pixels(source_rate[index], partition.labels))
            title.set_text(f"Normalized source rate  t={times[index] / scale:.1f} {suffix}")
            return image, title

        animation = FuncAnimation(fig, update, frames=indices, blit=False)
        gif = out.with_suffix(".gif")
        gif.parent.mkdir(parents=True, exist_ok=True)
        animation.save(gif, writer=PillowWriter(fps=8))
        plt.close(fig)
        written["animation"] = str(gif)
    except Exception as exc:  # writer missing or failed -- keep the montage
        written["animation_error"] = f"{type(exc).__name__}: {exc}"
    return written
