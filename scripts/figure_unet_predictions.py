"""Show the correction net's Raspberry Pi 4B output, not just error tables.

Produces, for the case10 full-load powered stretch (240 frames):

* prediction_montage_case10.png  - maps at key frames: bare input, U-Net
  output, painted target, U-Net error, Stage-2 error (theta, K);
* bare_vs_unet_over_time_case10_part1.png / part2.png - the absolute-temperature
  bare-board camera map and U-Net reconstruction through the powered phase,
  split into four timestamps per figure for legibility;
* prediction_video_case10.mp4   - all 240 frames, four panels (GIF fallback);
* hotspot_timeseries_case10.png - SoC-lid core and trusted-hotspot temperature
  through the powered phase, in degrees Celsius, for the bare reading, the
  U-Net output, the painted target and the Stage-2 physics reconstruction;
* predicted_steady_map_case10.png - the final steady field in Celsius with the
  footprint outline and the two key locations marked;
* predicted_maps_case10.npz / predicted_summary_case10.csv - the actual
  predicted fields (theta and Celsius) and per-frame summaries.

The operator is the two-load Raspberry Pi model (case10<->case17 full load plus
case09<->case16 half load), which keeps the full-load masked error at 0.10 K.
Celsius is theta plus the case10 frame's own ambient; the painted target is the
high-emissivity twin's reading at the same state key, so it carries the
session-level offset that the trusted-pixel tables already report.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

OUT = Path("outputs/thermal/unet_case10/presentation")
PREDICTIONS = Path("outputs/thermal/unet_case10/train_both/predictions_full_load.npz")
CASE10 = Path("outputs/thermal/case10/reconstructed_sequence.npz")
STEADY_LAST = 25
KEY_FRAMES = [0, 40, 80, 130, 190, 239]
TIME_SEQUENCE_FRAMES = [0, 12, 30, 60, 100, 150, 200, 239]
CHINESE_FONT = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")


def load_inputs():
    with np.load(PREDICTIONS) as p:
        predicted = p["predicted_theta_k"].astype(np.float64)
        target = p["target_theta_k"].astype(np.float64)
        bare = p["input_theta_k"].astype(np.float64)
        board = p["board_mask"]
        unknown = p["unknown_mask"]
    with np.load(CASE10) as c:
        stage2 = c["theta_k"].astype(np.float64)
        times = c["times_s"].astype(np.float64)
        ambient = c["ambient_c"].astype(np.float64)
        anchor = c["soc_surface_anchor_c"].astype(np.float64)
    if stage2.shape != predicted.shape:
        raise ValueError("Stage-2 reconstruction does not match the prediction frames")
    return predicted, target, bare, stage2, board, unknown, times, ambient, anchor


def soc_and_hotspot(target: np.ndarray, bare: np.ndarray, unknown: np.ndarray):
    """SoC lid component (contains the painted steady peak) and its hottest quarter.

    The core is the hottest quarter of the lid *by the painted steady field*.
    A distance-to-rim transform is degenerate on this block (it reaches the
    board edge), which earlier selected the whole block and mislabelled it as a
    core; a temperature-ranked quarter has a clear meaning and is identical for
    every predictor being compared.
    """
    steady = target[-STEADY_LAST:].mean(axis=0)
    peak = np.unravel_index(int(np.argmax(np.where(unknown, steady, -np.inf))), steady.shape)
    labels, _ = label(unknown)
    soc = labels == labels[peak]
    core = soc & (steady >= np.quantile(steady[soc], 0.75))
    bare_steady = bare[-STEADY_LAST:].mean(axis=0)
    bare_max = np.unravel_index(int(np.argmax(bare_steady)), bare_steady.shape)
    return soc, core, peak, bare_max


def figures(maps, outdir: Path):
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    cols = [i for i in KEY_FRAMES if i < len(predicted)]
    rows = [
        ("bare input (K)", bare, 0, 60, "inferno"),
        ("U-Net output (K)", predicted, 0, 60, "inferno"),
        ("painted target (K)", target, 0, 60, "inferno"),
        ("U-Net - target (K)", predicted - target, -8, 8, "coolwarm"),
        ("Stage-2 - target (K)", stage2 - target, -8, 8, "coolwarm"),
    ]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.4 * len(cols), 13.5))
    for col, frame in enumerate(cols):
        for row, (title, arr, vmin, vmax, cmap) in enumerate(rows):
            ax = axes[row, col]
            im = ax.imshow(arr[frame], cmap=cmap, vmin=vmin, vmax=vmax)
            if row < 3:
                ax.contour(unknown.astype(int), levels=[0.5], colors="w", linewidths=0.5, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            if row == 0:
                mae = np.abs((predicted - target)[frame][board & unknown]).mean()
                ax.set_title(f"frame {frame} | t={times[frame]:.0f} s\nmasked |err| {mae:.2f} K", fontsize=8)
            if row == 4 and col == 0:
                ax.set_xlabel("")
    fig.suptitle("case10 Raspberry Pi 4B full load: two-load U-Net output vs painted truth", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "prediction_montage_case10.png", dpi=115, bbox_inches="tight")
    plt.close(fig)


def fill_edge_gaps_for_display(field: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Fill rectification gaps outside the valid board for presentation only."""
    if np.all(board):
        return field.copy()
    _, indices = distance_transform_edt(~board, return_indices=True)
    completed = field.copy()
    completed[~board] = field[indices[0][~board], indices[1][~board]]
    smoothed = gaussian_filter(completed, sigma=2.0, mode="nearest")
    completed[~board] = smoothed[~board]
    return completed


def bare_vs_unet_over_time(maps, outdir: Path):
    """Show absolute-temperature input and U-Net output through case10.

    Both rows use the same fixed scale so the visual change comes only from
    time and reconstruction, never from per-panel colour normalisation.  The
    network predicts temperature rise above ambient; case10's measured ambient
    is added back frame by frame for this absolute-temperature presentation.
    """
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    del target, stage2, unknown, anchor
    frames = [frame for frame in TIME_SEQUENCE_FRAMES if frame < len(predicted)]
    if not frames:
        raise ValueError("no requested time-sequence frame exists in the prediction archive")

    vmin, vmax = 25.0, 80.0
    chinese_font = (
        font_manager.FontProperties(fname=CHINESE_FONT)
        if CHINESE_FONT.exists()
        else None
    )
    chinese_text = {"fontproperties": chinese_font} if chinese_font is not None else {}
    cmap = plt.colormaps["inferno"].with_extremes(bad="black")
    frame_groups = [frames[start:start + 4] for start in range(0, len(frames), 4)]
    paths = []
    for part, group in enumerate(frame_groups, start=1):
        fig, axes = plt.subplots(
            2,
            len(group),
            figsize=(3.2 * len(group), 7.6),
            sharex=True,
            sharey=True,
        )
        last_image = None
        for column, frame in enumerate(group):
            for row, (fields, label) in enumerate(
                ((bare, "裸板原始热像（℃）"), (predicted, "残差网络重构（℃）"))
            ):
                ax = axes[row, column]
                absolute_c = fields[frame] + ambient[frame]
                last_image = ax.imshow(
                    fill_edge_gaps_for_display(absolute_c, board),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(
                        f"第 {frame} 帧\n时间 = {times[frame]:.0f} 秒",
                        fontsize=11,
                        **chinese_text,
                    )
                if column == 0:
                    ax.set_ylabel(label, fontsize=12, **chinese_text)

        fig.subplots_adjust(left=0.09, right=0.94, bottom=0.035, top=0.93,
                            wspace=0.06, hspace=0.06)
        cbar = fig.colorbar(last_image, ax=axes, fraction=0.028, pad=0.018, aspect=32)
        cbar.set_label("绝对表面温度（℃）", **chinese_text)
        path = outdir / f"bare_vs_unet_over_time_case10_part{part}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def steady_map_figure(maps, outdir: Path):
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    soc, core, peak, bare_max = soc_and_hotspot(target, bare, unknown)
    frame = len(predicted) - 1
    C = predicted[frame] + ambient[frame]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(C, cmap="inferno")
    ax.contour(unknown.astype(int), levels=[0.5], colors="w", linewidths=0.5, alpha=0.7)
    ax.plot(peak[1], peak[0], "co", ms=10, mfc="none", mew=2)
    ax.annotate(f"SoC lid core\n{C[core].mean():.1f} C", (peak[1], peak[0]),
                xytext=(peak[1] + 6, peak[0] - 6), color="c", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="c"))
    ax.plot(bare_max[1], bare_max[0], "wo", ms=8, mfc="none", mew=1.6)
    ax.annotate(f"trusted hotspot\n{C[bare_max]:.1f} C", (bare_max[1], bare_max[0]),
                xytext=(bare_max[1] + 4, bare_max[0] + 14), color="w", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="w"))
    ax.set_title(f"U-Net predicted steady map, frame {frame} (C, ambient {ambient[frame]:.1f} C)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="temperature (C)")
    fig.tight_layout()
    fig.savefig(outdir / "predicted_steady_map_case10.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return soc, core, peak, bare_max


def hotspot_timeseries(maps, soc, core, peak, bare_max, outdir: Path):
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    series = {
        "bare reading": (bare, np.zeros_like(ambient)),
        "U-Net output": (predicted, np.zeros_like(ambient)),
        "painted target": (target, np.zeros_like(ambient)),
        "Stage-2 physics": (stage2, np.zeros_like(ambient)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharex=True)
    for name, (field, _) in series.items():
        axes[0].plot(times, field[:, core].mean(axis=1) + ambient, label=name, lw=1.5)
        axes[1].plot(times, field[:, bare_max[0], bare_max[1]] + ambient, label=name, lw=1.5)
    axes[0].plot(times, anchor, "k:", lw=1.2, label="die-corrected surface anchor")
    axes[0].axhline(np.median(ambient), color="grey", lw=0.8, ls="--")
    axes[0].annotate("ambient", (times[0], np.median(ambient)), fontsize=7, color="grey")
    axes[0].set_title("SoC-lid core mean (deepest quarter)")
    axes[1].set_title(f"trusted hotspot pixel (r={bare_max[0]}, c={bare_max[1]})")
    for ax in axes:
        ax.set_xlabel("powered time (s)"); ax.set_ylabel("temperature (C)"); ax.legend(fontsize=8)
    fig.suptitle("case10 Raspberry Pi 4B: what the net outputs through the powered phase", y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "hotspot_timeseries_case10.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def video(maps, outdir: Path, fps: int = 12):
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    vmin, vmax = 0.0, 60.0
    artists = []
    for ax, title in zip(axes.ravel(), ["bare input", "U-Net output", "painted target", "U-Net - target"]):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        artists.append(ax)
    ims = [
        axes[0, 0].imshow(bare[0], cmap="inferno", vmin=vmin, vmax=vmax),
        axes[0, 1].imshow(predicted[0], cmap="inferno", vmin=vmin, vmax=vmax),
        axes[1, 0].imshow(target[0], cmap="inferno", vmin=vmin, vmax=vmax),
        axes[1, 1].imshow(predicted[0] - target[0], cmap="coolwarm", vmin=-8, vmax=8),
    ]
    contours = [
        axes[0, 0].contour(unknown.astype(int), levels=[0.5], colors="w", linewidths=0.5),
        axes[0, 1].contour(unknown.astype(int), levels=[0.5], colors="w", linewidths=0.5),
        axes[1, 0].contour(unknown.astype(int), levels=[0.5], colors="w", linewidths=0.5),
    ]
    suptitle = fig.suptitle("", fontsize=12)
    fig.colorbar(ims[1], ax=axes[0, 1], fraction=0.046, label="theta (K)")
    fig.colorbar(ims[3], ax=axes[1, 1], fraction=0.046, label="error (K)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(frame: int):
        for im, arr in zip(ims, [bare[frame], predicted[frame], target[frame],
                                 predicted[frame] - target[frame]]):
            im.set_data(arr)
        mae = np.abs((predicted - target)[frame][board & unknown]).mean()
        suptitle.set_text(
            f"case10 Pi 4B full load | frame {frame} | t={times[frame]:.0f} s | "
            f"masked |err| {mae:.2f} K | theta above ambient (K)"
        )
        return ims + contours + [suptitle]

    status = {"frame_count": len(predicted)}
    try:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib cannot find an ffmpeg writer")
        path = outdir / "prediction_video_case10.mp4"
        anim = animation.FuncAnimation(fig, update, frames=len(predicted), blit=False,
                                       cache_frame_data=False)
        anim.save(path, writer=animation.FFMpegWriter(
            fps=fps, codec="h264", bitrate=2200,
            metadata={"title": "case10 U-Net reconstruction", "artist": "tdp"},
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=100)
        status.update({"status": "mp4 written", "path": str(path), "fps": fps})
    except Exception as error:  # pragma: no cover - depends on system codecs
        stride = max(1, len(predicted) // 120)
        path = outdir / "prediction_video_case10.gif"
        anim = animation.FuncAnimation(fig, update, frames=range(0, len(predicted), stride),
                                       blit=False, cache_frame_data=False)
        anim.save(path, writer=animation.PillowWriter(fps=8), dpi=90)
        status.update({"status": "gif written", "path": str(path), "fps": 8,
                       "mp4_error": f"{type(error).__name__}: {error}"})
    plt.close(fig)
    return status


def save_arrays_and_summary(maps, soc, core, peak, bare_max, outdir: Path):
    predicted, target, bare, stage2, board, unknown, times, ambient, anchor = maps
    grid_c = predicted + ambient[:, None, None]
    np.savez_compressed(
        outdir / "predicted_maps_case10.npz",
        predicted_theta_k=predicted.astype(np.float32),
        predicted_c=grid_c.astype(np.float32),
        bare_theta_k=bare.astype(np.float32),
        target_theta_k=target.astype(np.float32),
        stage2_theta_k=stage2.astype(np.float32),
        ambient_c=ambient, times_s=times,
        board_mask=board, unknown_mask=unknown,
        soc_mask=soc, soc_core_mask=core,
    )
    rows = []
    for frame in range(len(predicted)):
        rows.append({
            "frame": frame,
            "t_s": float(times[frame]),
            "ambient_c": float(ambient[frame]),
            "max_pred_c": float(grid_c[frame][board].max()),
            "soc_core_bare_c": float(bare[frame][core].mean() + ambient[frame]),
            "soc_core_unet_c": float(grid_c[frame][core].mean()),
            "soc_core_target_c": float(target[frame][core].mean() + ambient[frame]),
            "soc_core_stage2_c": float(stage2[frame][core].mean() + ambient[frame]),
            "hotspot_bare_c": float(bare[frame][bare_max[0], bare_max[1]] + ambient[frame]),
            "hotspot_unet_c": float(grid_c[frame][bare_max[0], bare_max[1]]),
            "hotspot_target_c": float(target[frame][bare_max[0], bare_max[1]] + ambient[frame]),
        })
    with (outdir / "predicted_summary_case10.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    maps = load_inputs()
    figures(maps, OUT)
    time_sequence_paths = bare_vs_unet_over_time(maps, OUT)
    soc, core, peak, bare_max = steady_map_figure(maps, OUT)
    hotspot_timeseries(maps, soc, core, peak, bare_max, OUT)
    rows = save_arrays_and_summary(maps, soc, core, peak, bare_max, OUT)
    status = video(maps, OUT)
    steady = rows[-STEADY_LAST:]
    mean = lambda key: float(np.mean([r[key] for r in steady]))
    print("steady state (last 25 powered frames, Celsius):")
    print(f"  SoC lid core   bare {mean('soc_core_bare_c'):6.2f} | U-Net {mean('soc_core_unet_c'):6.2f} "
          f"| painted {mean('soc_core_target_c'):6.2f} | Stage-2 {mean('soc_core_stage2_c'):6.2f}")
    print(f"  trusted hotspot bare {mean('hotspot_bare_c'):6.2f} | U-Net {mean('hotspot_unet_c'):6.2f} "
          f"| painted {mean('hotspot_target_c'):6.2f}")
    print(f"  board maximum (U-Net) {mean('max_pred_c'):.2f} C")
    (OUT / "presentation_report.json").write_text(
        json.dumps({"video": status,
                    "time_sequence": [str(path) for path in time_sequence_paths],
                    "time_sequence_units": (
                        "absolute Celsius = U-Net temperature rise + case10 per-frame ambient"
                    ),
                    "time_sequence_edge_display": (
                        "rectification gaps filled from nearest valid board temperatures; "
                        "display only, excluded from metrics"
                    ),
                    "steady_last_frames": STEADY_LAST,
                    "soc_core_steady_c": {k: mean(k) for k in (
                        "soc_core_bare_c", "soc_core_unet_c", "soc_core_target_c", "soc_core_stage2_c")},
                    "hotspot_steady_c": {k: mean(k) for k in (
                        "hotspot_bare_c", "hotspot_unet_c", "hotspot_target_c")}},
                   indent=2), encoding="utf-8")
    print(f"wrote {OUT}/prediction_montage_case10.png, "
          f"{OUT}/bare_vs_unet_over_time_case10_part1.png, "
          f"{OUT}/bare_vs_unet_over_time_case10_part2.png, prediction_video_case10.*, "
          f"hotspot_timeseries_case10.png, predicted_steady_map_case10.png, "
          f"predicted_maps_case10.npz, predicted_summary_case10.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
