"""Build the fixed-size Stage 3 forecast video and held-out error presentation.

This script never fits or selects a model.  It consumes the frozen Stage 3
artifacts, always writes a keyframe montage and numerical error arrays, then
tries MP4 and falls back to a sampled GIF if the video writer is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tdp-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tdp-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from PIL import Image

from tdp.thermal import (
    load_reconstruction_archive,
    model_range_fields,
    nearest_sample_indices,
    weighted_mae_by_frame,
)


MODEL_KEYS = {
    "equilibrium decay": "equilibrium_test_start_",
    "persistence": "persistence_test_start_",
    "constant source": "source_ablation_constant_source_start_",
    "smooth positive sources": "source_ablation_fitted_smooth_source_start_",
    "no source": "source_ablation_no_source_start_",
}
MODEL_COLOURS = {
    "equilibrium decay": "#1677b8",
    "persistence": "#6b7280",
    "constant source": "#d97706",
    "smooth positive sources": "#c2415d",
    "no source": "#5b21b6",
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _board_field(field: np.ndarray, board_mask: np.ndarray) -> np.ndarray:
    return np.ma.masked_where(~board_mask, np.asarray(field, dtype=np.float64))


def _aggregate_weighted_mae(
    predicted: np.ndarray,
    actual: np.ndarray,
    weight: np.ndarray,
) -> float:
    error = np.abs(np.asarray(predicted, dtype=np.float64) - actual)
    return float(np.sum(error * weight) / np.sum(weight))


def _prediction_key(files: list[str], prefix: str) -> tuple[str, int]:
    matches = [name for name in files if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one saved prediction matching {prefix!r}, got {matches}")
    key = matches[0]
    try:
        start = int(key.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"prediction key has no integer start frame: {key}") from exc
    return key, start


def _save_error_figure(
    path: Path,
    *,
    times_s: np.ndarray,
    frame0_weighted_mae_k: np.ndarray,
    frame0_direct_mae_k: np.ndarray,
    forecast_end_s: float,
    heldout_times_s: np.ndarray,
    heldout_error_by_model_k: dict[str, np.ndarray],
    heldout_aggregate_k: dict[str, float],
    board_mask: np.ndarray,
    truth_last: np.ndarray,
    equilibrium_last: np.ndarray,
    source_last: np.ndarray,
    test_start: int,
    test_stop: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=100, constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(times_s, frame0_weighted_mae_k, color="#1677b8", label="uncertainty weighted")
    ax.plot(times_s, frame0_direct_mae_k, color="#111827", alpha=0.75, label="direct pixels")
    ax.axvline(forecast_end_s, color="#c2415d", linestyle="--", linewidth=1.2,
               label="1% remaining")
    ax.set(title="Frame-0 forward error at captured times", xlabel="elapsed time [s]",
           ylabel="instantaneous MAE [K]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, values in heldout_error_by_model_k.items():
        ax.semilogy(
            heldout_times_s,
            values,
            label=name,
            color=MODEL_COLOURS[name],
            linewidth=2.2 if name == "equilibrium decay" else 1.4,
        )
    ax.set(title=f"Held-out forward window, start frame {test_start}",
           xlabel="absolute run time [s]", ylabel="instantaneous weighted MAE [K]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    names = list(heldout_aggregate_k)
    values = [heldout_aggregate_k[name] for name in names]
    bars = ax.bar(
        np.arange(len(names)), values,
        color=[MODEL_COLOURS[name] for name in names],
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)), [name.replace(" ", "\n") for name in names], fontsize=8)
    ax.set_ylabel("weighted MAE [K], logarithmic scale")
    ax.set_title("Frozen held-out score")
    ax.grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)

    temperature_values = truth_last[board_mask]
    temp_min, temp_max = np.percentile(temperature_values, [1.0, 99.0])
    source_error = source_last - truth_last
    equilibrium_error = equilibrium_last - truth_last
    error_limit = max(1.0, float(np.percentile(np.abs(source_error[board_mask]), 98.0)))

    ax = axes[1, 0]
    temperature_image = ax.imshow(
        _board_field(truth_last, board_mask), cmap="inferno", vmin=temp_min, vmax=temp_max
    )
    ax.set_title(f"Held-out target, frame {test_stop}\nreconstructed excess temperature")
    ax.axis("off")
    fig.colorbar(temperature_image, ax=ax, label="K above ambient", fraction=0.046)

    error_image = None
    for ax, error, title in (
        (axes[1, 1], equilibrium_error, "equilibrium decay error"),
        (axes[1, 2], source_error, "smooth positive-source error"),
    ):
        error_image = ax.imshow(
            _board_field(error, board_mask), cmap="RdBu_r",
            vmin=-error_limit, vmax=error_limit,
        )
        ax.set_title(f"{title}, frame {test_stop}\nshared scale; saturation is intentional")
        ax.axis("off")
    fig.colorbar(error_image, ax=axes[1, 1:], label="prediction - target [K]", fraction=0.03)

    fig.suptitle(
        "Stage 3 forecast diagnostics — test frames never select the settled field or rates\n"
        "Footprint errors are agreement with Stage 2 estimates, not measurements under lids",
        fontsize=14,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _build_video_figure(
    *,
    forecast_theta_k: np.ndarray,
    elapsed_s: np.ndarray,
    absolute_time_s: np.ndarray,
    settled_theta_k: np.ndarray,
    board_mask: np.ndarray,
    range_fields_for_indices,
    archive_times_s: np.ndarray,
    captured_theta_k: np.ndarray,
    captured_prediction_k: np.ndarray,
    captured_weighted_mae_k: np.ndarray,
    captured_direct_mae_k: np.ndarray,
    nearest_capture: np.ndarray,
    ambient_c: float,
):
    """Return one fixed 1280x720 canvas and an in-place frame updater."""

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.35))
    temp_ax = fig.add_subplot(grid[0, 0])
    deviation_ax = fig.add_subplot(grid[0, 1])
    range_ax = fig.add_subplot(grid[1, 0])
    error_ax = fig.add_subplot(grid[1, 1])
    curve_ax = fig.add_subplot(grid[:, 2])

    board_values = forecast_theta_k[:, board_mask]
    temp_min, temp_max = np.percentile(board_values, [1.0, 99.0])
    deviation_limit = max(
        0.1,
        float(np.percentile(np.abs(board_values - settled_theta_k[board_mask]), 99.0)),
    )
    start_range = range_fields_for_indices(np.array([0], dtype=np.int64))[0]
    end_range = range_fields_for_indices(
        np.array([len(forecast_theta_k) - 1], dtype=np.int64)
    )[0]
    range_limit = max(
        0.1,
        float(np.percentile(
            np.concatenate((start_range[board_mask], end_range[board_mask])), 99.0
        )),
    )
    relevant = archive_times_s <= absolute_time_s[-1]
    captured_error = captured_prediction_k[relevant] - captured_theta_k[relevant]
    error_limit = max(
        0.1, float(np.percentile(np.abs(captured_error[:, board_mask]), 99.0))
    )

    temperature_image = temp_ax.imshow(
        _board_field(forecast_theta_k[0], board_mask),
        cmap="inferno", vmin=temp_min, vmax=temp_max,
    )
    deviation_image = deviation_ax.imshow(
        _board_field(forecast_theta_k[0] - settled_theta_k, board_mask),
        cmap="RdBu_r", vmin=-deviation_limit, vmax=deviation_limit,
    )
    range_image = range_ax.imshow(
        _board_field(start_range, board_mask), cmap="viridis", vmin=0.0, vmax=range_limit
    )
    first_capture = int(nearest_capture[0])
    error_image = error_ax.imshow(
        _board_field(
            captured_prediction_k[first_capture] - captured_theta_k[first_capture], board_mask
        ),
        cmap="RdBu_r", vmin=-error_limit, vmax=error_limit,
    )
    for ax in (temp_ax, deviation_ax, range_ax, error_ax):
        ax.axis("off")
    fig.colorbar(temperature_image, ax=temp_ax, label="K above ambient", fraction=0.046)
    fig.colorbar(deviation_image, ax=deviation_ax, label="K", fraction=0.046)
    fig.colorbar(range_image, ax=range_ax, label="K", fraction=0.046)
    fig.colorbar(error_image, ax=error_ax, label="prediction - target [K]", fraction=0.046)

    captured_for_curve = relevant & (archive_times_s >= absolute_time_s[0])
    curve_times = archive_times_s[captured_for_curve]
    curve_ax.plot(
        curve_times, captured_weighted_mae_k[captured_for_curve],
        color="#1677b8", linewidth=2.0, label="uncertainty weighted",
    )
    curve_ax.plot(
        curve_times, captured_direct_mae_k[captured_for_curve],
        color="#111827", linewidth=1.4, alpha=0.75, label="direct pixels",
    )
    cursor = curve_ax.axvline(absolute_time_s[0], color="#c2415d", linewidth=1.8)
    point, = curve_ax.plot([], [], "o", color="#c2415d", markersize=5)
    curve_ax.set(
        title="Error against captured reconstructed frames",
        xlabel="absolute run time [s]",
        ylabel="instantaneous MAE [K]",
        xlim=(absolute_time_s[0], absolute_time_s[-1]),
        ylim=(0.0, max(0.1, float(np.max(captured_weighted_mae_k[captured_for_curve]) * 1.12))),
    )
    curve_ax.grid(alpha=0.25)
    curve_ax.legend(fontsize=8, loc="upper right")
    curve_note = curve_ax.text(
        0.03, 0.97, "", transform=curve_ax.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    title = fig.suptitle("", fontsize=14)

    def update(index: int):
        index = int(index)
        capture = int(nearest_capture[index])
        current_range = range_fields_for_indices(np.array([index], dtype=np.int64))[0]
        temperature_image.set_data(_board_field(forecast_theta_k[index], board_mask))
        deviation_image.set_data(
            _board_field(forecast_theta_k[index] - settled_theta_k, board_mask)
        )
        range_image.set_data(_board_field(current_range, board_mask))
        error_image.set_data(
            _board_field(
                captured_prediction_k[capture] - captured_theta_k[capture], board_mask
            )
        )
        temp_ax.set_title(f"Forecast excess temperature\n+{elapsed_s[index]:.0f} s")
        deviation_ax.set_title("Remaining deviation from settled field")
        range_ax.set_title("Declared model range (not a confidence interval)")
        error_ax.set_title(
            f"Error at nearest capture, t={archive_times_s[capture]:.0f} s"
        )
        cursor.set_xdata([absolute_time_s[index], absolute_time_s[index]])
        point.set_data(
            [archive_times_s[capture]], [captured_weighted_mae_k[capture]]
        )
        curve_note.set_text(
            f"nearest captured frame: {capture}\n"
            f"weighted MAE: {captured_weighted_mae_k[capture]:.3f} K\n"
            f"direct-pixel MAE: {captured_direct_mae_k[capture]:.3f} K"
        )
        title.set_text(
            "Forward-only equilibrium forecast from reconstructed frame 0\n"
            f"starting ambient held at {ambient_c:.2f} C as an external scenario"
        )
        return (
            temperature_image, deviation_image, range_image, error_image,
            cursor, point, curve_note, title,
        )

    update(0)
    fig.canvas.draw()
    if fig.canvas.get_width_height() != (1280, 720):
        raise RuntimeError(
            f"video canvas changed size: {fig.canvas.get_width_height()}, expected (1280, 720)"
        )
    return fig, update


def _canvas_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast", default="outputs/thermal/case10/forecast_from_frame_000.npz"
    )
    parser.add_argument(
        "--reconstruction", default="outputs/thermal/case10/reconstructed_sequence.npz"
    )
    parser.add_argument(
        "--whole-phase",
        default="outputs/thermal/case10/whole_phase_forward_at_measured_times.npz",
    )
    parser.add_argument(
        "--heldout", default="outputs/thermal/case10/heldout_forward_predictions.npz"
    )
    parser.add_argument(
        "--fit-report", default="outputs/thermal/case10/forecast_fit_report.json"
    )
    parser.add_argument("--outdir", default="outputs/thermal/case10")
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if args.video_fps <= 0 or args.video_stride <= 0 or args.gif_max_frames <= 1:
        raise SystemExit("video-fps and video-stride must be positive; gif-max-frames > 1")

    archive = load_reconstruction_archive(args.reconstruction)
    with np.load(args.forecast) as data:
        forecast = {name: np.array(data[name]) for name in data.files}
    forecast_theta = forecast["theta_k"].astype(np.float64)
    elapsed = forecast["elapsed_s"].astype(np.float64)
    absolute_times = forecast["absolute_time_s"].astype(np.float64)
    board_mask = forecast["board_mask"].astype(bool)
    if int(forecast["start_frame"]) != 0:
        raise ValueError("the headline error presentation currently requires start frame 0")
    if forecast_theta.shape != (len(elapsed), *archive.shape_hw):
        raise ValueError("forecast fields and elapsed times disagree")
    if not np.array_equal(board_mask, archive.board_mask):
        raise ValueError("forecast and reconstruction board masks disagree")
    if not np.allclose(forecast_theta[0], archive.theta_k[0], atol=1e-6, rtol=0.0):
        raise ValueError("saved forecast does not preserve its selected starting frame")

    with np.load(args.whole_phase) as data:
        captured_prediction = np.asarray(data["predicted_theta_k"], dtype=np.float64)
        captured_truth = np.asarray(data["measured_theta_k"], dtype=np.float64)
        captured_times = np.asarray(data["times_s"], dtype=np.float64)
    if (
        captured_prediction.shape != archive.theta_k.shape
        or captured_truth.shape != archive.theta_k.shape
        or not np.array_equal(captured_times, archive.times_s)
    ):
        raise ValueError("whole-phase artifact does not match the reconstruction archive")

    all_weight = archive.observation_weight()
    direct_mask = archive.board_mask & ~archive.rebuilt_mask
    direct_weight = np.broadcast_to(direct_mask, archive.theta_k.shape).astype(np.float64)
    frame0_weighted = weighted_mae_by_frame(
        captured_prediction, captured_truth, all_weight
    )
    frame0_direct = weighted_mae_by_frame(
        captured_prediction, captured_truth, direct_weight
    )
    rebuilt_weight = np.broadcast_to(
        archive.rebuilt_mask, archive.theta_k.shape
    ).astype(np.float64)
    frame0_rebuilt = weighted_mae_by_frame(
        captured_prediction, captured_truth, rebuilt_weight
    )

    with np.load(args.heldout) as data:
        heldout_files = list(data.files)
        heldout_payload = {name: np.array(data[name]) for name in data.files}
    prediction_by_model: dict[str, np.ndarray] = {}
    starts = set()
    for name, prefix in MODEL_KEYS.items():
        key, start = _prediction_key(heldout_files, prefix)
        prediction_by_model[name] = heldout_payload[key].astype(np.float64)
        starts.add(start)
    if len(starts) != 1:
        raise ValueError(f"held-out models use different starting frames: {sorted(starts)}")
    test_start = starts.pop()
    n_targets = len(next(iter(prediction_by_model.values())))
    if any(len(values) != n_targets for values in prediction_by_model.values()):
        raise ValueError("held-out predictions have different lengths")
    test_stop = test_start + n_targets
    test_truth = archive.theta_k[test_start + 1 : test_stop + 1]
    test_weight = all_weight[test_start + 1 : test_stop + 1]
    test_direct_weight = direct_weight[test_start + 1 : test_stop + 1]
    heldout_error = {
        name: weighted_mae_by_frame(values, test_truth, test_weight)
        for name, values in prediction_by_model.items()
    }
    heldout_direct_error = {
        name: weighted_mae_by_frame(values, test_truth, test_direct_weight)
        for name, values in prediction_by_model.items()
    }
    heldout_aggregate = {
        name: _aggregate_weighted_mae(values, test_truth, test_weight)
        for name, values in prediction_by_model.items()
    }

    with Path(args.fit_report).open(encoding="utf-8") as handle:
        fit_report = json.load(handle)
    reported = {
        "equilibrium decay": fit_report["equilibrium_forecast"]["test_forward"]["weighted_mae_k"],
        "persistence": fit_report["equilibrium_forecast"]["test_persistence"]["weighted_mae_k"],
        "constant source": fit_report["positive_smooth_source_ablation"]["test_forward_by_model"]["constant_source"]["weighted_mae_k"],
        "smooth positive sources": fit_report["positive_smooth_source_ablation"]["test_forward_by_model"]["fitted_smooth_source"]["weighted_mae_k"],
        "no source": fit_report["positive_smooth_source_ablation"]["test_forward_by_model"]["no_source"]["weighted_mae_k"],
    }
    for name in reported:
        if not np.isclose(heldout_aggregate[name], reported[name], atol=2e-5, rtol=0.0):
            raise ValueError(f"saved {name} prediction does not reproduce its reported score")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "forecast_presentation_metrics.npz"
    names = list(MODEL_KEYS)
    np.savez_compressed(
        metrics_path,
        captured_times_s=archive.times_s,
        frame0_weighted_mae_k=frame0_weighted,
        frame0_direct_mae_k=frame0_direct,
        frame0_rebuilt_estimate_agreement_mae_k=frame0_rebuilt,
        heldout_times_s=archive.times_s[test_start + 1 : test_stop + 1],
        heldout_model_names=np.asarray(names, dtype="U40"),
        heldout_weighted_mae_by_frame_k=np.stack([heldout_error[name] for name in names]),
        heldout_direct_mae_by_frame_k=np.stack(
            [heldout_direct_error[name] for name in names]
        ),
        heldout_aggregate_weighted_mae_k=np.asarray(
            [heldout_aggregate[name] for name in names]
        ),
        test_start_frame=test_start,
        test_stop_frame=test_stop,
        split_roles=heldout_payload["split_roles"],
    )

    error_figure_path = outdir / "forecast_error_and_ablations.png"
    _save_error_figure(
        error_figure_path,
        times_s=archive.times_s,
        frame0_weighted_mae_k=frame0_weighted,
        frame0_direct_mae_k=frame0_direct,
        forecast_end_s=float(absolute_times[-1]),
        heldout_times_s=archive.times_s[test_start + 1 : test_stop + 1],
        heldout_error_by_model_k=heldout_error,
        heldout_aggregate_k=heldout_aggregate,
        board_mask=archive.board_mask,
        truth_last=test_truth[-1],
        equilibrium_last=prediction_by_model["equilibrium decay"][-1],
        source_last=prediction_by_model["smooth positive sources"][-1],
        test_start=test_start,
        test_stop=test_stop,
    )

    start_range = forecast["start_model_range_k"].astype(np.float64)
    settled_range = forecast["settled_model_range_k"].astype(np.float64)
    empirical_error = float(forecast["heldout_forward_error_k"])
    range_decay = float(forecast["model_range_decay_rate_per_s"])

    def ranges_at_indices(indices: np.ndarray) -> np.ndarray:
        return model_range_fields(
            elapsed[np.asarray(indices, dtype=np.int64)],
            start_range,
            settled_range,
            empirical_error,
            range_decay,
        )

    nearest = nearest_sample_indices(archive.times_s, absolute_times)
    figure, update = _build_video_figure(
        forecast_theta_k=forecast_theta,
        elapsed_s=elapsed,
        absolute_time_s=absolute_times,
        settled_theta_k=forecast["settled_theta_k"].astype(np.float64),
        board_mask=archive.board_mask,
        range_fields_for_indices=ranges_at_indices,
        archive_times_s=archive.times_s,
        captured_theta_k=captured_truth,
        captured_prediction_k=captured_prediction,
        captured_weighted_mae_k=frame0_weighted,
        captured_direct_mae_k=frame0_direct,
        nearest_capture=nearest,
        ambient_c=float(forecast["ambient_c"][0]),
    )

    key_indices = np.unique(
        np.rint(np.linspace(0, len(forecast_theta) - 1, 4)).astype(np.int64)
    )
    rendered = []
    for index in key_indices:
        update(int(index))
        rendered.append(_canvas_rgb(figure))
    montage = Image.new("RGB", (2560, 1440), color="white")
    for tile, pixels in enumerate(rendered):
        row, column = divmod(tile, 2)
        montage.paste(Image.fromarray(pixels), (column * 1280, row * 720))
    montage_path = outdir / "forecast_video_montage.png"
    montage.save(montage_path)

    animation_status: dict[str, Any]
    video_indices = np.arange(0, len(forecast_theta), args.video_stride, dtype=np.int64)
    if video_indices[-1] != len(forecast_theta) - 1:
        video_indices = np.append(video_indices, len(forecast_theta) - 1)
    if args.skip_video:
        animation_status = {
            "status": "skipped by --skip-video",
            "frame_count": 0,
        }
    else:
        try:
            if not writers.is_available("ffmpeg"):
                raise RuntimeError("Matplotlib cannot find an ffmpeg writer")
            video_path = outdir / "forecast_from_frame_000.mp4"
            animation = FuncAnimation(
                figure, update, frames=video_indices, blit=False, cache_frame_data=False
            )
            writer = FFMpegWriter(
                fps=args.video_fps,
                codec="h264",
                bitrate=2200,
                metadata={"title": "case10 forward thermal forecast", "artist": "tdp"},
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            animation.save(video_path, writer=writer, dpi=100)
            animation_status = {
                "status": "mp4 written",
                "path": str(video_path),
                "frame_count": int(len(video_indices)),
                "forecast_frame_stride": int(args.video_stride),
                "fps": int(args.video_fps),
            }
        except Exception as video_error:
            try:
                gif_indices = np.unique(
                    np.rint(
                        np.linspace(
                            0, len(forecast_theta) - 1,
                            min(args.gif_max_frames, len(forecast_theta)),
                        )
                    ).astype(np.int64)
                )
                gif_path = outdir / "forecast_from_frame_000.gif"
                animation = FuncAnimation(
                    figure, update, frames=gif_indices, blit=False, cache_frame_data=False
                )
                animation.save(gif_path, writer=PillowWriter(fps=8), dpi=100)
                animation_status = {
                    "status": "sampled GIF fallback written",
                    "path": str(gif_path),
                    "frame_count": int(len(gif_indices)),
                    "forecast_frame_indices": gif_indices,
                    "fps": 8,
                    "mp4_error": f"{type(video_error).__name__}: {video_error}",
                }
            except Exception as gif_error:
                animation_status = {
                    "status": "animation unavailable; montage and arrays preserved",
                    "mp4_error": f"{type(video_error).__name__}: {video_error}",
                    "gif_error": f"{type(gif_error).__name__}: {gif_error}",
                    "frame_count": 0,
                }
    plt.close(figure)

    presentation_report = {
        "stage": "3c fixed-size forecast presentation",
        "model_fitting_or_selection_changed": False,
        "canvas_width_px": 1280,
        "canvas_height_px": 720,
        "forecast_fields": int(len(forecast_theta)),
        "forecast_spacing_s": float(np.median(np.diff(elapsed))),
        "forecast_start_elapsed_s": float(elapsed[0]),
        "forecast_stop_elapsed_s": float(elapsed[-1]),
        "animation": animation_status,
        "heldout_test": {
            "start_frame_supplied": int(test_start),
            "target_frame_start": int(test_start + 1),
            "target_frame_stop": int(test_stop),
            "weighted_mae_k": heldout_aggregate,
            "footprint_note": (
                "rebuilt-area error is agreement with Stage 2 estimates, not hidden truth"
            ),
        },
        "artifacts": {
            "raw_one_second_forecast": str(Path(args.forecast)),
            "presentation_metrics": str(metrics_path),
            "video_montage": str(montage_path),
            "error_and_ablations": str(error_figure_path),
            "animation": animation_status.get("path"),
        },
    }
    report_path = outdir / "forecast_presentation_report.json"
    report_path.write_text(
        json.dumps(presentation_report, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"canvas        : 1280 x 720 px")
    print(f"forecast      : {len(forecast_theta)} one-second fields")
    print(f"animation     : {animation_status['status']}")
    if "path" in animation_status:
        print(f"                {animation_status['path']}")
    print(f"montage       : {montage_path}")
    print(f"error figure  : {error_figure_path}")
    print(f"metrics       : {metrics_path}")
    print(f"report        : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
