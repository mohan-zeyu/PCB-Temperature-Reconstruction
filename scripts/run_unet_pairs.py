"""M1: build the case10(bare) <-> case17(painted) supervised pair archive.

Writes outputs/thermal/unet_case10/painted_pair_dataset.npz (+ .json), a
sample-pair figure, and a label-noise diagnostic figure.  Pairing follows the
rules in tdp/unet/pairs.py: theta fields, elapsed time since each session's own
>=90 % load onset as the state key, per-pixel interpolation of painted frames,
and a flagged steady-mean clamp beyond painted coverage.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tdp.dynamic_inverse.config import load_measurements
from tdp.thermal.calibration import (
    longest_high_load_interval,
    read_timestamped_values,
)

from tdp.unet import pairs
from tdp.unet.pairs import (
    CASE10_POWERED_FRAMES,
    load_sequence_arrays,
    powered_frames_by_clock,
    theta_series,
)

MEASUREMENTS_PATH = Path("configs/measurements_v3.yaml")
OUT = Path("outputs/thermal/unet_case10")
BARE = "case10_rpi4b_bare_full"
PAINTED = "case17_rpi4b_painted_full"


def _load_interval(sequence: str, measurements) -> tuple[float, float]:
    spec = measurements.sequences[sequence]
    log_rel = spec.logs["cpu_load"]
    log = read_timestamped_values(
        Path(measurements.dataset_root) / log_rel,
        value_columns=("total_percent", "cpu_usage_percent"),
    )
    interval = longest_high_load_interval(log)
    return float(interval.start_epoch_s), float(interval.stop_epoch_s)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements(MEASUREMENTS_PATH)

    bare = load_sequence_arrays(BARE)
    painted = load_sequence_arrays(PAINTED)
    bare_load_start, bare_load_stop = _load_interval(BARE, measurements)
    painted_load_start, painted_load_stop = _load_interval(PAINTED, measurements)

    # case10's powered stretch is a fixed project constant (frames 0-239).
    bare_epochs = bare["clock_start_epoch_s"] + bare["timestamps_s"]
    bare_powered = np.arange(CASE10_POWERED_FRAMES)
    if float(bare_epochs[bare_powered[-1]]) > bare_load_stop + 1.0:
        raise ValueError("case10 powered constant exceeds the logged load stop")
    painted_powered = powered_frames_by_clock(painted, painted_load_stop)

    with np.load("outputs/thermal/case10/reconstructed_sequence.npz") as archive:
        board = archive["board_mask"]
        unknown = archive["unknown_mask"]
        image_only = archive["image_only_unknown_mask"]

    meta = pairs.build_paired_archive(
        bare_arrays=bare,
        painted_arrays=painted,
        bare_powered=bare_powered,
        painted_powered=painted_powered,
        bare_load_start_epoch_s=bare_load_start,
        painted_load_start_epoch_s=painted_load_start,
        board=board,
        unknown_mask=unknown,
        out_path=OUT / "painted_pair_dataset.npz",
        load_value=1.0,
        label="case10 bare full load -> case17 painted full load",
    )

    # provenance arrays for later model-channel variants
    with np.load(OUT / "painted_pair_dataset.npz") as data:
        x = data["x_theta"]
        y = data["y_theta"]
        elapsed = data["elapsed_s"]
        flags = data["painted_flag"]
    np.savez_compressed(
        OUT / "paired_masks.npz",
        board_mask=board,
        unknown_mask=unknown,
        image_only_unknown_mask=image_only,
    )

    _figures(x, y, elapsed, flags, board, unknown, meta)
    print("meta:")
    for key in (
        "n_pairs", "n_interp", "n_steady_flag", "n_first_flag", "first_steady_index",
        "painted_powered_frames", "board_px", "trusted_px", "masked_px",
    ):
        print(f"  {key}: {meta[key]}")
    print("  trusted label-noise MAE by split:", meta["trusted_label_noise_mae_k_by_split"])
    print("  masked correction MAE by split:  ", meta["masked_correction_mae_k_by_split"])
    print(f"wrote {OUT/'painted_pair_dataset.npz'}")
    return 0


def _figures(x, y, elapsed, flags, board, unknown, meta) -> None:
    trusted = board & ~unknown
    residual = y - x
    pick = [8, 60, max(6, int(meta["first_steady_index"] or 0) - 8), 230]
    pick = sorted({min(max(int(p), 0), len(x) - 1) for p in pick})

    fig, axes = plt.subplots(3, len(pick), figsize=(4.4 * len(pick), 12))
    for col, i in enumerate(pick):
        titles = ["bare theta (K)", "painted theta (K)", "painted - bare (K)"]
        for row, (arr, title) in enumerate(
            [(x[i], titles[0]), (y[i], titles[1]), (residual[i], titles[2])]
        ):
            ax = axes[row, col]
            vmin, vmax = (0, 60) if row < 2 else (-6, 45)
            im = ax.imshow(arr, cmap="inferno" if row < 2 else "coolwarm",
                           vmin=vmin, vmax=vmax)
            ax.contour(unknown.astype(int), levels=[0.5], colors="w" if row < 2 else "k",
                       linewidths=0.5, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                flag = str(flags[i])
                ax.set_title(
                    f"frame {i}  t={elapsed[i]:.0f} s [{flag}]\n"
                    f"trusted mae {np.abs(residual[i][trusted]).mean():.2f} K | "
                    f"masked {np.abs(residual[i][unknown]).mean():.1f} K",
                    fontsize=9,
                )
    fig.suptitle("case10 bare -> case17 painted pairs (theta above own ambient)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "sample_pairs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharex=True)
    fr = np.arange(len(x))
    for ax, series, name in [
        (axes[0], np.abs(residual[:, trusted]).mean(axis=1), "trusted pixels (label noise)"),
        (axes[1], np.abs(residual[:, unknown]).mean(axis=1), "footprint pixels (correction to learn)"),
    ]:
        ax.plot(fr, series, lw=1.2)
        ax.set_title(name)
        ax.set_xlabel("powered frame index")
        ax.set_ylabel("|painted - bare| theta mean (K)")
    steady_start = meta["first_steady_index"]
    if steady_start is not None:
        for ax in axes:
            ax.axvline(steady_start, color="k", ls=":", lw=1)
            ax.annotate("painted coverage ends -> steady-mean target", (steady_start, 0),
                        xytext=(steady_start + 2, ax.get_ylim()[1] * 0.85), fontsize=8)
        for name, idx in (("val", meta["split_blocks"]["val"]), ("test", meta["split_blocks"]["test"])):
            pass
    axes[0].axhline(0.5, color="r", ls="--", lw=0.8)
    axes[0].annotate("0.5 K guide", xy=(0, 0.5), color="r", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "label_noise_vs_elapsed.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
