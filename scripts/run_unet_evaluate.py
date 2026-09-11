"""Score the trained U-Net against raw-copy and Stage-2 physics baselines.

Reads the training output predictions.npz, builds the honest scoreboard
(in-set fit quality vs painted labels; no case10 hold-out by owner decision),
writes evaluation.json and diagnostic figures.

Usage:
    python scripts/run_unet_evaluate.py [--train-dir outputs/thermal/unet_case10/train_default]
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import matplotlib.pyplot as plt
import numpy as np

from tdp.unet.evaluate import evaluate_archive, evaluation_table, write_evaluation_json

DEFAULT_ARCHIVE = "outputs/thermal/unet_case10/painted_pair_dataset.npz"
DEFAULT_TRAIN = "outputs/thermal/unet_case10/train_default"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    outdir = args.outdir or args.train_dir

    report = evaluate_archive(
        args.archive,
        f"{args.train_dir}/predictions.npz",
    )
    write_evaluation_json(outdir, report)
    print(evaluation_table(report))
    print()
    print("largest footprint components (masked MAE / bias per predictor):")
    comps = report["components"]
    for row in comps["stage2"]["regions"][:6]:
        line = f"  px {row['px']:5d}  stage2 {row['masked_mae_k']:5.2f}/{row['masked_bias_k']:+5.2f}  "
        for name in ("unet", "raw"):
            unet_row = next(c for c in comps[name]["regions"] if c["component"] == row["component"])
            line += f"{name} {unet_row['masked_mae_k']:5.2f}/{unet_row['masked_bias_k']:+5.2f}  "
        print(line)
    _figures(args.archive, args.train_dir, outdir, report)
    return 0


def _figures(archive, train_dir, outdir, report) -> None:
    with np.load(f"{archive}") as data:
        elapsed = data["elapsed_s"]
        flags = data["painted_flag"]
        board = data["board_mask"]
        unknown = data["unknown_mask"]
    with np.load(f"{train_dir}/predictions.npz") as pred:
        x = pred["input_theta_k"]
        y = pred["target_theta_k"]
        unet = pred["predicted_theta_k"]
    with np.load("outputs/thermal/case10/reconstructed_sequence.npz") as stage:
        stage2 = stage["theta_k"]

    trusted = board & ~unknown
    masked = unknown
    frames = np.arange(len(x))
    per_frame = {}
    for name, p in (("raw", x), ("unet", unet), ("stage2", stage2)):
        err = np.abs(p - y)
        per_frame[name] = {
            "masked": err[:, masked].mean(axis=1),
            "trusted": err[:, trusted].mean(axis=1),
        }

    # --- per-frame error curves ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    for name, col in (("raw", "0.6"), ("unet", "tab:blue"), ("stage2", "tab:green")):
        axes[0].plot(frames, per_frame[name]["masked"], label=name, color=col, lw=1.4)
        axes[1].plot(frames, per_frame[name]["trusted"], label=name, color=col, lw=1.4)
    axes[0].set_title("masked-area |error| vs painted target (K)")
    axes[1].set_title("trusted-area |error| vs painted target (K)")
    for ax in axes:
        ax.set_xlabel("powered frame index"); ax.legend(fontsize=8)
    axes[0].axvline(np.where(flags == "steady")[0][0], color="k", ls=":", lw=1)
    axes[1].axvline(np.where(flags == "steady")[0][0], color="k", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(f"{outdir}/error_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- example maps ---
    picks = [40, 120, 230]
    rows = ["bare input", "U-Net prediction", "painted target", "U-Net error", "Stage2 error"]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(4.0 * len(picks), 12))
    for col, i in enumerate(picks):
        for row, (title, arr) in enumerate(
            [
                ("bare input", x[i]),
                ("U-Net prediction", unet[i]),
                ("painted target", y[i]),
                ("U-Net error", unet[i] - y[i]),
                ("Stage2 error", stage2[i] - y[i]),
            ]
        ):
            ax = axes[row, col]
            vmin, vmax = (0, 60) if row < 3 else (-12, 12)
            im = ax.imshow(arr, cmap="inferno" if row < 3 else "coolwarm",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(f"frame {i}  t={elapsed[i]:.0f} s", fontsize=9)
    fig.suptitle("case10: U-Net vs painted target (theta above ambient, K)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{outdir}/example_maps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- per-component masked MAE, largest components ---
    comps = report["components"]
    rows = [c for c in comps["stage2"]["regions"]][:8]
    names = ["raw", "unet", "stage2"]
    pos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    width = 0.26
    for j, name in enumerate(names):
        means = [next(c for c in comps[name]["regions"]
                      if c["component"] == r["component"])["masked_mae_k"] for r in rows]
        ax.bar(pos + (j - 1) * width, means, width, label=name)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"#{r['component']} ({r['px']}px)" for r in rows], rotation=20, fontsize=7)
    ax.set_ylabel("masked MAE vs painted target (K)")
    ax.legend(fontsize=8)
    ax.set_title("largest footprint components")
    fig.tight_layout()
    fig.savefig(f"{outdir}/component_bars.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
