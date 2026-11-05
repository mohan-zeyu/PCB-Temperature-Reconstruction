"""Build Orange Pi 5 Pro pair archives: bare board -> black-tape twin.

Two loads, mirroring the Raspberry Pi work:
  full 0.5 ... case12_opi5pro_bare_half <-> case14_opi5pro_tape_half   (load 0.5)
  full 1.0 ... case13_opi5pro_bare_full <-> case15_opi5pro_tape_full   (load 1.0)

Both sessions of each load have their own CPU-load + die logs, so pairs are
matched by elapsed time since each session's own load onset (the same rule as
case10<->case17), with per-pixel interpolation of tape frames and a flagged
steady-mean clamp past tape coverage.

The bare-footprint mask combines the image-only detector on the bare full-load
steady frame with the paired attenuation audit (tape minus bare steady > 5 K),
exactly the two-part rule the Pi pipeline uses; it is built once on the full-load
pair and reused for both loads because it is the same physical board.

Writes pairs_full.npz, pairs_half.npz, pairs_both.npz and an audit figure/json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import matplotlib.pyplot as plt
import numpy as np

from tdp.data import emissivity as em
from tdp.dynamic_inverse.config import load_measurements
from tdp.thermal.calibration import (
    longest_high_load_interval,
    paired_attenuation_mask,
    read_timestamped_values,
    steady_frame_mask,
)
from tdp.unet.pairs import (
    build_paired_archive,
    load_interval_from_measurements,
    load_sequence_arrays,
    powered_frames_by_clock,
    theta_series,
)

MEASUREMENTS_PATH = "configs/measurements_opi5pro.yaml"
OUT = Path("outputs/thermal/unet_opi5pro")
FULL = ("case13_opi5pro_bare_full", "case15_opi5pro_tape_full", 1.0)
HALF = ("case12_opi5pro_bare_half", "case14_opi5pro_tape_half", 0.5)


def _load_interval(sequence: str, measurements):
    # The tape half-load log plateaus near 50 % CPU, so it needs the lower
    # threshold; the full-load sessions are read at the usual 90 %.
    threshold = 90.0 if sequence.endswith("full") else 40.0
    return load_interval_from_measurements(sequence, measurements, threshold_percent=threshold)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements(MEASUREMENTS_PATH)

    arrays = {
        name: load_sequence_arrays(name)
        for name in (FULL[0], FULL[1], HALF[0], HALF[1])
    }
    intervals = {name: _load_interval(name, measurements) for name in arrays}
    powered = {
        name: powered_frames_by_clock(arrays[name], intervals[name].stop_epoch_s)
        for name in arrays
    }
    theta = {name: theta_series(arrays[name]) for name in arrays}

    # shared board: physically valid in every session
    board = np.ones_like(arrays[FULL[0]]["valid_mask"], dtype=bool)
    for name in arrays:
        board &= arrays[name]["valid_mask"]
    if board.sum() == 0:
        raise ValueError("no shared board pixels across the Orange Pi sessions")

    # footprint mask: image-only detector on the bare full-load steady frame ...
    reference_index = int(powered[FULL[0]][-1])
    image_mask = em.detect_low_emissivity(theta[FULL[0]][reference_index], halo_px=1).mask & board

    # ... plus the paired tape-minus-bare attenuation audit at the full-load steady state
    bare_epochs = arrays[FULL[0]]["clock_start_epoch_s"] + arrays[FULL[0]]["timestamps_s"]
    tape_epochs = arrays[FULL[1]]["clock_start_epoch_s"] + arrays[FULL[1]]["timestamps_s"]
    bare_steady = steady_frame_mask(bare_epochs, intervals[FULL[0]], window_s=180.0)
    tape_steady = steady_frame_mask(tape_epochs, intervals[FULL[1]], window_s=180.0)
    audit = paired_attenuation_mask(
        theta[FULL[0]], bare_steady, theta[FULL[1]], tape_steady,
        board_mask=board, threshold_k=5.0,
    )
    unknown = (image_mask | audit.mask) & board
    trusted = board & ~unknown
    if trusted.sum() == 0 or (board & unknown).sum() == 0:
        raise ValueError("mask construction produced an empty region")

    np.savez_compressed(OUT / "paired_masks.npz", board_mask=board, unknown_mask=unknown)

    metas = {}
    for tag, (bare_name, tape_name, load) in (("full", FULL), ("half", HALF)):
        metas[tag] = build_paired_archive(
            bare_arrays=arrays[bare_name],
            painted_arrays=arrays[tape_name],
            bare_powered=powered[bare_name],
            painted_powered=powered[tape_name],
            bare_load_start_epoch_s=intervals[bare_name].start_epoch_s,
            painted_load_start_epoch_s=intervals[tape_name].start_epoch_s,
            board=board,
            unknown_mask=unknown,
            out_path=OUT / f"pairs_{tag}.npz",
            load_value=load,
            label=f"{bare_name} -> {tape_name}",
        )

    # combined archive for a two-load Orange Pi model
    with np.load(OUT / "pairs_full.npz") as f, np.load(OUT / "pairs_half.npz") as h:
        x = np.concatenate([f["x_theta"], h["x_theta"]])
        y = np.concatenate([f["y_theta"], h["y_theta"]])
        load_value = np.concatenate([f["load_value"][None], h["load_value"][None]])
        provenance = np.concatenate(
            [np.full(len(f["x_theta"]), "bare_full->tape_full", dtype="U20"),
             np.full(len(h["x_theta"]), "bare_half->tape_half", dtype="U20")]
        )
        flags = np.concatenate([f["painted_flag"], h["painted_flag"]])
    np.savez_compressed(
        OUT / "pairs_both.npz",
        x_theta=x, y_theta=y, board_mask=board, unknown_mask=unknown,
        load_value=load_value, provenance=provenance, painted_flag=flags,
    )

    audit_report = {
        "board_px": int(board.sum()),
        "trusted_px": int(trusted.sum()),
        "masked_px": int((board & unknown).sum()),
        "image_only_px": int(image_mask.sum()),
        "paired_audit_px": int(audit.mask.sum()),
        "paired_audit_added_px": int((audit.mask & ~image_mask).sum()),
        "paired_audit_report": audit.report,
        "pair_meta": metas,
    }
    (OUT / "pair_audit.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    _figure(arrays, powered, theta, board, unknown, intervals, metas)
    print(json.dumps({
        "board_px": audit_report["board_px"],
        "trusted_px": audit_report["trusted_px"],
        "masked_px": audit_report["masked_px"],
        "image_only_px": audit_report["image_only_px"],
        "paired_audit_added_px": audit_report["paired_audit_added_px"],
        "full": {k: metas["full"][k] for k in (
            "n_pairs", "n_interp", "n_steady_flag", "painted_powered_frames",
            "trusted_label_noise_mae_k_by_split", "masked_correction_mae_k_by_split")},
        "half": {k: metas["half"][k] for k in (
            "n_pairs", "n_interp", "n_steady_flag", "painted_powered_frames",
            "trusted_label_noise_mae_k_by_split", "masked_correction_mae_k_by_split")},
    }, indent=2))
    print(f"wrote {OUT}/pairs_{{full,half,both}}.npz")
    return 0


def _figure(arrays, powered, theta, board, unknown, intervals, metas) -> None:
    trusted = board & ~unknown
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    for col, (tag, (bare_name, tape_name, _)) in enumerate((("full", FULL), ("half", HALF))):
        bare_steady = theta[bare_name][powered[bare_name][-25:]].mean(axis=0)
        tape_steady = theta[tape_name][powered[tape_name][-25:]].mean(axis=0)
        diff = tape_steady - bare_steady
        for row, (arr, title, vmin, vmax, cmap) in enumerate([
            (bare_steady, f"{tag}: bare steady theta (K)", 0, 45, "inferno"),
            (tape_steady, f"{tag}: tape steady theta (K)", 0, 45, "inferno"),
            (diff, f"{tag}: tape - bare (K)", -5, 40, "coolwarm"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(unknown.astype(int), levels=[0.5], colors="w" if row < 2 else "k",
                       linewidths=0.5, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
        print(f"{tag}: per-pair trusted |residual| mean "
              f"{np.mean(metas[tag]['trusted_label_noise_mae_k_by_split']['train']):.2f} K")
    # level curves
    for col, (tag, (bare_name, tape_name, _)) in enumerate((("full", FULL), ("half", HALF)), start=2):
        ax = axes[0, col]
        ax.plot(powered[bare_name], np.median(theta[bare_name][powered[bare_name]][:, trusted], axis=1),
                label="bare")
        ax.plot(powered[tape_name], np.median(theta[tape_name][powered[tape_name]][:, trusted], axis=1),
                label="tape")
        ax.set_title(f"{tag}: trusted level (K)"); ax.legend(fontsize=7)
        ax.set_xlabel("powered frame")
    axes[1, 2].axis("off"); axes[2, 2].axis("off"); axes[1, 3].axis("off"); axes[2, 3].axis("off")
    fig.suptitle("Orange Pi 5 Pro: bare vs black-tape steady maps (theta above ambient)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "pair_audit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
