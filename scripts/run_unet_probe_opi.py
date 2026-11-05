"""Leave-one-load-out transfer probe on the Orange Pi 5 Pro.

Applies an already-trained correction operator to the bare session of a load
level and scores it against the black-tape twin's steady mean, using the
generalised steady_probe helper.  Also reports the same operator's steady
in-domain score when that level was part of its training archive.

Usage examples:
    python scripts/run_unet_probe_opi.py --model .../train_full/model_best.pt \
        --bare case12_opi5pro_bare_half --tape case14_opi5pro_tape_half \
        --load half --outdir .../probe_half_from_full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import matplotlib.pyplot as plt
import numpy as np

from tdp.dynamic_inverse.config import load_measurements
from tdp.unet.pairs import (
    load_interval_from_measurements,
    load_sequence_arrays,
    powered_frames_by_clock,
)
from tdp.unet.probe import steady_probe

MEASUREMENTS_PATH = "configs/measurements_opi5pro.yaml"
MASKS = Path("outputs/thermal/unet_opi5pro/paired_masks.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bare", required=True)
    parser.add_argument("--tape", required=True)
    parser.add_argument("--load", choices=["half", "full"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    measurements = load_measurements(MEASUREMENTS_PATH)
    bare = load_sequence_arrays(args.bare)
    tape = load_sequence_arrays(args.tape)
    threshold = 90.0 if args.load == "full" else 40.0
    bare_load = load_interval_from_measurements(args.bare, measurements, threshold_percent=threshold)
    tape_load = load_interval_from_measurements(args.tape, measurements, threshold_percent=threshold)
    bare_powered = powered_frames_by_clock(bare, bare_load.stop_epoch_s)
    tape_powered = powered_frames_by_clock(tape, tape_load.stop_epoch_s)
    with np.load(MASKS) as masks:
        board = masks["board_mask"]
        unknown = masks["unknown_mask"]

    result = steady_probe(
        bare_arrays=bare,
        twin_arrays=tape,
        bare_powered=bare_powered,
        twin_powered=tape_powered,
        board=board,
        unknown=unknown,
        model_path=args.model,
        device=args.device,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "bare": args.bare,
        "tape": args.tape,
        "load": args.load,
        "model": args.model,
        "steady_frames": {"bare": int(len(result["bare"])), "tape": 25},
        "session_level_offset_k": result["session_level_offset_k"],
        "summary": result["summary"],
    }
    (outdir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        outdir / "probe_predictions.npz",
        bare_theta_k=result["bare"].astype(np.float32),
        predicted_theta_k=result["pred"].astype(np.float32),
        tape_steady_theta_k=result["twin_steady"].astype(np.float32),
        board_mask=board,
        unknown_mask=unknown,
    )
    _figure(result, outdir, args.load)

    print(f"probe {args.load}: bare {args.bare} vs tape {args.tape}")
    print(f"  session level offset (tape - bare, trusted): {result['session_level_offset_k']:+.2f} K")
    for name, metrics in result["summary"].items():
        print(f"  {name:24s} board {metrics['board_mae_k']:6.2f} | trusted {metrics['trusted_mae_k']:5.2f} "
              f"| masked {metrics['masked_mae_k']:6.2f} (bias {metrics['masked_bias_k']:+5.2f}) K")
    return 0


def _figure(result, outdir: Path, load: str) -> None:
    board = result["board"]; unknown = result["unknown"]
    pred = result["pred"].mean(axis=0)
    y = result["twin_steady"]
    bare = result["bare"].mean(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, arr, title, cmap, vmin, vmax in [
        (axes[0], bare, "bare steady theta (K)", "inferno", 0, 45),
        (axes[1], pred, "operator output (K)", "inferno", 0, 45),
        (axes[2], y, "tape steady theta (K)", "inferno", 0, 45),
        (axes[3], np.where(board, pred - y, np.nan), "operator - tape (K)", "coolwarm", -8, 8),
    ]:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    s = result["summary"]
    fig.suptitle(
        f"Orange Pi {load} load: masked MAE raw {s['raw']['masked_mae_k']:.2f} K -> "
        f"operator {s['unet']['masked_mae_k']:.2f} K "
        f"(offset-corrected {s['unet_offset_corrected']['masked_mae_k']:.2f} K)",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(outdir / "probe_maps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
