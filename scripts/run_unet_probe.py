"""M4: cross-load probe - apply the full-load-trained U-Net to case09 (half load).

The operator was trained on case10(bare) -> case17(painted) pairs at full load,
with the case10 footprint mask as a static input channel.  Here it is applied,
without retraining, to the *bare half-load* session case09, and its output is
scored against the painted half-load truth (case16) at the steady state.

Why steady only:
* case16 has no CPU/SoC logs and its IR shows a stepped load (three heating
  episodes), so elapsed-time matching of the transient is not trustworthy.
* The trustworthy painted half-load data is its final near-steady segment
  (frames 167-191).  case09 ran a constant ~51 % CPU and is near-steady at its
  powered end (frames 215-239 used here).
* At steady, trusted pixels of the two sessions agree to ~0.6-1.9 K in theta
  (same board, same layout), which is the label floor the probe inherits.

The reported numbers are therefore *steady-state cross-load transfer*: they
test whether the learned artifact correction transfers to a cooler, unseen
field; they say nothing about transient behaviour at other loads.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tdp.unet.evaluate import component_metrics, mask_metrics
from tdp.unet.model import build_residual_unet
from tdp.unet.train import THETA_SCALE_K

BARE_HALF = "case09_rpi4b_bare_half"
PAINTED_HALF = "case16_rpi4b_painted_half"
STEADY_BARE_FRAMES = np.arange(215, 240)     # last 25 powered case09 frames
STEADY_PAINTED_FRAMES = np.arange(167, 192)  # last 25 powered case16 frames


def _arrays(sequence: str) -> dict:
    base = f"outputs/dynamic_inverse/{sequence}/latest"
    with np.load(f"{base}/temperature_corrected_sequence.npz") as data:
        return {k: data[k] for k in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="outputs/thermal/unet_case10/train_default/model_best.pt"
    )
    parser.add_argument("--outdir", default="outputs/thermal/unet_case10/probe_half")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with np.load("outputs/thermal/case10/reconstructed_sequence.npz") as archive:
        board = archive["board_mask"]
        unknown = archive["unknown_mask"]

    bare = _arrays(BARE_HALF)
    painted = _arrays(PAINTED_HALF)

    def theta(seq: dict, indices: np.ndarray) -> np.ndarray:
        raw = seq["raw_temperature_c"].astype(np.float64)
        ambient = seq["ambient_c"].astype(np.float64)
        return raw[indices] - ambient[indices][:, None, None]

    x = theta(bare, STEADY_BARE_FRAMES)
    y_painted_frames = theta(painted, STEADY_PAINTED_FRAMES)
    y = y_painted_frames.mean(axis=0, keepdims=True)  # constant steady target

    model = build_residual_unet(in_channels=3)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=True))
    model.eval()
    torch_device = torch.device(args.device)

    x = np.where(board[None], x, 0.0)
    board_ch = np.asarray(board, dtype=np.float32)[None]
    mask_ch = np.asarray(unknown, dtype=np.float32)[None]
    pred = []
    with torch.no_grad():
        for i in range(len(x)):
            ch0 = torch.tensor(x[i : i + 1] / THETA_SCALE_K, dtype=torch.float32)[:, None]
            static = torch.tensor(
                np.concatenate([board_ch, mask_ch], axis=0)[None], dtype=torch.float32
            )
            inp = torch.cat([ch0, static], dim=1).to(torch_device)
            corr = model(inp)[0, 0].cpu().numpy()
            pred.append((x[i] / THETA_SCALE_K + corr) * THETA_SCALE_K)
    pred = np.stack(pred)

    y_full = np.broadcast_to(y, pred.shape).copy()
    err = {"unet": pred - y_full, "raw": x - y_full}
    summary = {name: mask_metrics(e, board, unknown) for name, e in err.items()}
    components = {name: component_metrics(e, unknown, board) for name, e in err.items()}

    import pathlib
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "probe_predictions.npz",
        bare_theta_k=x.astype(np.float32),
        predicted_theta_k=pred.astype(np.float32),
        painted_steady_theta_k=y[0].astype(np.float32),
        board_mask=board,
        unknown_mask=unknown,
    )
    (outdir / "probe_report.json").write_text(
        json.dumps(
            {
                "design": __doc__.splitlines()[1] if False else (
                    "cross-load steady-state transfer: full-load-trained operator "
                    "applied to bare half-load case09, scored vs painted half-load case16"
                ),
                "bare_frames": [int(i) for i in STEADY_BARE_FRAMES],
                "painted_steady_frames": [int(i) for i in STEADY_PAINTED_FRAMES],
                "summary": summary,
                "components": components,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _figure(x, pred, y[0], board, unknown, outdir, summary)
    print("probe summary (half-load steady, |err| vs painted truth):")
    for name in ("raw", "unet"):
        s = summary[name]
        print(f"  {name:5s} board {s['board_mae_k']:6.2f} | trusted {s['trusted_mae_k']:5.2f} "
              f"| masked {s['masked_mae_k']:6.2f} (bias {s['masked_bias_k']:+5.2f}) K")
    print("largest components:")
    for row in components["unet"]["regions"][:5]:
        unet_bias = row["masked_bias_k"]
        print(f"  px {row['px']:5d}  unet {row['masked_mae_k']:5.2f}/{unet_bias:+5.2f}")
    return 0


def _figure(x, pred, y, board, unknown, outdir, summary) -> None:
    trusted = board & ~unknown
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    i = len(x) // 2
    maps = [
        ("case09 bare half-load theta", x[i], 0, 45, "inferno"),
        ("case16 painted steady theta", y, 0, 45, "inferno"),
        ("U-Net prediction (trained at full load)", pred[i], 0, 45, "inferno"),
        ("U-Net error (K)", pred[i] - y, -8, 8, "coolwarm"),
    ]
    for r, (title, arr, vmin, vmax, cmap) in enumerate(maps):
        ax = axes[r, 0]
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(title, fontsize=8)
    axes[0, 1].scatter(y[trusted], x[i][trusted], s=1, alpha=0.3, color="tab:blue")
    axes[0, 1].plot([0, 40], [0, 40], "r--", lw=0.8)
    axes[0, 1].set_title("raw (bare) trusted vs painted"); axes[0, 1].set_xlim(0, 40); axes[0, 1].set_ylim(0, 40)
    axes[1, 1].scatter(y[trusted], pred[i][trusted], s=1, alpha=0.3, color="tab:green")
    axes[1, 1].plot([0, 40], [0, 40], "r--", lw=0.8)
    axes[1, 1].set_title("U-Net trusted vs painted"); axes[1, 1].set_xlim(0, 40); axes[1, 1].set_ylim(0, 40)
    for c, (name, e) in enumerate([("raw masked error", (x[i] - y)), ("U-Net masked error", pred[i] - y)]):
        ax = axes[2, c]
        vmin, vmax = (0, 12) if name.startswith("raw") else (-8, 8)
        cm = "viridis" if name.startswith("raw") else "coolwarm"
        im = ax.imshow(np.where(unknown, e if not name.startswith("raw") else np.abs(e), np.nan),
                       cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name)
    axes[3, 0].axis("off")
    axes[0, 2].axis("off"); axes[1, 2].axis("off"); axes[2, 2].axis("off"); axes[3, 2].axis("off")
    text = "\n".join(
        f"{name}: masked MAE {s['masked_mae_k']:.2f} K (bias {s['masked_bias_k']:+.2f})\n"
        f"    trusted MAE {s['trusted_mae_k']:.2f} K"
        for name, s in summary.items()
    )
    axes[3, 1].text(0.02, 0.5, text, fontsize=10, family="monospace", va="center")
    axes[3, 1].axis("off")
    fig.suptitle("Cross-load probe: operator trained on full-load pairs, applied to half-load case09", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "probe_maps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
