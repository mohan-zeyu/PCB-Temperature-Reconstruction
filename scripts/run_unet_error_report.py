"""Report error and uncertainty of the current U-Net path against the painted proxy.

The true temperature map is not measured anywhere: the closest physical reference
is a matte-black-painted twin of the same board read by the same infrared camera
(case17 painted full load <-> case10 bare full load; case16 painted half load <->
case09 bare half load).  Painted-board IR maps are the per-pixel labels the net
is trained and scored against, and the imager's own datasheet uncertainty is the
anchor that says how tightly those labels can represent the unknown truth.

Imager anchor (datasheet, HIKMICRO H21ProS+, model HM-TP52+-3AQF/W-H21ProS+):
  * NETD < 0.050 K @ 25 C, F#=1.0   - per-pixel temporal noise of one frame
  * accuracy +-2 C or +-2 % of reading, whichever is larger
    - level anchor of the blackpaint board (2 C dominates below ~100 C object)
  * source: HIKMICRO H21ProS+ specification (256x192 VOx, 7.5-14 um), e.g.
    https://www.dfzk.com/prod_hikmicro_tp52-3aqfw-h21pros.html
  The CSV exports carry no radiometric parameter block; 0.95 emissivity on the
  painted surface is the project definition of truth, not a fitted value.

Two scored sets, kept apart on purpose:
  * full-load archive (case10 -> case17, 240 powered frames).  The net trained
    on the whole archive (owner decision), so these numbers are in-set fit
    quality against the painted labels, NOT independent generalization.
  * half-load steady probe (case09 -> case16): the full-load-trained operator
    applied, without retraining, to 25 steady bare half-load frames and scored
    against the case16 painted steady mean.  This is the closest thing to an
    honest out-of-sample error against a painted map the archive allows.

Every reported quantity carries its unit.  Errors are temperature above ambient
(theta, K); painted maps are theta too, so session ambient offsets cancel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

from tdp.unet.evaluate import component_metrics, mask_metrics  # region helpers

OUT = Path("outputs/thermal/unet_case10/error_report")
ARCHIVE = Path("outputs/thermal/unet_case10/painted_pair_dataset.npz")
TRAIN_NPZ = Path("outputs/thermal/unet_case10/train_default/predictions.npz")
PROBE_NPZ = Path("outputs/thermal/unet_case10/probe_half/probe_predictions.npz")
STAGE2 = Path("outputs/thermal/case10/reconstructed_sequence.npz")

CAMERA = {
    "make_model": "HIKMICRO HM-TP52+-3AQF/W-H21ProS+ (H21ProS+)",
    "sensor": "256 x 192 uncooled VOx microbolometer, 7.5-14 um",
    "source": "H21ProS+ datasheet via dfzk.com: https://www.dfzk.com/prod_hikmicro_tp52-3aqfw-h21pros.html",
    "netd_k": 0.050,
    "netd_note": "NETD < 50 mK @ 25 C, F#=1.0 - one-frame per-pixel temporal noise floor (spec)",
    "accuracy_note": "+-2 C or +-2 % of reading, whichever is larger - absolute level band",
    "level_anchor_k": 2.0,
    "level_anchor_at": "object ~80 C (2 C dominates over 2 % = 1.6 C below ~100 C)",
}


def region_mask(board: np.ndarray, unknown: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "board": board,
        "trusted": board & ~unknown,
        "masked": board & unknown,
    }


def load_archive() -> dict:
    with np.load(ARCHIVE) as d:
        return {
            "n": int(d["x_theta"].shape[0]),
            "train_idx": d["train_idx"],
            "val_idx": d["val_idx"],
            "test_idx": d["test_idx"],
            "elapsed_s": d["elapsed_s"],
            "flag": d["painted_flag"],
            "board": d["board_mask"],
            "unknown": d["unknown_mask"],
        }


def archive_metrics() -> dict:
    with np.load(TRAIN_NPZ) as p:
        pred = p["predicted_theta_k"].astype(np.float64)
        y = p["target_theta_k"].astype(np.float64)
        x = p["input_theta_k"].astype(np.float64)
        board = p["board_mask"]
        unknown = p["unknown_mask"]
    meta = load_archive()
    with np.load(STAGE2) as s:
        stage2 = s["theta_k"].astype(np.float64)
    if stage2.shape != pred.shape:
        raise ValueError("stage2 reconstruction does not cover the archive frames")

    names = {"unet": pred - y, "raw": x - y, "stage2": stage2 - y}
    summary = {name: mask_metrics(e, board, unknown) for name, e in names.items()}
    components = {name: component_metrics(e, unknown, board) for name, e in names.items()}

    # contiguous-block and painted-target-kind detail (unet only, vs painted)
    unet_err = names["unet"]
    trusted = board & ~unknown
    masked = board & unknown
    splits = {"train": meta["train_idx"], "val": meta["val_idx"], "test": meta["test_idx"]}
    split_metrics = {
        name: {
            "frames": [int(idx[0]), int(idx[-1]) + 1],
            "masked_mae_k": float(np.abs(unet_err[idx][:, masked]).mean()),
            "masked_bias_k": float(unet_err[idx][:, masked].mean()),
            "trusted_mae_k": float(np.abs(unet_err[idx][:, trusted]).mean()),
        }
        for name, idx in splits.items()
    }
    steady = np.where(meta["flag"] == "steady")[0]
    interp = np.where(meta["flag"] == "interp")[0]
    flag_metrics = {
        "steady_mean_target": {
            "frames": [int(steady[0]), int(steady[-1]) + 1],
            "masked_mae_k": float(np.abs(unet_err[steady][:, masked]).mean()),
            "trusted_mae_k": float(np.abs(unet_err[steady][:, trusted]).mean()),
        },
        "interp_target": {
            "frames": [int(interp[0]), int(interp[-1]) + 1],
            "masked_mae_k": float(np.abs(unet_err[interp][:, masked]).mean()),
            "trusted_mae_k": float(np.abs(unet_err[interp][:, trusted]).mean()),
        },
    }
    # pixel maps over all frames: time mean error and frame scatter
    pixel_mean_err = unet_err.mean(axis=0)
    pixel_frame_std = unet_err.std(axis=0)
    per_frame = {
        "masked_mae_k": np.abs(unet_err[:, masked]).mean(axis=1).tolist(),
        "trusted_mae_k": np.abs(unet_err[:, trusted]).mean(axis=1).tolist(),
    }
    # level structure of masked error: pooled pixels of every steady-target frame
    rng = np.random.default_rng(0)
    pooled_level = y[steady][:, masked].ravel()
    pooled_err = unet_err[steady][:, masked].ravel()
    sample = rng.choice(len(pooled_level), size=min(80_000, len(pooled_level)), replace=False)
    levels, meds, q = [], [], []
    edges = np.unique(np.quantile(pooled_level, np.linspace(0, 1, 13)))
    for left, right in zip(edges[:-1], edges[1:]):
        sel = (pooled_level >= left) & (pooled_level < right)
        if sel.any():
            levels.append(float(np.median(pooled_level[sel])))
            meds.append(float(np.median(pooled_err[sel])))
            q.append(float(np.percentile(np.abs(pooled_err[sel]), 90)))
    return {
        "region_summary": summary,
        "components": components,
        "split_metrics": split_metrics,
        "target_kind": flag_metrics,
        "pixel_mean_err_k": pixel_mean_err,
        "pixel_frame_std_k": pixel_frame_std,
        "per_frame": per_frame,
        "level_bins": {"level_k": levels, "median_err_k": meds, "p90_abs_err_k": q},
        "masks": {"board": board, "unknown": unknown},
    }


def probe_metrics() -> dict:
    with np.load(PROBE_NPZ) as p:
        bare = p["bare_theta_k"].astype(np.float64)
        pred = p["predicted_theta_k"].astype(np.float64)
        y_steady = p["painted_steady_theta_k"].astype(np.float64)
        board = p["board_mask"]
        unknown = p["unknown_mask"]
    y = np.broadcast_to(y_steady[None], pred.shape).copy()
    names = {"unet": pred - y, "raw": bare - y}
    summary = {name: mask_metrics(e, board, unknown) for name, e in names.items()}
    components = {name: component_metrics(e, unknown, board) for name, e in names.items()}
    masked = board & unknown
    trusted = board & ~unknown
    pred_scatter = pred.std(axis=0)  # model wobble over the 25 steady frames
    # per-frame series
    per_frame = {
        "masked_mae_k": np.abs((pred - y)[:, masked]).mean(axis=1).tolist(),
        "trusted_mae_k": np.abs((pred - y)[:, trusted]).mean(axis=1).tolist(),
    }
    return {
        "region_summary": summary,
        "components": components,
        "per_frame": per_frame,
        "pred_frame_std_k_p95": float(np.percentile(pred_scatter[board & ~unknown], 95)),
        "pred_frame_std_k_max": float(pred_scatter[board].max()),
        "masks": {"board": board, "unknown": unknown},
        "maps": {"pred": pred, "y_steady": y_steady, "bare": bare},
    }


def _stats_block(name: str, s: dict) -> str:
    return (
        f"| {name:7s} | {s['board_mae_k']:.3f} | {s['trusted_mae_k']:.3f} | "
        f"{s['trusted_bias_k']:+.3f} | {s['masked_mae_k']:.3f} | "
        f"{s['masked_bias_k']:+.3f} | {s['masked_rmse_k']:.3f} |"
    )


def write_report(full: dict, probe: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# U-Net path: error and uncertainty against the painted temperature proxy\n")
    lines.append("Scored 2026-09-09 on branch `experiment/unet-painted-truth-reconstruction`.\n")
    lines.append(
        "**The true temperature map is not measured anywhere.** The closest reference is the "
        "matte-black-painted twin of the same board, read by the same camera, on the same "
        "canonical grid (case17 full load <-> case10; case16 half load <-> case09). All errors "
        "below are **model output minus the painted-board IR map** (theta above each session's "
        "own ambient, in K), and the imager datasheet below is the anchor that says how tightly "
        "that painted map can represent the unknown truth.\n"
    )

    lines.append("## 1. Infrared-imager anchor (datasheet, not measured here)\n")
    lines.append(f"- Camera: **{CAMERA['make_model']}**, {CAMERA['sensor']}.")
    lines.append(f"- {CAMERA['netd_note']} -> per-pixel temporal noise of a single frame < 0.050 K.")
    lines.append(
        f"- {CAMERA['accuracy_note']}. {CAMERA['level_anchor_at']} -> the blackpaint board anchors "
        f"an absolute level only to about +-{CAMERA['level_anchor_k']:.0f} K on its own."
    )
    lines.append(f"- Source: {CAMERA['source']}")
    lines.append(
        "- Note: the CSV exports carry no radiometric parameter block; emissivity 0.95 on the "
        "painted surface is the project definition of truth, not a fitted value.\n"
    )
    lines.append(
        "Consequence: differences between the model and the painted map that are smaller than "
        "about 0.05-0.1 K are inside the imager's one-frame noise and carry no information; "
        "differences below the +-2 K absolute band say the model agrees with the *camera's* "
        "painted reading, not that it is within +-2 K of the unknown true map. Only a contact "
        "or blackbody reference could tighten the absolute level.\n"
    )

    lines.append("## 2. Full-load archive: case10 bare -> case17 painted (240 powered frames)\n")
    lines.append(
        "The net trained on the whole archive (owner decision), so these are **in-set fit "
        "quality against the painted labels**, not independent generalization.\n"
    )
    lines.append(
        "Region MAE / bias / RMSE in K, all frames, model output minus painted target "
        "(`masked` = on-board footprint pixels):\n"
    )
    lines.append("| predictor | board MAE | trusted MAE | trusted bias | masked MAE | masked bias | masked RMSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("unet", "raw", "stage2"):
        lines.append(_stats_block(name, full["region_summary"][name]))
    lines.append("\nPer contiguous split (guard gaps kept; steady-only from frame 132):\n")
    lines.append("| block | frames | masked MAE | masked bias | trusted MAE |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in full["split_metrics"].items():
        lo, hi = m["frames"]
        lines.append(
            f"| {name} | {lo}-{hi} | {m['masked_mae_k']:.3f} | {m['masked_bias_k']:+.3f} | {m['trusted_mae_k']:.3f} |"
        )
    lines.append("")
    for kind, m in full["target_kind"].items():
        lo, hi = m["frames"]
        lines.append(
            f"- {kind} target, frames {lo}-{hi}: masked MAE {m['masked_mae_k']:.3f} K, "
            f"trusted MAE {m['trusted_mae_k']:.3f} K"
        )
    lines.append("\nLargest footprint components (masked MAE / bias / p90 of |err|, K):\n")
    lines.append("| component | px | masked MAE | masked bias | p90 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in full["components"]["unet"]["regions"][:8]:
        lines.append(
            f"| {row['component']} | {row['px']} | {row['masked_mae_k']:.3f} | "
            f"{row['masked_bias_k']:+.3f} | {row['p90_abs_k']:.3f} |"
        )
    lines.append("")

    lines.append("## 3. Half-load steady probe: full-load operator on case09 vs case16 painted\n")
    lines.append(
        "25 steady case09 bare frames at ~51 % CPU load, operator trained at full load only, "
        "scored against the case16 painted steady mean. This is the honest out-of-sample "
        "transfer number the archive allows.\n"
    )
    lines.append("| predictor | board MAE | trusted MAE | trusted bias | masked MAE | masked bias | masked RMSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("unet", "raw"):
        lines.append(_stats_block(name, probe["region_summary"][name]))
    lines.append("\nLargest footprint components (probe, K):\n")
    lines.append("| component | px | masked MAE | masked bias | p90 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in probe["components"]["unet"]["regions"][:8]:
        lines.append(
            f"| {row['component']} | {row['px']} | {row['masked_mae_k']:.3f} | "
            f"{row['masked_bias_k']:+.3f} | {row['p90_abs_k']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"- Model output is stable across the 25 steady frames: trusted-pixel frame scatter "
        f"p95 = {probe['pred_frame_std_k_p95']:.3f} K, max on board = {probe['pred_frame_std_k_max']:.3f} K.\n"
    )

    lines.append("## 4. What these mean against the imager anchor\n")
    lines.append(
        "- In-set masked MAE of order 0.1 K (full-load archive) is **below the +-2 K datasheet "
        "absolute band**: it says the net reproduces the painted IR frames, not that the board "
        "is known to +-0.1 K. On the trusted pixels the painted map itself only agrees with the "
        "bare session to about 1.2-1.9 K (state mismatch + imager noise), so in-set fit errors "
        "of 0.1 K are also far below the pairing's own label noise."
    )
    lines.append(
        "- The cross-load probe masked MAE "
        f"({probe['region_summary']['unet']['masked_mae_k']:.2f} K) is larger than the imager's "
        "+-2 K level band: the learned correction does not transfer level for level to a colder "
        "board state, so out-of-sample model error dominates the uncertainty budget there."
    )
    lines.append(
        "- Imager per-pixel temporal noise (NETD < 0.05 K) is negligible in both comparisons "
        "once any spatial or systematic term is present; it only limits single-frame, "
        "single-pixel claims. Absolute level claims against the unknown true map are limited by "
        "the camera's +-2 C / +-2 % accuracy unless a contact/blackbody anchor is added.\n"
    )
    lines.append("## 5. Notes\n")
    lines.append(
        "- All errors use theta (temperature above each session's own ambient); raw degrees "
        "Celsius across sessions whose ambient differs by ~2 K must never be compared directly."
    )
    lines.append(
        "- `train_metrics.json` masked rows are internally inconsistent with the saved "
        "`predictions.npz` it accompanies (history masked MAE ~0.62 K vs 0.10 K recomputed "
        "here from the npz); the numbers above are recomputed from the saved prediction arrays."
    )

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    report = {
        "camera_anchor": CAMERA,
        "full_load": {k: v for k, v in full.items() if k not in ("pixel_mean_err_k", "pixel_frame_std_k", "per_frame", "masks")},
        "probe": {k: v for k, v in probe.items() if k not in ("maps", "per_frame", "masks")},
        "note": "errors are model output minus painted-board IR map (theta, K); true map is not measured; painted proxy carries the datasheet anchor above",
    }
    report["full_load"]["split_metrics"] = {
        k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        for k, v in full["split_metrics"].items()
    }
    report["full_load"]["level_bins"] = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in full["level_bins"].items()
    }
    (outdir / "error_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    return report


def figure_archive(full: dict, outdir: Path) -> None:
    board = full["masks"]["board"]
    unknown = full["masks"]["unknown"]
    mean_map = full["pixel_mean_err_k"]
    std_map = full["pixel_frame_std_k"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    vmax = max(0.05, float(np.nanpercentile(np.abs(mean_map[board]), 99)))
    ax = axes[0, 0]
    ax.imshow(
        np.where(board, mean_map, np.nan), cmap="RdBu_r",
        norm=colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), interpolation="nearest")
    ax.contour(unknown.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    ax.set_title(f"full load: time-mean U-Net error (K), +-{vmax:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
    vmax2 = max(0.05, float(np.nanpercentile(std_map[board], 99)))
    ax = axes[0, 1]
    ax.imshow(np.where(board, std_map, np.nan), cmap="magma", vmin=0, vmax=vmax2, interpolation="nearest")
    ax.set_title(f"full load: frame scatter of error (K), to {vmax2:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[0, 2]
    ax.plot(full["per_frame"]["masked_mae_k"], lw=1.2, color="tab:red", label="masked")
    ax.plot(full["per_frame"]["trusted_mae_k"], lw=1.2, color="tab:blue", label="trusted")
    ax.axvline(132, color="k", ls=":", lw=1)
    ax.text(133, ax.get_ylim()[1] * 0.9, "steady target", fontsize=8)
    ax.set_xlabel("powered frame"); ax.set_ylabel("|error| mean (K)")
    ax.set_title("per-frame error vs painted target"); ax.legend(fontsize=8)

    # level structure of masked error at steady
    lb = full["level_bins"]
    ax = axes[1, 0]
    ax.plot(lb["level_k"], lb["median_err_k"], "-o", color="tab:red", ms=3, label="median error")
    ax.plot(lb["level_k"], lb["p90_abs_err_k"], "--o", color="tab:purple", ms=3, label="p90 |error|")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("painted theta level (K)")
    ax.set_ylabel("masked error (K)")
    ax.set_title("steady masked error vs painted level"); ax.legend(fontsize=8)

    comp = full["components"]["unet"]["regions"][:8]
    ax = axes[1, 1]
    names = [f"c{r['component']}" for r in comp]
    ax.bar(names, [r["masked_mae_k"] for r in comp], color="tab:blue")
    ax.set_ylabel("masked MAE (K)")
    ax.set_title("largest footprint components (in-set fit)")
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1, 2]
    names3 = ("unet", "raw", "stage2")
    ax.bar(range(3), [full["region_summary"][n]["masked_mae_k"] for n in names3],
           color=("tab:blue", "0.6", "tab:green"))
    ax.set_xticks(range(3))
    ax.set_xticklabels(names3)
    ax.set_ylabel("masked MAE (K)")
    ax.set_title("archive baselines vs painted")
    fig.suptitle(
        "Full-load archive (case10 bare -> case17 painted): U-Net error vs painted proxy, "
        "in-set fit (no hold-out)", y=0.995
    )
    fig.tight_layout()
    fig.savefig(outdir / "figure_error_archive.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_probe(probe: dict, outdir: Path) -> None:
    board = probe["masks"]["board"]
    unknown = probe["masks"]["unknown"]
    y = probe["maps"]["y_steady"]
    pred = probe["maps"]["pred"].mean(axis=0)
    err = pred - y
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, arr, title, cmap, vmin, vmax in [
        (axes[0, 0], y, "case16 painted steady theta (K)", "inferno", 0, 45),
        (axes[0, 1], pred, "U-Net output, 25-frame mean (K)", "inferno", 0, 45),
        (axes[0, 2], np.where(board, err, np.nan), "U-Net - painted (K)", "RdBu_r", -8, 8),
    ]:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].plot(probe["per_frame"]["masked_mae_k"], color="tab:red", lw=1.4, label="masked")
    axes[1, 0].plot(probe["per_frame"]["trusted_mae_k"], color="tab:blue", lw=1.4, label="trusted")
    axes[1, 0].set_xlabel("steady bare frame"); axes[1, 0].set_ylabel("|error| (K)")
    axes[1, 0].set_title("probe per-frame error vs case16 painted mean"); axes[1, 0].legend(fontsize=8)
    comp = probe["components"]["unet"]["regions"][:8]
    axes[1, 1].bar([f"c{r['component']}" for r in comp], [r["masked_mae_k"] for r in comp], color="tab:blue")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_ylabel("masked MAE (K)")
    axes[1, 1].set_title("footprint components (cross-load probe)")
    for name, col in (("raw", "0.6"), ("unet", "tab:green")):
        s = probe["region_summary"][name]
        axes[1, 2].bar(name, s["masked_mae_k"], color=col)
    axes[1, 2].set_ylabel("masked MAE (K)")
    axes[1, 2].set_title("probe: raw vs corrected, masked area")
    fig.suptitle(
        "Cross-load probe: full-load-trained operator on steady case09, vs case16 painted truth",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(outdir / "figure_error_probe.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    full = archive_metrics()
    probe = probe_metrics()
    write_report(full, probe, OUT)
    figure_archive(full, OUT)
    figure_probe(probe, OUT)

    def line(name: str, s: dict) -> str:
        return (
            f"  {name:7s} board {s['board_mae_k']:6.3f} | trusted {s['trusted_mae_k']:6.3f} "
            f"(bias {s['trusted_bias_k']:+5.3f}) | masked {s['masked_mae_k']:6.3f} "
            f"(bias {s['masked_bias_k']:+5.3f}, rmse {s['masked_rmse_k']:6.3f}) K"
        )

    print(f"camera anchor: NETD < {CAMERA['netd_k']:.3f} K; accuracy +-{CAMERA['level_anchor_k']:.0f} K (or +-2 %)")
    print("full-load archive (in-set fit vs painted, all 240 frames):")
    for name in ("unet", "raw", "stage2"):
        print(line(name, full["region_summary"][name]))
    print("  splits (masked MAE):", {k: f"{v['masked_mae_k']:.3f}" for k, v in full["split_metrics"].items()})
    print("half-load steady probe (cross-load vs painted):")
    for name in ("unet", "raw"):
        print(line(name, probe["region_summary"][name]))
    print(f"wrote {OUT}/report.md, error_report.json, figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
