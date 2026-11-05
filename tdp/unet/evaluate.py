"""Score the trained correction net against honest baselines on the archive.

Because the owner chose whole-archive training (no case10 hold-out), the scores
here describe *fit quality against the painted labels*, not independent
generalization.  The report is written that way on purpose.  Independent
generalization is measured later by the case09 <-> case16 cross-load probe.

Predictors compared on the very same 240 frames and the very same painted
target:

* ``raw``        - keep the bare reading (identity).  Masked-area MAE of this
                   predictor is the artifact the other predictors must remove.
* ``unet``       - the residual U-Net trained on the whole archive.
* ``stage2``     - the production physics reconstruction
                   (outputs/thermal/case10/reconstructed_sequence.npz), which
                   already used case17 as its anchor - so it is a strong,
                   physics-based in-set baseline, not an independent method.

Metrics are masked-area and per-footprint-component first (a good board average
must never hide a wrong lid); trusted-area error is reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from tdp.unet.train import THETA_SCALE_K, PairedSet

STAGE2_ARCHIVE = Path("outputs/thermal/case10/reconstructed_sequence.npz")


def mask_metrics(err_k: np.ndarray, board: np.ndarray, unknown: np.ndarray) -> dict:
    """One predictor's error against the painted target (frames x H x W).

    Masked-area statistics use board & unknown only; footprint pixels outside
    the shared board carry no label and are excluded (see train.mask_metrics).
    """
    trusted = board & ~unknown
    masked = board & unknown
    return {
        "board_mae_k": float(np.abs(err_k[:, board]).mean()),
        "trusted_mae_k": float(np.abs(err_k[:, trusted]).mean()),
        "trusted_bias_k": float(err_k[:, trusted].mean()),
        "masked_mae_k": float(np.abs(err_k[:, masked]).mean()),
        "masked_bias_k": float(err_k[:, masked].mean()),
        "masked_rmse_k": float(np.sqrt((err_k[:, masked] ** 2).mean())),
    }


def component_metrics(err_k: np.ndarray, unknown: np.ndarray, board: np.ndarray, min_px: int = 40) -> dict:
    """Per-footprint-component masked MAE/bias, restricted to the shared board."""
    masked = unknown & board
    labels, n = label(masked, structure=np.ones((3, 3), dtype=bool))
    rows = []
    for comp in range(1, n + 1):
        px = labels == comp
        if int(px.sum()) < min_px:
            continue
        values = err_k[:, px]
        rows.append({
            "component": int(comp),
            "px": int(px.sum()),
            "masked_mae_k": float(np.abs(values).mean()),
            "masked_bias_k": float(values.mean()),
            "p90_abs_k": float(np.percentile(np.abs(values), 90)),
        })
    rows.sort(key=lambda row: -row["px"])
    return {"regions": rows, "n_regions": len(rows)}


def evaluate_archive(
    archive_path: str | Path,
    predictions_npz: str | Path,
    stage2_archive: str | Path = STAGE2_ARCHIVE,
) -> dict:
    dataset = PairedSet(archive_path)
    with np.load(predictions_npz) as data:
        pred_unet = data["predicted_theta_k"].astype(np.float64)
        y = data["target_theta_k"].astype(np.float64)
        x = data["input_theta_k"].astype(np.float64)
    board = dataset.board.numpy()
    unknown = dataset.unknown.numpy()
    if pred_unet.shape != y.shape:
        raise ValueError("prediction archive does not match the target archive")

    with np.load(stage2_archive) as stage:
        pred_stage2 = stage["theta_k"].astype(np.float64)
    if pred_stage2.shape[:1] != (len(y),):
        raise ValueError("stage2 reconstruction frame count differs from the archive")

    predictors = {
        "raw": x,
        "unet": pred_unet,
        "stage2": pred_stage2,
    }
    summary = {}
    components = {}
    for name, pred in predictors.items():
        summary[name] = mask_metrics(pred - y, board, unknown)
        components[name] = component_metrics(pred - y, unknown, board)
    return {
        "summary": summary,
        "components": components,
        "note": (
            "in-set fit quality against painted labels; no case10 hold-out "
            "(owner decision). Independent generalization is the cross-load "
            "probe, not these numbers."
        ),
    }


def write_evaluation_json(outdir: str | Path, report: dict) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def evaluation_table(report: dict) -> str:
    """Human-readable scoreboard (board / trusted / masked MAE per predictor)."""
    lines = ["predictor      board_mae  trusted_mae  masked_mae  masked_bias"]
    for name, metrics in report["summary"].items():
        lines.append(
            f"{name:14s} {metrics['board_mae_k']:8.3f}  {metrics['trusted_mae_k']:9.3f}  "
            f"{metrics['masked_mae_k']:10.3f}  {metrics['masked_bias_k']:+9.3f}"
        )
    return "\n".join(lines)
