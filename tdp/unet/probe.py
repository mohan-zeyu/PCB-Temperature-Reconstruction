"""Steady-state transfer probe for the correction net, board agnostic.

Given a bare session and its high-emissivity twin (painted or taped) at the same
load, this applies an already-trained operator (no retraining) to the bare
session's steady frames and scores it against the twin's steady mean field.
Used for the Orange Pi leave-one-load-out generalization tests; the Raspberry Pi
probe in scripts/run_unet_probe.py is the same idea with its own fixed windows.

Honesty notes:
* the target is the high-emissivity twin's camera reading, not a measured truth;
* the bare/twin pair carries a session-level offset (state mismatch), which the
  trusted-pixel score measures and which bounds the masked score;
* steady frames of one session are near-duplicates, so the effective sample is
  much smaller than the frame count.
"""

from __future__ import annotations

import numpy as np
import torch

from tdp.unet.evaluate import mask_metrics
from tdp.unet.model import build_residual_unet
from tdp.unet.pairs import theta_series
from tdp.unet.train import THETA_SCALE_K


def load_operator(model_path: str, device: str = "cpu"):
    model = build_residual_unet(in_channels=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(torch.device(device))


def _predict(model, x_theta: np.ndarray, board: np.ndarray, unknown: np.ndarray, device="cpu"):
    """Apply the operator frame by frame; returns predicted theta (T,H,W)."""
    board_ch = np.asarray(board, dtype=np.float32)[None]
    mask_ch = np.asarray(unknown, dtype=np.float32)[None]
    static = torch.tensor(
        np.concatenate([board_ch, mask_ch], axis=0)[None], dtype=torch.float32
    ).to(torch.device(device))
    x = np.where(board[None], x_theta, 0.0)
    out = np.empty_like(x, dtype=np.float64)
    with torch.no_grad():
        for i in range(len(x)):
            ch0 = torch.tensor(
                x[i : i + 1] / THETA_SCALE_K, dtype=torch.float32
            )[:, None].to(torch.device(device))
            correction = model(torch.cat([ch0, static], dim=1))[0, 0].cpu().numpy()
            out[i] = (x[i] / THETA_SCALE_K + correction) * THETA_SCALE_K
    return out


def steady_probe(
    *,
    bare_arrays: dict,
    twin_arrays: dict,
    bare_powered: np.ndarray,
    twin_powered: np.ndarray,
    board: np.ndarray,
    unknown: np.ndarray,
    model_path: str,
    steady_last: int = 25,
    device: str = "cpu",
) -> dict:
    """Apply the operator to the bare steady frames and score vs the twin mean."""
    bare_theta = theta_series(bare_arrays)
    twin_theta = theta_series(twin_arrays)
    x = bare_theta[bare_powered[-steady_last:]]
    y_frames = twin_theta[twin_powered[-steady_last:]]
    y = y_frames.mean(axis=0, keepdims=True)
    y = np.broadcast_to(y, x.shape).copy()

    # a session-level level offset measured where both sessions are trustworthy
    trusted = board & ~unknown
    bare_level = np.median(x[:, trusted], axis=1)
    twin_level = np.median(y_frames[:, trusted], axis=1)
    offset_k = float(np.median(twin_level) - np.median(bare_level))

    model = load_operator(model_path, device=device)
    pred = _predict(model, x, board, unknown, device=device)

    summary = {
        "raw": mask_metrics(x - y, board, unknown),
        "unet": mask_metrics(pred - y, board, unknown),
        "unet_offset_corrected": mask_metrics(pred - y - offset_k, board, unknown),
    }
    return {
        "summary": summary,
        "session_level_offset_k": offset_k,
        "n_frames": int(len(x)),
        "pred": pred,
        "bare": x,
        "twin_steady": y[0],
        "board": board,
        "unknown": unknown,
    }
