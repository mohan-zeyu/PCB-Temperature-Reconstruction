"""Train the residual U-Net on the full bare->painted pair archive.

The owner's decision for this experiment: no train/test split; the network
trains on the whole case10 <-> case17 archive (240 powered frames).  Honest
generalization therefore cannot be claimed from case10 frames alone; it is
tested separately by the case09 <-> case16 cross-load probe (M4), never by a
random frame split (which this project forbids).

Loss is the owner's choice: complete-map L1 between predicted and painted theta
over the shared board pixels.  Masked-area and trusted-area MAE are always
reported as diagnostics, but the objective itself is the full board map.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tdp.unet.model import build_residual_unet

# Fixed unit for the network: theta in kelvin divided by 60 K.  Never per-map
# normalisation: the artifact correction depends on absolute temperature.
THETA_SCALE_K = 60.0


class PairedSet:
    """Whole-archive tensors (no split) for the residual correction net."""

    def __init__(self, archive_path: str | Path, mask_source: str = "combined"):
        with np.load(archive_path) as data:
            x_theta = data["x_theta"].astype(np.float64)
            y_theta = data["y_theta"].astype(np.float64)
            board = data["board_mask"]
            unknown = data["unknown_mask"]
            elapsed = (
                data["elapsed_s"]
                if "elapsed_s" in data.files
                else np.arange(len(x_theta))
            )
            for key in ("painted_flag", "full_flag", "provenance"):
                if key in data.files:
                    flags = data[key]
                    break
            else:
                flags = np.full(len(x_theta), "", dtype="U8")
        if mask_source == "image-only":
            with np.load(Path(archive_path).with_name("paired_masks.npz")) as masks:
                unknown = masks["image_only_unknown_mask"]
        elif mask_source != "combined":
            raise ValueError(f"unknown mask_source {mask_source!r}")
        x_theta = np.where(board[None], x_theta, 0.0)
        y_theta = np.where(board[None], y_theta, 0.0)
        self.x_norm = torch.tensor(x_theta / THETA_SCALE_K, dtype=torch.float32)
        self.y_norm = torch.tensor(y_theta / THETA_SCALE_K, dtype=torch.float32)
        board_ch = np.asarray(board, dtype=np.float32)[None]
        mask_ch = np.asarray(unknown, dtype=np.float32)[None]
        self.static_channels = torch.tensor(
            np.concatenate([board_ch, mask_ch], axis=0), dtype=torch.float32
        )
        self.board = torch.tensor(board, dtype=torch.bool)
        self.unknown = torch.tensor(unknown, dtype=torch.bool)
        self.elapsed_s = elapsed
        self.flags = flags

    def __len__(self) -> int:
        return int(self.x_norm.shape[0])

    def inputs(self, indices: np.ndarray) -> torch.Tensor:
        ch0 = self.x_norm[indices][:, None, :, :]
        static = self.static_channels.unsqueeze(0).expand(len(indices), -1, -1, -1)
        return torch.cat([ch0, static], dim=1)

    def targets(self, indices: np.ndarray) -> torch.Tensor:
        return self.y_norm[indices][:, None, :, :]


def mask_metrics(pred_k: np.ndarray, y_k: np.ndarray, board: np.ndarray, unknown: np.ndarray) -> dict:
    """Per-region-style diagnostics for one predictor over frames.

    ``masked`` means pixels inside a footprint *and* on the shared board: a few
    masked pixels sit on rows where the bare board's valid mask has no surface,
    the input was zeroed there by construction, and no label exists - counting
    them would pollute every masked statistic.
    """
    trusted = board & ~unknown
    masked = board & unknown
    err = pred_k - y_k
    return {
        "board_mae_k": float(np.abs(err[:, board]).mean()),
        "trusted_mae_k": float(np.abs(err[:, trusted]).mean()),
        "trusted_bias_k": float(err[:, trusted].mean()),
        "masked_mae_k": float(np.abs(err[:, masked]).mean()),
        "masked_bias_k": float(err[:, masked].mean()),
        "masked_rmse_k": float(np.sqrt((err[:, masked] ** 2).mean())),
    }


def train_full_set(
    archive_path: str | Path,
    outdir: str | Path,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 12,
    seed: int = 0,
    device: str = "cpu",
    mask_source: str = "combined",
    base_channels: int = 24,
    depth: int = 3,
    report_every: int = 10,
) -> dict:
    """Train on the whole archive; return final metrics and save artifacts."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    dataset = PairedSet(archive_path, mask_source=mask_source)
    n = len(dataset)
    all_idx = np.arange(n)
    model = build_residual_unet(in_channels=3, base_channels=base_channels, depth=depth)
    model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.L1Loss(reduction="none")

    config = {
        "archive": str(archive_path),
        "n_frames": n,
        "split": "none (owner decision: train on the whole archive)",
        "loss": "complete-map L1 over shared board pixels, kelvin",
        "theta_scale_k": THETA_SCALE_K,
        "mask_source": mask_source,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "base_channels": base_channels,
        "depth": depth,
        "n_parameters": model.n_parameters(),
    }

    def evaluate() -> dict:
        model.eval()
        pred_k = np.empty((n,) + tuple(dataset.x_norm.shape[1:]), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, 64):
                idx = np.arange(start, min(start + 64, n))
                x = dataset.inputs(idx).to(torch_device)
                corr = model(x)[:, 0].cpu().numpy()
                pred_k[idx] = (dataset.x_norm[idx].numpy() + corr) * THETA_SCALE_K
        metrics = mask_metrics(pred_k, dataset.y_norm.numpy() * THETA_SCALE_K,
                               dataset.board.numpy(), dataset.unknown.numpy())
        return metrics, pred_k

    history = []
    best = {"epoch": -1, "board_mae_k": float("inf")}
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(n)
        epoch_loss_k = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            x = dataset.inputs(idx).to(torch_device)
            y = dataset.targets(idx).to(torch_device)
            corr = model(x)
            pred = dataset.x_norm[idx][:, None].to(torch_device) + corr
            board = dataset.board.to(torch_device)
            loss = loss_fn(pred, y)[:, 0][:, board].mean() * THETA_SCALE_K
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_k += float(loss.detach())
            n_batches += 1
        scheduler.step()
        if epoch % report_every == 0 or epoch == epochs:
            metrics, _ = evaluate()
            history.append({"epoch": epoch, "train_l1_k": epoch_loss_k / n_batches, **metrics})
            if metrics["board_mae_k"] < best["board_mae_k"]:
                best = {"epoch": epoch, "board_mae_k": metrics["board_mae_k"]}
                torch.save(model.state_dict(), out / "model_best.pt")
            print(f"epoch {epoch:4d}  train {epoch_loss_k / n_batches:6.3f} K | "
                  f"board {metrics['board_mae_k']:.3f}  trusted {metrics['trusted_mae_k']:.3f}  "
                  f"masked {metrics['masked_mae_k']:.3f} K")

    model.load_state_dict(torch.load(out / "model_best.pt", weights_only=True))
    final_metrics, pred_k = evaluate()
    history.append({"epoch": best["epoch"], **final_metrics})

    np.savez_compressed(
        out / "predictions.npz",
        predicted_theta_k=pred_k.astype(np.float32),
        target_theta_k=(dataset.y_norm.numpy() * THETA_SCALE_K).astype(np.float32),
        input_theta_k=(dataset.x_norm.numpy() * THETA_SCALE_K).astype(np.float32),
        board_mask=dataset.board.numpy(),
        unknown_mask=dataset.unknown.numpy(),
    )
    with (out / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "history": history, "best": best,
                   "final": final_metrics}, handle, indent=2)
    return {"config": config, "best": best, "final": final_metrics}
