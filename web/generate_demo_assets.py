"""Export a small, truthful presentation sample from existing model archives.

Run from the repository root with ``.venv/bin/python web/generate_demo_assets.py``.
These assets are the **read-only fallback** used when no backend is running.  With
``web/server.py`` the browser calls the real model instead; the fallback never
pretends to train or to infer a new frame.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets"
SAMPLES = Path(__file__).resolve().parent / "samples"


def packed_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode("ascii")


def frame_values(field: np.ndarray) -> list[int]:
    # 0.1 K precision keeps the browser asset small without changing the source.
    return np.rint(field.ravel() * 10).astype(np.int16).tolist()


def fill_edge_gaps_for_display(field: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Mirror the nearest-valid then Gaussian presentation fill in figure_unet_predictions."""
    if np.all(board):
        return field.copy()
    _, indices = distance_transform_edt(~board, return_indices=True)
    completed = field.copy()
    completed[~board] = field[indices[0][~board], indices[1][~board]]
    smoothed = gaussian_filter(completed, sigma=2.0, mode="nearest")
    completed[~board] = smoothed[~board]
    return completed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        {
            "id": "raspberry-pi-4b",
            "name": "Raspberry Pi 4B",
            "path": ROOT / "outputs/thermal/unet_case10/presentation/predicted_maps_case10.npz",
            "raw_key": "bare_theta_k",
            "pred_key": "predicted_theta_k",
            "indices": [0, 40, 100, 190, 239],
            "pair_count": 432,
            "training_path": ROOT / "outputs/thermal/unet_case10/train_both/train_metrics.json",
            "archive_note": "case10 / case17 full load and case09 / case16 half load",
            "ambient_path": None,
            "sample_case": "case10_树莓派4B裸板_满负载",
            "sample_offset": 0,
        },
        {
            "id": "orange-pi-5-pro",
            "name": "Orange Pi 5 Pro",
            "path": ROOT / "outputs/thermal/unet_opi5pro/probe_full_from_both/probe_predictions.npz",
            "raw_key": "bare_theta_k",
            "pred_key": "predicted_theta_k",
            "indices": [0, 6, 12, 18, 24],
            "pair_count": 480,
            "training_path": ROOT / "outputs/thermal/unet_opi5pro/train_both/train_metrics.json",
            "archive_note": "case12 / case14 half load and case13 / case15 full load",
            "ambient_path": ROOT / "outputs/dynamic_inverse/case13_opi5pro_bare_full/latest/temperature_corrected_sequence.npz",
            "sample_case": "case13_香橙派5Pro裸板_满负载",
            "sample_offset": 215,
        },
    ]
    for spec in sources:
        with np.load(spec["path"]) as archive:
            raw = archive[spec["raw_key"]]
            predicted = archive[spec["pred_key"]]
            board = archive["board_mask"]
            unknown = archive["unknown_mask"]
            if spec["ambient_path"] is None:
                ambient = archive["ambient_c"]
            else:
                # The full-load probe uses the final 25 powered frames, 215-239.
                # These are the same frames used to make probe_predictions.npz.
                with np.load(spec["ambient_path"]) as prepared:
                    ambient = prepared["ambient_c"][215:240]
            source_files = sorted((ROOT / "实验数据（黑胶带版）" / spec["sample_case"] / "data").glob("HM*.csv"))
            sample_dir = SAMPLES / spec["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for index in spec["indices"]:
                source = source_files[spec["sample_offset"] + index]
                destination = sample_dir / source.name
                shutil.copy2(source, destination)
                frames.append({
                    "index": int(index),
                    "ambient_c": round(float(ambient[index]), 2),
                    "source_csv": source.name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "raw": frame_values(raw[index]),
                    "predicted": frame_values(predicted[index]),
                    "raw_display": frame_values(fill_edge_gaps_for_display(raw[index], board)),
                    "predicted_display": frame_values(fill_edge_gaps_for_display(predicted[index], board)),
                })
        with spec["training_path"].open(encoding="utf-8") as handle:
            training = json.load(handle)
        if spec["id"] == "orange-pi-5-pro":
            with np.load(ROOT / "outputs/thermal/unet_opi5pro/pairs_full.npz") as paired:
                paired_preview = {
                    "bare": frame_values(paired["x_theta"][120]),
                    "painted": frame_values(paired["y_theta"][120]),
                }
        else:
            paired_preview = None
        result = {
            "id": spec["id"],
            "name": spec["name"],
            "unit": "degrees Celsius; source arrays are K above ambient",
            "width": int(board.shape[1]),
            "height": int(board.shape[0]),
            "mask": packed_mask(board),
            "unknown_mask": packed_mask(unknown),
            "board_pixels": int(board.sum()),
            "unknown_pixels": int((board & unknown).sum()),
            "pair_count": spec["pair_count"],
            "archive_note": spec["archive_note"],
            "training": {
                "epochs": training["config"]["epochs"],
                "history": [
                    {
                        "epoch": row["epoch"],
                        "train_l1_k": round(row["train_l1_k"], 3),
                        "board_mae_k": round(row["board_mae_k"], 3),
                        "masked_mae_k": round(row["masked_mae_k"], 3),
                    }
                    for row in training["history"]
                    if "train_l1_k" in row
                ],
                "best_epoch": training["best"]["epoch"],
                "best_board_mae_k": round(training["best"]["board_mae_k"], 3),
            },
            "frames": frames,
            "paired_preview": paired_preview,
        }
        (OUT / f"{spec['id']}.json").write_text(
            json.dumps(result, separators=(",", ":")), encoding="utf-8"
        )
        print(f"wrote {OUT / (spec['id'] + '.json')}")


if __name__ == "__main__":
    main()
