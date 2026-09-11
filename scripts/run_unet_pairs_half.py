"""Build the combined two-load training archive for the correction net.

Load level 1.0: the existing case10(bare) -> case17(painted) archive, 240 pairs,
matched by elapsed time since each session's own load onset.

Load level 0.5: case09(bare, constant ~51 % CPU) -> case16(painted).  case16 has
no CPU/SoC logs and its IR shows a stepped load, so elapsed matching is not
sound there: each of case16's powered frames is instead paired with the case09
powered frame whose trusted-pixel temperature level is closest (state matching,
see tdp.unet.pairs.state_matched_pairs).  The trusted residual of every pair is
reported, never assumed.

Outputs the combined archive with a per-frame load_value column and provenance
so the model can later be compared with a load-conditioned variant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

import numpy as np

from tdp.unet.pairs import (
    load_sequence_arrays,
    state_matched_pairs,
    theta_series,
)

OUT = Path("outputs/thermal/unet_case10")
FULL_ARCHIVE = OUT / "painted_pair_dataset.npz"
COMBINED_ARCHIVE = OUT / "painted_pair_dataset_both.npz"
BARE_HALF = "case09_rpi4b_bare_half"
PAINTED_HALF = "case16_rpi4b_painted_half"
HALF_POWERED_FRAMES = 240   # case09 powered 0..239 (phase labels)
HALF_PAINTED_FRAMES = 192   # case16 powered 0..191 (cooling starts at 192)


def _powered(sequence: str) -> np.ndarray:
    with np.load(
        f"outputs/dynamic_inverse/{sequence}/latest/temperature_corrected_sequence.npz"
    ) as data:
        phase = data["phase_labels"]
    idx = np.arange(len(phase))
    cooling = idx[phase == "free_cooling_candidate"]
    stop = int(cooling[0]) if cooling.size else len(phase)
    return np.arange(stop)


def main() -> int:
    # ---- load level 1.0 pairs (existing archive) ----
    with np.load(FULL_ARCHIVE) as full:
        x_full = full["x_theta"].astype(np.float64)
        y_full = full["y_theta"].astype(np.float64)
        board = full["board_mask"]
        unknown = full["unknown_mask"]
        full_flags = full["painted_flag"]
    trusted = board & ~unknown

    # ---- load level 0.5 pairs by state matching ----
    bare09 = load_sequence_arrays(BARE_HALF)
    painted16 = load_sequence_arrays(PAINTED_HALF)
    powered09 = _powered(BARE_HALF)
    powered16 = _powered(PAINTED_HALF)
    if len(powered09) != HALF_POWERED_FRAMES or len(powered16) != HALF_PAINTED_FRAMES:
        raise ValueError(
            f"half-load powered frames changed: case09 {len(powered09)}, case16 {len(powered16)}"
        )
    bare_theta = theta_series(bare09)
    painted_theta = theta_series(painted16)
    valid09 = bare09["valid_mask"]
    valid16 = painted16["valid_mask"]
    common = board & valid09 & valid16
    level_pixels = common & ~unknown
    masked_pixels = common & unknown
    if level_pixels.sum() == 0 or masked_pixels.sum() == 0:
        raise ValueError("no shared trusted/masked pixels between half-load sessions")

    from scipy.ndimage import label as _label

    # reusable trusted-pixel set that is valid in every session
    bare_level = np.median(bare_theta[powered09][:, level_pixels], axis=1)
    painted_level = np.median(painted_theta[powered16][:, level_pixels], axis=1)
    bare_idx, painted_idx, half_diag = state_matched_pairs(
        bare_theta,
        painted_theta,
        powered09,
        powered16,
        bare_level,
        painted_level,
        trusted=level_pixels,
        masked=masked_pixels,
        label="case09 bare half -> case16 painted half (state matched)",
    )
    x_half = np.where(common[None], bare_theta[powered09][bare_idx], 0.0)
    y_half = np.where(common[None], painted_theta[powered16][painted_idx], 0.0)

    # ---- combine ----
    x = np.concatenate([x_full, x_half], axis=0)
    y = np.concatenate([y_full, y_half], axis=0)
    load_value = np.concatenate(
        [np.full(len(x_full), 1.0), np.full(len(x_half), 0.5)]
    )
    provenance = np.concatenate(
        [
            np.full(len(x_full), "case10->case17", dtype="U16"),
            np.full(len(x_half), "case09->case16", dtype="U16"),
        ]
    )
    # zero x outside the board so the input channel is consistent for all frames
    x = np.where(board[None], x, 0.0)

    np.savez_compressed(
        COMBINED_ARCHIVE,
        x_theta=x.astype(np.float32),
        y_theta=y.astype(np.float32),
        board_mask=board,
        unknown_mask=unknown,
        load_value=load_value.astype(np.float64),
        provenance=provenance,
        full_flag=np.concatenate(
            [np.asarray(full_flags), np.full(len(x_half), "state", dtype="U8")]
        ),
    )
    meta = {
        "label": "combined two-load training archive",
        "n_full_load_pairs": int(len(x_full)),
        "n_half_load_pairs": int(len(x_half)),
        "board_px": int(board.sum()),
        "trusted_px": int(trusted.sum()),
        "masked_px": int(unknown.sum()),
        "loss": "complete-map L1 over shared board pixels",
        "half_load_diagnostics": half_diag,
        "state_match_note": (
            "case16 has no logs and a stepped load; each painted case16 frame is "
            "paired with the case09 bare frame at the closest trusted level"
        ),
    }
    (COMBINED_ARCHIVE.with_suffix(".json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    # figure: level matching audit
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(powered09, bare_level, label="case09 bare level")
    axes[0].plot(powered16, painted_level, label="case16 painted level")
    axes[0].set_xlabel("powered frame"); axes[0].set_ylabel("trusted median theta (K)")
    axes[0].legend(); axes[0].set_title("levels before matching")
    axes[1].hist(bare_level[bare_idx] - painted_level, bins=30)
    axes[1].set_xlabel("matched level gap (K)"); axes[1].set_title("state-match level gaps")
    axes[2].hist(np.abs(y_half - x_half)[:, trusted].mean(axis=1), bins=30)
    axes[2].set_xlabel("per-pair trusted |residual| (K)")
    axes[2].set_title("half-load pairing noise")
    fig.tight_layout()
    fig.savefig(OUT / "half_load_state_match.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(meta, indent=2))
    print(f"wrote {COMBINED_ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
