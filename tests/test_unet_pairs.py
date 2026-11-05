"""Unit tests for the bare->painted pair archive builder (M1, U-Net branch).

These lock the pairing conventions: contiguous splits with guard gaps, powered
frames cut at the load stop, theta fields above each frame's ambient, per-pixel
interpolation inside painted coverage, and a flagged steady-mean clamp beyond
it.  No real dataset is required except in the artifact-dependent test, which
skips when the archive is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdp.unet.pairs import (
    contiguous_split,
    powered_frames_by_clock,
    theta_series,
)


def test_contiguous_split_enforces_guard_gaps():
    with pytest.raises(ValueError, match="guard"):
        contiguous_split(240, train=(0, 100), val=(99, 130))
    with pytest.raises(ValueError, match="guard"):
        contiguous_split(240, train=(0, 130), val=(130, 180))
    with pytest.raises(ValueError, match="past"):
        contiguous_split(240, train=(0, 130), val=(135, 180), test=(185, 300))


def test_contiguous_split_returns_ordered_disjoint_blocks():
    blocks = contiguous_split(240)
    assert {name: int(len(idx)) for name, idx in blocks.items()} == {
        "train": 135, "val": 42, "test": 53,
    }
    all_idx = np.concatenate([blocks[k] for k in ("train", "val", "test")])
    assert len(set(all_idx.tolist())) == 230  # 10 guard frames excluded
    # blocks are chronological and separated by >= 5 guard frames
    assert blocks["train"][-1] + 5 < blocks["val"][0]
    assert blocks["val"][-1] + 5 < blocks["test"][0]


def test_theta_series_subtracts_per_frame_ambient():
    arrays = {
        "raw_temperature_c": np.arange(6, dtype=float).reshape(2, 1, 3) + 30.0,
        "ambient_c": np.array([28.0, 29.0]),
    }
    theta = theta_series(arrays)
    assert np.allclose(theta[0], [2.0, 3.0, 4.0])
    assert np.allclose(theta[1], [4.0, 5.0, 6.0])  # raw also rose by 1 K


def test_powered_frames_by_clock_cuts_at_load_stop():
    arrays = {
        "clock_start_epoch_s": 1000.0,
        "timestamps_s": np.arange(10, dtype=float) * 6.0,
    }
    powered = powered_frames_by_clock(arrays, load_stop_epoch_s=1000.0 + 6.0 * 5.5)
    assert list(powered) == [0, 1, 2, 3, 4, 5]  # last frame with epoch <= stop


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "outputs/thermal/unet_case10/painted_pair_dataset.npz"
    ).exists(),
    reason="run scripts/run_unet_pairs.py first",
)
def test_painted_pair_archive_contract():
    with np.load("outputs/thermal/unet_case10/painted_pair_dataset.npz") as data:
        x = data["x_theta"]
        y = data["y_theta"]
        assert x.shape == (240, 157, 103) == y.shape
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        flags = data["painted_flag"]
        counts = {k: int((flags == k).sum()) for k in ("interp", "steady", "first")}
        assert counts["interp"] + counts["steady"] == 240
        assert counts["first"] == 0
        assert data["val_idx"].min() > data["train_idx"].max() + 4
        assert data["test_idx"].min() > data["val_idx"].max() + 4
