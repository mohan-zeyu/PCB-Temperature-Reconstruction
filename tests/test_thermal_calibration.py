"""Cross-case calibration must fix radiometric ambiguity without hiding it."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from tdp.thermal.calibration import (
    LoadInterval,
    align_to_frames,
    longest_high_load_interval,
    paired_attenuation_mask,
    read_timestamped_values,
    transfer_surface_anchor,
)


def test_timestamped_logs_align_and_find_the_longest_load_interval(tmp_path):
    start = datetime(2026, 1, 2, 3, 4, 5)
    rows = ["timestamp,cpu_usage_percent,soc_temp_c"]
    loads = [2, 100, 100, 3, 100, 100, 100, 2]
    for index, load in enumerate(loads):
        stamp = start + timedelta(seconds=2 * index)
        rows.append(f"{stamp.isoformat(sep=' ')},{load},{45 + index}")
    path = tmp_path / "combined.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    load_log = read_timestamped_values(
        path, value_columns=("total_percent", "cpu_usage_percent")
    )
    interval = longest_high_load_interval(load_log)
    assert interval.n_samples == 3
    assert interval.start_epoch_s == pytest.approx((start + timedelta(seconds=8)).timestamp())
    temperature_log = read_timestamped_values(
        path, value_columns=("temp_c", "soc_temp_c")
    )
    frames = np.array(
        [(start + timedelta(seconds=value)).timestamp() for value in (1, 5, 9)]
    )
    assert np.allclose(align_to_frames(temperature_log, frames), [45.5, 47.5, 49.5])


def test_alignment_refuses_silent_temperature_log_extrapolation(tmp_path):
    path = tmp_path / "temperature.csv"
    path.write_text(
        "datetime,temp_c\n2026-01-01 00:00:00,40\n2026-01-01 00:00:02,42\n",
        encoding="utf-8",
    )
    log = read_timestamped_values(path, value_columns=("temp_c", "soc_temp_c"))
    with pytest.raises(ValueError, match="does not cover"):
        align_to_frames(log, np.array([log.epoch_s[0] - 1.0]))


def test_painted_pair_flags_a_low_emissivity_source_even_when_it_still_looks_hot():
    shape = (32, 28)
    bare = np.full((8, *shape), 20.0)
    painted = bare.copy()
    source = np.zeros(shape, dtype=bool)
    source[10:20, 9:19] = True
    # The bare source is still 10 K hotter than its neighbours, so a cold-drop
    # detector cannot seed it.  Painted truth shows that it was suppressed by
    # another 15 K and therefore must not be called trustworthy.
    bare[:, source] = 30.0
    painted[:, source] = 45.0
    steady = np.ones(8, dtype=bool)
    result = paired_attenuation_mask(
        bare,
        steady,
        painted,
        steady,
        threshold_k=5.0,
        smooth_px=0.0,
        min_region_px=6,
    )
    assert result.mask[source].mean() > 0.9
    assert result.mask[11:19, 10:18].all()
    assert not result.mask[0, 0]
    assert result.report["n_flagged_px"] >= int(0.9 * source.sum())


def test_painted_die_to_surface_difference_is_transferred_by_load_elapsed_time():
    reference_frames = np.arange(0.0, 121.0, 10.0)
    reference_load = LoadInterval(10.0, 110.0, 101)
    reference_die = 60.0 + 20.0 * (1.0 - np.exp(-np.maximum(reference_frames - 10.0, 0) / 20.0))
    difference = np.where(reference_frames < 80.0, 2.0 + 0.025 * reference_frames, 4.0)
    reference_surface = reference_die - difference

    target_frames = np.arange(20.0, 181.0, 10.0)
    target_load = LoadInterval(20.0, 180.0, 161)
    target_die = np.full(target_frames.shape, 82.0)
    result = transfer_surface_anchor(
        target_frames,
        target_die,
        target_load,
        reference_frames,
        reference_die,
        reference_surface,
        reference_load,
        steady_window_s=30.0,
        uncertainty_floor_k=1.5,
    )
    target_steady = target_frames >= 150.0
    assert np.allclose(result.surface_temperature_c[target_steady], 78.0)
    assert np.all(result.uncertainty_k == 1.5)
    assert result.report["die_to_surface_offset_mean_k"] == pytest.approx(4.0)
