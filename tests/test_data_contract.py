"""The data contract and the pipeline stages, on synthetic and on real files.

Tests that need the recorded campaign are skipped when it is absent, so the
suite still runs in a checkout without the data tree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdp.dynamic_inverse.config import REPO_ROOT, as_dict, load_analysis, load_measurements
from tdp.dynamic_inverse.data import ThermalSequence, estimate_board_angle, estimate_board_window
from tdp.dynamic_inverse.derived import characteristic_length, diffusivity_proxy, time_constants
from tdp.dynamic_inverse.phases import COOLING, HEATING, NEAR_STEADY, detect_phases
from tdp.dynamic_inverse.physics import RegionGraph
from tdp.dynamic_inverse.regions import aggregate_series, build_regions, expand_to_pixels

MEASUREMENTS = REPO_ROOT / "configs" / "measurements_v3.yaml"
ANALYSIS = REPO_ROOT / "configs" / "dynamic_inverse.yaml"


def _dataset_root() -> Path | None:
    if not MEASUREMENTS.exists():
        return None
    root = REPO_ROOT / load_measurements(MEASUREMENTS).dataset_root
    return root if root.exists() else None


needs_data = pytest.mark.skipif(_dataset_root() is None, reason="recorded campaign not present")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_configs_parse_and_round_trip():
    measurements = load_measurements(MEASUREMENTS)
    analysis = load_analysis(ANALYSIS)
    assert measurements.sequences, "measurement config defines no sequences"
    for name, spec in measurements.sequences.items():
        assert spec.name == name
        assert spec.case_dir
        assert 0 < spec.radiometry.assumed_emissivity <= 1.0
    assert analysis.split.train_fraction + analysis.split.val_fraction < 1.0
    assert as_dict(analysis)["regions"]["n_regions"] == analysis.regions.n_regions


def test_unknown_config_field_is_rejected():
    """A typo in a YAML key must fail loudly, not be silently ignored."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write("regions:\n  n_regionz: 12\n")
        path = handle.name
    with pytest.raises(ValueError, match="no field"):
        load_analysis(path)


def test_emissivity_is_documented_as_an_assumption_not_a_measurement():
    """The camera setting was not recorded for these sessions; say so."""
    measurements = load_measurements(MEASUREMENTS)
    for spec in measurements.sequences.values():
        assert spec.radiometry.recorded is False
        assert spec.radiometry.sensitivity_sweep


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_board_angle_is_recovered_from_a_synthetic_tilted_rectangle():
    from scipy.ndimage import rotate as ndrotate

    frame = np.full((256, 192), 25.0)
    frame[40:200, 45:150] = 55.0
    for truth in (-6.0, -2.0, 0.0, 3.0, 7.0):
        tilted = ndrotate(frame, -truth, reshape=False, order=1, mode="nearest")
        assert abs(estimate_board_angle(tilted, ambient_c=25.0) - truth) < 0.8


def test_board_window_forces_the_known_aspect_and_reports_the_residual():
    frame = np.full((256, 192), 25.0)
    frame[40:200, 45:150] = 55.0  # 160 x 105, aspect 1.524 vs 85/56 = 1.518
    window = estimate_board_window(frame, ambient_c=25.0, edges_derotated={
        "top": 40, "bottom": 200, "left": 45, "right": 150
    })
    height, width = window["size_hw"]
    assert abs(height / width - 85.0 / 56.0) < 1e-9
    assert window["aspect_residual"] < 0.02
    assert window["edge_source"] == "config"


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def _synthetic_phase_curve():
    times = np.arange(0.0, 3600.0, 6.0)
    theta = np.zeros((len(times), 4))
    for k in range(4):
        rise = 20.0 * (1.0 - np.exp(-(times - 0) / 200.0))
        theta[:, k] = np.where(times < 1500.0, rise, 20.0 * np.exp(-(times - 1500.0) / 400.0))
    return theta, times


def test_phase_detection_finds_heating_then_steady_then_decay():
    theta, times = _synthetic_phase_curve()
    result = detect_phases(theta, times, smooth_seconds=60.0, min_phase_seconds=120.0)
    labels = [s["label"] for s in result.segments]
    assert labels[0] == HEATING
    assert COOLING in labels
    assert NEAR_STEADY in labels
    assert result.segments[labels.index(COOLING)]["start_time"] == pytest.approx(1500.0, abs=120.0)


def test_decay_is_never_labelled_free_cooling_outright():
    """The label must stay a *candidate*: no power-down was confirmed."""
    theta, times = _synthetic_phase_curve()
    result = detect_phases(theta, times)
    assert COOLING == "free_cooling_candidate"
    assert set(np.unique(result.labels)) <= {HEATING, NEAR_STEADY, COOLING, "invalid_transition"}
    assert "candidate" in result.report["cooling_label_note"]


def test_manual_overrides_win_over_detection():
    theta, times = _synthetic_phase_curve()
    result = detect_phases(
        theta, times, overrides=[{"label": "invalid_transition", "start_index": 0, "stop_index": 30}]
    )
    assert np.all(result.labels[:31] == "invalid_transition")
    assert result.report["n_frames_changed_by_overrides"] > 0


def test_a_decay_interval_never_contains_a_rewarm():
    """The guarantee the thermal fit depends on.

    Whatever label the warming stretch ends up with, it must not be inside a
    ``free_cooling_candidate`` interval -- a fit that assumes decay would
    otherwise be handed a rising trajectory.  case10 shows this for real: the
    SoC warms again near the end of the recording while the board is nominally
    idling.
    """
    times = np.arange(0.0, 3000.0, 6.0)
    decay = 20.0 * np.exp(-times / 300.0)
    rewarm = decay + np.clip((times - 1800.0) / 100.0, 0.0, 12.0)
    theta = np.repeat(rewarm[:, None], 4, axis=1)
    result = detect_phases(theta, times, rewarm_tolerance_k=0.5)
    for segment in result.segments:
        if segment["label"] != COOLING:
            continue
        block = result.mean_theta[segment["start_index"] : segment["stop_index"] + 1]
        # No frame may sit more than the tolerance above the running minimum
        # of everything before it: that is what "still decaying" means.
        assert np.all(block <= np.minimum.accumulate(block) + 0.5 + 1e-6)
        assert segment["stop_time"] <= 1900.0


def test_a_local_rewarm_under_a_falling_mean_is_marked_invalid():
    """The board mean can keep falling while most regions turn around.

    That is the case the mean alone cannot catch, and it is why the detector
    also tracks the fraction of regions with a positive slope.
    """
    times = np.arange(0.0, 2400.0, 6.0)
    theta = np.zeros((len(times), 10))
    # Two large regions dominate the mean and keep falling throughout; eight
    # small ones turn around halfway.  The mean never rises, so only the
    # per-region agreement can catch the turnaround.
    theta[:, :2] = (60.0 * np.exp(-times / 2000.0))[:, None]
    turnaround = 4.0 * np.exp(-times / 250.0) + np.clip((times - 1200.0) * 1e-3, 0.0, 1.5)
    theta[:, 2:] = turnaround[:, None]
    result = detect_phases(theta, times, rewarm_tolerance_k=0.2, min_phase_seconds=180.0)

    assert np.all(np.diff(result.mean_theta) < 0), "the board mean must keep falling"
    assert result.positive_fraction[times > 1400].mean() > 0.5
    assert "invalid_transition" in {s["label"] for s in result.segments}
    for segment in result.segments:
        if segment["label"] == COOLING:
            assert segment["stop_time"] < 1600.0


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------
def _toy_sequence(height=40, width=30, frames=25) -> ThermalSequence:
    rows, cols = np.indices((height, width))
    times = np.arange(frames, dtype=np.float64) * 6.0
    hot = np.exp(-((rows - 12) ** 2 + (cols - 10) ** 2) / 40.0)
    temperature = 25.0 + 20.0 * hot[None] * np.exp(-times / 90.0)[:, None, None]
    valid = np.ones((height, width), bool)
    trust = np.ones((height, width), np.float32)
    trust[25:, :] = 0.3
    return ThermalSequence(
        name="toy",
        temperature_c=temperature.astype(np.float32),
        raw_temperature_c=temperature.astype(np.float32),
        valid_mask=valid,
        trust_weight=trust,
        repair_mask=np.zeros((height, width), bool),
        emissivity=np.full((height, width), 0.95, np.float32),
        timestamps_s=times,
        ambient_c=np.full(frames, 25.0, np.float32),
        pixel_size_mm=0.5,
        time_unit="s",
    )


def test_region_graph_is_valid_at_several_resolutions():
    sequence = _toy_sequence()
    for n_regions in (8, 16, 32):
        partition = build_regions(sequence, n_regions=n_regions)
        graph = partition.graph
        assert graph.n_nodes >= 2
        assert np.all(graph.edge_index[0] < graph.edge_index[1])
        assert graph.n_edges > 0
        assert partition.labels.max() == graph.n_nodes
        assert np.all(graph.areas_px > 0)


def test_aggregation_is_confidence_weighted():
    """A region built from reconstructed pixels must carry less weight."""
    sequence = _toy_sequence()
    partition = build_regions(sequence, n_regions=16)
    theta, weight, trust = aggregate_series(sequence, partition.labels)
    assert theta.shape[0] == sequence.n_frames
    assert np.all(np.isfinite(theta))
    top = partition.graph.centroids_px[:, 0] < 20
    assert trust[top].mean() > trust[~top].mean()
    assert np.all(weight >= 0)


def test_expand_to_pixels_round_trips_region_values():
    sequence = _toy_sequence()
    partition = build_regions(sequence, n_regions=16)
    values = np.arange(partition.n_regions, dtype=float)
    grid = expand_to_pixels(values, partition.labels)
    for region in range(partition.n_regions):
        assert np.all(grid[partition.labels == region + 1] == region)


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------
def test_derived_quantities_and_their_units():
    graph = RegionGraph(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]]),
        centroids_px=np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0]]),
        areas_px=np.ones(3),
        edge_length_px=np.ones(2),
    )
    edge_rate = np.array([0.01, 0.02])
    alpha = diffusivity_proxy(graph, edge_rate)
    # a * d^2 with d = 4 px: the middle node averages both of its edges.
    assert alpha[0] == pytest.approx(0.01 * 16.0)
    assert alpha[1] == pytest.approx(0.015 * 16.0)

    beta = np.array([0.001, 0.002, 0.0])
    tau, low = time_constants(beta)
    assert np.all(np.isfinite(tau)), "a near-zero rate must be floored, never infinite"
    assert low[2], "the floored region must be flagged low confidence"
    length, low_length = characteristic_length(alpha, beta)
    assert np.all(np.isfinite(length))
    assert low_length[2]


def test_nothing_reports_absolute_conductance_or_power():
    """Guard the naming discipline the harness makes non-negotiable."""
    from tdp.dynamic_inverse import derived, report

    banned = ("absolute_power", "true_conductivity", "watt")
    for module in (derived, report):
        text = (module.__doc__ or "").lower()
        for word in banned:
            assert f"{word}(" not in text
    summary = derived.summarise(
        RegionGraph(
            n_nodes=2,
            edge_index=np.array([[0], [1]]),
            centroids_px=np.array([[0.0, 0.0], [0.0, 3.0]]),
            areas_px=np.ones(2),
            edge_length_px=np.ones(1),
        ),
        np.array([0.01]),
        np.array([0.001, 0.002]),
        "s",
        0.5,
    )
    assert summary["units"]["alpha_proxy"] == "px^2/s"
    assert summary["units"]["beta"] == "1/s"
    assert "proxy" in summary["naming_note"].lower()
    assert "not a material thermal conductivity" in summary["naming_note"]


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.parametrize("name", ["case10_rpi4b_bare_full", "case17_rpi4b_painted_full"])
def test_real_sequence_satisfies_the_data_contract(name):
    from tdp.dynamic_inverse.data import load_sequence

    measurements = load_measurements(MEASUREMENTS)
    sequence = load_sequence(
        measurements.sequences[name], measurements, root=_dataset_root(), frame_stride=24
    )
    assert sequence.temperature_c.ndim == 3
    assert sequence.temperature_c.dtype == np.float32
    assert sequence.valid_mask.shape == sequence.shape_hw
    assert sequence.emissivity.shape == sequence.shape_hw
    assert sequence.timestamps_s.shape == (sequence.n_frames,)
    assert sequence.ambient_c.shape == (sequence.n_frames,)
    assert np.all(np.diff(sequence.timestamps_s) > 0), "timestamps must be monotonic"
    assert np.isfinite(sequence.temperature_c).all()
    assert sequence.time_unit == "s"
    assert 0.5 < sequence.pixel_size_mm < 0.6
    quality = sequence.metadata["quality"]
    assert quality["n_nan"] == 0 and quality["n_inf"] == 0
    assert quality["pixel_size"]["calibrated"] is False
    assert 0.0 <= quality["repair"]["masked_fraction"] < 0.7


@needs_data
def test_bare_board_needs_more_repair_than_the_painted_one():
    """The physical claim behind using case17 as a cross-check.

    Matte black paint covers the low-emissivity metal, so detection should find
    far less to repair there than on the bare board.  If this ever inverts,
    the repair is keying on something other than emissivity.
    """
    from tdp.dynamic_inverse.data import load_sequence

    measurements = load_measurements(MEASUREMENTS)
    root = _dataset_root()
    bare = load_sequence(
        measurements.sequences["case10_rpi4b_bare_full"], measurements, root=root, frame_stride=48
    )
    painted = load_sequence(
        measurements.sequences["case17_rpi4b_painted_full"], measurements, root=root, frame_stride=48
    )
    assert bare.repair_mask.mean() > 1.5 * painted.repair_mask.mean()
    assert bare.emissivity.min() < painted.emissivity.min()


@needs_data
def test_real_phase_detection_finds_the_expected_cycle():
    from tdp.dynamic_inverse.pipeline import prepare

    measurements = load_measurements(MEASUREMENTS)
    analysis = load_analysis(ANALYSIS)
    prepared = prepare(
        measurements.sequences["case10_rpi4b_bare_full"],
        measurements,
        analysis,
        frame_stride=4,
        dataset_root=_dataset_root(),
    )
    labels = [s["label"] for s in prepared.phases.segments]
    assert labels[0] == HEATING
    assert NEAR_STEADY in labels
    assert COOLING in labels
    cooling = [s for s in prepared.phases.segments if s["label"] == COOLING][0]
    # cpu_load_100.csv puts load-off about 24.9 min after the first frame.
    assert 22.0 * 60 < cooling["start_time"] < 28.0 * 60
    assert cooling["duration"] > 8 * 60
