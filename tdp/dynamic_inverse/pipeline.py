"""Shared stage plumbing: load -> phases -> regions -> series.

The CLI subcommands each need a prefix of the same chain, and a run directory
that can be resumed from.  Keeping the chain here means ``prepare``,
``fit-thermal``, ``fit-source`` and ``evaluate`` cannot drift apart in how they
build the data they operate on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tdp.dynamic_inverse.config import AnalysisConfig, MeasurementConfig, SequenceSpec
from tdp.dynamic_inverse.data import ThermalSequence, load_sequence
from tdp.dynamic_inverse.phases import PhaseResult, detect_phases
from tdp.dynamic_inverse.regions import RegionPartition, aggregate_series, build_regions
from tdp.dynamic_inverse.train_thermal import NodeSeries


@dataclass
class Prepared:
    sequence: ThermalSequence
    phases: PhaseResult
    partition: RegionPartition
    series: NodeSeries

    def phase_payload(self) -> dict[str, Any]:
        return {
            "segments": self.phases.segments,
            "report": self.phases.report,
            "counts": self.phases.report["counts"],
        }


def prepare(
    spec: SequenceSpec,
    measurements: MeasurementConfig,
    analysis: AnalysisConfig,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    n_regions: int | None = None,
    region_seed: int = 0,
    region_jitter: float = 0.0,
    dataset_root: Path | str | None = None,
) -> Prepared:
    sequence = load_sequence(
        spec, measurements, root=dataset_root, frame_stride=frame_stride, max_frames=max_frames
    )
    phases = detect_phases(
        sequence.theta.reshape(sequence.n_frames, -1),
        sequence.timestamps_s,
        np.broadcast_to(
            np.where(sequence.valid_mask, sequence.trust_weight, 0.0).ravel()[None, :],
            (sequence.n_frames, sequence.shape_hw[0] * sequence.shape_hw[1]),
        ),
        smooth_seconds=analysis.phases.smooth_seconds,
        steady_slope=analysis.phases.steady_slope_k_per_s,
        slope_threshold_relative=analysis.phases.slope_threshold_relative,
        min_phase_seconds=analysis.phases.min_phase_seconds,
        rewarm_tolerance_k=analysis.phases.rewarm_tolerance_k,
        overrides=spec.phase_overrides,
        time_unit=sequence.time_unit,
    )
    partition = build_regions(
        sequence,
        n_regions=int(n_regions or analysis.regions.n_regions),
        method=analysis.regions.method,
        compactness=analysis.regions.compactness,
        use_emissivity=analysis.regions.use_emissivity_feature,
        min_valid_fraction=analysis.regions.min_valid_fraction,
        seed=region_seed,
        jitter=region_jitter,
    )
    series = make_series(sequence, partition, phases)
    return Prepared(sequence=sequence, phases=phases, partition=partition, series=series)


def make_series(
    sequence: ThermalSequence, partition: RegionPartition, phases: PhaseResult
) -> NodeSeries:
    theta, weight, trust = aggregate_series(sequence, partition.labels)
    return NodeSeries(
        theta=theta,
        times=sequence.timestamps_s,
        weight=weight,
        trust=trust,
        phase=phases.labels,
        powered=np.isin(phases.labels, ["heating", "near_steady"]),
        time_unit=sequence.time_unit,
    )


def run_dir(base: Path | str, sequence_name: str, tag: str = "latest") -> Path:
    path = Path(base) / sequence_name / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
        return str(value)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
