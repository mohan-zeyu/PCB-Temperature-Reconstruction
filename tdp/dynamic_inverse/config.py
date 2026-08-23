"""Typed configuration loading for the dynamic-inversion pipeline.

Two YAML files carry different kinds of knowledge and are kept apart on
purpose.  ``configs/measurements_v3.yaml`` records what the *experiment* was --
which directory holds which sequence, how the board sits in the frame, what the
camera was set to -- and changing it means restating a measurement.
``configs/dynamic_inverse.yaml`` records what the *analysis* does -- region
count, loss weights, epochs -- and changing it means rerunning a fit.  Mixing
them makes an ablation look like a re-measurement.

Every field is optional at the call site: :func:`load_config` deep-merges the
file over the dataclass defaults, so a config may set only what it overrides
and a missing file still yields a runnable default configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# measurement-side configuration
# --------------------------------------------------------------------------
@dataclass
class BoardWindow:
    """Where the PCB sits in the raw frame.

    ``angle_deg`` and ``center_rc``/``size_hw`` are estimated from the data when
    left at ``None``; supplying them pins a sequence to a window agreed once, so
    a rerun cannot silently re-frame the board.  ``output_hw`` is the canonical
    board grid every sequence is resampled onto, which is what makes the pixel
    scale a property of the analysis rather than of the shot.
    """

    center_rc: tuple[float, float] | None = None
    size_hw: tuple[float, float] | None = None
    angle_deg: float | None = None
    output_hw: tuple[int, int] = (157, 103)
    search_angle_deg: float = 20.0
    # PCB edges read off the *derotated* frame, as {top, bottom, left, right}
    # row/column indices.  Stating them beats auto-detection on a bare board:
    # the ports protrude past the PCB outline and the warm air halo has no edge
    # of its own, so an automatic outline is a few pixels ambiguous.  When
    # omitted the loader falls back to detection and records the aspect
    # residual so a bad fallback is visible rather than silent.
    edges_derotated: dict[str, float] | None = None


@dataclass
class Radiometry:
    """Camera-side radiometric settings.

    ``assumed_emissivity`` is what the camera divided by when it wrote the CSV.
    The 2026-07-26/27 sessions did not record it (the Chinese-dialect export
    omits the parameter block), so it is an assumption, and
    ``sensitivity_sweep`` exists to show how far the results move across the
    plausible range instead of hiding the choice.
    """

    assumed_emissivity: float = 0.95
    reflected_temperature_c: float | None = None  # None -> use the ambient estimate
    band_exponent: float = 4.0
    sensitivity_sweep: tuple[float, ...] = (0.90, 0.93, 0.95, 1.00)
    recorded: bool = False


@dataclass
class RepairSettings:
    """How the low-emissivity repair is applied across a sequence.

    The mask is detected once on a reference frame and reused for every frame:
    the board does not move within a session, the footprints therefore do not
    move, and a fixed mask is the only way the repaired series is free of
    frame-to-frame mask flicker that would look like real dynamics.
    """

    enabled: bool = True
    method: str = "conduction"
    fin_length_px: float = 100.0
    reference_quantile: float = 0.9  # pick the reference frame from the hot end
    trust_decay_px: float = 6.0
    max_extrapolation_px: float = 24.0
    infer_sources: bool = False


@dataclass
class SequenceSpec:
    """One recorded sequence."""

    name: str = ""
    case_dir: str = ""
    device: str = ""
    surface: str = ""
    load: str = ""
    board: BoardWindow = field(default_factory=BoardWindow)
    radiometry: Radiometry = field(default_factory=Radiometry)
    repair: RepairSettings = field(default_factory=RepairSettings)
    logs: dict[str, str] = field(default_factory=dict)
    phase_overrides: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass
class MeasurementConfig:
    dataset_root: str = "实验数据（黑胶带版）"
    board_size_mm: tuple[float, float] = (85.0, 56.0)
    board_size_note: str = "Raspberry Pi 4B PCB outline"
    sequences: dict[str, SequenceSpec] = field(default_factory=dict)


# --------------------------------------------------------------------------
# analysis-side configuration
# --------------------------------------------------------------------------
@dataclass
class PhaseConfig:
    smooth_seconds: float = 120.0
    # The steady/moving boundary is set as a fraction of the sequence's own
    # peak slope rather than as an absolute K/s.  A fixed constant cannot be
    # right for both a 6 s and a 60 s cadence, for both "s" and "frame" time
    # units, and for a bare board and a painted one -- and on this data a
    # constant tight enough for one sequence shatters the other into a dozen
    # spurious phases.  `steady_slope_k_per_s` remains as an absolute floor.
    slope_threshold_relative: float = 0.25
    steady_slope_k_per_s: float = 1.0e-4
    min_phase_seconds: float = 180.0
    idle_window_seconds: float = 300.0
    rewarm_tolerance_k: float = 0.5


@dataclass
class RegionConfig:
    n_regions: int = 64
    method: str = "slic"  # slic | grid
    compactness: float = 0.5
    resolutions: tuple[int, ...] = (32, 64, 128)
    use_emissivity_feature: bool = True
    use_trajectory_feature: bool = True
    min_valid_fraction: float = 0.25


@dataclass
class ModelConfig:
    node_embedding_dim: int = 32
    hidden_dim: int = 64
    source_hidden_dim: int = 32
    source_time_features: int = 8
    dropout: float = 0.0


@dataclass
class LossWeights:
    rollout: float = 1.0
    observed: float = 0.5
    source_l1: float = 3e-3
    source_tv_time: float = 3e-3
    edge_reg: float = 1e-4
    spatial_smooth: float = 1e-3
    stability: float = 1.0
    # Applied to the source the operator implies for the *measured* steady
    # profile, s = -A theta_steady.  These are what separate conduction from
    # convection: the decay alone cannot, because uncoupled regions each
    # relaxing at their own beta look almost identical to coupled ones relaxing
    # together.  Set both to zero for the ablation.
    steady_source_negativity: float = 1.0
    steady_source_spread: float = 1.0


@dataclass
class TrainConfig:
    epochs_thermal: int = 400
    epochs_source: int = 400
    epochs_joint: int = 100
    lr_thermal: float = 5e-3
    # Learning rate for the two global log-magnitudes.  Large on purpose: they
    # move multiplicatively, so this covers a decade in about fifty steps.
    lr_scale: float = 5e-2
    lr_source: float = 1e-2
    lr_joint: float = 5e-4
    rollout_steps: int = 12
    rollout_stride: int = 4
    batch_windows: int = 16
    integrator: str = "rk4"
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    patience: int = 80


@dataclass
class SplitConfig:
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    # "interleaved": whole contiguous chunks assigned to train/val/test, so
    # training covers the fast head and the slow tail of a single decay.
    # "sequential": first 60% / next 20% / last 20% -- pure forward
    # extrapolation, reported alongside as long_rollout_drift_k.
    mode: str = "interleaved"
    n_blocks: int = 5
    # Frames dropped either side of every block boundary.  Without a guard the
    # split leaks: the last training frame and the first evaluation frame are
    # 6 s apart and nearly the same measurement.
    guard_frames: int = 3


@dataclass
class UncertaintyConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    bootstrap_windows: int = 4
    region_resolutions: tuple[int, ...] = (48, 64, 96)
    high_cv_threshold: float = 0.5


@dataclass
class AnalysisConfig:
    phases: PhaseConfig = field(default_factory=PhaseConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    output_dir: str = "outputs/dynamic_inverse"
    seed: int = 0
    device: str = "cpu"


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------
def _merge(target: Any, patch: Any) -> Any:
    """Deep-merge ``patch`` (plain dicts from YAML) onto a dataclass instance."""
    if patch is None:
        return target
    if not dataclasses.is_dataclass(target):
        return patch
    if not isinstance(patch, dict):
        raise TypeError(f"cannot merge {type(patch).__name__} onto {type(target).__name__}")
    known = {f.name: f for f in dataclasses.fields(target)}
    unknown = set(patch) - set(known)
    if unknown:
        raise ValueError(
            f"{type(target).__name__} has no field(s) {sorted(unknown)}; "
            f"known fields are {sorted(known)}"
        )
    updates: dict[str, Any] = {}
    for key, value in patch.items():
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            updates[key] = _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return dataclasses.replace(target, **updates)


def _read_yaml(path: Path | str | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded or {}


def load_measurements(path: Path | str | None) -> MeasurementConfig:
    raw = _read_yaml(path)
    sequences_raw = raw.pop("sequences", {}) or {}
    config = _merge(MeasurementConfig(), raw)
    sequences: dict[str, SequenceSpec] = {}
    for name, spec_raw in sequences_raw.items():
        spec = _merge(SequenceSpec(name=name), spec_raw or {})
        sequences[name] = dataclasses.replace(spec, name=name)
    return dataclasses.replace(config, sequences=sequences)


def load_analysis(path: Path | str | None) -> AnalysisConfig:
    return _merge(AnalysisConfig(), _read_yaml(path))


def as_dict(config: Any) -> dict:
    """Dataclass tree -> plain dict, for run_metadata.json."""
    if dataclasses.is_dataclass(config):
        return {f.name: as_dict(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if isinstance(config, dict):
        return {k: as_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [as_dict(v) for v in config]
    if isinstance(config, Path):
        return str(config)
    return config
