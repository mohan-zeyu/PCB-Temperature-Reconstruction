"""Heat-source recovery and temperature forecasting from infrared board images.

Four stages, in order:

1. :mod:`tdp.thermal.sources` -- measure the board's rates and find where it makes
   heat, using the images alone.
2. :mod:`tdp.thermal.calibration` -- audit case10's reliability mask against the
   painted case17 session and convert the logged SoC die temperature into a
   frame-aligned surface-temperature constraint.
3. :mod:`tdp.thermal.reconstruct` -- build a full temperature map for every frame,
   driven by those sources and the calibrated SoC level rather than interpolated
   from footprint edges.
4. :mod:`tdp.thermal.patterns` / :mod:`tdp.thermal.forecast` -- compress the
   reconstructed field into fixed physical patterns and forecast forward from
   any moment with a closed-form plate solve.

Everything here is relative. The governing equation has been divided by the areal
heat capacity, which is what makes the remaining quantities measurable from a
temperature movie and what removes the absolute scale. Rates are in kelvin per
second; nothing in this package is a power, and heat capacity varies across the
board, so even ratios between hot spots are not power ratios.
"""

from tdp.thermal.calibration import (
    LoadInterval,
    PairedAttenuation,
    SurfaceAnchorSeries,
    TimestampedValues,
    align_to_frames,
    longest_high_load_interval,
    paired_attenuation_mask,
    read_timestamped_values,
    sequence_start_epoch,
    steady_frame_mask,
    transfer_surface_anchor,
)

from tdp.thermal.reconstruct import (
    BlockSource,
    InteriorTemperatureAnchor,
    Reconstruction,
    TruthCheck,
    heating_rate_from_reconstruction,
    plate_residual,
    block_labels,
    reconstruct_sequence,
    validate_against_truth,
    warming_rate,
)
from tdp.thermal.forecast import (
    ClosedFormPlateModel,
    ReconstructionArchive,
    load_reconstruction_archive,
)
from tdp.thermal.patterns import (
    PatternLibrary,
    build_pattern_library,
    gaussian_source_shape,
    load_pattern_library,
    plate_laplacian,
    steady_source_response,
)
from tdp.thermal.presentation import (
    model_range_fields,
    nearest_sample_indices,
    weighted_mae_by_frame,
)
from tdp.thermal.train_forecast import (
    EquilibriumForecastFit,
    FrozenForecastSplit,
    SourceHistoryFit,
    fit_equilibrium_forecast,
    fit_source_histories,
    make_frozen_split,
    rollout_to_equilibrium,
)
from tdp.thermal.sources import (
    ASSUMED_EMISSIVITY,
    BAND_EXPONENT,
    CoolingFit,
    FootprintVerdict,
    HeatingRateMap,
    Hotspot,
    LoadDrop,
    SpreadFit,
    cooling_rate_map,
    cross_check,
    emissivity_gain,
    find_hotspots,
    find_load_drop,
    fit_cooling_rate,
    fit_spread_rate,
    judge_footprints,
    measure_heating_rate,
)

__all__ = [
    "ASSUMED_EMISSIVITY",
    "BAND_EXPONENT",
    "BlockSource",
    "ClosedFormPlateModel",
    "InteriorTemperatureAnchor",
    "Reconstruction",
    "TruthCheck",
    "CoolingFit",
    "EquilibriumForecastFit",
    "FootprintVerdict",
    "FrozenForecastSplit",
    "HeatingRateMap",
    "Hotspot",
    "LoadInterval",
    "LoadDrop",
    "PairedAttenuation",
    "PatternLibrary",
    "ReconstructionArchive",
    "SpreadFit",
    "SourceHistoryFit",
    "SurfaceAnchorSeries",
    "TimestampedValues",
    "align_to_frames",
    "cooling_rate_map",
    "cross_check",
    "emissivity_gain",
    "find_hotspots",
    "find_load_drop",
    "fit_cooling_rate",
    "fit_equilibrium_forecast",
    "fit_source_histories",
    "fit_spread_rate",
    "gaussian_source_shape",
    "heating_rate_from_reconstruction",
    "judge_footprints",
    "load_reconstruction_archive",
    "load_pattern_library",
    "measure_heating_rate",
    "model_range_fields",
    "make_frozen_split",
    "nearest_sample_indices",
    "plate_residual",
    "plate_laplacian",
    "longest_high_load_interval",
    "paired_attenuation_mask",
    "read_timestamped_values",
    "block_labels",
    "reconstruct_sequence",
    "rollout_to_equilibrium",
    "sequence_start_epoch",
    "steady_frame_mask",
    "build_pattern_library",
    "steady_source_response",
    "validate_against_truth",
    "transfer_surface_anchor",
    "weighted_mae_by_frame",
    "warming_rate",
]
