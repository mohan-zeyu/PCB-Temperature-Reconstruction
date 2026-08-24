"""Heat-source recovery and temperature forecasting from infrared board images.

Three stages, in order:

1. :mod:`tdp.thermal.sources` -- measure the board's rates and find where it makes
   heat, using the images alone.
2. :mod:`tdp.thermal.reconstruct` -- build a full temperature map for every frame,
   driven by those sources rather than interpolated from footprint edges.
3. :mod:`tdp.thermal.patterns` / :mod:`tdp.thermal.model` -- forecast forward from
   any moment.

Everything here is relative. The governing equation has been divided by the areal
heat capacity, which is what makes the remaining quantities measurable from a
temperature movie and what removes the absolute scale. Rates are in kelvin per
second; nothing in this package is a power, and heat capacity varies across the
board, so even ratios between hot spots are not power ratios.
"""

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
    "CoolingFit",
    "FootprintVerdict",
    "HeatingRateMap",
    "Hotspot",
    "LoadDrop",
    "SpreadFit",
    "cooling_rate_map",
    "cross_check",
    "emissivity_gain",
    "find_hotspots",
    "find_load_drop",
    "fit_cooling_rate",
    "fit_spread_rate",
    "judge_footprints",
    "measure_heating_rate",
]
