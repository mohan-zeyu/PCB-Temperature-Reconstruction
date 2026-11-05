# Reconstruction after heat-source detection

**Document role:** implemented Stage 2 method record. This describes the current code; it
is not an open proposal. `CLAUDE.md` remains authoritative for owner decisions and status.

This document describes the current Stage 2 method. It starts after the powered
part of the sequence, the low-emissivity footprint mask, the cooling rate, the
spread rate, and the confident heat-source blocks have been identified.

The method reconstructs every masked footprint in every powered infrared frame.
For `case10`, that means frames 0 through 239. It does not estimate emissivity,
conductivity, heat capacity, or power.

## 1. Inputs

The default case10 reconstruction uses:

- the raw infrared temperature above the frame's ambient temperature;
- the exact capture time of every frame;
- the painted case17 infrared sequence at the same full load;
- the absolute sequence clocks, CPU-load logs, and SoC die-temperature logs for
  both sessions;
- a binary mask formed from the image-detected footprints plus the paired
  attenuation audit;
- the trusted-board heating-rate map measured outside those footprints;
- the measured cooling rate and the selected spread rate; and
- the footprint IDs confidently classified as heat sources by the edge test.

The raw case10 footprint readings are never used as temperature data because
reflection and low emissivity make them unreliable. `--no-paired-calibration`
keeps the previous image-only reconstruction as an explicit ablation.

## 2. Paired reliability audit and SoC surface anchor

A hot apparent pixel is evidence for a heat source, but it is not evidence that
the reported temperature is radiometrically trustworthy. A low-emissivity source
can remain hotter than its surroundings while still reading many kelvin too cold.

The case10 mask therefore receives a second check. The code takes the final
180 seconds of the longest interval at or above 90% logged load in case10 and
case17, subtracts each frame's ambient temperature, and compares the two median
maps. A compact region whose painted reading exceeds its bare reading by more
than 5 K is added to the unreliable mask. This is a binary reliability audit,
not a fitted emissivity map. On the current data it flags 3,991 pixels, of which
43 were not already in the 6,428-pixel image-only mask.

The SoC lid's absolute level needs a separate constraint because exterior board
temperatures do not uniquely determine a low-emissivity interior level. Both die
logs are linearly aligned to the infrared timestamps without silent extrapolation.
For case17, the painted surface temperature is measured in the deepest quarter
of the case10 SoC footprint. Its die-to-surface difference is mapped by elapsed
time since load onset and becomes constant at the reference steady-state mean.
Subtracting that difference from the case10 die log gives the case10 surface
anchor. A 1.5 K uncertainty floor declares the unmeasured session-to-session
transfer error.

The two recordings pass the spatial check: all 162 pixels in case17's largest
steady top-one-percent hot patch fall inside the case10 SoC footprint. case17 is
therefore a calibration input and no longer an independent held-out validation
set. Its hide-and-reconstruct score remains useful only as a within-session shape
diagnostic.

## 3. Plate relation used for each frame

After division by the unknown areal heat capacity, the temperature movie obeys

```text
warming rate
    = spread rate x spatial curvature
      - cooling rate x temperature above ambient
      + heating rate
```

For a single frame this is rearranged as

```text
spread rate x spatial curvature
    - cooling rate x temperature above ambient
    + (heating rate - warming rate) = 0
```

The term in parentheses is called the **source rate**. Moving the measured
warming rate into this term is why each frame can be reconstructed independently;
the method does not assume that frame 0 was cold and does not march forward from
an invented initial state.

The warming rate is calculated at every frame by fitting a local straight line to
the nearby frames and their real timestamps.

## 4. Fitting one source rate for each block and frame

The heating rate and warming rate cannot be trusted inside a footprint. Only their
difference enters the per-frame equation, so the current method fits one source
rate per sufficiently large block per frame.

An eight-pixel ring of trusted board around the union of the footprints is
temporarily hidden from the solver. For each frame:

1. The footprint and held-out ring are treated as unknown.
2. Temperature farther outside is fixed to the measured trusted-board value.
3. The measured source rate is supplied on the held-out ring.
4. One influence field is precomputed for every block: it is the temperature
   raised by one unit of source rate in that block.
5. A linear least-squares fit chooses the block source rates whose combined
   influence best reproduces the measured temperature on the hidden ring.

This fit turns the temperature response around each component into a quantitative
source rate. Small blocks with too little surrounding evidence are assigned zero
source rate and filled from their surroundings.

## 5. Provisional source shape and fitted level

The ring fit supplies one mean source rate, but the solver still needs a spatial
shape before it can translate that rate into a provisional temperature level.

- Blocks not confidently classified as heat sources use a uniform source shape.
- A confident heat-source block uses a centre-weighted shape based on distance
  from that block's rim. Currently this applies only to the SoC block in
  `case10`.

The centre weighting represents a small die under a much larger lid. It is now
used only to estimate the block's mean level from the exterior ring. Its bell
shape is not accepted as the final lid temperature: `case17` showed that it made
the middle too hot and the broad area around the middle too cold.

## 6. Production reconstruction of passive blocks

After fitting the rates, the held-out ring is restored as trusted data. For each
frame:

1. The plate equation is solved with only the footprint mask unknown.
2. Every pixel outside the footprint mask stays exactly equal to its measured
   infrared value, except for the explicit one-pixel transition around a
   heat-producing lid described below.
3. The fitted source influence from all blocks is added to the boundary-driven
   solution.
4. Every pixel inside the mask, including the footprint edge and its one-pixel
   halo, is replaced by the solved temperature.

Passive blocks retain this plate solution. Heat-producing lids receive the
broad-source plate reconstruction below.

## 7. Broad-source plate reconstruction for a heat-producing lid

The final SoC fill deliberately stays simple. It keeps the provisional solution's
mean level, but replaces its unjustified central-bump pattern with a second plate
solve driven by a positive broad source:

1. Extend the reconstructed lid by one pixel so the residual low-emissivity rim
   just outside the detected footprint is not mistaken for trusted board. This
   small extra area is saved separately as `lid_transition_mask`.
2. Treat the original footprint and this transition rim as one unknown conducting
   plate domain. The measured board immediately outside it supplies the boundary.
3. Distribute one non-negative source uniformly over the original lid footprint.
4. Precompute the lid's response to one unit of that source.
5. For each frame, choose the source amplitude that matches the calibrated central
   surface temperature when an anchor is supplied; otherwise preserve the
   ring-fitted provisional footprint mean.
6. Use the resulting plate solution directly. Do not paste a curve, flatten a
   core, or blend temperatures afterward.

The footprint border is therefore no longer a fixed-temperature seam: pixels on
both sides of that old border are solved together. The discrete conduction and
cooling equation determines their slopes and leaves the small interior curvature
needed to carry heat outward. The visible plateau is broad, but it is not exactly
flat and the source cannot disappear because its fitted amplitude is constrained
to be non-negative.

In the current case17 within-session diagnostic, the broad-source lid gives
2.78 K mean SoC error and 1.77 K error over the whole deliberately hidden area.
The corresponding centre-weighted values are 2.93 K and 1.79 K. These diagnose
spatial shape only because case17 also supplies the absolute calibration.

In case10's final 180-second steady window, case17 measures 78.28 C at the SoC
surface core, 79.25 C at its surface peak, and 83.04 C at the die. The transferred
anchor gives case10 a 77.78 C core and 78.29 C reconstructed surface peak. The
same combined mask without the anchor reaches only 69.02 C. After allowing for
the sessions' different die temperatures, the case17-based expected peak is
78.75 C, so the reconstruction differs by -0.46 K and passes the 1.5 K limit.

## 8. Uncertainty: what the infrared data did not determine

Every reconstructed frame now carries `uncertainty_k` over the footprint pixels.
It is explicitly a **model range, not a statistical confidence interval**.
For an unanchored lid it combines:

- the error left on the trusted ring that was hidden during fitting, enlarged as
  distance from trusted data increases; and
- the difference between the provisional central-source solution and the final
  broad-source plate solution.

For the anchored SoC, the corrected surface series replaces the unanchored level.
The size of that correction is saved as a diagnostic rather than added back as
uncertainty. Its range combines the local ring/shape term with the 1.5 K
cross-session anchor floor in quadrature.

## 9. Central-temperature interface and case10 default

`InteriorTemperatureAnchor` accepts one frame-aligned central temperature and its
uncertainty for a named footprint. When supplied, it sets the mean of the deepest
quarter of the lid. The same broad-source plate solve then determines the complete
spatial field; the sensor value is not pasted into individual pixels.

The input is temperature above ambient. `vcgencmd measure_temp` reports the SoC
die, not the visible lid surface. A known die-to-surface difference must be removed
first. If it is not known, it belongs in the supplied uncertainty; the die reading
must not silently be pasted in as an exact surface temperature.

`scripts/run_reconstruct.py` now constructs this anchor by default for case10 using
`transfer_surface_anchor`. The generic interface remains available to other callers.
An anchor below the source-free boundary solution is rejected because satisfying it
would require the declared heat source to become a negative source.

```python
from tdp.thermal import InteriorTemperatureAnchor, reconstruct_sequence

soc_anchor = InteriorTemperatureAnchor(
    region_id=soc_region_id,
    temperature_above_ambient_k_by_frame=aligned_surface_temperature_k,
    uncertainty_k=surface_temperature_uncertainty_k,
    label="corrected SoC sensor",
)

result = reconstruct_sequence(
    # regular inputs omitted here
    lid_fill="broad-source-plate",
    heat_source_ids=[soc_region_id],
    interior_temperature_anchors=[soc_anchor],
)
```

## 10. Implementation locations

- Pipeline driver: `scripts/run_reconstruct.py`
- Log alignment, paired audit, and surface transfer: `tdp/thermal/calibration.py`
- Per-frame fit and reconstruction: `tdp/thermal/reconstruct.py`
- Sparse plate solver: `tdp/data/emissivity.py::solve_plate`
- Saved result: `outputs/thermal/case10/reconstructed_sequence.npz`

The two source series are deliberately distinct:

- `BlockSource.source_rate_by_frame` and the saved
  `ring_fitted_source_rate_by_frame` are the provisional rates fitted from the
  exterior ring;
- `BlockSource.reconstruction_source_rate_by_frame` and the saved
  `source_rate_by_frame` are the rates actually used by the final plate solve.

They differ for a broad-source lid because source shape or the temperature anchor
changes the rate needed by the final solve. The archive also saves
`image_only_unknown_mask`, `paired_attenuation_mask`, `paired_attenuation_k`, the
combined `unknown_mask`, `lid_transition_mask`, their union as `rebuilt_mask`, the
SoC surface anchor and its uncertainty, and the model range as
`reconstruction_uncertainty_k`. `reconstruction_report.json` records all temporal,
spatial, calibration, and acceptance diagnostics.
