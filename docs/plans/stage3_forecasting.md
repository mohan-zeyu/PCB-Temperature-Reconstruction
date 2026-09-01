# Stage 3: fixed physical patterns and equilibrium-centred forecasting

**Document role:** implemented Stage 3 method record. This describes the completed
load-blind case10 model; it is not the future fixed-load plan. `CLAUDE.md` remains
authoritative for owner decisions and status.

## Outcome

Stage 3 now provides a forward-only forecast from any reconstructed powered frame to a
settled state. The selected starting field is reproduced exactly, the broad SoC plateau is
retained, and the internal field relaxes smoothly under a stable plate operator.

Run the stages in order:

```bash
python scripts/run_reconstruct.py
python scripts/run_patterns.py
python scripts/run_forecast_fit.py
python scripts/run_forecast.py --start-frame 0
```

`--start-frame` may be any index from 0 through 239. The last command writes a compressed
array and a montage at one-second spacing by default.

## Input contract and uncertainty weights

`load_reconstruction_archive` checks that temperatures, timestamps, ambient values,
masks, source series, and Stage 2 model ranges have consistent shapes. It also requires
strictly increasing timestamps, unchanged directly observed pixels, and a non-negative
model range that is zero outside the rebuilt area.

The forecasting domain is the union of the old valid mask and `rebuilt_mask`. Directly
observed pixels have observation weight one. Rebuilt pixels are downweighted using the
Stage 2 model range and can never acquire measurement weight one. Consequently, an error
reported for a rebuilt footprint is agreement with the Stage 2 estimate, not accuracy
against hidden truth.

## Stage 3a: fixed spatial library

The library is built without reading a measured or reconstructed temperature frame. Its
candidates are:

1. the 64 slowest natural modes of the four-neighbour board plate; and
2. settled responses for a broad board source, the SoC footprint, and the trusted-board
   hot spot at trial spread rates 0.5, 1, 2, and 4 px²/s.

Pivoted QR removes redundant candidates at a declared rank threshold. On case10, 76
candidates become 72 orthonormal patterns. Including source responses is mandatory:
smooth modes alone lose compact heating even when their board-average error looks small.

Every retained source response is compared with a fine-grid plate solve at the active
rate. The acceptance threshold is 0.10 K over the source region; the current worst error
is below `1e-10 K`. The fixed library's archive projection MAE is 0.596 K. That is an
honest truncation diagnostic, not a forecast score.

## Closed-form physical evolution

In pattern coordinates, the state rate is the reduced plate operator acting on the state,
plus any declared source drive. Positive cooling and non-negative spreading make every
unforced mode stable. A matrix exponential advances the model over any positive time gap,
and repeated interval widths reuse the same transition matrix.

The general `ClosedFormPlateModel` still exposes constant-source rollout and direct
settled-state solves. The accepted case10 forecast, however, does not infer its settled
state from only three source amplitudes. The data show that this is too restrictive.

## Stage 3b split and self-contained export

The split is fixed before model selection and uses contiguous blocks with three-frame
guard gaps:

| role | inclusive frame blocks |
|---|---|
| training | 0-44, 51-92, 147-188 |
| validation | 99-140 |
| held-out test | 195-239 |

No validation or test frame contributes to the settled field. The export
`stage3_training_export.npz` contains the reconstructed fields, uncertainty, source table,
pattern library, measured rates, split roles, configuration, reports, and input digests.
It can reproduce the fit without reopening the raw image tree.

## The source-history proposal was tested and rejected

The initially proposed model used three non-negative sources, 12 time knots, a temporal
smoothness penalty, bounded spread and cooling corrections, and multi-step forward
training from starts distributed across the training blocks. This is physically clean,
but it is not expressive enough for this board:

| held-out test model | weighted MAE |
|---|---:|
| positive smooth source histories | 2.145 K |
| persistence from the starting frame | 0.270 K |

Three source channels cannot balance the 72 modes created by spatially heterogeneous
capacity, cooling, and imperfectly observed sources. Adding a large free signed residual
source would improve fit only by allowing that term to take over the physics. The source
fit is therefore saved as a failed ablation, not used for production.

## Accepted equilibrium-centred model

Later frames show the property identified by the owner: the SoC area approaches nearly
the same local level as its surroundings. The production forecast uses that information
directly while retaining physical relaxation.

The full settled temperature field is the mean of the last training block only, frames
147-188 (922-1178 s). For a named starting frame:

```text
initial deviation = starting field - settled field
resolved deviation at time = stable plate transition x resolved initial deviation
unresolved deviation at time = unresolved initial deviation x cooling decay
forecast = settled field + resolved deviation + unresolved deviation
```

This construction has four important properties:

- the forecast equals the reconstructed starting frame exactly at elapsed time zero;
- every deviation decays, so the result converges to the full settled field;
- the source-shaped patterns keep the broad SoC plateau in the resolved dynamics; and
- small details outside the 72-pattern basis are declared and decay smoothly rather than
  being disguised as fitted heat generation.

Validation compared spread multipliers 0.8, 1.0, and 1.2; cooling multipliers 0.9, 1.0,
and 1.1; and residual decay multipliers 0.5, 1.0, and 2.0. Any candidate within 0.01 K of
the numerical validation minimum is indistinguishable at the files' 0.1 K resolution.
Inside that band the model keeps the independently measured values: spread 2.0 px²/s,
cooling 0.006851/s, and nominal residual decay.

## Held-out results

| score | weighted MAE |
|---|---:|
| validation forward prediction | 0.185 K |
| held-out test forward prediction | 0.191 K |
| held-out test persistence | 0.270 K |
| frame 0 rollout over all 240 measured powered frames | 0.422 K |

The directly measured part of the held-out test also gives 0.191 K. The SoC footprint
value is 0.155 K, but it is explicitly agreement with Stage 2's estimate rather than a
measurement under the low-emissivity lid.

These scores test forward windows: the starting frame is supplied and only subsequent
frames are scored. The test block does not choose rates, the settled field, or any other
parameter.

## Exact starts, ambient scenario, and declared model range

The frame-0 default contains 674 fields from elapsed 0 through 673 s. Its first field is
exact within saved floating-point precision, and the last is below 1% remaining deviation.
Other start frames use the same rule and never integrate backward.

The model forecasts temperature above ambient. The rendering holds the starting-frame
ambient constant and records that choice in the output. This is an external scenario, not
an ambient-temperature prediction. Absolute temperature is recovered as:

```text
temperature in Celsius = forecast above ambient + declared ambient scenario
```

The per-pixel range is also explicit. It transitions from the Stage 2 starting range to
the training-only settled range and adds the measured held-out forward discrepancy as the
forecast takes over. The settled range has median 0.338 K and 95th percentile 2.522 K.
It is a sensitivity/model range, not a calibrated confidence interval.

## Public interface

```python
from tdp.thermal import (
    fit_equilibrium_forecast,
    load_pattern_library,
    load_reconstruction_archive,
    make_frozen_split,
    rollout_to_equilibrium,
)

archive = load_reconstruction_archive(
    "outputs/thermal/case10/reconstructed_sequence.npz"
)
library = load_pattern_library(
    "outputs/thermal/case10/pattern_library.npz"
)
split = make_frozen_split(archive.n_frames)
fit = fit_equilibrium_forecast(archive, library, split=split)

elapsed_s = [0.0, 1.0, 2.0, 3.0]
forecast = rollout_to_equilibrium(
    fit,
    archive.theta_k[start_frame],
    elapsed_s,
)
```

For reproducible case10 work, prefer the command-line scripts and saved fit so validation
selection is not repeated accidentally.

## Artifacts

- `pattern_library.npz` and `pattern_report.json`: fixed Stage 3a library and checks.
- `stage3_training_export.npz`: self-contained frozen training input.
- `source_history_fit.npz`: rejected positive-source ablation.
- `equilibrium_forecast_fit.npz`: selected production fit.
- `forecast_fit_report.json`: candidate selection, validation, test, and limitations.
- `heldout_forward_predictions.npz`: saved validation/test predictions and persistence.
- `whole_phase_forward_at_measured_times.npz`: frame-0 forecast at observed timestamps.
- `forecast_from_frame_000.npz`: one-second production forecast.
- `forecast_from_frame_000_montage.png`: start, midpoint, settled view, and model range.
- `forecast_from_frame_000.mp4`: all 674 one-second fields, H.264, 1280x720 at 24 fps.
- `forecast_video_montage.png`: four full presentation keyframes on fixed canvases.
- `forecast_error_and_ablations.png`: held-out curves, scores, and error maps.
- `forecast_presentation_metrics.npz`: frame-aligned errors behind the presentation.
- `forecast_presentation_report.json`: codec, dimensions, frame count, and reproduced scores.

## Stage 3c presentation contract

`scripts/figure_forecast.py` reads saved fields only. It does not rebuild the split, fit a
source, change a rate, or select a candidate. Before plotting, it recomputes each aggregate
held-out score from the saved predictions and requires agreement with
`forecast_fit_report.json` within 0.00002 K.

The movie synchronizes five views: forecast temperature, remaining deviation from the
settled field, declared model range, prediction error at the nearest captured time, and
the measured error curve. Its 674 video frames correspond one-for-one to the 674 one-second
forecast fields; playback speed is 24 fps. A fixed-size four-keyframe montage and the
underlying error arrays are written before encoding. If FFmpeg is unavailable, the script
tries a sampled GIF and still preserves the montage and numerical artifacts.
