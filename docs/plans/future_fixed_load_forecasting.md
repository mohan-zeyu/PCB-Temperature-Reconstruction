# Plan for a fixed-load-conditioned temperature forecast

## Status

This is the approved future design and data-collection gate. It is **not implemented**.
`CLAUDE.md` is authoritative if this document ever becomes stale. The current load-blind
case10 forecast remains unchanged until this plan has enough data and passes its acceptance
tests.

The main blocker is insufficient training coverage. The current relevant recordings provide
approximately two fixed powered-load values:

- case09: load `0.5`;
- case10: load `1.0`.

Two load values cannot identify a reliable continuous response over the complete interval
from 0 to 1. A model could accept other numbers, but its behaviour there would be an
unsupported assumption rather than something learned from data.

## Clarified model contract

Load is one scalar input in the closed interval `[0, 1]`. It is fixed during one recorded
or predicted session. The intended interface is:

```text
forecast(starting temperature field, fixed load, future times)
```

The model does not need a continuously changing load schedule. If a future application
changes load, it should begin a new forecast from the latest temperature field with the new
fixed value. This preserves temperature continuity.

## Core idea

Load should enter the thermal model as an input that changes the settled thermal state:

```text
fixed load -> learned settled pattern coordinates ---------+
                                                            |
starting temperature -> existing stable plate evolution -> future temperature fields
```

The learned part should not manually interpolate complete temperature images. It maps load
to a settled state expressed in the existing physical pattern library. This is equivalent to
a load-conditioned effective heating drive under the fixed stable operator. The existing
plate evolution then turns that learned destination into a smooth temperature trajectory.

The temperature at elapsed time zero must remain exactly equal to the supplied starting
field. Load changes the future derivative, not the current temperature.

## Existing code to reuse

Most present interfaces should be reused:

1. Heat-source detection and footprint reconstruction, after their assumptions have been
   validated for each new session.
2. The fixed physical pattern library.
3. The stable reduced plate operator and closed-form interval solve.
4. Starting-frame projection and explicit unresolved residual handling.
5. Stage 2 uncertainty weights for directly measured and rebuilt pixels.
6. Forward-window evaluation, persistence baselines, saved arrays, and presentation tools.

The only new learned model component should be a small load encoder between the scalar load
and the existing physical solver. Session ingestion, reconstruction, and alignment still
need per-session validation rather than blind reuse of case10 constants.

## Proposed learned component

The preferred first version is a deliberately small load encoder:

```text
one scalar load
    -> small smooth constrained encoder
    -> load-conditioned settled coordinates across the retained pattern library
    -> stable plate evolution from the exact supplied starting field
```

It should remain much smaller than the spatial model. A large neural network would only
memorise recording sessions because the available number of distinct load values is small.

Small here means low capacity as a function of one scalar; it does **not** mean returning
only the three existing source amplitudes. That source-only model has already been rejected:
three smooth positive source histories scored 2.145 K held-out error against 0.270 K for
persistence because they could not balance all 72 board modes. The load-conditioned state
must have enough spatial freedom across the retained library to describe the board-wide
response. It may use shared or low-rank structure, but the allowed rank must be validated
rather than fixed to the three named sources.

The load encoder should obey these restrictions:

- accept only finite values from 0 through 1;
- produce a smooth response as load changes;
- generally prevent higher load from making the board colder overall;
- keep the existing plate operator stable and unchanged;
- avoid a large unrestricted signed correction that can replace the physics;
- preserve broad source structures such as the SoC plateau; and
- retain enough spatial freedom to beat the rejected three-source bottleneck.

The exact encoder form should be selected after data collection. Candidates include a
small monotonic network or a monotonic spline. A flexible encoder should not be chosen
before there are enough distinct load levels to validate it.

## Why case09 and case10 alone are insufficient

With training data only at loads 0.5 and 1.0, many different functions fit equally well:

```text
load response A: almost linear
load response B: flat below 0.5 and steep above it
load response C: strongly curved between the two observations
```

All three can reproduce the two known loads while predicting very different temperatures
at 0.25 or 0.75. Validation windows from the same two recordings cannot resolve this.

There is also a session-confounding problem. Half and full load were recorded in different
sessions, so a difference between them may include:

- load;
- initial board temperature;
- ambient and reflected temperature;
- camera alignment or drift;
- reconstruction differences; and
- ordinary run-to-run hardware variation.

Repeating multiple load values across sessions is necessary to separate load response from
session identity.

## Data that should be collected

At minimum, collect aligned bare-board Raspberry Pi sessions at:

| Nominal load | Purpose |
|---:|---|
| 0.00 | idle baseline |
| 0.25 | lower intermediate response |
| 0.50 | connect to case09 |
| 0.75 | upper intermediate response |
| 1.00 | connect to case10 |

Load 0 means the normalized workload is at idle. It does **not** mean that the Raspberry Pi
is unpowered or that its heating rate is zero; the existing board still dissipates heat at
idle.

The minimum validation-grade collection is **three independent sessions at each load: 15
powered sessions total**. One repeat can contribute to fitting, one to model selection, and
one can remain untouched for final evaluation. Two repeats per load, or 10 total sessions,
are useful only for an exploratory prototype; they cannot provide a clean
training/validation/test separation and a credible run-to-run range.

If the existing case09 and case10 runs pass the same-board, same-workload, alignment,
load-stability, and metadata checks, each may count as one session at 0.50 and 1.00. The
minimum new collection would then be 13 sessions: three each at 0.00, 0.25, and 0.75, plus
two more each at 0.50 and 1.00. A safer target is four repeats at the same five loads, or 20
sessions total. Add load levels before adding encoder complexity if the response is visibly
nonlinear.

For every session:

1. Use the same board, surface treatment, camera pose, grid, and camera settings.
2. Begin from a reproducible idle condition.
3. Record the actual average load as a float, not only a name such as "half".
4. Keep that load fixed throughout the powered part of the session.
5. Record at least five measured response times and include a clear near-steady interval.
   Case10's 146 s response gives about 730 s, or 12.2 minutes, as a planning floor;
   matching its roughly 25-minute powered interval is safer.
6. Synchronise load timestamps and thermal-frame timestamps.
7. Preserve ambient measurements and any corrected central-temperature sensor data.
8. Record the load generator and workload type because equal CPU utilisation can produce
   different spatial heating on CPU, memory, GPU, and I/O blocks.

The training label should preferably be the measured time-average load during the powered
interval. Its variation should also be saved so a nominally fixed session can be rejected
if the load was not actually stable.

## Training sample format

Each forward training window should contain:

```text
starting reconstructed temperature field
fixed session load in [0, 1]
elapsed target times
later reconstructed temperature fields
Stage 2 per-pixel model ranges
session identifier and acquisition metadata
```

Many starting frames may be drawn from one session, but they all share that session's fixed
load label. The session identifier is metadata for splitting and diagnostics, not a model
input that lets the model memorise the answer.

## Training and validation split

Random frame splitting is not acceptable because neighbouring thermal frames are nearly
duplicates. The main evaluation must hold out complete sessions.

Many starting windows from one recording are useful training examples, but they are not
independent experiments and do not increase the session count above.

Two complementary tests are needed:

1. **Held-out session:** use separate complete repeats for fitting, model selection, and an
   untouched final check at seen loads. This measures run-to-run generalisation.
2. **Held-out load:** remove every session at one intermediate load, train on the remaining
   values, and test on the removed value. Rotate this check across the intermediate loads.
   This measures whether the continuous load input actually works.

The held-out-load test is the decisive test. Without it, the model has only demonstrated
that it can reproduce load values it already saw.

## Losses and physical restrictions

Training should use multi-step forward predictions from many starting frames. The primary
loss remains full-field temperature error under the existing uncertainty weights.

Additional restrictions should be small and explicit:

- load-response smoothness across `[0, 1]`;
- non-decreasing board-average settled temperature with load;
- non-negative growth of confidently identified source regions unless data contradict it;
- stable decay of all unforced plate modes;
- exact preservation of the supplied starting field; and
- a reported cap on any correction outside the physical pattern response.

Directly measured pixels retain measurement weight one. Rebuilt footprint pixels remain
estimates and cannot be presented as hidden ground truth.

## Uncertainty

The forecast must report greater uncertainty where load support is weak. The model range
should combine:

- the Stage 2 reconstruction range at the starting frame;
- held-out-session forecast error;
- disagreement across repeated-session fits;
- disagreement across acceptable load-encoder forms; and
- distance from the nearest well-sampled training load.

This remains a model range unless enough repeated sessions exist to calibrate a statistical
coverage claim.

## Acceptance criteria

The load-conditioned model should not replace the current model unless it satisfies all of
the following:

1. Exact starting-frame reproduction.
2. Smooth internal fields without sharp footprint borders.
3. A broad SoC plateau where the reconstruction indicates one.
4. Stable convergence for every input load from 0 to 1.
5. Better held-out-session error than persistence and a load-blind forecast.
6. Better held-out-load error than a simple linear baseline.
7. No large unrestricted correction term dominating the physical solver.
8. Explicitly larger model range at poorly sampled loads.

The comparison with a simple linear load baseline is important. A learned encoder is only
justified if it improves held-out-load predictions by more than the thermal files' 0.1 K
resolution and remains stable.

## Ordered implementation plan after data collection

1. Audit load stability and timestamp alignment for every candidate session.
2. Validate and parameterize the powered interval, footprint reference, alignment, spread
   rate, cooling rate, and reconstruction diagnostics for every accepted session. Reuse the
   Stage 2 algorithm and interfaces, not unverified case10 constants.
3. Align all sessions to the same board grid and verify trusted-pixel correlation.
4. Freeze complete-session and held-out-load splits.
5. Build one self-contained multi-load training export.
6. Add the scalar load encoder for a settled state across the retained pattern library while
   keeping the stable plate operator fixed.
7. Train first with the smallest smooth constrained encoder and explicitly verify that its
   spatial rank is not the rejected three-source bottleneck.
8. Compare against persistence, load-blind forecasting, and a linear load baseline.
9. Open the held-out sessions and held-out load only after model selection.
10. Generate arbitrary-start forecasts, uncertainty maps, and presentation artifacts.

## Decisions still needed

Before implementation, record answers to these questions:

1. Which aligned bare-board Raspberry Pi recording represents load 0?
2. Is the input CPU utilisation, total system utilisation, or a workload-specific control
   variable?
3. Should two workloads with the same utilisation but different CPU/GPU/I/O activity share
   one load number?
4. What load stability tolerance makes a session acceptable?
5. Can the validation-grade minimum of five loads and three independent sessions per load
   be collected, or must the first result be labelled exploratory?

Until these are resolved and the additional sessions exist, the correct action is to keep
the current case10 forecast unchanged and treat load conditioning as planned future work.
