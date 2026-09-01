# CLAUDE.md

Guidance for working in this repository. **The instructions below come from the
project owner. Follow them exactly; they override any default behaviour.**

## What this project is

Infrared thermography of PCBs (currently the Raspberry Pi 4B). The camera exports
HIKMICRO CSV frames; there is no visible-light modality (`case*/photo/*.jpeg` are the
camera's own colourised renderings of the same thermal frame).

## The owner's instruction, in their words

Two things were missing from the reconstruction: **emissivity** and **conductivity**.

1. Deriving emissivity from the time series was too noisy and performed badly.
2. The conductivity-first approach could not get the number and position of heat
   sources, was hard to generalise to other boards, and — most importantly — **never
   constructed a temperature; it only estimated parameters.**

The chosen way out:

> Because the IR imager we used considers the emissivity to be 0.95 and ambient
> temperature, we regard the "hot" areas to be true.

So the pipeline is:

1. Recognise the footprint (untrustworthy) areas, as before.
2. **Additionally recognise the number and position of heat sources**, using the
   direction of derivatives. Note they are **not all inside footprint areas** — some
   are "shiny" in the image.
3. Reconstruct the whole temperature map by filling the blank places with the same
   plate-conduction method as before.

This deliberately avoids two problems at the start: no emissivity is ever estimated,
and conductivity is divided out rather than measured. After this, we have a real
temperature for every frame.

Then a **time-series model** takes the heat-source data, the reconstructed temperature
map, and the exact time each frame was shot, and predicts the board's temperature at
any moment. Training compares predictions against the measured maps at other times.
The model must understand the physics; it may predict the residual against the current
temperature rather than the temperature itself.

Two results are required:

1. **A reconstructed image for every raw infrared frame.**
2. **Pick one state and generate the temperature images across the whole phase, from
   the beginning through to the steady ending.**

Focus on **case10** first.

> For anything you think wrong, you are supposed to propose and discuss with me.

That last line is binding. Disagreement is wanted, but as a proposal — raise it,
give the evidence, and let the owner decide.

## Decisions the owner has made since

- **Ignore the cooling phase.** Only the powered stretch is used: frames 0-239 on
  case10, 0 to 1498 s. Reconstruct those 240 frames; nothing after the load stops.
- **Forward prediction only**, and it must be able to start from *any* moment.
- **Infrared images are the default input.** No CPU-load or system log is used in
  the case10 reconstruction. A separate, optional
  `InteriorTemperatureAnchor` interface exists for boards that expose a corrected
  central temperature; it is not used in the case10 benchmark.
- **The next model may use one fixed load value as an additional input.** That value
  is a float from 0 through 1 and stays constant within one recording or forecast
  session. This is approved future work, not a change to the completed load-blind
  case10 benchmark.
- **Relative only — no watts, ever.** See "Units" below.
- **Plain language, never Greek symbols.** See "Naming" below.
- Board state is represented by a small **pattern library** with a closed-form
  solution over any time gap, which is what allows starting from any moment.

### One place the work has diverged, with the owner's approval

The original instruction said *"don't train the model on local device, I would put it
in remote GPU."* The model turned out to be a roughly 100x100 matrix: a full fit is
about 20 seconds on a CPU, and GPU launch overhead on matrices that small exceeds the
computation. The approved plan therefore trains locally and keeps `--device` plumbed.
A self-contained export is still built so a run is reproducible off-machine. **If the
model grows, revisit this rather than assuming it still holds.**

## Documentation map and authority

All thermal design and planning documents live in `docs/plans/`. They have different
roles and must not be treated as four competing task lists:

1. **`CLAUDE.md` is authoritative.** It records the owner's current decisions, the
   implemented state, hard constraints, and the next approved direction. If another
   Markdown file conflicts with it, this file wins.
2. **`docs/plans/stage2_reconstruction.md` is an implemented-method record.** It
   explains the reconstruction code that already exists; it is not a proposal to redo
   Stage 2.
3. **`docs/plans/stage3_forecasting.md` is an implemented-method record.** It explains
   the completed load-blind case10 forecast, frozen split, rejected source-only model,
   and saved artifacts.
4. **`docs/plans/future_fixed_load_forecasting.md` is the approved future plan.** It is
   blocked on additional fixed-load sessions and must not silently change the current
   model before that data exists.
5. **`docs/plans/case10_original_plan.md` is historical provenance.** Its completed
   status and old checkpoints explain how the project reached the current design, but
   its body is not an active instruction list.

`docs/dynamic_inverse_zh.md` documents the older dynamic-inverse comparison path. It is
not one of the current thermal plans and must not be allowed to consume reconstructed
Stage 2 estimates silently.

## Where the work is (2026-09-01, Stage 3 committed; future load plan documented)

The current status is summarized here. The detailed implemented methods and historical
record are linked in the documentation map above.

**Done** — `tdp/thermal/sources.py`: the load stop is found from the images (frame 239
on case10), the cooling rate comes from the warm-up alone (0.00685 /s, response 146 s),
and heat sources are found in two parts. On case10 that gives one heat-source block
(the SoC, 269 mm² at (26.1, 32.7) mm, 100 % outward) and one trusted-board hot spot at
(17.9, 10.3) mm. Figure: `outputs/thermal/case10/phase1_sources.png`.

**Done** — `tdp/thermal/reconstruct.py`: all 240 powered frames are reconstructed.
Passive blocks use the ring-fitted source solution; the SoC uses a positive broad
source in one smooth plate solve across its footprint and transition rim. The saved
archive includes both the provisional ring-fit and final reconstruction source series,
the complete rebuilt mask, and a per-pixel model-uncertainty range.

**Done** — `tdp/thermal/patterns.py` + `tdp/thermal/forecast.py`: the fixed library uses
64 slow plate modes and source responses for the broad background, SoC footprint and
trusted-board hot spot at four trial spread rates. Pivoted QR leaves 72 independent
patterns from 76 candidates. The fine-grid source-response gate is below 1e-10 K against a
0.10 K limit. The reduced plate advances by matrix exponential over any positive time
gap and solves its settled state directly. Stage 2 uncertainty becomes an observation
weight, never measurement confidence. An arbitrary starting frame is exact at time zero;
detail outside the fixed basis is carried as an explicit initial residual.

**Done** — `tdp/thermal/train_forecast.py`: the export and time split are frozen, with
three-frame guard gaps around contiguous validation and test blocks. The intuitive fit of
three non-negative, smooth source histories was tested and rejected: it scores 2.145 K
weighted test MAE against 0.270 K persistence because three source channels cannot balance
all 72 board modes. Keep this result as the source-only ablation; do not rescue it with a
large unconstrained signed correction.

**Done** — the production forecast decays deviations toward a settled full field estimated
only from the final training block, frames 147-188. It keeps the named starting frame exact,
uses the stable reduced plate operator, and retains the measured spread 2.0 px^2/s and
cooling 0.006851 /s under a 0.01 K parsimony rule. It scores 0.185 K validation and 0.191 K
held-out test weighted MAE, beating persistence. Run `python scripts/run_forecast_fit.py`,
then `python scripts/run_forecast.py --start-frame N`; the default frame-0 artifact has 674
one-second fields through less than 1% remaining deviation.

**Done** — `scripts/figure_forecast.py`: the fixed 1280x720 H.264 presentation contains
all 674 one-second forecast fields at 24 fps, plus temperature, remaining deviation, model
range, captured-frame error, and error-curve panels. The saved-field ablation figure reports
equilibrium 0.191 K, persistence 0.270 K, constant source 1.774 K, smooth positive sources
2.145 K, and no source 15.336 K. It never refits the model. MP4 failure falls back to a
sampled GIF; the montage and arrays are always written.

**Then** — do not tune case10 further on its held-out test. New accuracy work needs new
evidence: another painted/truth board or a corrected central-temperature sensor on hardware
that exposes one. The old `tdp.dynamic_inverse` path remains a legacy comparison and does
not silently consume Stage 2 estimates.

**The three numerical deliverables are done** — 240 reconstructed powered frames, a
forward one-second forecast from any named frame, and `hotspots.csv`.

## Future fixed-load-conditioned model — approved, not implemented

Do this only after the multi-load data gate below is satisfied. The completed case10
forecast remains the reference result until a replacement passes held-out-session and
held-out-load tests.

### Model contract

The public input is:

```text
forecast(starting temperature field, fixed load from 0 through 1, future times)
```

The load is one scalar and is constant for the whole session. The model does not need a
time-varying load schedule. If an application changes load, start a new forecast from the
latest temperature field using the new fixed value. The field at elapsed time zero must
remain exactly equal to the supplied starting field; load changes the future trajectory,
not the current temperature.

Reuse the Stage 2 reconstruction and uncertainty weights, the fixed spatial pattern
library, the stable reduced plate operator, the arbitrary-start projection, the explicit
unresolved residual, the saved-array interfaces, and the existing evaluation tools.

The learned component maps the fixed load to a load-conditioned settled state in the
retained physical pattern coordinates. The existing stable plate operator then relaxes
the supplied starting field toward that state. Start with the smallest smooth constrained
encoder, and compare it with a learned linear-load baseline before adding capacity.

**Do not reduce this to the three existing source amplitudes.** That exact spatial
bottleneck already lost to persistence by a wide margin: 2.145 K against 0.270 K. A small
load encoder may be low-capacity as a function of one scalar, but its output must have
enough spatial freedom across the retained pattern library to represent the board-wide
settled response. It must also preserve the SoC plateau and may not hide missing physics
inside a large unrestricted signed correction.

### Minimum data-collection reference

The minimum validation-grade collection is **five load levels with three independent
sessions per level: 15 powered sessions total**.

| fixed load | minimum independent sessions | role |
|---:|---:|---|
| 0.00 | 3 | idle-powered baseline |
| 0.25 | 3 | lower unseen-load check |
| 0.50 | 3 | connects to case09 |
| 0.75 | 3 | upper unseen-load check |
| 1.00 | 3 | connects to case10 |

Here, load 0 means the normalized workload is at idle. It does **not** mean that the board
is unpowered or that its heating rate is zero; this board still dissipates heat at idle.

Three independent sessions are the floor because frames from one session are highly
correlated. One session can support fitting, a second model selection, and a third an
untouched final check. **Two repeats per load, 10 sessions total, are useful only for an
exploratory prototype; they do not provide a clean training/validation/test separation or
a credible run-to-run uncertainty estimate.** A safer collection target is four repeats at
the same five loads, or 20 sessions total. Add load levels before adding encoder complexity
if the response is visibly nonlinear.

If case09 and case10 pass the same-board, same-workload, alignment, load-stability, and
metadata checks, each may count as one session at 0.50 and 1.00. Under that assumption the
minimum new collection is **13 sessions**: three each at 0.00, 0.25, and 0.75, plus two
additional repeats each at 0.50 and 1.00.

Each powered recording must last for at least **five response times** and include a clear
near-steady interval. Using case10's measured 146 s response as a planning reference gives
about 730 s, or 12.2 minutes, as the mathematical floor; matching the existing roughly
25-minute powered recording is safer. Duration is governed by settling evidence, not only
the clock. Record the actual mean load, its variation, exact thermal timestamps, ambient
conditions, initial state, workload type, camera configuration, and any corrected central
temperature sensor. Equal nominal utilization under different CPU/GPU/I/O workloads is
not automatically the same scalar input.

### Data split and implementation order

- Split by complete sessions, never random frames. Multiple starting windows from one
  recording remain one experimental unit and do not increase the independent sample count.
- Hold out complete repeats at seen loads to measure run-to-run generalization.
- Hold out every session at one intermediate load in turn to test whether the continuous
  load input works between observed values.
- Audit the powered interval, footprint reference, alignment, spread rate, cooling rate,
  and reconstruction diagnostics for every session. The case10 driver contains measured
  case10 defaults; do not assume the reconstruction is unchanged on new sessions.
- Build one self-contained multi-session export, freeze the session and held-out-load
  splits, then fit the smallest load encoder.
- Compare against persistence, the completed load-blind forecast, and a learned linear-load
  baseline. A more flexible encoder is justified only if its held-out-load improvement is
  larger than the files' 0.1 K resolution and remains stable for every load from 0 to 1.
- Report larger model range where repeated-session evidence is weak or the requested load
  is far from well-sampled values. Rebuilt footprints remain estimates, never hidden truth.

The complete future design, sample schema, acceptance criteria, and unresolved metadata
questions are in `docs/plans/future_fixed_load_forecasting.md`.

### What Stage 2 measured, and where it stops

Scored on case17 — the same board painted matte black, so the parts case10 must guess at
are *measured* there. Imposing case10's mask and rebuilding the 39.5% it hides:

| | mean error | bias |
|---|---|---|
| edge-only fill (the old way) | 3.68 K | **−2.10 K**, cold every frame |
| source-driven, source spread uniformly over each block | 2.63 K | **+0.01 K** |
| provisional **centre-weighted** source shape | **2.37 K** | +0.37 K |
| final **broad-source plate** lid | **2.29 K** | +0.37 K |

Killing that systematic cold bias is the point of the whole stage. **These numbers replace
an earlier table here that read 4.34 / −3.22 / 2.72 / +0.12 K; those were stale and did not
match what the code produced.** Docstring, figure and this file now all report the same run.

The remaining 2.29 K splits into a **wrong level** and a **wrong pattern** inside each block:

- Fitting each block's source to case17's *hidden truth* — the best any single number per
  block can do — leaves 1.57 K, of which 1.56 K is pattern and 0.20 K level. **No extra
  source parameter can help the pattern.**
- **The board-wide bias hides the per-block level errors.** They reach **±4.4 K** and
  cancel: block 1 is +4.5 K hot while the SoC block is −4.4 K cold, and the average comes
  out near zero. Always report the per-block level, never only the aggregate.
- An earlier claim here — "the ring fit's own gap is not a conditioning problem: condition
  number 81, and per-sector, ridge and plain least squares agree to within 0.04 K" — was
  **reasoned wrongly and is withdrawn.** Three estimators agreeing shows they extract the
  same information, not that the information suffices. What sets the level error is the
  ring residual times the amplification: **1.87 K of residual against condition number 81.**

### The source's shape inside a block, not just its size (2026-08-29)

A die under an SoC lid dissipates over a few mm; the lid spreads it over ten times that
area. A **uniform** source over the whole 269 mm² therefore builds a plateau — about right
at the block's rim, which measured board pins from a pixel away, and several kelvin cold in
the middle. That is exactly what the owner saw. Weighting the source by distance from the
block's own rim (exponent 1.0, `CONCENTRATION_EXPONENT`) matches the geometry at **no extra
parameter**: the SoC block goes **4.38 → 2.51 K** and its level bias **−4.38 → −1.83 K**.

Choose the shape **per block from the edge test**, never globally:

- concentrating *every* block gives 2.98 K, **worse** than uniform — a connector bank
  really is distributed.
- the exponent optimum is flat (0.5 → 2.43 K, 1.0 → 2.36 K, 1.5 → 2.40 K) and 2.0 is
  already too sharp (2.46 K, SoC back to 3.16 K).
- the error asymmetry runs strongly against false positives: a source block left uniform
  costs **0.03 K**, a passive block wrongly concentrated costs **several K**. Pass only
  confident `heat source` verdicts. On case10 that is exactly one block, the SoC, at 100%
  outward over 22 of 24 sectors.

`--uniform-sources` on `scripts/run_reconstruct.py` is the ablation.

### The final lid shape is now a broad-source plate solve (2026-08-30)

The owner rejected the centre-weighted bell as the final visible temperature of a
heat-spreading lid. case17 confirms the problem: at frame 120 it is 0.76 K cold at the
SoC edge, 3.75 K cold 4-5 px inside, and 3.83 K hot at the centre. The fitted bell is now
used only to retain the exterior-ring estimate of the block's **mean level**.

For a confidently heat-producing lid, the production fill first takes one extra rim pixel
into an explicit `lid_transition_mask`; case10's nominally trusted first ring is still 0.5 K
colder than the second and is residual lid-rim contamination. The original footprint and
this rim are then solved as one plate domain with a non-negative source distributed over
the lid. Its amplitude is chosen per frame to preserve the provisional footprint mean.
There is no pasted curve, blending band, or exactly flat core: the broad plateau and its
transition to the measured board are generated by the same conduction-and-cooling
equation. On case17 this changes the SoC error **2.51 -> 1.99 K** and the original
masked area **2.37 -> 2.29 K**; including the rebuilt rim gives **2.26 K**. On case10
frame 200 the result is still unmistakably source-driven: the SoC core is **41.66 K**
above ambient against **29.45 K** for a source-free solve. The source-shaped result
remains the ablation and one side of the reported model range.

Every reconstruction now reports a per-pixel uncertainty range. It is explicitly a
**model range, not a statistical confidence interval**: held-out-ring error enlarged by
distance, plus disagreement between the provisional central-source and final broad-source
plate solutions. An optional
`InteriorTemperatureAnchor` supplies a frame-aligned central surface temperature and its
uncertainty. `vcgencmd` measures the die, not the lid surface, so a die-to-surface
difference must be removed or included in that uncertainty.

### Reading the pattern under a lid: tried four ways, all lose

The owner asked for this to be tried. It was, and it fails for a measurable reason. All
scored on case17 with the gain and the reflection offset calibrated from the block's own
rim, never from truth, and with the lid's roughness set to what case10's lids actually
carry:

| what was tried | error |
|---|---|
| add the pattern that grew over the run (reflection-free: a fixed lid reflecting a fixed room is constant in time, so a time difference cancels it exactly) | 2.63 → **2.44 K** at the absolute ceiling |
| use only reflection-free time differences | **2.60 K** |
| rebuild each block straight from its lid reading | **6.59 K** |
| take only the block's level from the lid reading | **4.83 K** |

Why, in order of how much each one bites:

- **97% of the squared error is static.** Anything built from differences of lid frames can
  only reach the other 3%.
- **The lid's gain is recoverable to 20% at best** (worst block 98%). The only reference for
  the true temperature under a block is the fill being corrected, so the calibration is
  circular. A 10% gain error on a 30 K field is already 3 K.
- **The static reflection offset and the fill's own static level error are the same
  unknown.** The lid reading gives one static equation per block for two static unknowns.
  This is structural, not a tuning failure.
- Real lids are not smooth enough to be temperature anyway: case10's lid interiors carry
  **0.5–2.0 K of 2 px structure** above the camera's 0.42 K floor, and it does not fall off
  as you erode deeper, so it is not a rim artifact. Heat spreading 16 px cannot sustain 2 px
  structure — that reading is reflection, not temperature.

One useful by-product, kept: the **growth pattern** (late mean − early mean, normalised per
block) recovers a block's true internal temperature shape at correlation **0.93–1.00** and
is completely immune to reflection scatter. It is the right tool if a future stage ever
needs a shape under a lid; it just cannot supply the *level*, which is what was missing.

Also measured and closed: **case17's fitted rates do not transfer to case10** (ratios −7.5
to +27, signs disagreeing on 4 of 11 blocks), so borrowing the painted board's numbers is
not an option. Within one run, though, a single rate per block for the whole powered
stretch is nearly as good as per-frame rates (1.57 → 1.75 K) — useful for Stage 3.

### One Part-1 verdict that Stage 2 contradicts, with case17 as referee

Block 4 (175 mm² at (9.9, 42.9) mm) is called **passive** by the Part-1 edge test (33%
outward, −0.20 K) but gets the *highest* fitted source rate of any block (+0.344 K/s).
case17 settles it: that block's true mean is 29.9 K, the second hottest of all the blocks.
**Stage 2 is right and the edge test is wrong there.** The edge test still wins on the
connector bank, which both methods and the owner agree is passive. Treat `judge_footprints`
as a screen, not a verdict.

## Units — the rule that must never be broken

The governing equation is divided through by the areal heat capacity. That division is
what makes the remaining quantities measurable from a temperature movie, and it is
also what destroys the absolute scale:

```
heat capacity x (temperature change per second)
    = (spreading) - (loss to air) + (power generated)

temperature change per second
    = (spread rate) x curvature - (cooling rate) x temperature + (heating rate)
```

Scaling heat capacity, conductivity, air-loss and power by one common constant leaves
all three measured rates unchanged, so every temperature trajectory is identical.

**Never output or claim** `W`, `W/K`, `W/(m*K)` or `W/m^2`, and never claim to have
recovered hidden copper geometry. Also never say "component X supplies Y% of board
power": heat capacity **varies across the board** (copper-backed areas differ from bare
laminate), so even the ratios between hot spots are not power ratios. Rank and compare
hot spots instead.

Adding neural-network capacity does not make an unidentifiable quantity identifiable.

## Naming — plain words, never Greek

The owner rejected a document for containing "strange alpha, beta chars". Use:

| Use this | Not this | Unit |
|---|---|---|
| spread rate — how fast heat creeps sideways | alpha, diffusivity | px^2/s |
| cooling rate — how fast heat leaks to air | beta | 1/s |
| heating rate — how fast a spot warms itself | s, source | K/s |
| response time | tau | s |
| spread distance | L, fin length | mm |
| temperature above ambient | theta | K |
| hot spot | source atom | |
| pattern library | basis, modes | |

Write equations as sentences where possible. Every reported quantity carries its unit.

## Facts about this dataset that keep mattering

- The 2026-07-26/27 CSV exports carry **no radiometric parameter block**, so the
  camera's emissivity setting is an assumption (`recorded: false`), not a measurement.
  Under the current design that is fine — 0.95 is the *definition* of truth on
  well-behaved pixels, not a fitted value.
- **The board is never powered off.** The load stops but the board idles at 13.90 K
  above ambient, so a residual source is always running and zero power must never be
  asserted.
- There is **no measured pixel scale**. It is derived from the 85x56 mm PCB outline
  resampled onto the canonical 157x103 grid (0.5425 mm/px); `pixel_size_calibrated`
  stays `false`.
- 44.7% of case10's board is untrustworthy, the deepest point 20 mm from good data.
  case17 (painted matte black) has only 16% and is the natural cross-check.

## Things that broke and must not be reintroduced

- **Do not apply the spreading correction across an emissivity step.** A metal lid's
  rim is a step in apparent temperature, and its curvature is 6x that of real board
  (0.558 vs 0.092 K/px^2). The correction then injects ~0.35 K/s of pure artifact along
  every rim — a third of the real signal — and *manufactures hot spots there*. Without
  the guard, seven of ten leading hot spots sat exactly on a lid edge. Pass
  `footprint_mask` to `measure_heating_rate` and exclude rim pixels in `find_hotspots`.
- **A cross-check between two estimators does not catch a shared artifact.** Both the
  rise and onset maps come from the same raw derivative, so a rim artifact appears in
  both and scores highly against a ring that spans the step. Confirmation across
  estimators is necessary, not sufficient; check geometry too.
- **Do not use the repaired field or an emissivity-derived gain in the source
  measurement.** Both smuggle emissivity estimation back in, and both measurably make
  it worse — every leading hot spot became a lid pixel with none confirmed.
- **Never look for a hot spot inside a low-emissivity block.** The reading there is
  wrong by an unknown factor, so a bump in it carries no information about a source.
  On case10 that put two detections inside the 970 mm^2 connector bank -- passive
  metal, which the owner identified on sight. Blocks are excluded from peak search
  *entirely*, interior included, and judged separately by `judge_footprints`.
- **Do not look for sources as bumps in a single frame.** The spatial second
  derivative finds lid edges on raw frames, and on repaired frames it finds nothing
  under the lids by construction, because the repair set the source to zero there.
  Use the time derivative.
- **Do not trust a fitted rate whose confidence interval includes zero.** The
  second-moment spread estimator returned *negative* spreading for four of six
  candidates — it was measuring where the baseline was put, not diffusion.
- **Never give `contour` an `imshow` extent.** `imshow` reads `(0, W, H, 0)` as
  `(left, right, bottom, top)`, so row 0 lands at the top. `contour` reads the same tuple
  as `(x0, x1, y0, y1)` and maps row 0 to `y = H`, so **every outline is drawn mirrored
  top-to-bottom over the image it annotates.** This shipped in
  `phase2_reconstruction.png` and made the block outlines miss their blocks — which is what
  the owner spotted and reported as the two runs looking 180° apart. The data was never
  affected; only the overlay. Pass explicit coordinate arrays
  (`ax.contour(grid_x, grid_y, Z)`); a test now fails if `contour` is given an `extent`.
- **A board-average bias can hide per-block errors that cancel.** Stage 2's +0.01 K bias
  covers ±4.4 K per-block level errors of opposite sign. Choose the per-block metric first.
- **Never split frames randomly.** Adjacent frames on a ~6 s cadence are near
  duplicates. Use contiguous blocks with guard bands.
- **Make regularisers dimensionless.** An absolute penalty on a rate of order 1e-3
  contributes ~1e-6 to a loss of order 1, so its configured weight does nothing.

## The two-part source detection (the owner's design)

1. **Low-emissivity blocks are judged from their edges, not their interiors.** If a
   block generates heat, that heat must leave through its boundary, so the board
   temperature falls as you walk away -- **all the way around**. Consistency is the
   test, not net flux: a passive lump sitting in the board's gradient takes heat in
   one side and lets it out the other, so its net is near zero and so is a weak
   source's. Measured on case10: the SoC block is 100% outward over 22 of 24 sectors
   at +2.10 K, while the connector bank is 22% outward at -0.71 K. Around the SoC the
   board falls 37.5 -> 29.5 K with distance; around the connector bank it *rises*
   18 -> 26 K. This test never touches an untrustworthy pixel.
2. **Everywhere else, the time-derivative heating rate applies**, because the readings
   there are trustworthy.

Keep the footprint halo at **1 px**. At 2 px separate blocks merge (10 regions instead
of 12) and 5 more points of the board are masked, for no benefit -- the ring probes
start 2 px out, so the camera blur the halo was protecting against is already excluded.

## Report findings honestly, including against your own earlier claims

This project has repeatedly produced results that looked good on an average metric
while being wrong on the quantity that mattered. Choose the success metric **before**
seeing results. For source recovery that metric is **hot-spot accuracy and
cross-estimator agreement, not board-average error** — the background is broad and hot
spots are small, so average error barely moves whatever happens.

If a number you previously reported turns out to be an artifact, say so plainly and
correct it.

## Commands

```bash
python -m pytest tests/ -q                                       # 92 tests, ~38 s
python -m tdp.dynamic_inverse run --smoke --sequence case10_rpi4b_bare_full
```

Configuration is split on purpose: `configs/measurements_v3.yaml` records what the
*experiment* was, `configs/dynamic_inverse.yaml` records what the *analysis* does.
Mixing them makes an ablation look like a re-measurement.

## Style

Module docstrings explain *why* a choice was made and what it prevents; comments carry
measured evidence rather than restating the code; every reported quantity carries its
unit.
## For token usage

First of all, make sure tasks completed with quality and quantity guaranteed

### Token-efficient tool usage

- Search before reading. Use Glob, Grep, or `rg` to identify candidate files,
  symbols, and line numbers before opening files.
- For large, generated, minified, or unfamiliar files, use Read with
  `offset` and `limit`. Start with the relevant range and enough surrounding
  context; expand only when necessary.
- Read small source and configuration files completely when their overall
  structure affects correctness.
- Do not re-read unchanged content already present in the active context.
  After an edit, inspect the diff or changed ranges instead of rereading the
  entire file. Re-read when exact text is required or context was compacted.
- Prefer dedicated Glob, Grep, Read, and Edit tools over Bash equivalents when
  they can retrieve the required information more precisely.
- Use `rg --files` or a targeted directory listing instead of recursive `ls`
  or unrestricted `find`.

#### Command output

- Anticipate noisy commands. Use quiet, concise, or summary options provided by
  the command before applying generic truncation.
- Do not place complete build logs, test logs, dependency trees, recursive
  listings, or large JSON responses into the conversation.
- Filter known structured output using targeted tools such as `rg`, `jq`,
  `head`, or `tail`, and impose reasonable result limits.
- Never truncate a command's output before preserving information needed to
  determine success or diagnose failure.
- For potentially noisy builds and tests, save complete stdout and stderr to a
  temporary log, preserve the original exit status, and initially inspect only:
  1. the command summary,
  2. matched errors or failed tests, and
  3. a short tail of the log.
- If the result is unsuccessful or ambiguous, inspect additional targeted
  ranges from the saved log. Do not rerun a command merely to recover output
  that was discarded.
- Never infer success from the absence of errors in filtered output. Check the
  original command's exit status.

#### Responses

- Summarize tool results instead of repeating them.
- Reference file paths and relevant line ranges rather than reproducing large
  sections already available in files.
- Keep progress updates and final responses concise unless the user requests a
  detailed explanation.
