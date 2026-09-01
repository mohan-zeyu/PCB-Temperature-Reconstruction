# Heat-source recovery and temperature forecasting for PCB thermal images (case10)

> ## STATUS — read this first (updated 2026-08-30)
>
> **Phase 0, Phase 1, Stage 2, Stage 3a, Stage 3b, and the Stage 3c presentation are DONE.** Everything below the status block is
> the original plan; where it conflicts with these numbers, these numbers win.
>
> ### Measured and settled (do not re-derive)
>
> | Quantity | Value |
> |---|---|
> | Load stops at | frame **239** (t = 1498 s), found from images alone |
> | Powered stretch | frames **0-239**, 240 frames — the only data used |
> | Cooling rate | **0.00685 /s**, response time **146 s** |
> | Board settles at | **26.63 K** above ambient (fit residual 0.223 K) |
> | Board idle level | 13.90 K measured; warm-up fit predicts 13.70 K unseen |
> | Spread rate | ~0.3-0.6 px^2/s; spread distance **5-9 mm, poorly resolved** |
> | Footprint halo | **1 px** (not 2): 12 blocks instead of 10, **37.8% of valid case10 board** masked |
> | Field dimensionality | ~**8 patterns** gives 0.079 K truncation, below the 0.1 C file resolution |
> | Accuracy to beat | old pipeline's **0.309 K** test error |
>
> ### The heat sources found on case10
>
> * **Part 1 (blocks, judged from the trusted board outside them):** exactly one is a
>   heat source — region 8 at halo=1, **269 mm^2 centred (26.1, 32.7) mm**, 100 % outward
>   over 22 of 24 sectors, +2.10 K. That is the SoC. The 970 mm^2 connector bank is
>   **passive** (22 % outward, -0.71 K) — the owner identified it on sight as passive
>   metal. Four small blocks also passive.
> * **Part 2 (trusted board only):** one hot spot at **(17.9, 10.3) mm**, +1.1 sd.
>
> ### What is built
>
> `tdp/thermal/sources.py` + `tdp/thermal/__init__.py`.
> Figure: `outputs/thermal/case10/phase1_sources.png`.
> Functions: `find_load_drop`, `fit_cooling_rate`, `fit_spread_rate`,
> `measure_heating_rate`, `find_hotspots`, `cross_check`, `judge_footprints`,
> `cooling_rate_map`, `emissivity_gain`.
>
> ### Stage 2 final lid refinement — DONE (2026-08-30)
>
> The centre-weighted source remains only the provisional exterior-ring fit for the SoC
> level. The production field now uses one non-negative broad source in a plate solve over
> the SoC footprint plus an explicit one-pixel transition rim. There is no pasted flat core.
> On case17 the SoC error is **1.99 K**, the original hidden-area error **2.29 K**, and the
> error including the rebuilt rim **2.26 K**. On case10 frame 200 the SoC core is **41.66 K**
> above ambient against **29.45 K** for a source-free solve.
>
> The output now separates the provisional ring-fitted source series from the source rate
> actually used by the final reconstruction, and saves `rebuilt_mask` plus a per-pixel model
> range. `InteriorTemperatureAnchor` is an optional interface for future boards with a
> corrected central-temperature sensor; case10 itself remains infrared-only.
>
> ### Stage 3a — fixed patterns + closed-form plate solver — DONE (2026-08-30)
>
> `tdp/thermal/patterns.py` builds the spatial library without reading any temperature
> frame: 64 slow plate modes plus source-response shapes for the broad background, SoC
> footprint, and trusted-board hot spot at four trial spread rates. Pivoted QR reduces
> 76 candidates to **72 independent patterns**. The fine-grid source-response gate is
> **below 1e-10 K worst-case**, far below the predeclared 0.10 K limit.
>
> `tdp/thermal/forecast.py` validates the Stage 2 archive, carries its uncertainty into
> observation weights, advances the reduced plate by a matrix exponential over any
> positive time gap, and obtains the settled state with one linear solve. Starting-frame
> detail outside the fixed basis is **kept exactly at time zero** and remains an explicit
> residual; it is never hidden in a fitted source. The fixed basis
> alone has 0.596 K archive projection MAE, recorded honestly as truncation rather than
> presented as a forecast score. Run `python scripts/run_patterns.py`; outputs are
> `pattern_library.npz` and `pattern_report.json`.
>
> ### Stage 3b — frozen export + equilibrium-centred forecast — DONE (2026-08-30)
>
> The first proposed fit was implemented rather than assumed: three non-negative source
> histories, smooth in time, with bounded plate-rate corrections and multi-step training.
> It fails the held-out test at **2.145 K weighted MAE**, versus **0.270 K** for simply
> holding the starting frame. Three spatial source channels cannot balance all 72 modes of
> this heterogeneous board. It is retained as the named source-only ablation, not padded
> with a large free signed correction and presented as physics.
>
> The production model uses the observed late-run property the owner identified. Its
> settled field is the mean of the final **training-only** block, frames **147-188**
> (922-1178 s). From any named frame, deviations from that field decay through the stable
> reduced plate operator; the unresolved initial residual is exact at time zero and decays
> at the measured cooling rate. Validation selected the measured spread and cooling rates
> by a 0.01 K parsimony band: the numerical alternatives differ below the file's 0.1 K
> resolution. Weighted forward MAE is **0.185 K validation** and **0.191 K held-out test**,
> beating persistence at **0.270 K**. Directly measured test pixels give **0.191 K**; the
> **0.155 K** SoC value is only agreement with the Stage 2 estimate, not hidden truth.
>
> Run `python scripts/run_forecast_fit.py`, then
> `python scripts/run_forecast.py --start-frame N`. The frame-0 result contains **674**
> forward fields at one-second spacing, is exact at its start, and ends below 1% remaining
> deviation after 673 s. The starting ambient is held fixed as a declared external
> scenario; ambient itself is not predicted. The propagated per-pixel range is a model
> range, not a confidence interval (settled median **0.338 K**, p95 **2.522 K**).
>
> ### Stage 3c — fixed-size video + held-out diagnostics — DONE (2026-08-30)
>
> `scripts/figure_forecast.py` consumes saved predictions and does not refit or select
> anything. It writes a **1280x720 H.264 MP4**, 674 frames at 24 fps, using every one-second
> field from 0 through 673 s. The synchronized panels show temperature, remaining
> deviation, declared model range, error at the nearest captured frame, and the captured
> error curve. A four-keyframe montage and numerical presentation arrays are always saved;
> a sampled GIF is the fallback if MP4 encoding is unavailable.
>
> `forecast_error_and_ablations.png` reproduces the frozen held-out scores directly from
> their saved fields: equilibrium decay **0.191 K**, persistence **0.270 K**, constant
> source **1.774 K**, smooth positive sources **2.145 K**, and no source **15.336 K**.
> The source-only error map visibly creates the artificial hot-SoC/cold-surroundings shape;
> the accepted equilibrium error map is nearly neutral on the same scale. **92 tests pass.**
>
> ### Stage 2 refinement — DONE (2026-08-29)
>
> The owner asked for two things: check a suspected 180-degree mismatch on case17, and try
> reading the pattern under a lid because the SoC looked wrong at its edges.
>
> **The 180-degree report was a figure bug, not a data one.** `contour(Z, extent=(0,W,H,0))`
> draws mirrored top-to-bottom relative to `imshow` with the same extent, so every block
> outline in `phase2_reconstruction.png` missed its block. Fixed with explicit coordinate
> arrays; a test fails if `contour` is ever given an `extent` again. The rectified grids
> themselves are aligned: all 162 px of case17's independently-located hottest patch fall
> inside case10's SoC outline and none in any other block; whole-board correlation on
> trusted pixels **+0.956 as stored against −0.550 rotated**.
>
> **The SoC really was reconstructed too cold — by 4.4 K — and it is now fixed.** Not by
> reading the lid but by shaping the source: a die under a spreading lid is not a uniform
> source over 269 mm2. Centre-weighting the source of blocks the edge test calls heat
> sources takes the SoC block **4.38 -> 2.51 K** and the whole masked area **2.63 -> 2.37 K**,
> at no extra parameter, with the worst change on any other block +0.03 K.
>
> **Reading the pattern under a lid was tried four ways and loses** (2.44 K ceiling / 2.60 /
> 6.59 / 4.83 against 2.63 K). 97% of the error is static, the lid's gain is recoverable
> only to 20%, and the reflection offset is the same unknown as the fill's own level error.
> Full numbers in `CLAUDE.md`.
>
> **Two corrections to earlier claims in this file and in CLAUDE.md:** the case17 table read
> 4.34 / -3.22 / 2.72 / +0.12 K, which never matched what the code produced (3.68 / -2.10 /
> 2.63 / +0.01 K); and "the ring fit is not a conditioning problem" was reasoned wrongly --
> estimator agreement is not sufficiency. What sets the level error is 1.87 K of ring
> residual times condition number 81. Per-block level errors reach +-4.4 K and cancel to a
> near-zero board average.
>
> Tests at that checkpoint: 71 green. `judge_footprints`
> and `reconstruct_sequence` now share one
> labelling (`tdp.thermal.block_labels`); they used 4- and 8-connectivity respectively and
> the driver joined their tables on the id.
>
> ### Stage 2 — DONE (2026-08-24), commit 38db605
>
> `tdp/thermal/reconstruct.py`. All **240 powered frames** rebuilt in ~5 s CPU. One source
> rate per block per frame, fitted by holding out a ring of trusted board and matching its
> measured temperature. The transient moves into the source rather than being marched in
> time, so no initial condition is needed — frame 0 is already 11.3 K above ambient.
> Run: `python scripts/run_reconstruct.py`.
>
> Every block gets its source *fitted*, including ones Part 1 called passive — a fitted
> rate near zero is a testable result where an assumed zero is not. Chosen spread rate
> **2.0 px^2/s = 9.27 mm** with the final cooling fit; the reconstruction moves only
> 6 % over a fivefold range of it.
>
> **Scored on case17** (same board painted matte black, so the hidden parts are measured):
> edge-only fill 4.34 K mean / bias **-3.22 K**; source-driven 2.72 K mean / bias
> **+0.12 K**. Killing the systematic cold bias is the whole point of the stage.
>
> **Where it stops, measured not asserted.** The remaining 2.6 K splits into wrong *level*
> and wrong *pattern* inside a block. Fitting sources to case17's hidden truth — the best
> one number per block can do — takes level 1.95 -> 0.20 K but pattern only 1.91 -> 1.56 K.
> So no extra source parameter helps, and the ring fit is not ill-conditioned either
> (condition number 81; per-sector, ridge and plain least squares agree to 0.04 K). Only
> the *pattern under a lid* can do better, which needs the owner's decision.
>
> **Contradiction worth keeping:** block 4 (175 mm^2 at (9.9, 42.9) mm) is "passive" to the
> Part-1 edge test but gets the highest fitted source of all. case17 says its true mean is
> 29.9 K, second hottest. Stage 2 wins there; the edge test wins on the connector bank.
> Treat `judge_footprints` as a screen, not a verdict.
>
> Tests: `tests/test_thermal_reconstruct.py`, 13 new (65 total), synthetic plate with a
> negative control.
>
> ### NEXT DECISION — new evidence, not more case10 tuning
>
> The case10 reconstruction, forecast, video, and frozen ablations are complete. Further
> accuracy work needs new information: either validate on another painted/truth board or
> use the optional corrected central-temperature sensor interface on hardware that exposes
> it. Do not tune case10 on the held-out test frames or claim the rebuilt SoC score as truth.
>
> ### Three deliverables the owner is waiting for
>
> 1. **DONE:** a reconstructed image for every one of the 240 powered frames.
> 2. **DONE:** from any named frame, a forward prediction to the settled state at
>    one-second steps. Forward only; no backward integration.
> 3. **DONE:** `hotspots.csv` — the block verdicts and trusted-board hot spots.
>
> ### Dropped by the owner
>
> The caveat that Part 1 gives a verdict per block rather than a position inside it is
> explicitly **not a concern**. Do not spend effort splitting blocks further.

## Context

Two things block the current pipeline, and this plan removes both.

**Surface reflectivity was being guessed.** The camera converts what it receives into
temperature assuming every surface radiates like a good emitter (a setting of 0.95).
Metal lids and connector shields do not, so they read far too cold. The old code tried
to *fit* how reflective each patch was — an unstable calculation exactly where the
readings are worst. New approach, per your instruction: treat the well-behaved pixels as
simply true, throw the rest away, and never estimate reflectivity again.

**Heat sources were never found, and the temperature was never actually built from
them.** The shipped run spreads the source almost uniformly over the board (its own
diagnostic says it concentrates into 58.8 "effective regions" out of 64 — i.e. it
localises nothing), and a stripped-down variant beats the full model on accuracy.

The new pipeline reverses the order: **find the heat sources first, then construct the
temperature from them.**

### What was decided

- Everything comes from the infrared images alone. No CPU-load or system logs are used.
- Plain-language names throughout, no Greek symbols. Recorded in `CLAUDE.md`.
- One approved divergence from the original instruction: training runs locally rather
  than on a remote GPU, because the model is a ~100x100 matrix and GPU launch overhead
  exceeds its compute. Revisit if the model grows.
- **The cooling phase is ignored.** Only the powered stretch is used: **frames 0–239**,
  0 to 1498 seconds. All 240 get reconstructed; nothing after the load stops is touched.
- Forward prediction only, and it must be able to **start from any moment** in that range.
- Everything stays relative. No watts, ever.
- The on-die chip temperature log is used **only as a held-out scorecard line**, never
  as an input.

---

## Answering your question: does this need conductivity? No — and that is what makes it tractable

The full physics carries four unknown material quantities: heat capacity, conductivity,
how well the surface sheds heat to air, and how much power each part makes. You cannot
get four unknowns out of a temperature movie.

But divide the whole equation through by heat capacity and only three survive — and every
one of them is directly measurable from the images:

| Quantity | Plain meaning | How it is measured | Measured on case10 |
|---|---|---|---|
| **spread rate** | how fast heat creeps sideways through the board | watch a hot spot widen over 13–35 s | 0.6 px²/s, upper bound 2.0 |
| **cooling rate** | how fast heat leaks away into the air | how fast the board levels off | 0.00685 per second (response time 146 s) |
| **heating rate** | how fast a spot warms itself | the warming rate, corrected for cooling | ~1 K/s at the strongest hot spot |

No material constant is ever needed. That is *simpler* than a conductivity-based model,
not harder — there is nothing left to look up or calibrate.

The honest price: because everything was divided by heat capacity, the model **predicts
temperature correctly but cannot state any component's dissipation in watts.** Hot spots
get ranked and compared against each other, never converted. You chose this trade
deliberately.

### The consequence that drives the whole design

Spread rate and cooling rate together give a **spread distance** — how far heat travels
before it is lost to the air. **CORRECTED after implementation: 5.1 mm, upper bound
9.1 mm**, and only 3 of 5 candidate hot spots resolve it at all. (The original
second-moment estimator was unusable — it measured where the background baseline was
put, returning *negative* spreading for four of six candidates. Replaced with a fitted
bump on a tilted background.)

The deepest point inside a masked footprint is **20 mm** from the nearest trustworthy
pixel — two to four spread distances, so edge information decays to between 14% and 2%
before it reaches the middle. Weaker than the 6–12 originally claimed, same conclusion.

**So a deep footprint cannot be filled in from its edges at all. It can only be filled
from the heat source underneath it.** This is exactly why the old edge-only fill lands
about 4 K too cold on the main chip, and it is why measuring the sources is not a
refinement here — it is the only thing that can reconstruct those regions.

---

## What was measured (all verified against case10)

| Fact | Value |
|---|---|
| Usable data | 240 frames, 157×103 grid, 0.5425 mm per pixel, ~6 s apart, 0–1498 s |
| Untrustworthy pixels | 44.7% of the board, 10 footprints, deepest 20 mm from good data |
| Ambient drift | 25.5 → 27.7 °C across the run — carried explicitly, never absorbed silently |
| Where the information is | The board is ~97% settled by frame 70; the rates come from the first ~70 frames, the rest pins the final level |

**The obvious detector does not work.** Looking for sources as bumps in a single image
(the spatial second derivative) fails in both directions: on raw images its peaks are the
*edges of the metal lids* — a sharp step has a huge second derivative — and on repaired
images it finds nothing under the lids at all, because the repair puts zero source there
by construction. This is why "the direction of the derivatives" has to mean *change over
time*, not curvature in space.

**The working detector.** For every pixel, over the rise:

```
heating rate  =  (how fast this pixel is warming)
              +  (cooling rate) x (how far above ambient it is)
              -  (spread rate) x (local curvature)
```

Solved as a least-squares fit for one constant heating rate per pixel across the rise.
It is stable — the hot-spot list barely changes between using 10 frames and 70:

| frames used | 1–10 | 1–20 | 1–40 | 1–70 |
|---|---|---|---|---|
| strongest hot spot | 0.417 K/s | 0.375 | 0.359 | 0.352 |
| positions | stable to 1–3 px across all four windows | | | |

**CORRECTED after implementation.** An earlier draft claimed roughly seven hot spots
survive; that came from eyeballing two peak lists as agreeing. Scored properly -- each
hot spot's height above its own surroundings in the *independent* estimator -- case10
yields **3 hot spots, best +2.2 standard deviations, all marginal, none strong**. Two
artifacts were inflating the earlier count:

* **Lid rims.** The spreading correction applied to raw pixels injects ~0.35 K/s of
  artifact along every metal-lid rim, where curvature is 6x that of real board. Without
  a guard, seven of ten leading hot spots sat exactly on a lid edge. A cross-check
  between the two estimators does *not* catch this -- both derive from the same raw
  time derivative, so the artifact appears in both and scores well against a ring that
  spans the step.
* **Board border.** Five of seven unconfirmed peaks sat within 8 px of the frame edge,
  where rectification mixes board with background.

Both are now guarded. What remains is real but thin.

**Second, independent estimator** for cross-checking: over the first 3–5 frames the
warming rate dominates, giving a sharper but noisier map that needs no cooling rate at
all. Each hot spot is scored by its height above its own surrounding ring in that map,
in standard deviations — *not* by whether it also appears in the other map's top N,
which conflates absence with low rank and gave a badly pessimistic answer.

⚠️ **Agreement between these two is necessary but not sufficient.** Both are built on
the same raw time derivative, so an artifact in that derivative — a lid rim — appears in
both and passes the check. Geometry has to be checked too: a hot spot sitting on a lid
boundary is an artifact regardless of how well the estimators agree about it.

**The board is not nearly isothermal, and the temperature profile does not prove it is.**
A flat temperature profile is often read as "heat spreads easily". That inference only
holds if the source is concentrated. With a spread-out source, the settled temperature is
simply source ÷ cooling rate **everywhere, whatever the spread rate** — a flat profile
then says nothing at all about spreading. Measuring the spreading directly (how fast a hot
spot widens) gives the small value above, and independently: at a large spread rate, two
hot spots 28 px apart would have blurred together within 13 s, and they demonstrably do not.

---

## Stage 1 — measure the rates and find the hot spots

New module `tdp/thermal/sources.py`.

```python
def fit_cooling_rate(board_mean_temp, times) -> CoolingFit
    """Fit the levelling-off curve. Returns cooling rate, final level, response time."""

def fit_spread_rate(temps, times, hotspot_rc, windows_s=(13,19,25,32,44,60,90,120)) -> SpreadFit
    """Track how a hot spot widens with time. Returns spread rate WITH a confidence
    interval and the size range it is actually valid over (currently ~2-6 mm)."""

def measure_heating_rate(temps, times, cooling_rate, spread_rate, n_frames=70) -> HeatingRateMap
def find_hotspots(heating_map, mask, min_area_px=6) -> list[Hotspot]
def cross_check(map_a, map_b) -> dict   # position agreement between the two estimators
```

`Hotspot` records position (pixels and mm), width, heating rate, whether it sits under a
lid, and how far it moves between the two estimators — a per-hot-spot confidence, not just
one global number.

**Report hot spots as a ranked, stability-filtered list, never a bare count.** The count
depends on where you put the threshold (14 hot spots at 10 frames, 25 at 70). The
defensible statement is "these seven appear in every window and in both estimators".

**A free by-product worth reporting on its own:** once the heating rate is known, on the
flat parts of the board `cooling rate = heating rate ÷ temperature above ambient`,
pixel by pixel, with no fitting at all. That is a cooling-rate map obtained directly.

**Honest dependency:** hot-spot *positions* are robust to the assumed cooling rate,
because the correction it applies is smooth. The *split* between hot spots and the
broad background is not. Sweep it and report the spread.

## Stage 2 — build every frame's full temperature map

New module `tdp/thermal/reconstruct.py`.

The masked footprints are treated as unknowns, the good pixels around them as fixed
boundary values, and the measured heating rate is injected underneath. Reuses the existing
solver `emissivity.solve_plate` **unchanged** — it already accepts a source term. The
footprint mask still comes from `emissivity.detect_low_emissivity`, and the confidence
weighting from `emissivity.distance_to_trusted`, both untouched.

Two changes from the old fill:

1. **The heat source is put in**, so the fill can carry a peak in the middle of a
   footprint. Without a source term it is mathematically impossible for the filled region
   to be hotter than its own edges — the old method's central failure.
2. **The measured rate of change is included**, so each frame is filled as the transient
   it actually is rather than as a frozen steady state. The rate of change *is* measurable
   under the lids even though the temperature is not, so this is well-founded.

Output: 240 reconstructed frames at full pixel resolution, plus the arrays behind them.

**Declared limitation:** the fill inside deep footprints rests almost entirely on the
measured source, since edge information cannot reach there. Sweep the reconstruction across
the spread-rate confidence interval and publish a per-footprint error bar.

## Stage 3 — the forecasting model

The board's temperature field is far simpler than its pixel count suggests. Compressing
the measured sequence shows it is **effectively about 8-dimensional**:

| patterns kept | 3 | **8** | 32 |
|---|---|---|---|
| error from discarding the rest | 0.168 K | **0.079 K** | 0.049 K |

At 8 patterns the loss is already below the 0.1 °C resolution of the source files. So the
model tracks **a few dozen pattern strengths** rather than 16,171 pixels.

**The pattern library** (`tdp/thermal/patterns.py`) is fixed in advance and built from
physics, not from the data:

- ~64 natural heat-distribution shapes of the board (its vibration-mode equivalents), plus
- **one shape per detected hot spot** — the steady temperature pattern that hot spot alone
  would produce, at 3–4 trial spread rates so the library never depends on the fitted value.

**Including the hot-spot shapes is not optional.** Benchmarked against a fine-grained
reference calculation:

| library | size | error at the hot spot |
|---|---|---|
| smooth shapes only | 256 | **−3.21 K** |
| 32 smooth + hot-spot shapes | **40** | **−0.05 K** |

Smooth shapes alone cannot represent a 2 px hot spot and systematically under-predict it by
3 K — silently recreating the exact failure this work exists to fix, while showing an
excellent *average* error. A 40-pattern library with hot-spot shapes beats a 256-pattern
smooth one by 60× on the number that matters.

**The model** (`tdp/thermal/model.py`) then says, in these compressed coordinates:

> how fast each pattern's strength changes = (heat spreading between pattern regions)
> − (heat lost to air) + (heat made by the hot spots)

Because that relationship is linear and unchanging within a training step, the solution
over any time gap has a **closed form** (a matrix exponential on a ~100×100 matrix). That
buys three things directly:

- **any starting moment, any time step** — including 1-second output, at no extra cost;
- the **final settled state without simulating toward it**, solved directly;
- no chain of thousands of tiny simulation steps to differentiate through.

What is learned — always the *coefficients* of the physical relationship, never the images
themselves:

| Learned | How it is kept physical |
|---|---|
| spread rate and cooling rate across the board | forced positive; started at the measured values above; allowed only smooth variation |
| each hot spot's strength over time | forced positive, smooth in time |
| the broad background source | forced positive |
| a small correction term | hard-capped and penalised, and **its share of the behaviour is reported** so it cannot quietly take over |

**Training.** Start from many different frames, run forward, compare against what was
actually measured, weighted so deeply-extrapolated pixels count for less. Blocks of
consecutive frames are held out with gaps either side — never a random scatter of frames,
since neighbours 6 s apart are near-copies of each other. Reuses
`train_thermal.contiguous_split` unchanged.

Because the model must start from any moment, training samples start frames uniformly
across the run rather than always beginning at frame 0.

---

## Deliverables

1. **240 reconstructed frames** + video. Four panels: raw / reconstructed / hot-spot
   overlay / difference, on one fixed colour scale, with contours showing how far each
   region is from trustworthy data so the 20 mm extrapolation is visible rather than
   hidden.
2. **A forward prediction from any frame you name**, rendered at 1-second steps through to
   the settled state, with a measured-vs-predicted error panel over the frames that have
   ground truth. Headline version seeded at frame 0.
3. **`hotspots.csv`** — rank, position in pixels and mm, width, heating rate, under-lid
   flag, and cross-estimator agreement.

Note: the existing figure helper crops to fit the title text, so frames come out at
different sizes and no video encoder will accept them. Frames need a fixed-size saver.
Follow the existing animation fallback pattern — try the video writer, fall back to GIF,
and **always** write the montage and the raw arrays, so a missing codec never loses results.

## Running it

A GPU is not needed and I would not use one. The model is a ~100×100 matrix; a full
400-step fit is about **20 seconds on a CPU**, and GPU launch overhead on matrices that
small exceeds the actual computation. Stage 1 and 2 are sparse solves, also CPU work.

I will still build the self-contained export — one ~20 MB file holding the pattern library,
reconstructed frames, hot-spot table, measured rates, and the frozen train/test split — so a
run is reproducible anywhere without the 3.1 GB image set, and so a remote rerun cannot
silently re-split the data and produce numbers that no longer compare. `--device` stays
plumbed if you want it later.

**Before anything else:** the repository has no commits and the 3.1 GB dataset is not
excluded, so provenance on every run is currently blank and `git add -A` would try to
stage the whole image set. One `.gitignore` line and one initial commit.

## Checking it works

- **Synthetic first.** Build a fake board with known hot spots, known rates, and a known
  mask, and require the pipeline to recover them before it is pointed at real data. Include
  a **negative control** — a board with a perfectly even source — where it must report *no*
  hot spots. A method that invents hot spots there will invent them everywhere.
- **case17** — the same board painted matte black, so every pixel is trustworthy and only
  16% needs repair instead of 44.7%. Hot-spot positions should match case10's; the cooling
  rate should not, because paint changes the surface. This replaces the cross-check lost by
  dropping the cooling phase, and it is a stronger test — a different surface, same silicon.
- **Held-out chip sensor** — never fitted to. Does the reconstruction close the 4 K gap
  under the main chip, and does the reconstructed surface stay *below* the die temperature
  everywhere, as physics requires?
- **Two-estimator agreement** on hot-spot positions, reported per hot spot.
- **Leave-part-out** — fit on the first half of the rise, predict the second half.
- **Ablations, decided in advance**: hot-spot shapes in the library vs not (expect −3 K at
  the hot spot without); source-driven fill vs edge-only fill (expect the 4 K chip gap to
  close); hot spots vs background alone; and a no-source-measurement control. That last one
  goes in the results table from day one, so it cannot quietly go unrun the way the current
  pipeline's losing comparison did.
- Existing test suite (52 tests, ~37 s) stays green.

## The success metric, chosen before any results exist

**Hot-spot reconstruction accuracy and cross-estimator agreement — not board-average
error.** The hot spots are small and the background is broad, so average error is dominated
by the background and will barely move whatever we do. That is precisely how the current
pipeline ended up with a stripped-down variant beating the full model. Fixing the metric
now is the single most important guard against repeating it.

Reference points: current average error 0.309 K; source-file resolution 0.1 K; pattern
library truncation 0.079 K. The forecasting model must beat 0.309 K on average **and** hold
hot-spot error under 0.10 K, which is the criterion that actually bites.

## What the results must never claim

- No watts, and no "component X supplies Y% of board power". Everything is temperature per
  second divided by an unknown heat capacity — and that capacity **varies across the board**
  (copper-backed areas differ from bare laminate), so even the *ratios* between components
  are not power ratios. The existing `relative_component_power.csv` overstates this today
  and gets renamed.
- No claim that N hot spots means N components — two adjacent chips merge into one hot spot
  at this resolution.
- No single "the board's spread distance is X" — it is measured over a 2–6 mm range and is
  not established outside it.
- No claim that spreading varies across the board unless an ablation proves it; a threefold
  variation changes predictions by less than 10%, so the data cannot tell.

## Order of work

**Phase 1 is built and committed** (`tdp/thermal/sources.py`). Its gate is **only partly
met**: the rates are solid, the localisation is marginal. Running Phase 1 on case17
before starting Phase 2 is the recommended way to settle whether that is a case10
limitation or a method limitation.

| Phase | Content | Gate |
|---|---|---|
| 0 | `.gitignore` the dataset, initial commit | provenance works |
| 1 | Cooling rate, spread rate, heating-rate map, hot spots | DONE. Rates solid; localisation marginal (3 hot spots, best +2.2 sd) |
| 2 | Reconstruct 240 frames; validate on case17 and the chip sensor | chip gap closes; surface stays below die |
| 3 | Pattern library + closed-form solver, **no learning yet** | hot-spot error under 0.10 K vs a fine reference |
| 4 | Self-contained export | Stage 5 runs from the export alone, dataset unmounted |
| 5 | Train the forecasting model | beats 0.309 K average *and* 0.10 K at hot spots |
| 6 | Figures, video, report, ablations | |

## Risks

| Risk | Handling |
|---|---|
| Average error barely improves | Success metric is hot-spot accuracy, fixed in advance |
| Deep-footprint fill rests entirely on the measured source | Sweep the spread-rate interval; publish per-footprint error bars |
| The learned rates drift to exploit pattern-library error | Report library adequacy every training step; allow only smooth variation |
| Only ~70 frames carry rate information | Keep the model small — the field is only ~8-dimensional anyway |
| The correction term swallows the physics | Hard cap, penalty, and its share is reported |
| Ambient drifting 2.2 K acts as a phantom source | Carried explicitly per frame |
| Hot-spot count is threshold-dependent | Ranked stability-filtered list, never a bare number |
