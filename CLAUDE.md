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
- **Infrared images only.** No CPU-load or system log is an input. `soc_log_100.csv`
  may appear as a held-out scoreboard line, never as an input.
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
python -m pytest tests/ -q                                       # 52 tests, ~37 s
python -m tdp.dynamic_inverse run --smoke --sequence case10_rpi4b_bare_full
```

Configuration is split on purpose: `configs/measurements_v3.yaml` records what the
*experiment* was, `configs/dynamic_inverse.yaml` records what the *analysis* does.
Mixing them makes an ablation look like a re-measurement.

## Style

Module docstrings explain *why* a choice was made and what it prevents; comments carry
measured evidence rather than restating the code; every reported quantity carries its
unit.
