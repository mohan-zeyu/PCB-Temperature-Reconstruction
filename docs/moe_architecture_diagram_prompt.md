# Drawing the mixture-of-experts architecture — status, ground truth, and prompts

## Read this first: the MoE does not exist in the code yet

The prompt below produces a **target-design** figure. Per the owner's decision, the
artwork itself carries no implementation-status text; the caption and body text carry it.
That distinction matters, because a paper-style MoE diagram otherwise reads as "we built
this".

Evidence in this repo:

- `tmp/report_outline_supplement.md:14` — "**"Unet + MoE"目前没有实现。** 仓库里今天只有
  "残差 U-Net"（539k 参数 ...），没有任何 MoE 代码。"
- `tmp/experimental_principles_draft.md` §3.3.5 — titled "按板型路由：混合专家架构（**规划中**）".
  It defines the design and then says the experiments cover a single board type (K=1), so
  the routing structure "属于后续工作" (is future work).
- `CLAUDE.md` — the approved future direction is the **fixed-load-conditioned single
  model**, not a mixture.
- A grep for `moe|expert|router|gating|top-k|dispatch` across the repo hits nothing but
  those two draft documents. There is no router module, no gate, no dispatch layer.

**What does exist**, and is the honest basis for the figure: one separately trained
residual U-Net per board type, plus the paired archives to train them.

```text
outputs/thermal/unet_case10/train_default/model_best.pt                      <- Pi 4B expert
outputs/thermal/unet_opi5pro/train_half|train_full|train_both/model_best.pt  <- OPi 5 Pro experts
```

So: the **experts exist as checkpoints**, the **router does not exist**. The figure shows
the target design.

## Ground truth the diagram must match

| item | fact | source |
|---|---|---|
| router key | the session's `device` field — the board type is recorded metadata, not inferred from the image | `configs/measurements_v3.yaml`, `configs/measurements_opi5pro.yaml` |
| board type 1 | `raspberry_pi_4b`, outline 85 x 56 mm, canonical grid **157 x 103** | `configs/measurements_v3.yaml:13,20` |
| board type 2 | `orange_pi_5_pro`, outline 89 x 56 mm, grid **164 x 103** | `configs/measurements_opi5pro.yaml:17,24` |
| board types 3..K | any further board; each expert keeps its own grid, so the input resolution is per-expert, never shared | by design |
| expert architecture | `ResidualUNet`, `base_channels=24`, `depth=3`, levels 24/48/96, **538,753 parameters**, one set of weights per board type | `tdp/unet/model.py` |
| expert input | 3 channels: temperature above ambient / 60 K, board mask, untrustworthy-footprint mask | `tdp/unet/train.py` (`PairedSet.inputs`) |
| expert output | one correction field; prediction = input temperature field + correction (residual contract) | `tdp/unet/model.py` |
| expert loss | complete-map L1 over board pixels, in kelvin | `tdp/unet/train.py` |
| expert training | Adam 1e-3, cosine annealing to 1e-5, batch 12, 200 epochs, CPU, best checkpoint by board MAE | `tdp/unet/train.py` |
| Pi paired data | case10 (bare, full) <-> case17 (painted, full) = 240 pairs; case09 <-> case16 (half) = 192 pairs; archive 432 maps, `157 x 103` | `outputs/thermal/unet_case10/painted_pair_dataset_both.npz` |
| Orange Pi paired data | case12/13 (bare, half/full) <-> case14/15 (black tape) = 480 pairs, `164 x 103` | `outputs/thermal/unet_opi5pro/pairs_both.npz` |
| routing rule | deterministic by board index `w_i` in `{1..K}`; **no gating weights**; `x_hat_i = X_i + G_phi_{w_i}(X_i)`; equivalently an indicator sum in which exactly one term is non-zero | `tmp/experimental_principles_draft.md` eq. (3.4), (3.5) |
| fallback | a board type with no expert falls back to the generic/single-expert model and the result is explicitly flagged "uncovered board type" | same section |
| evaluation | leave-one-board-out; no cross-board generalization claim before that passes | same section |
| forbidden in this project | Greek symbols (write "temperature above ambient"), and any mention of watts, emissivity, conductivity, or power | `CLAUDE.md` units + naming rules |

## What "looks like a real MoE diagram" means here

Real MoE figures (Switch Transformer, GShard, Mixtral) share a visual grammar: a token or
sample enters, a **router/gateway** computes a discrete selection, K **expert** columns sit
below it, only the selected experts are drawn active, and the outputs are combined. The
project's version keeps that grammar and changes one thing: the selection is a known
one-hot index, not a learned softmax score.

So draw it as a gateway with a one-hot vector inside, one solid highlighted path and the
remaining paths grey dashed. That is visually indistinguishable from a Switch-style figure
and is also exactly what §3.3.5 defines. Do **not** draw softmax weights, top-k, or a
weighted sum.

The expert set is **open-ended**: show the two board types that have paired data today,
then a vertical ellipsis, then a generic expert K, so the figure reads as "one expert per
board type, K of them" rather than a closed two-way switch. Load levels are deliberately
out of scope for this figure.

## Prompt 1 — main prompt, paper-style MoE figure (give this to a coding agent)

```text
Produce a publication-quality vector architecture figure of a mixture-of-experts
reconstruction model for infrared PCB thermal imaging. Draw it as the target design: one
expert per board type, K board types, selected by a deterministic one-hot router. The
artwork must read as open-ended in the number of boards, not as a closed set of two.

Read these files first if you have repo access, and do not contradict them:
  tdp/unet/model.py                 (the expert network)
  tdp/unet/train.py                 (expert input contract, loss, optimiser)
  tmp/experimental_principles_draft.md, section 3.3.5  (the routing definition)
  configs/measurements_v3.yaml, configs/measurements_opi5pro.yaml  (board types)

DELIVERABLE
A standalone TikZ file figures/moe_architecture.tex that compiles with pdflatex, plus
figures/moe_architecture.svg. Vector only, no raster. Full-width figure in a two-column
paper, readable at 17 cm wide.

PANEL (a) - the routing level, main panel
- Left: a stack of three input maps labelled "temperature above ambient / 60 K",
  "board mask", "untrustworthy-footprint mask", annotated "each expert keeps the grid of
  its own board: 157 x 103 for the Raspberry Pi 4B, 164 x 103 for the Orange Pi 5 Pro,
  and its own resolution for every further board type". Below it a metadata tag:
  "session record: device = board type".
- A router / gateway box in the middle, labelled "Router (gateway)" with the subtitle
  "deterministic board-type selection". Inside the box show a one-hot vector
  [1 0 ... 0] of K entries, with the caption "w_i in {1..K}: the index is known from the
  session record, not inferred from the image". Draw NO softmax, NO probability bars, NO
  top-k, NO learned gate: this router is a lookup, and the figure must not imply
  otherwise.
- K expert columns to the right, each a compact residual U-Net drawn as an
  encoder-decoder U shape. Show, top to bottom:
    Expert 1: "G_1  Raspberry Pi 4B expert", annotation "157 x 103, paired archive
    case10 <-> case17, 240 full-load pairs + 192 half-load pairs"
    Expert 2: "G_2  Orange Pi 5 Pro expert", annotation "164 x 103, paired archive
    case12/case13 <-> case14/case15, 480 pairs"
    a vertical ellipsis spanning the expert column, centred in the column
    Expert K: "G_K  generic expert - fallback for an uncovered board type", drawn with a
    dashed outline and no data annotation; add a dashed vertical ellipsis above it if the
    panel needs it for spacing
  Label the group with a width bracket reading "K board types, one expert each, K is
  open-ended". All experts must be drawn as the SAME architecture with SEPARATE weights:
  same encoder-decoder shape, same three levels; label each
  "24 / 48 / 96 channels, group norm, SiLU, 538,753 parameters".
- Routing: from the router box, one solid highlighted arrow to the selected expert and
  grey dashed arrows to every other expert, with legend entries "active path (one-hot)"
  and "inactive path (not evaluated)", and a small count label "K - 1 inactive paths".
  Add the side note "only one expert runs per sample; this is a switch, not an ensemble,
  and expert outputs are never averaged or weighted".
- Right end: a circled plus labelled "residual add". Two inputs: a long prominent arrow
  from the input temperature channel running beneath the panels (the residual contract is
  the point of the design), and the selected expert's correction output. Output arrow to
  "predicted temperature map". Word equation:
  "prediction = input temperature field + selected expert's correction".
- Do NOT put any implementation-status text in the artwork: no "planned", no "not
  implemented", no "K = 1 today". The paper text and caption carry that.

PANEL (b) - inside one expert, zoomed
- A zoom callout from Expert 1 showing the actual residual U-Net: 3-channel input, head
  3 -> 24, encoder levels 24 / 48 / 96 with stride-2 downsampling, bottleneck at 96,
  decoder with 2x2 stride-2 transposed convolutions and skip concatenation back to 24,
  1x1 tail to 1 channel, then the residual add. Annotate spatial sizes
  104 x 160 -> 52 x 80 -> 26 x 40 after reflect-padding to a multiple of 8, and note the
  crop back to 157 x 103.
- Mark "identical topology, independent weights; every expert keeps its own board grid".
  Annotate the loss: "complete-map L1 over board pixels, in kelvin".

PANEL (c) - optional if space allows
- A compact training/evaluation schematic: paired sessions grouped by board type feeding
  their own expert, and a leave-one-board-out evaluation fold. Keep it small; drop it if
  it crowds panels (a) and (b).

TYPOGRAPHY AND STYLE
- White background, thin dark strokes, one accent colour for the active routing path,
  a second for the residual path, grey for inactive paths. Colour-blind safe palette.
- All text horizontal, sans-serif, no rotated labels. Consistent box widths, even vertical
  spacing, straight orthogonal arrows with right-angle bends. Textbook figure look.
- Vector PDF output matters: this goes into a paper. No gradients, no shadows, no 3D.
- Write equations as words. Never use Greek symbols: say "temperature above ambient",
  never theta; never alpha or beta. Name the experts G_1, G_2, G_K.
- Never show or mention emissivity, conductivity, watts, power, or energy anywhere: each
  expert outputs a temperature correction in kelvin and nothing else. There is no
  emissivity output head and no power prediction.
- Caption underneath the artwork (outside the drawn figure), 3-4 sentences, paper style:
  per-board expert design, deterministic one-hot routing by recorded board type, the
  residual add, and that the routing layer is a target design whose current experiments
  use K = 1.

COMMON MISTAKES TO AVOID - a generic MoE template gets all of these wrong
1. Drawing a softmax gate, probability weights, or top-k. The routing here is a known
   one-hot index, and equation (3.4) of the project draft explicitly introduces no gating
   weights.
2. Summing expert outputs with weights. Only one term of the indicator sum is non-zero.
3. Giving every expert the same input resolution. They differ per board: 157 x 103,
   164 x 103, and so on.
4. Drawing the experts as different architectures. Same topology, separate weights.
5. Adding an image-based board classifier in front of the router. The board type is read
   from the session record, so no classification head exists.
6. Omitting the residual add, or drawing it as an optional skip.
7. Closing the expert set at two, or drawing the router as a fixed two-way switch with no
   ellipsis. It must read as K board types with K open-ended.
8. Putting implementation-status text inside the artwork; that belongs in the caption and
   body text.
9. Claiming cross-board generalisation in the figure. With the two boards that have paired
   data, leave-one-board-out has not been run.

Finally, audit your own figure: render it, read it back, and check every number, channel
count, and arrow against tdp/unet/model.py and section 3.3.5. Report every mismatch you
found and how you fixed it.
```

## Prompt 2 — variant with a learned gate (only if you change the model definition)

Marked separately because it contradicts §3.3.5. Use it only after deciding that the
router becomes a trained classifier over board type or sample statistics.

```text
Same task and same figure structure as before, but the router is now a learned gate.
Draw the router as a small network that consumes the input maps and emits a softmax
distribution over K experts, with a top-1 or top-2 selection, and combine the selected
expert outputs with their gate weights. Add, in the figure itself, a clearly marked
"design change" note: the project's draft section 3.3.5 defines routing as a
deterministic lookup by recorded board type with no gating weights, so this variant is a
different model that has not been written down or implemented. Also annotate the extra
risks this variant introduces: routing error becomes a new error source, board type must
now be inferred from data, and the gate needs its own training signal. Do not silently
blend the two designs in one figure.
```

## Prompt 3 — variant with experts per fixed load level

The alternative definition suggested in `tmp/report_outline_supplement.md:64`. Not part of
the main figure, which is board-typed.

```text
Same task, but the experts are selected by fixed load level instead of board type: one
expert per normalized load, for example load 0.50 and load 1.00, routed by the load value
recorded for the session. Annotate the honest data status on the figure: the merged
two-load archive is case09 <-> case16 (192 pairs) plus case10 <-> case17 (240 pairs), each
load level has only ONE independent session, and a mixture selected by load needs several
sessions per level before it is defensible. The approved alternative in the project is a
small capacity load-conditioned single model, so label this panel "not implemented,
weakly supported by data". Do not present it as the adopted design.
```

## Prompt 4 — audit prompt (run on any render)

```text
Here is a rendered mixture-of-experts architecture figure, plus the project's routing
definition and model source. Audit it as a reviewer, not a designer. Report every
factual problem: a gate, softmax, top-k, or weighting that the definition does not have,
an expert set drawn as closed at two with no ellipsis, wrong or shared input resolutions,
experts drawn with different topologies, a missing fallback path, a missing or misrouted
residual add, any implementation-status text inside the artwork, any claim of cross-board
generalisation, and any appearance of emissivity, power, watts, or Greek symbols. Also
state which parts of the figure are implemented today versus proposed, so the text can be
written honestly. Do not comment on aesthetics.
```

## Claims checklist before this figure goes in the paper

- Keep the artwork free of status text, but write "target design, not yet implemented" in
  the caption and body; do not write "we built a mixture of experts".
- Say the experts are per board type and that separate checkpoints already exist for the
  two boards with paired data.
- Say routing is a deterministic lookup from recorded metadata, with no learned gate.
- Present K as open-ended; note that K = 1 in the reported experiments.
- Do not claim cross-board generalisation: leave-one-board-out has not been run.
- Keep every quantity in kelvin; no watts, no emissivity, no conductivity, no power.
