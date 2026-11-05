# Drawing the `ResidualUNet` architecture — tool choice + ready prompts

Source of truth: `tdp/unet/model.py` (branch `experiment/unet-painted-truth-reconstruction`),
plus `tdp/unet/train.py` for the training contract. Everything below is taken from those
files, not from a generic U-Net.

## The facts the diagram must get right

| item | value |
|---|---|
| class / builder | `ResidualUNet` / `build_residual_unet(in_channels=3, base_channels=24, depth=3)` |
| job | predicts a temperature **correction**; prediction = input temperature field + correction |
| input | 3 channels, **103 rows x 157 columns** on the canonical grid (0.5425 mm/px) |
| channel 0 | temperature above ambient / 60 K (fixed unit, never per-map normalised) |
| channel 1 | board mask |
| channel 2 | untrustworthy-footprint mask |
| padding | reflect-pad to a multiple of 8 (104 x 160), cropped back at the end |
| head | 3 -> 24 conv 3x3, GroupNorm(8), SiLU |
| encoder levels | 24 / 48 / 96 channels, depth 3 |
| ConvBlock | two conv 3x3 (padding 1) + GroupNorm(8) + SiLU, same channel count |
| downsample | conv 3x3 stride 2 + GroupNorm + SiLU (24->48, 48->96) |
| bottleneck | ConvBlock at 96 channels (26 x 40) |
| upsample | ConvTranspose2d kernel 2 stride 2 + GroupNorm + SiLU |
| decoder | concat(upsampled, same-scale encoder skip) -> ConvBlock (2C in, C out) |
| skip connections | encoder output at scale L -> decoder at scale L (long skips) |
| tail | conv 1x1, 24 -> 1 |
| output | one correction field, 103 x 157, in kelvin (after x 60 K) |
| parameters | **538,753** at the defaults |
| training | 240 powered frames, whole archive (no split), complete-map L1 on board pixels, Adam 1e-3, cosine annealing to 1e-5, batch 12, 200 epochs, CPU |
| must NOT appear | emissivity, conductivity, watts, power, or any Greek symbol (repo rules: plain words only) |

Handy per-module parameter split for an annotation callout: head 696, encoders 218,400,
downsamples 52,128, bottleneck 166,272, upsamples 23,184, decoders 78,048, tail 25.

## Which AI route to use

Ranked for a methods figure that reviewers will read closely:

1. **Diagram-as-code, written by a coding agent — best default.**
   Give Claude Code / Codex / Cursor the file `tdp/unet/model.py` and ask for **TikZ or a
   hand-written SVG**. It is deterministic, text is exact, blocks and arrows are aligned,
   it is editable forever, and it exports vector PDF for LaTeX. This is the only route
   where "538,753 parameters" and "103 x 157" cannot be silently falsified.
2. **SVG from a frontier chat model, then tidy in Inkscape/Figma.** Same idea, no LaTeX
   needed. Good for slides.
3. **Image models (Nano Banana Pro / Gemini 3 Pro Image, GPT-Image-1, Ideogram, Recraft).**
   Excellent aesthetics, unreliable text and geometry: they garble long strings, drop
   numbers, and invent skip connections. Use them **only after the code-drawn version is
   correct**, either as a style reference or on a text-free version, then overlay the real
   labels in Inkscape. Recraft V3 is the exception worth trying because it exports SVG.
4. **DiagramGPT / Eraser, Mermaid Chart, Excalidraw AI, Whimsical.** Fast and clean for
   README/blog-level diagrams; too little control over tensor shapes and skip routing for
   a paper figure.
5. **Vision redraw loop.** Screenshot any candidate figure, feed it back to a vision model
   with `model.py`, and ask it to list every factual mismatch. Cheap and it catches the
   usual errors (wrong channel counts, missing residual add, a skip that is drawn as a
   plain arrow into the wrong level).

Recommended pipeline: 1 -> render PNG -> 5 -> fix -> optionally 3 for polish.

## Prompt 1 — coding agent, produce TikZ and SVG (main prompt)

```text
Read tdp/unet/model.py and tdp/unet/train.py in this repo before writing anything.
Draw the architecture of the ResidualUNet defined there as a publication-quality
vector figure. Output two files: figures/resunet_architecture.tex (standalone TikZ,
compiles with pdflatex) and figures/resunet_architecture.svg. Do not invent layers,
numbers, or connections; if something is ambiguous in the code, leave a comment
instead of guessing.

WHAT THE FIGURE MUST SHOW
- Classic U shape. Left column = encoder, bottom = bottleneck, right column = decoder.
- Input strip on the left with three stacked channels, each labelled:
  "temperature above ambient / 60 K", "board mask", "untrustworthy-footprint mask".
  All three are H=103 x W=157. Annotate that the field is reflect-padded to 104 x 160
  (multiple of 8) and cropped back at the end.
- head: 3 -> 24, conv 3x3, GroupNorm(8), SiLU.
- Three encoder levels with 24, 48, 96 channels. Each level contains one ConvBlock,
  drawn as a box labelled "conv 3x3 x2, GroupNorm(8), SiLU".
- Downsample between levels: "conv 3x3, stride 2". Annotate spatial sizes
  104 x 160 -> 52 x 80 -> 26 x 40 next to the feature maps.
- Bottleneck at 96 channels, 26 x 40, one ConvBlock.
- Decoder mirrors the encoder: "conv transpose 2x2, stride 2" then a box
  "concat with encoder skip -> conv 3x3 x2, GroupNorm(8), SiLU" going 96 -> 48 -> 24.
- Draw the two long skip connections as clearly separate curved arrows from each
  encoder block to the matching decoder block, routed over the top of the U, and
  label the first one "skip: concatenated on channels".
- tail: conv 1x1, 24 -> 1, labelled "correction field, 103 x 157, kelvin".
- End with an explicit residual add: a circled plus where the input temperature
  channel and the correction field meet, producing
  "predicted temperature = input temperature + correction". This residual path is the
  whole point of the design (capacity concentrates on the ~38% masked pixels rather
  than relearning the ~62% of trusted pixels), so make it visually prominent with a
  long arrow running under the U from the input strip to the plus node.
- One small annotation box, dashed border: "538,753 parameters at base_channels=24,
  depth=3" and "trained on 240 powered frames, complete-map L1 over board pixels,
  Adam 1e-3 with cosine annealing, batch 12, 200 epochs, CPU".

STYLE
- Vector only, no raster. White background, thin dark strokes, one accent colour for
  the skip connections and a second for the residual path; keep it colour-blind safe.
- All text horizontal, no rotated labels, sans-serif. Readable when printed at one
  column width in a two-column paper (target figure width 8.5 cm, but produce it so it
  also scales to a full page).
- Group the figure with a light-shaded container for the network and a separate
  container for the input strip and the residual add.
- Write equations as words, not symbols. Use "temperature above ambient", never theta,
  never alpha/beta. Never show or mention emissivity, conductivity, watts, or power:
  the network outputs a temperature correction only.
- Add a caption underneath, 2-3 sentences, that states the residual contract and the
  fixed 60 K unit.

Then verify yourself: render the SVG to PNG, read the PNG back, and check every number
and every arrow against model.py. Report any mismatch you found and fixed.
```

## Prompt 2 — image model (Nano Banana Pro / GPT-Image-1 / Ideogram / Recraft)

Keep the rendered labels short; image models corrupt long text. Best used on the
correct code-drawn layout as a style reference, or on a text-free version that you
label yourself afterwards.

```text
A clean scientific architecture diagram of a small residual U-Net for infrared
thermal image correction, flat vector style, white background, wide landscape canvas
(16:9), thin dark grey outlines, muted blue encoder, muted orange decoder, light grey
skip arrows arcing over the top, one green arrow running underneath the whole network.

Layout, left to right: a stack of three small identical rectangles labelled
"temperature / 60 K", "board mask", "mask"; then a descending staircase of three
rectangles (labels "24", "48", "96 channels"); a wide rectangle at the bottom
("96 channels"); then an ascending staircase of three rectangles (labels "96", "48",
"24 channels"); then a small rectangle labelled "1 x 1 conv". Curved arrows arc over
the top from each descending rectangle to its matching ascending rectangle. At the far
right a small circle with a plus sign receives two arrows: one short arrow from the
last rectangle and one long arrow running beneath the whole network from the input
stack. A final arrow exits the plus circle to a rectangle labelled
"predicted temperature".

Style: no gradients, no 3D, no glow, no drop shadows, no watermark, no logo, no title
text, no decorative elements. Consistent rectangle sizes and even spacing, textbook
figure look, crisp edges, high resolution. Render every quoted label exactly as
written and do not add any other text.
```

## Prompt 3 — Mermaid / D2 for README and issues

```text
From tdp/unet/model.py, write a Mermaid flowchart (left to right) of the ResidualUNet
forward pass. Show the 3-channel input, the 24/48/96 encoder levels with their
downsample steps, the 96-channel bottleneck, the decoder levels with conv-transpose
upsampling and explicit skip edges from each encoder level, the 1x1 conv tail, and a
residual addition node whose other input is channel 0 of the input. Label edges with
tensor shapes (channels and H x W). Use only plain-word descriptions, no Greek
symbols, and no mention of emissivity, conductivity, or power. Keep it under 30 nodes.
```

## Prompt 4 — verification pass (run after any render)

```text
Here is a rendered architecture figure and the file tdp/unet/model.py. Compare them as
an auditor, not as a designer. List every factual mismatch: wrong channel counts, wrong
spatial sizes, a layer that does not exist in the code, a missing GroupNorm or SiLU, a
skip connection drawn from or to the wrong level, a missing or misplaced residual add,
and any number that differs from the code. State plainly if the figure is accurate.
Do not comment on aesthetics.
```
