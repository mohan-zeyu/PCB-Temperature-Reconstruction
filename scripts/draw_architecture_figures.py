"""Draw the ResUNet expert and the board-routed mixture-of-experts architecture.

Why this file exists
--------------------
The architecture figures must not drift from the code.  Every number drawn here is
taken from ``tdp/unet/model.py`` (topology, channel widths, 538,753 parameters),
``tdp/unet/train.py`` (three input channels, the fixed 60 K unit, L1 loss, optimiser)
and ``configs/measurements_*.yaml`` (the two board types and their grid sizes).  The two
named experts are the board types that currently have paired archives; the expert set is
drawn open-ended because more boards are expected.

Both figures are written as vector PDF and SVG, which is what a paper needs, plus a PNG
for quick inspection.  Run:

    python scripts/draw_architecture_figures.py

Outputs land in ``outputs/thermal/figures/``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_arch_figures")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Plain words, no Greek symbols, exactly as the project's naming rule requires.
INK = "#1f2933"
GREY = "#9aa5b1"
ENC = "#cfe0f3"
DEC = "#fbe3cc"
BOT = "#e4dcf2"
INP = "#eceff1"
OUT = "#d7ecd9"
ROUTER = "#fff3d6"
ACTIVE = "#c0392b"
RESIDUAL = "#2e7d32"
SKIP = "#5b7c99"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # DejaVu Sans first: it is the only installed family that carries the arrow,
        # times and vertical-ellipsis glyphs used below, so nothing renders as tofu.
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 6.0,
        "text.color": INK,
        "axes.edgecolor": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _box(ax, cx, cy, w, h, fc, *, ec=INK, lw=0.8, ls="-", z=2):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.5",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, p0, p1, *, color=INK, lw=1.0, ls="-", ms=7.0, z=4):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=z,
        )
    )


def _block(ax, cx, cy, w, h, title, lines, *, fc, title_fs=5.9, line_fs=5.2):
    """A labelled stage box: bold title plus up to two detail lines."""
    _box(ax, cx, cy, w, h, fc)
    ax.text(cx, cy + h * 0.28, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", zorder=6)
    for i, line in enumerate(lines):
        ax.text(cx, cy + h * 0.02 - i * h * 0.24, line, ha="center", va="center",
                fontsize=line_fs, color="#39424e", zorder=6)


def _plus_node(ax, cx, cy, r=2.9):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=INK, linewidth=1.0,
                        zorder=5))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=8.5, zorder=6)


def _draw_unet(ax, *, strip=True):
    """The ResidualUNet forward pass: 3 levels, long skips, residual add.

    Rows are centred on the middle channel of the input stack, so the head arrow leaves
    the stacked tensor straight into the first encoder block.
    """
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 100)
    ax.axis("off")

    row = (76.0, 53.0, 30.0)
    enc_x, bot_x, dec_x = 41.0, 65.0, 89.0
    plus_x, out_x = 104.0, 119.0

    # --- input: one temperature field plus two indicator fields ---
    for text, cy in (
        ("temperature above ambient / 60 K", row[0] + 8.0),
        ("board mask", row[0]),
        ("untrustworthy footprint mask", row[0] - 8.0),
    ):
        _box(ax, 13, cy, 24, 6.4, INP)
        ax.text(13, cy, text, ha="center", va="center", fontsize=5.1, zorder=6)
    ax.text(13, row[0] + 14.5, "input: 3 channels", ha="center", fontsize=5.6,
            fontweight="bold")
    ax.text(7.5, row[0] - 13.5, "one frame of one board type", ha="left", fontsize=4.9,
            color=GREY)

    _arrow(ax, (25, row[0]), (32, row[0]), color=ENC, lw=1.6)
    ax.text(28.5, row[0] + 4.5, "head\n3 → 24", ha="center", va="center", fontsize=4.7,
            color="#39424e", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"))

    # --- encoder ---
    _block(ax, enc_x, row[0], 18, 11, "24 channels", ["103 × 157", "2 × conv 3×3"], fc=ENC)
    _block(ax, enc_x, row[1], 18, 11, "48 channels", ["52 × 80", "2 × conv 3×3"], fc=ENC)
    _block(ax, enc_x, row[2], 18, 11, "96 channels", ["26 × 40", "2 × conv 3×3"], fc=ENC)
    for i, (lo, hi) in enumerate(((24, 48), (48, 96))):
        _arrow(ax, (enc_x, row[i] - 5.5), (enc_x, row[i + 1] + 5.5), color=ENC, lw=1.6)
        ax.text(enc_x + 2.5, (row[i] + row[i + 1]) / 2,
                f"downsample\nconv 3×3, stride 2\n{lo} → {hi}",
                ha="left", va="center", fontsize=4.9, color="#39424e")

    # --- bottleneck at the coarsest scale ---
    _block(ax, bot_x, row[2], 18, 11, "96 channels", ["26 × 40", "2 × conv 3×3"], fc=BOT)
    _arrow(ax, (enc_x + 9, row[2]), (bot_x - 9, row[2]), color=BOT, lw=1.6)
    ax.text(bot_x, row[2] + 8.5, "bottleneck", ha="center", fontsize=4.9,
            color="#39424e")

    # --- decoder ---
    _block(ax, dec_x, row[1], 18, 11, "48 channels", ["52 × 80", "2 × conv 3×3"], fc=DEC)
    _block(ax, dec_x, row[0], 18, 11, "24 channels", ["103 × 157", "2 × conv 3×3"], fc=DEC)
    _arrow(ax, (bot_x + 9, row[2]), (dec_x - 9, row[1]), color=DEC, lw=1.6)
    ax.text(bot_x + 16, (row[1] + row[2]) / 2,
            "upsample\nconv transpose 2×2\n96 → 48", ha="left", va="center", fontsize=4.9,
            color="#39424e")
    _arrow(ax, (dec_x, row[1] + 5.5), (dec_x, row[0] - 5.5), color=DEC, lw=1.6)
    ax.text(dec_x - 2.5, (row[0] + row[1]) / 2,
            "upsample\nconv transpose 2×2\n48 → 24", ha="right", va="center", fontsize=4.9,
            color="#39424e")

    # --- long skip connections: encoder features concatenated at equal scale ---
    _arrow(ax, (enc_x + 9, row[0]), (dec_x - 9, row[0]), color=SKIP, lw=1.2, ls=(0, (4, 2)))
    ax.text((enc_x + dec_x) / 2, row[0] + 3.6, "skip: concatenate encoder channels",
            ha="center", fontsize=5.0, color="#3d5a73")
    _arrow(ax, (enc_x + 9, row[1]), (dec_x - 9, row[1]), color=SKIP, lw=1.2, ls=(0, (4, 2)))
    ax.text((enc_x + dec_x) / 2, row[1] + 3.6, "skip", ha="center", fontsize=5.0,
            color="#3d5a73")

    # --- tail, residual add, output ---
    _arrow(ax, (dec_x + 9, row[0]), (plus_x - 2.9, row[0]), color=DEC, lw=1.6)
    ax.text(100.0, row[0] + 8.5, "tail: 1×1 conv\n24 → 1", ha="center", va="center",
            fontsize=4.9, color="#39424e")
    _plus_node(ax, plus_x, row[0])

    # The residual path is the point of the design, so it is drawn long and prominent.
    ax.plot([5, 5, plus_x], [row[0] - 11.2, 20, 20], color=RESIDUAL, lw=1.7, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    _arrow(ax, (plus_x, 20), (plus_x, row[0] - 2.9), color=RESIDUAL, lw=1.7)
    ax.text((5 + plus_x) / 2, 22.4,
            "residual: prediction = input temperature field + correction",
            ha="center", va="center", fontsize=5.2, color=RESIDUAL, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    _arrow(ax, (plus_x + 2.9, row[0]), (out_x - 9.5, row[0]), color=INK, lw=1.2)
    _block(ax, out_x, row[0], 19, 11, "prediction",
           ["temperature", "103 × 157, kelvin"], fc=OUT, title_fs=5.6, line_fs=4.9)

    if strip:
        _box(ax, 65, 9.5, 126, 13.0, "#f7f9fa", ec="#d6dde3", lw=0.7, z=1)
        facts = (
            "Expert: base 24 channels, depth 3, levels 24 / 48 / 96, group norm (8 groups), SiLU — 538,753 parameters.",
            "Training: paired sessions of one board type, complete-map L1 over board pixels in kelvin; Adam 1e-3, cosine annealing to 1e-5, batch 12, 200 epochs.",
            "Grid: 103 × 157, reflect-padded to a multiple of 8 and cropped back; the fixed 60 K unit keeps the correction dependent on absolute temperature.",
            "Output: a temperature correction only — no emissivity, no conductivity, and no power is produced anywhere in the network.",
        )
        for i, line in enumerate(facts):
            ax.text(4, 13.9 - i * 2.75, line, ha="left", va="center", fontsize=4.9,
                    color="#39424e", zorder=6)


def draw_resunet(path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    _draw_unet(ax, strip=True)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    _save_fig(fig, path, "resunet_architecture", dpi)


def _draw_moe(ax):
    """Board-routed mixture of experts: one residual U-Net per board type."""
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 78)
    ax.axis("off")

    expert_x, expert_w = 76.0, 28.0
    expert_left = expert_x - expert_w / 2
    bus_x, plus_x, out_x = 95.0, 103.0, 118.0
    rows = (65.0, 52.0, 35.0, 21.5)

    # --- input ---
    for text, cy in (
        ("temperature above ambient / 60 K", 46.5),
        ("board mask", 39.0),
        ("untrustworthy footprint mask", 31.5),
    ):
        _box(ax, 13, cy, 24, 6.4, INP)
        ax.text(13, cy, text, ha="center", va="center", fontsize=5.2, zorder=6)
    ax.text(13, 53.5, "input: 3 channels", ha="center", fontsize=5.6, fontweight="bold")
    ax.text(14, 25.0, "session record:\ndevice = board type", ha="left", va="center",
            fontsize=4.9, color=GREY)

    # --- router / gateway: a deterministic one-hot lookup, not a learned gate ---
    _box(ax, 44, 39, 22, 20, ROUTER, ec="#b98b1e")
    ax.text(44, 45.6, "Router (gateway)", ha="center", fontsize=6.0, fontweight="bold")
    ax.text(44, 42.2, "deterministic board-type selection", ha="center", fontsize=4.8,
            color="#39424e")
    for val, cx in (("1", 40.0), ("0", 44.0), ("…", 48.0)):
        _box(ax, cx, 37.8, 3.6, 3.6, "white", ec="#b98b1e", lw=0.7, z=5)
        ax.text(cx, 37.8, val, ha="center", va="center", fontsize=5.4, zorder=7)
    ax.text(44, 34.2, "index of the recorded board type", ha="center", fontsize=4.8,
            color="#39424e")
    ax.text(44, 31.6, "not inferred from the image", ha="center", fontsize=4.8, color=GREY)

    _arrow(ax, (25, 39), (32.6, 39), color=INK, lw=1.2)

    # --- the open-ended expert set, one residual U-Net per board type ---
    _arrow(ax, (55, 42), (expert_left, rows[0]), color=ACTIVE, lw=1.5)
    for cy in rows[1:]:
        _arrow(ax, (55, 42), (expert_left, cy), color=GREY, lw=1.0, ls=(0, (4, 2)))

    _block(ax, expert_x, rows[0], expert_w, 9.5, "Expert 1 (G1) — Raspberry Pi 4B",
           ["157 × 103 grid, 240 + 192 paired frames",
            "residual U-Net, 538,753 parameters"], fc=ENC, title_fs=5.4, line_fs=4.7)
    _block(ax, expert_x, rows[1], expert_w, 9.5, "Expert 2 (G2) — Orange Pi 5 Pro",
           ["164 × 103 grid, 480 paired frames",
            "residual U-Net, 538,753 parameters"], fc=ENC, title_fs=5.4, line_fs=4.7)
    ax.text(expert_x, 43.0, "⋮", ha="center", va="center", fontsize=10.5, color="#6b7480")
    _block(ax, expert_x, rows[2], expert_w, 9.5, "Expert K (GK) — board type K",
           ["its own grid and its own paired archive",
            "residual U-Net, 538,753 parameters"], fc=ENC, title_fs=5.4, line_fs=4.7)
    _block(ax, expert_x, rows[3], expert_w, 9.5, "generic expert — fallback",
           ["used when a board type has no expert",
            "the result is flagged as uncovered"], fc="#f1f3f5", title_fs=5.4, line_fs=4.7)

    # --- routing legend, placed in clear space left of the expert column ---
    ax.plot([29, 35], [59.5, 59.5], color=ACTIVE, lw=1.5)
    ax.text(36.2, 59.5, "active path (one-hot)", ha="left", va="center", fontsize=5.0,
            color=ACTIVE)
    ax.plot([29, 35], [55.5, 55.5], color=GREY, lw=1.0, ls=(0, (4, 2)))
    ax.text(36.2, 55.5, "inactive paths, not evaluated", ha="left", va="center",
            fontsize=5.0, color=GREY)

    # --- the expert group is open-ended ---
    ax.plot([expert_left, expert_x + expert_w / 2], [70.6, 70.6], color=GREY, lw=0.7)
    ax.plot([expert_left, expert_left], [69.8, 71.4], color=GREY, lw=0.7)
    ax.plot([expert_x + expert_w / 2] * 2, [69.8, 71.4], color=GREY, lw=0.7)
    ax.text(expert_x, 73.4, "K board types, one expert each (K is open-ended)",
            ha="center", fontsize=5.2, color="#39424e")

    # --- one-hot switch: exactly one expert output continues ---
    ax.plot([bus_x, bus_x], [rows[3], rows[0]], color=GREY, lw=1.1, zorder=3)
    for cy in rows:
        ax.plot([expert_x + expert_w / 2, bus_x], [cy, cy], color=GREY, lw=0.9, zorder=3)
    _arrow(ax, (bus_x, 39), (plus_x - 2.9, 39), color=INK, lw=1.3)
    ax.text(bus_x + 1.5, 46.5, "one-hot switch", ha="left", va="center", fontsize=4.9,
            color="#39424e", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"))

    _plus_node(ax, plus_x, 39)
    _arrow(ax, (plus_x + 2.9, 39), (out_x - 10, 39), color=INK, lw=1.2)
    _block(ax, out_x, 39, 20, 11, "prediction", ["temperature", "103 × 157, kelvin"],
           fc=OUT, title_fs=5.6, line_fs=4.9)

    # --- residual path from the input temperature field ---
    ax.plot([5, 5, plus_x], [28.3, 12, 12], color=RESIDUAL, lw=1.7, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    _arrow(ax, (plus_x, 12), (plus_x, 39 - 2.9), color=RESIDUAL, lw=1.7)
    ax.text(55, 14.4,
            "residual: prediction = input temperature field + selected expert's correction",
            ha="center", va="center", fontsize=5.1, color=RESIDUAL, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    ax.text(65, 5.0,
            "Only one expert runs per sample: this is a switch, not an ensemble, and the experts' outputs are never averaged or weighted.",
            ha="center", va="center", fontsize=4.9, color="#39424e")


def draw_moe(path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.15))
    _draw_moe(ax)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    _save_fig(fig, path, "moe_architecture", dpi)


def _save_fig(fig, outdir: Path, name: str, dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {outdir / name}.pdf | .svg | .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outputs/thermal/figures")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    out = Path(args.outdir)
    draw_resunet(out, args.dpi)
    draw_moe(out, args.dpi)


if __name__ == "__main__":
    main()
