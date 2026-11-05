"""Second reflectivity group: bare vs painted raw steady maps, labelled.

The two inferno panels are the same maps as the first two panels of
tmp/unet_findings/figure_steady_mismatch.py (median of the last 25 powered
frames of each session's raw camera field, shared 30-82 C scale), without
the trusted/untrusted contour and with a short label over each panel.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures/reflectivity/combined_processed_bare_vs_painted_maps.png")
SESSIONS = (
    ("case10_rpi4b_bare_full", "Bare PCB"),
    ("case17_rpi4b_painted_full", "Painted PCB"),
)


def steady_map(seq: str) -> np.ndarray:
    with np.load(f"outputs/dynamic_inverse/{seq}/latest/temperature_corrected_sequence.npz") as d:
        raw = d["raw_temperature_c"].astype(float)
        phases = d["phase_labels"]
    free_cooling = np.flatnonzero(phases == "free_cooling_candidate")
    powered_end = int(free_cooling[0]) if free_cooling.size else len(phases)
    return np.median(raw[powered_end - 25:powered_end], axis=0)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    for ax, (seq, label) in zip(axes, SESSIONS):
        m = steady_map(seq)
        ax.imshow(m, cmap="inferno", vmin=30, vmax=82)
        y, x = np.unravel_index(int(np.argmax(m)), m.shape)
        ax.plot(x, y, "o", ms=9, mfc="none", mec="c", mew=2)
        ax.annotate(f"max {m.max():.1f} C", (x, y), color="c",
                    xytext=(x + 4, y - 2), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="c", lw=0.8))
        ax.set_title(label, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.90, bottom=0.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
