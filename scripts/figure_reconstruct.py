"""Figure for Stage 2: what the fill does on case10, and how far off it is on case17."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from tdp import thermal as th
from tdp.data import emissivity as em
from tdp.data.emissivity import CROSS, solve_plate

PX = 0.5425
SPREAD = 2.0
OUT = Path("outputs/thermal/case10")


def case17_check():
    """Impose case10's mask on the painted board, where the truth underneath is known."""
    data = np.load("outputs/dynamic_inverse/case17_rpi4b_painted_full/latest/temperature_corrected_sequence.npz")
    raw = data["raw_temperature_c"].astype(np.float64)
    ambient = data["ambient_c"].astype(np.float64)
    times = data["timestamps_s"].astype(np.float64)
    board = data["valid_mask"]
    theta = raw - ambient[:, None, None]
    own = em.detect_low_emissivity(theta[215], halo_px=1).mask
    hidden = np.load(OUT / "reconstructed_sequence.npz")["unknown_mask"] & board & ~own

    weight = np.where(board, 1.0, 0.0) * (~hidden)
    mean = (theta * weight).reshape(len(raw), -1).sum(1) / weight.sum()
    n = th.find_load_drop(mean, times).frame_index + 1
    cooling = th.fit_cooling_rate(mean[:n], times[:n]).cooling_rate_per_s
    heating = th.measure_heating_rate(
        theta[:n], times[:n], cooling_rate_per_s=cooling,
        spread_rate_px2_per_s=SPREAD, footprint_mask=hidden, method="rise").rate_k_per_s
    result = th.reconstruct_sequence(
        theta[:n], times[:n], hidden, heating, cooling_rate_per_s=cooling,
        spread_rate_px2_per_s=SPREAD, board_mask=board, pixel_size_mm=PX)
    check = th.validate_against_truth(result, theta[:n])
    edge = np.stack([solve_plate(theta[i], hidden, result.spread_distance_px)
                     for i in range(0, n, 5)])
    truth = theta[0:n:5][:, hidden]

    # The best one number per block could possibly do: fit it to the truth this
    # stage is not allowed to see.  What survives that is a limit, not an error.
    blocks, _ = label(hidden, structure=np.ones((3, 3)))
    big = [b for b in np.unique(blocks[hidden]) if (blocks == b).sum() >= 12]
    influence = np.stack([solve_plate(np.zeros(hidden.shape), hidden, result.spread_distance_px,
                                      source=np.where(blocks == b, 1.0 / SPREAD, 0.0)) for b in big])
    design = influence[:, hidden].T
    solver = np.linalg.pinv(design)
    oracle = result.theta_k.copy()
    for i in range(n):
        base = solve_plate(theta[i], hidden, result.spread_distance_px)
        oracle[i][hidden] = base[hidden] + design @ (solver @ (theta[i][hidden] - base[hidden]))
    oracle_check = th.validate_against_truth(
        th.Reconstruction(theta_k=oracle, times_s=times[:n], unknown_mask=hidden,
                          blocks=result.blocks, spread_rate_px2_per_s=SPREAD,
                          cooling_rate_per_s=cooling, spread_distance_px=result.spread_distance_px,
                          ring_error_k=result.ring_error_k,
                          ring_error_no_source_k=result.ring_error_no_source_k),
        theta[:n])
    return dict(
        oracle=oracle_check,
        truth=theta[:n], built=result.theta_k, mask=hidden, board=board, check=check,
        edge_bias=float((edge[:, hidden] - truth).mean()),
        edge_error=float(np.abs(edge[:, hidden] - truth).mean()),
        frame=min(120, n - 1),
    )


def main() -> int:
    run = np.load(OUT / "reconstructed_sequence.npz")
    built, measured = run["theta_k"], run["measured_theta_k"]
    mask, board, times = run["unknown_mask"], run["board_mask"], run["times_s"]
    ring, ring0 = run["ring_error_k"], run["ring_error_no_source_k"]
    rates, ids = run["source_rate_by_frame"], run["region_ids"]
    labels, _ = label(mask, structure=np.ones((3, 3)))
    edge = mask ^ binary_dilation(mask, CROSS)
    depth = distance_transform_edt(mask) * PX
    show = 200
    c17 = case17_check()

    extent = (0, mask.shape[1] * PX, mask.shape[0] * PX, 0)
    fig, axes = plt.subplots(2, 4, figsize=(21, 13))
    lo, hi = 0.0, float(built[show][board].max())

    def board_map(ax, field, title, cmap="inferno", vmin=lo, vmax=hi, mm=True):
        canvas = np.where(board, field, np.nan)
        image = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("mm" if mm else "")
        plt.colorbar(image, ax=ax, fraction=0.046)
        return image

    a = axes[0, 0]
    board_map(a, measured[show], "1. What the camera reports, frame %d (t=%.0f s)\n"
              "outlined blocks read far too cold: metal lids, not cold parts" % (show, times[show]))
    a.contour(np.where(edge, 1.0, 0.0), levels=[0.5], colors="cyan", linewidths=0.8, extent=extent)

    a = axes[0, 1]
    board_map(a, built[show], "2. Reconstructed, same frame\n"
              "blocks filled by conduction with a fitted heat source underneath")
    a.contour(np.where(edge, 1.0, 0.0), levels=[0.5], colors="cyan", linewidths=0.8, extent=extent)

    a = axes[0, 2]
    diff = np.where(mask, built[show] - measured[show], np.nan)
    top = float(np.nanmax(np.abs(diff)))
    image = a.imshow(np.where(board, diff, np.nan), cmap="RdBu_r", vmin=-top, vmax=top, extent=extent)
    a.contour(depth, levels=[5, 10, 15], colors="k", linewidths=0.7, extent=extent)
    a.set_title("3. How much was added, kelvin\n"
                "contours: 5, 10, 15 mm from the nearest trustworthy pixel", fontsize=11)
    a.set_xlabel("mm")
    plt.colorbar(image, ax=a, fraction=0.046)

    a = axes[0, 3]
    order = np.argsort(rates.mean(axis=1))
    med = np.median(rates, axis=1)[order]
    spread_bar = np.abs(np.percentile(rates, [16, 84], axis=1)[:, order] - med)
    area = np.array([(labels == i).sum() for i in ids])[order] * PX ** 2
    colours = ["#c1272d" if v > 0.15 else "#e8a33d" if v > 0.02 else "#7f8fa4" for v in med]
    a.barh(range(len(med)), med, xerr=spread_bar, color=colours, height=0.7,
           error_kw=dict(lw=1, capsize=2))
    a.set_yticks(range(len(med)))
    a.set_yticklabels(["block %d  (%.0f mm2)" % (i, s) for i, s in zip(ids[order], area)], fontsize=8)
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("fitted source rate, kelvin per second")
    a.set_title("4. What each block puts into the board\n"
                "bars: middle 68% of the 240 frames.  No watts: every rate here\n"
                "was divided by an unknown, varying heat capacity", fontsize=11)

    a = axes[1, 0]
    a.plot(times, ring0, color="#7f8fa4", lw=1.4, label="every block source set to zero")
    a.plot(times, ring, color="#c1272d", lw=1.6, label="with the fitted block sources")
    a.set_xlabel("seconds since the run started")
    a.set_ylabel("error on the held-out ring, kelvin")
    a.set_title("5. The ring of real board that was hidden from the solve\n"
                "median %.2f K, against %.2f K with no source" % (np.median(ring), np.median(ring0)),
                fontsize=11)
    a.legend(fontsize=8, loc="upper right")
    a.grid(alpha=0.3)
    a.set_ylim(bottom=0)

    frame17 = c17["frame"]
    m17, b17 = c17["mask"], c17["board"]
    hi17 = float(c17["truth"][frame17][b17].max())
    a = axes[1, 1]
    board_map(a, c17["truth"][frame17], "6. case17, the painted board: MEASURED\n"
              "here the parts case10 must guess at are trustworthy", vmax=hi17)
    a.contour(np.where(m17 ^ binary_dilation(m17, CROSS), 1.0, 0.0), levels=[0.5],
              colors="cyan", linewidths=0.8, extent=extent)

    a = axes[1, 2]
    board_map(a, c17["built"][frame17], "7. case17 with case10's mask imposed, RECONSTRUCTED\n"
              "the same 39.5% of the board hidden and rebuilt", vmax=hi17)
    a.contour(np.where(m17 ^ binary_dilation(m17, CROSS), 1.0, 0.0), levels=[0.5],
              colors="cyan", linewidths=0.8, extent=extent)

    a = axes[1, 3]
    a.axis("off")
    check = c17["check"]
    a.text(0.0, 1.0, "8. Scored against case17's truth\n", fontsize=13, weight="bold",
           va="top", transform=a.transAxes)
    oracle = c17["oracle"]
    a.text(0.0, 0.93, (
        "Filling the blocks from their edges only, the old way:\n"
        "     mean error   %.2f K\n"
        "     bias         %+.2f K   <- cold, every frame\n\n"
        "Putting the measured heat source underneath:\n"
        "     mean error   %.2f K\n"
        "     bias         %+.2f K   <- the cold bias is gone\n"
        "     worst 5 in 100                %.2f K\n\n"
        "Where that remaining %.2f K sits:\n"
        "     block sits at the wrong level    %.2f K\n"
        "     wrong pattern inside the block   %.2f K\n\n"
        "Now fit each block's source to case17's HIDDEN TRUTH\n"
        "instead of to the ring -- the best one number per\n"
        "block can possibly do:\n"
        "     wrong level    %.2f K   <- nearly all recovered\n"
        "     wrong pattern  %.2f K   <- barely moves\n\n"
        "So the two halves have different causes:\n"
        " - the LEVEL is limited by what the board around a\n"
        "   block can say about what is under it.\n"
        " - the PATTERN cannot be fixed by any per-block\n"
        "   source at all.  A uniform source makes a smooth\n"
        "   dome; the real blocks span 13-26 K inside.\n\n"
        "Reading the pattern under a lid is the only way past\n"
        "this.  Stage 2 deliberately does not -- that reading\n"
        "is wrong by an unknown emissivity factor.  Worth a\n"
        "decision rather than a silent choice."
    ) % (c17["edge_error"], c17["edge_bias"], check.mean_absolute_k, check.bias_k,
         check.p95_absolute_k, check.mean_absolute_k, check.level_error_k, check.shape_error_k,
         oracle.level_error_k, oracle.shape_error_k),
        fontsize=10, va="top", family="monospace", transform=a.transAxes)

    fig.suptitle(
        "case10 Raspberry Pi 4B, full load -- every one of the 240 powered frames rebuilt "
        "from its heat sources, and checked on a board where the answer is known",
        fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "phase2_reconstruction.png", dpi=110)
    print("wrote", OUT / "phase2_reconstruction.png")

    picks = [0, 12, 30, 60, 100, 150, 200, 239]
    fig2, axes2 = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 8.2))
    for column, index in enumerate(picks):
        for row, (fieldset, name) in enumerate(((measured, "camera"), (built, "reconstructed"))):
            ax = axes2[row, column]
            ax.imshow(np.where(board, fieldset[index], np.nan), cmap="inferno",
                      vmin=0, vmax=hi, extent=extent)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title("frame %d\nt = %.0f s" % (index, times[index]), fontsize=10)
            if column == 0:
                ax.set_ylabel(name, fontsize=12)
    fig2.suptitle("Every powered frame is rebuilt, not just the settled one "
                  "(8 of 240 shown, one colour scale throughout)", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2.savefig(OUT / "phase2_frames.png", dpi=105)
    print("wrote", OUT / "phase2_frames.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
