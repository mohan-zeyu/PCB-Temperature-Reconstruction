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
    # The edge test is re-run on case17's own images, not copied from case10, so the
    # centre-weighted source is chosen the same way it would be on any new board.
    verdicts = th.judge_footprints(theta[min(120, n - 1)], hidden, pixel_size_mm=PX)
    sources = [v.region_id for v in verdicts if v.verdict == "heat source"]
    labels17, _ = th.block_labels(hidden)
    centred = th.reconstruct_sequence(
        theta[:n], times[:n], hidden, heating, cooling_rate_per_s=cooling,
        spread_rate_px2_per_s=SPREAD, board_mask=board, pixel_size_mm=PX,
        heat_source_ids=sources, lid_fill="source-shaped")
    centred_check = th.validate_against_truth(centred, theta[:n], region_labels=labels17)
    result = th.reconstruct_sequence(
        theta[:n], times[:n], hidden, heating, cooling_rate_per_s=cooling,
        spread_rate_px2_per_s=SPREAD, board_mask=board, pixel_size_mm=PX,
        heat_source_ids=sources, lid_fill="broad-source-plate")
    check = th.validate_against_truth(result, theta[:n], region_labels=labels17)
    # The ablation: the same fit with every source spread uniformly, which is what
    # this figure reported before 2026-08-29.
    flat = th.reconstruct_sequence(
        theta[:n], times[:n], hidden, heating, cooling_rate_per_s=cooling,
        spread_rate_px2_per_s=SPREAD, board_mask=board, pixel_size_mm=PX)
    flat_check = th.validate_against_truth(flat, theta[:n], region_labels=labels17)
    per_block = {}
    for blk in result.blocks:
        region = labels17 == blk.region_id
        if region.sum() < 12:
            continue
        per_block[blk.region_id] = dict(
            area_mm2=blk.area_mm2, shape=blk.source_shape,
            broad=float(np.abs(result.theta_k[:, region] - theta[:n][:, region]).mean()),
            centred=float(np.abs(centred.theta_k[:, region] - theta[:n][:, region]).mean()),
            uniform=float(np.abs(flat.theta_k[:, region] - theta[:n][:, region]).mean()),
        )
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
    # Where case17's own hottest patch is, located with no reference to case10 at
    # all.  This is the alignment check: if the two runs were 180 degrees apart it
    # would land on the connector bank instead of inside the SoC outline.
    late = theta[min(120, n - 1)]
    patch = late > np.percentile(late[board], 99.0)
    plab, pn = label(patch)
    biggest = max(range(1, pn + 1), key=lambda k: (plab == k).sum())
    patch = plab == biggest
    soc_id = max(sources, key=lambda i: (labels17 == i).sum()) if sources else None
    soc = labels17 == soc_id if soc_id is not None else np.zeros(hidden.shape, bool)
    rr, cc = np.nonzero(patch)
    return dict(
        oracle=oracle_check, per_block=per_block, flat_check=flat_check,
        centred_check=centred_check,
        truth=theta[:n], built=result.theta_k, mask=hidden,
        transition=result.lid_transition_mask, check=check,
        edge_bias=float((edge[:, hidden] - truth).mean()),
        edge_error=float(np.abs(edge[:, hidden] - truth).mean()),
        frame=min(120, n - 1), soc=soc, soc_id=soc_id,
        patch_rc=(float(rr.mean()), float(cc.mean())), patch_px=int(patch.sum()),
        patch_in_soc=int((patch & soc).sum()),
        patch_rot_rc=tuple(float(v.mean()) for v in np.nonzero(np.rot90(patch, 2))),
    )


def main() -> int:
    run = np.load(OUT / "reconstructed_sequence.npz")
    built, measured = run["theta_k"], run["measured_theta_k"]
    mask, times = run["unknown_mask"], run["times_s"]
    transition = run["lid_transition_mask"] if "lid_transition_mask" in run else np.zeros_like(mask)
    rebuilt_mask = run["rebuilt_mask"] if "rebuilt_mask" in run else mask | transition
    ring, ring0 = run["ring_error_k"], run["ring_error_no_source_k"]
    rates, ids = run["source_rate_by_frame"], run["region_ids"]
    labels, _ = label(mask, structure=np.ones((3, 3)))
    edge = rebuilt_mask ^ binary_dilation(rebuilt_mask, CROSS)
    depth = distance_transform_edt(rebuilt_mask) * PX
    show = 200
    c17 = case17_check()

    extent = (0, mask.shape[1] * PX, mask.shape[0] * PX, 0)
    # imshow reads extent as (left, right, bottom, top), so (0, W, H, 0) puts row 0
    # at the top.  contour reads the same tuple as (x0, x1, y0, y1) and maps row 0
    # to y0 = H, which draws every outline mirrored top-to-bottom over the image it
    # is supposed to annotate.  Pass explicit coordinates instead of an extent.
    grid_y = np.arange(mask.shape[0]) * PX
    grid_x = np.arange(mask.shape[1]) * PX

    def outline(ax, field, **kwargs):
        return ax.contour(grid_x, grid_y, np.asarray(field, dtype=np.float64), **kwargs)

    fig, axes = plt.subplots(2, 4, figsize=(21, 13))
    lo, hi = 0.0, float(np.nanmax(built[show]))

    def board_map(ax, field, title, cmap="inferno", vmin=lo, vmax=hi, mm=True):
        # The analysis-valid mask rejects deep extrapolations, but those pixels
        # still belong to the rectified PCB and were rebuilt, so display them.
        canvas = np.ma.masked_invalid(field)
        image = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("mm" if mm else "")
        plt.colorbar(image, ax=ax, fraction=0.046)
        return image

    a = axes[0, 0]
    board_map(a, measured[show], "1. What the camera reports, frame %d (t=%.0f s)\n"
              "outlined blocks read far too cold: metal lids, not cold parts" % (show, times[show]))
    outline(a, np.where(edge, 1.0, 0.0), levels=[0.5], colors="cyan", linewidths=0.8)

    a = axes[0, 1]
    board_map(a, built[show], "2. Reconstructed, same frame\n"
              "blocks filled by conduction with a fitted heat source underneath")
    outline(a, np.where(edge, 1.0, 0.0), levels=[0.5], colors="cyan", linewidths=0.8)

    a = axes[0, 2]
    diff = np.where(rebuilt_mask, built[show] - measured[show], np.nan)
    top = float(np.nanmax(np.abs(diff)))
    image = a.imshow(diff, cmap="RdBu_r", vmin=-top, vmax=top, extent=extent)
    outline(a, depth, levels=[5, 10, 15], colors="k", linewidths=0.7)
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
    a.set_xlabel("final reconstruction source rate, kelvin per second")
    a.set_title("4. Source used by the final plate reconstruction\n"
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
    m17 = c17["mask"]
    rebuilt17 = m17 | c17["transition"]
    hi17 = float(np.nanmax(c17["truth"][frame17]))
    a = axes[1, 1]
    board_map(a, c17["truth"][frame17], "6. case17, painted: MEASURED\n"
              "its own hottest patch (+) falls %d of %d px inside\n"
              "case10's SoC outline: the two runs are aligned"
              % (c17["patch_in_soc"], c17["patch_px"]), vmax=hi17)
    outline(a, np.where(rebuilt17 ^ binary_dilation(rebuilt17, CROSS), 1.0, 0.0), levels=[0.5],
            colors="cyan", linewidths=0.8)
    soc = c17["soc"]
    outline(a, np.where(soc, 1.0, 0.0), levels=[0.5], colors="lime", linewidths=2.0)
    pr, pc = c17["patch_rc"]
    a.plot(pc * PX, pr * PX, "w+", markersize=16, markeredgewidth=2.5)
    rr, rc = c17["patch_rot_rc"]
    a.plot(rc * PX, rr * PX, "x", color="#7f8fa4", markersize=13, markeredgewidth=2.0)
    a.text(rc * PX + 2, rr * PX, "where it would be\nif rotated 180", fontsize=7,
           color="#d0d0d0", va="center")

    a = axes[1, 2]
    board_map(a, c17["built"][frame17], "7. case17, case10's mask imposed: RECONSTRUCTED\n"
              "the same 39.5% hidden plus one explicit rim pixel. The SoC has\n"
              "a broad plateau generated by one smooth plate-equation solve", vmax=hi17)
    outline(a, np.where(rebuilt17 ^ binary_dilation(rebuilt17, CROSS), 1.0, 0.0), levels=[0.5],
            colors="cyan", linewidths=0.8)

    a = axes[1, 3]
    a.axis("off")
    check = c17["check"]
    a.text(0.0, 1.0, "8. Scored against case17's truth\n", fontsize=13, weight="bold",
           va="top", transform=a.transAxes)
    oracle, flat, centred = c17["oracle"], c17["flat_check"], c17["centred_check"]
    blocks = c17["per_block"]
    soc = max(blocks, key=lambda i: blocks[i]["area_mm2"] if blocks[i]["shape"] != "uniform" else -1)
    worst = max(abs(blocks[i]["broad"] - blocks[i]["centred"]) for i in blocks if i != soc)
    a.text(0.0, 0.94, (
        "Filling the blocks from their edges only, the old way:\n"
        "    mean %.2f K, bias %+.2f K  <- cold in every frame\n\n"
        "Putting the measured heat source underneath:\n"
        "    spread uniformly over the block   %.2f K  (bias %+.2f)\n"
        "    provisional centre-weighted       %.2f K  (bias %+.2f)\n"
        "    broad-source plate solve          %.2f K  (bias %+.2f)\n\n"
        "The final shape change touches one block, the SoC:\n"
        "    SoC block, uniform source         %.2f K\n"
        "    SoC block, centre-weighted        %.2f K\n"
        "    SoC block, broad-source plate     %.2f K\n"
        "    worst change on any other block   %+.2f K\n"
        "The exterior ring estimates the block's mean level. The\n"
        "unobserved bell shape made the centre hot and the broad\n"
        "area around it cold. The new fill preserves that mean,\n"
        "is generated by a broad source and the plate equation.\n\n"
        "Where the remaining %.2f K sits:\n"
        "    block at the wrong level   %.2f K\n"
        "    wrong pattern inside it    %.2f K\n"
        "Fitting each source to case17's HIDDEN truth instead of to\n"
        "the ring -- the best one number per block can do -- leaves\n"
        "level %.2f K and pattern %.2f K.\n\n"
        "The saved per-pixel uncertainty is a model range, not a\n"
        "confidence interval. It combines held-out-ring error with\n"
        "the disagreement between central- and broad-source plate solutions."
    ) % (c17["edge_error"], c17["edge_bias"],
         flat.mean_absolute_k, flat.bias_k, centred.mean_absolute_k, centred.bias_k,
         check.mean_absolute_k, check.bias_k,
         blocks[soc]["uniform"], blocks[soc]["centred"], blocks[soc]["broad"], worst,
         check.mean_absolute_k, check.level_error_k, check.shape_error_k,
         oracle.level_error_k, oracle.shape_error_k),
        fontsize=8.8, va="top", family="monospace", transform=a.transAxes)

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
            ax.imshow(np.ma.masked_invalid(fieldset[index]), cmap="inferno",
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
