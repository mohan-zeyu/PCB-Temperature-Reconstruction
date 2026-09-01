"""Stage 2 on one sequence: reconstruct every powered frame from fitted block sources.

Writes the reconstructed stack, the block source table and the hot-spot table.  The
cooling phase is deliberately not touched -- only the powered stretch, which on
case10 is frames 0-239.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tdp import thermal as th
from tdp.data import emissivity as em

PIXEL_SIZE_MM = 0.5425  # 85x56 mm outline on the 157x103 grid; not an instrument reading

# Chosen on measured evidence, not fitted: Stage 1's direct widening measurement puts
# the spread distance at 5-9 mm, refilling held-out holes in known case10 board bottoms
# out above 5.5 mm, and case17's reconstruction error moves by 6 % over a fivefold
# range of spread rate.  With the final cooling fit, 2.0 px^2/s = 9.27 mm and
# sits at the upper end where all three checks agree.
DEFAULT_SPREAD_RATE = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="case10_rpi4b_bare_full")
    parser.add_argument("--spread-rate", type=float, default=DEFAULT_SPREAD_RATE)
    parser.add_argument("--halo-px", type=int, default=1)
    parser.add_argument("--ring-px", type=int, default=8)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--uniform-sources",
        action="store_true",
        help="spread every block's source uniformly, the pre-2026-08-29 behaviour;"
        " the ablation for the centre-weighted source",
    )
    parser.add_argument(
        "--source-shaped-lids",
        action="store_true",
        help="keep the fitted centre-weighted bump inside a heat-producing lid;"
        " the default solves a smooth plate driven by a broad distributed source",
    )
    args = parser.parse_args()

    source = Path("outputs/dynamic_inverse") / args.sequence / "latest" / "temperature_corrected_sequence.npz"
    data = np.load(source)
    raw = data["raw_temperature_c"].astype(np.float64)
    ambient = data["ambient_c"].astype(np.float64)
    times = data["timestamps_s"].astype(np.float64)
    board = data["valid_mask"]
    theta = raw - ambient[:, None, None]

    reference = min(215, len(raw) - 1)
    footprints = em.detect_low_emissivity(theta[reference], halo_px=args.halo_px).mask

    weight = np.where(board, 1.0, 0.0) * (~footprints)
    board_mean = (theta * weight).reshape(len(raw), -1).sum(1) / weight.sum()
    drop = th.find_load_drop(board_mean, times)
    n_powered = drop.frame_index + 1
    cooling = th.fit_cooling_rate(board_mean[:n_powered], times[:n_powered])

    print(f"{args.sequence}: load stops at frame {drop.frame_index} (t={drop.time_s:.0f} s)")
    print(f"  cooling rate {cooling.cooling_rate_per_s:.5f} /s, response {cooling.response_time_s:.0f} s")
    print(f"  {footprints[board].mean() * 100:.1f}% of the board is untrustworthy")

    heating = th.measure_heating_rate(
        theta[:n_powered],
        times[:n_powered],
        cooling_rate_per_s=cooling.cooling_rate_per_s,
        spread_rate_px2_per_s=args.spread_rate,
        footprint_mask=footprints,
        method="rise",
    ).rate_k_per_s

    # The edge test runs first: which blocks generate heat decides how each one's
    # source is shaped inside it, and only confident "heat source" calls are used.
    # A source block left uniform costs 0.03 K; a passive block wrongly
    # concentrated costs several kelvin, so only the top verdict qualifies.
    verdicts = {
        v.region_id: v
        for v in th.judge_footprints(theta[reference], footprints, pixel_size_mm=PIXEL_SIZE_MM)
    }
    heat_source_ids = [i for i, v in verdicts.items() if v.verdict == "heat source"]
    print(f"  edge test calls {len(heat_source_ids)} block(s) a heat source: {heat_source_ids}")

    result = th.reconstruct_sequence(
        theta[:n_powered],
        times[:n_powered],
        footprints,
        heating,
        cooling_rate_per_s=cooling.cooling_rate_per_s,
        spread_rate_px2_per_s=args.spread_rate,
        ring_px=args.ring_px,
        board_mask=board,
        pixel_size_mm=PIXEL_SIZE_MM,
        heat_source_ids=() if args.uniform_sources else heat_source_ids,
        lid_fill="source-shaped" if args.source_shaped_lids else "broad-source-plate",
    )
    report = result.report
    print(f"  spread distance {report['spread_distance_mm']:.2f} mm")
    print(f"  {report['n_blocks']} blocks, {report['n_fitted']} given a fitted source rate")
    print(
        f"  held-out ring error {report['ring_error_median_k']:.3f} K"
        f"  (with every source set to zero: {report['ring_error_no_source_median_k']:.3f} K)"
    )

    spots = th.cross_check(
        th.find_hotspots(
            th.measure_heating_rate(
                theta[:n_powered], times[:n_powered],
                cooling_rate_per_s=cooling.cooling_rate_per_s,
                spread_rate_px2_per_s=args.spread_rate,
                footprint_mask=footprints, method="rise",
            ),
            footprint_mask=footprints, pixel_size_mm=PIXEL_SIZE_MM,
        ),
        th.measure_heating_rate(
            theta[:n_powered], times[:n_powered],
            cooling_rate_per_s=cooling.cooling_rate_per_s,
            spread_rate_px2_per_s=args.spread_rate,
            footprint_mask=footprints, method="onset", n_frames=4,
        ),
    )

    outdir = Path(args.outdir or f"outputs/thermal/{args.sequence.split('_')[0]}")
    outdir.mkdir(parents=True, exist_ok=True)
    fitted_rates = np.stack([b.source_rate_by_frame for b in result.blocks])
    reconstruction_rates = np.stack([
        b.source_rate_by_frame
        if b.reconstruction_source_rate_by_frame is None
        else b.reconstruction_source_rate_by_frame
        for b in result.blocks
    ])
    np.savez_compressed(
        outdir / "reconstructed_sequence.npz",
        theta_k=result.theta_k.astype(np.float32),
        measured_theta_k=theta[:n_powered].astype(np.float32),
        times_s=result.times_s,
        ambient_c=ambient[:n_powered],
        unknown_mask=result.unknown_mask,
        lid_transition_mask=result.lid_transition_mask,
        rebuilt_mask=result.rebuilt_mask,
        board_mask=board,
        heating_rate_k_per_s=heating.astype(np.float32),
        source_rate_by_frame=reconstruction_rates,
        ring_fitted_source_rate_by_frame=fitted_rates,
        region_ids=np.array([b.region_id for b in result.blocks]),
        ring_error_k=result.ring_error_k,
        ring_error_no_source_k=result.ring_error_no_source_k,
        reconstruction_uncertainty_k=result.uncertainty_k.astype(np.float32),
        spread_rate_px2_per_s=result.spread_rate_px2_per_s,
        cooling_rate_per_s=result.cooling_rate_per_s,
        pixel_size_mm=PIXEL_SIZE_MM,
    )

    with (outdir / "hotspots.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["kind", "id", "area_mm2", "row_mm", "col_mm", "deepest_mm",
             "ring_fitted_source_rate_k_per_s", "ring_fitted_source_rate_scatter_k_per_s",
             "reconstruction_source_rate_k_per_s",
             "reconstruction_source_rate_scatter_k_per_s",
             "edge_test_verdict", "edge_test_outward_fraction", "edge_test_gradient_k",
             "confidence_sd", "touches_border", "source_shape", "interior_fill"]
        )
        for block in sorted(
            result.blocks, key=lambda b: -float(b.reconstruction_source_rate_k_per_s)
        ):
            verdict = verdicts.get(block.region_id)
            writer.writerow([
                "block", block.region_id, f"{block.area_mm2:.1f}",
                f"{block.centroid_mm[0]:.1f}", f"{block.centroid_mm[1]:.1f}",
                f"{block.deepest_mm:.1f}",
                f"{block.source_rate_k_per_s:.4f}" if block.fitted else "",
                f"{block.source_rate_spread_k_per_s:.4f}" if block.fitted else "",
                f"{block.reconstruction_source_rate_k_per_s:.4f}",
                f"{block.reconstruction_source_rate_spread_k_per_s:.4f}",
                verdict.verdict if verdict else "not judged",
                f"{verdict.outward_fraction:.2f}" if verdict else "",
                f"{verdict.mean_gradient_k:.2f}" if verdict else "",
                "", block.touches_border, block.source_shape, block.interior_fill,
            ])
        for spot in spots:
            writer.writerow([
                "board hot spot", spot.rank, "", f"{spot.row_mm:.1f}", f"{spot.col_mm:.1f}",
                "0.0", "", "", f"{spot.heating_rate_k_per_s:.4f}", "", "", "", "",
                f"{spot.confidence_sd:+.1f}" if spot.confidence_sd is not None else "", False, "", "",
            ])

    (outdir / "reconstruction_report.json").write_text(
        json.dumps(
            {
                "sequence": args.sequence,
                "n_powered_frames": n_powered,
                "last_powered_frame": drop.frame_index,
                "last_powered_time_s": drop.time_s,
                "cooling_rate_per_s": cooling.cooling_rate_per_s,
                "response_time_s": cooling.response_time_s,
                "spread_rate_px2_per_s": args.spread_rate,
                "pixel_size_mm": PIXEL_SIZE_MM,
                "pixel_size_calibrated": False,
                **report,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {outdir}/reconstructed_sequence.npz ({n_powered} frames), hotspots.csv, reconstruction_report.json")

    print("\n block   area mm2   centre mm     deepest   final source K/s   scatter   edge test        interior fill")
    for block in sorted(
        result.blocks, key=lambda b: -float(b.reconstruction_source_rate_k_per_s)
    ):
        verdict = verdicts.get(block.region_id)
        print(
            "   %2d     %7.1f   (%4.1f,%4.1f)   %5.1f mm    %+.4f     %.4f    %-14s  %s"
            % (block.region_id, block.area_mm2, block.centroid_mm[0], block.centroid_mm[1],
               block.deepest_mm, block.reconstruction_source_rate_k_per_s,
               block.reconstruction_source_rate_spread_k_per_s,
               verdict.verdict if verdict else "not judged", block.interior_fill)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
