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
from scipy.ndimage import distance_transform_edt, label

from tdp import thermal as th
from tdp.data import emissivity as em
from tdp.dynamic_inverse.config import load_measurements

PIXEL_SIZE_MM = 0.5425  # 85x56 mm outline on the 157x103 grid; not an instrument reading

# Chosen on measured evidence, not fitted: Stage 1's direct widening measurement puts
# the spread distance at 5-9 mm, refilling held-out holes in known case10 board bottoms
# out above 5.5 mm, and case17's reconstruction error moves by 6 % over a fivefold
# range of spread rate.  With the final cooling fit, 2.0 px^2/s = 9.27 mm and
# sits at the upper end where all three checks agree.
DEFAULT_SPREAD_RATE = 2.0
MEASUREMENTS_PATH = Path("configs/measurements_v3.yaml")
DEFAULT_PAIRED_REFERENCE = "case17_rpi4b_painted_full"
DEFAULT_STEADY_WINDOW_S = 180.0
PAIRED_ACCEPTANCE_LIMIT_K = 1.5


def _sequence_paths(sequence: str) -> tuple[Path, Path]:
    directory = Path("outputs/dynamic_inverse") / sequence / "latest"
    return directory / "temperature_corrected_sequence.npz", directory / "run_metadata.json"


def _log_path(measurements, sequence: str, kind: str) -> Path:
    try:
        relative = measurements.sequences[sequence].logs[kind]
    except KeyError as exc:
        raise ValueError(f"{sequence} has no {kind!r} log in {MEASUREMENTS_PATH}") from exc
    path = Path(measurements.dataset_root) / relative
    if not path.exists():
        raise FileNotFoundError(f"configured {kind} log does not exist: {path}")
    return path


def _largest_patch(mask: np.ndarray) -> np.ndarray:
    labels, count = label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="case10_rpi4b_bare_full")
    parser.add_argument("--spread-rate", type=float, default=DEFAULT_SPREAD_RATE)
    parser.add_argument("--halo-px", type=int, default=1)
    parser.add_argument("--ring-px", type=int, default=8)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--no-paired-calibration",
        action="store_true",
        help="keep the image-only reconstruction as an explicit ablation; for case10 "
        "the default uses case17 plus the recorded SoC logs",
    )
    parser.add_argument(
        "--paired-reference-sequence", default=DEFAULT_PAIRED_REFERENCE,
        help="painted high-emissivity recording used to audit the mask and calibrate "
        "the SoC die-to-surface difference",
    )
    parser.add_argument(
        "--paired-attenuation-threshold-k", type=float, default=5.0,
        help="minimum matched steady temperature suppression added to the unreliable mask",
    )
    parser.add_argument(
        "--steady-window-s", type=float, default=DEFAULT_STEADY_WINDOW_S,
    )
    parser.add_argument(
        "--anchor-uncertainty-floor-k", type=float, default=1.5,
        help="minimum uncertainty for transferring one painted session to another run",
    )
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

    source, source_metadata = _sequence_paths(args.sequence)
    with np.load(source) as data:
        raw = data["raw_temperature_c"].astype(np.float64)
        ambient = data["ambient_c"].astype(np.float64)
        times = data["timestamps_s"].astype(np.float64)
        board = data["valid_mask"].astype(bool)
    theta = raw - ambient[:, None, None]

    reference = min(215, len(raw) - 1)
    base_footprints = em.detect_low_emissivity(
        theta[reference], halo_px=args.halo_px
    ).mask
    footprints = base_footprints.copy()
    paired_mask = np.zeros_like(footprints)
    paired_attenuation = np.zeros_like(theta[0])
    calibration_context = None
    calibration_enabled = (
        not args.no_paired_calibration
        and args.sequence == "case10_rpi4b_bare_full"
    )
    if calibration_enabled:
        measurements = load_measurements(MEASUREMENTS_PATH)
        if args.paired_reference_sequence not in measurements.sequences:
            raise ValueError(
                f"unknown paired reference sequence {args.paired_reference_sequence!r}"
            )
        reference_source, reference_metadata = _sequence_paths(
            args.paired_reference_sequence
        )
        with np.load(reference_source) as reference_data:
            reference_raw = reference_data["raw_temperature_c"].astype(np.float64)
            reference_ambient = reference_data["ambient_c"].astype(np.float64)
            reference_times = reference_data["timestamps_s"].astype(np.float64)
            reference_board = reference_data["valid_mask"].astype(bool)
        reference_theta = reference_raw - reference_ambient[:, None, None]
        if reference_theta.shape[1:] != theta.shape[1:]:
            raise ValueError(
                "paired bare and painted sequences do not share the canonical board grid"
            )

        target_start = th.sequence_start_epoch(source_metadata)
        reference_start = th.sequence_start_epoch(reference_metadata)
        target_frame_epoch = target_start + times
        reference_frame_epoch = reference_start + reference_times
        target_load_log = th.read_timestamped_values(
            _log_path(measurements, args.sequence, "cpu_load"),
            value_columns=("total_percent", "cpu_usage_percent"),
        )
        reference_load_log = th.read_timestamped_values(
            _log_path(measurements, args.paired_reference_sequence, "cpu_load"),
            value_columns=("total_percent", "cpu_usage_percent"),
        )
        target_load = th.longest_high_load_interval(target_load_log)
        reference_load = th.longest_high_load_interval(reference_load_log)
        target_steady = th.steady_frame_mask(
            target_frame_epoch, target_load, window_s=args.steady_window_s
        )
        reference_steady = th.steady_frame_mask(
            reference_frame_epoch, reference_load, window_s=args.steady_window_s
        )
        paired = th.paired_attenuation_mask(
            theta,
            target_steady,
            reference_theta,
            reference_steady,
            board_mask=(board | base_footprints) & reference_board,
            threshold_k=args.paired_attenuation_threshold_k,
        )
        paired_mask = paired.mask
        paired_attenuation = paired.attenuation_k
        footprints |= paired_mask
        calibration_context = {
            "measurements": measurements,
            "reference_raw": reference_raw,
            "reference_theta": reference_theta,
            "reference_ambient": reference_ambient,
            "reference_times": reference_times,
            "reference_board": reference_board,
            "reference_frame_epoch": reference_frame_epoch,
            "reference_load": reference_load,
            "reference_steady": reference_steady,
            "target_frame_epoch": target_frame_epoch,
            "target_load": target_load,
            "target_steady": target_steady,
            "paired_report": paired.report,
        }

    weight = np.where(board, 1.0, 0.0) * (~footprints)
    board_mean = (theta * weight).reshape(len(raw), -1).sum(1) / weight.sum()
    drop = th.find_load_drop(board_mean, times)
    image_n_powered = drop.frame_index + 1
    if calibration_context is None:
        n_powered = image_n_powered
    else:
        frame_epoch = calibration_context["target_frame_epoch"]
        stop_epoch = calibration_context["target_load"].stop_epoch_s
        powered_indices = np.flatnonzero(frame_epoch <= stop_epoch)
        if not powered_indices.size:
            raise ValueError("the logged powered interval contains no infrared frame")
        n_powered = int(powered_indices[-1] + 1)
        if abs(n_powered - image_n_powered) > 1:
            raise ValueError(
                "image and load log disagree on the powered interval: "
                f"{image_n_powered} versus {n_powered} frames"
            )
    cooling = th.fit_cooling_rate(board_mean[:n_powered], times[:n_powered])

    print(f"{args.sequence}: load stops at frame {n_powered - 1} (t={times[n_powered - 1]:.0f} s)")
    if calibration_context is not None:
        print(
            "  paired calibration: image/load-log powered-frame counts "
            f"{image_n_powered}/{n_powered}"
        )
        print(
            "  paired reliability audit adds "
            f"{int((paired_mask & ~base_footprints).sum())} pixels to the footprint mask"
        )
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

    anchors: list[th.InteriorTemperatureAnchor] = []
    anchor_surface_c = np.empty(0, dtype=np.float64)
    anchor_uncertainty_k = np.empty(0, dtype=np.float64)
    soc_region_id = -1
    soc_region = np.zeros_like(footprints)
    soc_core = np.zeros_like(footprints)
    calibration_report: dict = {
        "enabled": False,
        "mode": (
            "paired attenuation audit without a SoC temperature anchor"
            if calibration_context is not None
            else "image-only reconstruction"
        ),
    }
    if calibration_context is not None and not args.uniform_sources:
        if args.source_shaped_lids:
            raise ValueError(
                "paired SoC calibration requires the broad-source plate lid; "
                "use --no-paired-calibration for the source-shaped ablation"
            )
        if not heat_source_ids:
            raise ValueError("paired calibration found no heat-producing footprint to anchor")
        labels, _ = th.block_labels(footprints)
        soc_region_id = max(
            heat_source_ids, key=lambda region_id: int((labels == region_id).sum())
        )
        soc_region = labels == soc_region_id
        depth = distance_transform_edt(soc_region)
        soc_core = soc_region & (depth >= max(0.75 * float(depth.max()), 1.0))
        if not soc_core.any():
            raise ValueError("SoC footprint has no central pixels for the temperature anchor")

        context = calibration_context
        measurements = context["measurements"]
        target_die_log = th.read_timestamped_values(
            _log_path(measurements, args.sequence, "soc_temp"),
            value_columns=("temp_c", "soc_temp_c"),
        )
        reference_die_log = th.read_timestamped_values(
            _log_path(measurements, args.paired_reference_sequence, "soc_temp"),
            value_columns=("temp_c", "soc_temp_c"),
        )
        target_die = th.align_to_frames(
            target_die_log, context["target_frame_epoch"][:n_powered]
        )
        reference_powered = (
            context["reference_frame_epoch"] <= context["reference_load"].stop_epoch_s
        )
        reference_anchor_epoch = context["reference_frame_epoch"][reference_powered]
        reference_die = th.align_to_frames(
            reference_die_log, reference_anchor_epoch
        )
        reference_surface = context["reference_raw"][reference_powered][:, soc_core].mean(axis=1)
        surface_anchor = th.transfer_surface_anchor(
            context["target_frame_epoch"][:n_powered],
            target_die,
            context["target_load"],
            reference_anchor_epoch,
            reference_die,
            reference_surface,
            context["reference_load"],
            steady_window_s=args.steady_window_s,
            uncertainty_floor_k=args.anchor_uncertainty_floor_k,
        )
        anchor_surface_c = surface_anchor.surface_temperature_c
        anchor_uncertainty_k = surface_anchor.uncertainty_k
        anchors.append(
            th.InteriorTemperatureAnchor(
                region_id=soc_region_id,
                temperature_above_ambient_k_by_frame=(
                    anchor_surface_c - ambient[:n_powered]
                ),
                uncertainty_k=anchor_uncertainty_k,
                label=(
                    f"{args.sequence} die log corrected by "
                    f"{args.paired_reference_sequence} painted surface"
                ),
            )
        )

        reference_steady_map = np.median(
            context["reference_raw"][context["reference_steady"]], axis=0
        )
        reference_board = context["reference_board"]
        hottest = _largest_patch(
            (reference_steady_map > np.percentile(reference_steady_map[reference_board], 99.0))
            & reference_board
        )
        hottest_overlap = float((hottest & soc_region).sum() / max(int(hottest.sum()), 1))
        if hottest_overlap < 0.8:
            raise ValueError(
                "paired spatial correspondence failed: less than 80% of the painted "
                "steady hot patch lies in the case10 SoC footprint"
            )
        calibration_report = {
            "enabled": True,
            "mode": "paired painted-board surface calibration plus target die log",
            "target_sequence": args.sequence,
            "reference_sequence": args.paired_reference_sequence,
            "soc_region_id": soc_region_id,
            "soc_area_px": int(soc_region.sum()),
            "soc_core_px": int(soc_core.sum()),
            "paired_attenuation": {
                **context["paired_report"],
                "new_pixels_beyond_image_only_mask": int(
                    (paired_mask & ~base_footprints).sum()
                ),
                "image_only_mask_px": int(base_footprints.sum()),
                "combined_mask_px": int(footprints.sum()),
            },
            "temporal_correspondence": {
                "target_first_ir_seconds_after_90_percent_load_start": float(
                    context["target_frame_epoch"][0]
                    - context["target_load"].start_epoch_s
                ),
                "reference_first_ir_seconds_after_90_percent_load_start": float(
                    context["reference_frame_epoch"][0]
                    - context["reference_load"].start_epoch_s
                ),
                "target_load_duration_s": float(
                    context["target_load"].stop_epoch_s
                    - context["target_load"].start_epoch_s
                ),
                "reference_load_duration_s": float(
                    context["reference_load"].stop_epoch_s
                    - context["reference_load"].start_epoch_s
                ),
            },
            "spatial_correspondence": {
                "painted_hottest_patch_px": int(hottest.sum()),
                "painted_hottest_patch_fraction_inside_soc": hottest_overlap,
            },
            "surface_anchor": surface_anchor.report,
            "reference_surface_steady_core_mean_c": float(
                np.mean(reference_surface[context["reference_steady"][reference_powered]])
            ),
            "reference_surface_steady_peak_mean_c": float(
                np.mean(
                    np.max(
                        context["reference_raw"][context["reference_steady"]][:, soc_region],
                        axis=1,
                    )
                )
            ),
            "reference_die_steady_mean_c": float(
                np.mean(reference_die[context["reference_steady"][reference_powered]])
            ),
        }
        print(
            f"  SoC block {soc_region_id} anchored to painted-surface calibration; "
            f"target steady core {surface_anchor.report['target_surface_steady_mean_c']:.2f} C"
        )

    reconstruction_kwargs = dict(
        cooling_rate_per_s=cooling.cooling_rate_per_s,
        spread_rate_px2_per_s=args.spread_rate,
        ring_px=args.ring_px,
        board_mask=board,
        pixel_size_mm=PIXEL_SIZE_MM,
        heat_source_ids=() if args.uniform_sources else heat_source_ids,
        lid_fill="source-shaped" if args.source_shaped_lids else "broad-source-plate",
    )
    unanchored_result = th.reconstruct_sequence(
        theta[:n_powered],
        times[:n_powered],
        footprints,
        heating,
        **reconstruction_kwargs,
    )
    result = (
        unanchored_result
        if not anchors
        else th.reconstruct_sequence(
            theta[:n_powered],
            times[:n_powered],
            footprints,
            heating,
            **reconstruction_kwargs,
            interior_temperature_anchors=anchors,
        )
    )
    report = {**result.report}
    if anchors:
        steady = np.asarray(calibration_context["target_steady"][:n_powered], dtype=bool)
        reconstructed_c = result.theta_k + ambient[:n_powered, None, None]
        unanchored_c = unanchored_result.theta_k + ambient[:n_powered, None, None]
        reconstructed_peak = np.max(reconstructed_c[:, soc_region], axis=1)
        unanchored_peak = np.max(unanchored_c[:, soc_region], axis=1)
        expected_peak = (
            calibration_report["surface_anchor"]["target_surface_steady_mean_c"]
            + calibration_report["reference_surface_steady_peak_mean_c"]
            - calibration_report["reference_surface_steady_core_mean_c"]
        )
        actual_peak = float(np.mean(reconstructed_peak[steady]))
        peak_error = float(actual_peak - expected_peak)
        acceptance_passed = abs(peak_error) <= PAIRED_ACCEPTANCE_LIMIT_K
        calibration_report["result"] = {
            "unanchored_soc_steady_peak_mean_c": float(
                np.mean(unanchored_peak[steady])
            ),
            "anchored_soc_steady_core_mean_c": float(
                np.mean(reconstructed_c[:, soc_core][steady])
            ),
            "anchored_soc_steady_peak_mean_c": actual_peak,
            "anchored_soc_last_powered_peak_c": float(reconstructed_peak[-1]),
            "anchored_whole_powered_peak_c": float(np.max(reconstructed_c[:, board])),
            "case17_adjusted_expected_peak_c": float(expected_peak),
            "steady_peak_error_against_adjusted_reference_k": peak_error,
            "acceptance_limit_k": PAIRED_ACCEPTANCE_LIMIT_K,
            "acceptance_passed": acceptance_passed,
            "note": "surface temperature, not the hotter SoC die reading",
        }
        if not acceptance_passed:
            raise ValueError(
                "paired SoC temperature acceptance failed: steady surface peak "
                f"{actual_peak:.2f} C versus adjusted reference {expected_peak:.2f} C "
                f"(limit {PAIRED_ACCEPTANCE_LIMIT_K:.2f} K)"
            )
        print(
            "  SoC steady surface peak "
            f"{actual_peak:.2f} C (same-mask unanchored "
            f"{np.mean(unanchored_peak[steady]):.2f} C; "
            f"paired expectation {expected_peak:.2f} C)"
        )
    report["calibration"] = calibration_report
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
        image_only_unknown_mask=base_footprints,
        paired_attenuation_mask=paired_mask,
        paired_attenuation_k=paired_attenuation.astype(np.float32),
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
        soc_region_id=np.int64(soc_region_id),
        soc_surface_anchor_c=anchor_surface_c,
        soc_surface_anchor_uncertainty_k=anchor_uncertainty_k,
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
                "last_powered_frame": n_powered - 1,
                "last_powered_time_s": float(times[n_powered - 1]),
                "image_detected_last_powered_frame": drop.frame_index,
                "image_detected_last_powered_time_s": drop.time_s,
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
