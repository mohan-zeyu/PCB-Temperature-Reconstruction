"""Stage 3a: build and validate the fixed physical pattern library for case10.

This step does not fit a forecasting model.  It consumes Stage 2's reconstructed
archive and hot-spot table only to obtain the board/source geometry and measured
rates.  The spatial basis itself is generated from the plate operator, never
from the reconstructed temperature frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tdp.thermal.forecast import ClosedFormPlateModel, load_reconstruction_archive
from tdp.thermal.patterns import (
    build_pattern_library,
    gaussian_source_shape,
    steady_source_response,
)
from tdp.thermal.reconstruct import block_labels


def _source_geometry(archive, hotspot_csv: Path, point_width_px: float):
    labels, _ = block_labels(archive.unknown_mask)
    names = ["broad_background"]
    shapes = [archive.board_mask.astype(np.float64)]
    evidence = [{"name": names[0], "kind": "broad background", "origin": "whole board"}]

    with hotspot_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] == "block" and row["edge_test_verdict"] == "heat source":
                region_id = int(row["id"])
                shape = labels == region_id
                if not shape.any():
                    raise ValueError(f"hotspots.csv names absent footprint block {region_id}")
                name = f"footprint_block_{region_id}"
                names.append(name)
                shapes.append(shape.astype(np.float64))
                evidence.append({
                    "name": name,
                    "kind": "heat-producing footprint",
                    "origin": f"edge-test verdict for block {region_id}",
                })
            elif row["kind"] == "board hot spot":
                row_px = float(row["row_mm"]) / archive.pixel_size_mm
                col_px = float(row["col_mm"]) / archive.pixel_size_mm
                name = f"trusted_board_hotspot_{row['id']}"
                names.append(name)
                shapes.append(gaussian_source_shape(
                    archive.shape_hw,
                    (row_px, col_px),
                    width_px=point_width_px,
                    board_mask=archive.board_mask,
                ))
                evidence.append({
                    "name": name,
                    "kind": "trusted-board hot spot",
                    "origin": f"hotspots.csv at ({row['row_mm']}, {row['col_mm']}) mm",
                    "centre_rc_px": [row_px, col_px],
                    "width_px": point_width_px,
                })
    return names, shapes, evidence


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="outputs/thermal/case10/reconstructed_sequence.npz"
    )
    parser.add_argument("--hotspots", default="outputs/thermal/case10/hotspots.csv")
    parser.add_argument("--outdir", default="outputs/thermal/case10")
    parser.add_argument("--smooth-patterns", type=int, default=64)
    parser.add_argument("--trial-spread-rates", default="0.5,1,2,4")
    parser.add_argument("--point-source-width-px", type=float, default=2.0)
    parser.add_argument("--rank-relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--gate-hotspot-error-k", type=float, default=0.10)
    args = parser.parse_args()

    input_path = Path(args.input)
    hotspot_path = Path(args.hotspots)
    outdir = Path(args.outdir)
    archive = load_reconstruction_archive(input_path)
    trial_rates = tuple(float(value) for value in args.trial_spread_rates.split(","))
    source_names, source_shapes, source_evidence = _source_geometry(
        archive, hotspot_path, args.point_source_width_px
    )

    library = build_pattern_library(
        archive.board_mask,
        source_shapes,
        source_names=source_names,
        n_smooth=args.smooth_patterns,
        trial_spread_rates=trial_rates,
        cooling_rate_per_time=archive.cooling_rate_per_s,
        rank_relative_tolerance=args.rank_relative_tolerance,
    )
    model = ClosedFormPlateModel.build(
        library,
        spread_rate_px2_per_time=archive.spread_rate_px2_per_s,
        cooling_rate_per_time=archive.cooling_rate_per_s,
    )

    gates = []
    for source_index, (name, shape) in enumerate(zip(source_names, source_shapes, strict=True)):
        reference = steady_source_response(
            archive.board_mask,
            shape,
            spread_rate_px2_per_time=archive.spread_rate_px2_per_s,
            cooling_rate_per_time=archive.cooling_rate_per_s,
        )
        rates = np.zeros(len(source_names), dtype=np.float64)
        rates[source_index] = 1.0
        reduced = model.steady_field(rates)
        active = np.asarray(shape) > 0.5 * float(np.max(shape))
        gates.append({
            "source": name,
            "peak_region_max_abs_error_k": float(
                np.max(np.abs(reduced[active] - reference[active]))
            ),
            "board_mae_k": float(
                np.mean(np.abs(reduced[archive.board_mask] - reference[archive.board_mask]))
            ),
        })
    worst_gate = max(item["peak_region_max_abs_error_k"] for item in gates)

    coordinates = library.project(archive.theta_k)
    projected = library.reconstruct(coordinates)
    absolute_error = np.abs(projected - archive.theta_k)
    confidence = archive.observation_weight()
    weighted_error = float(
        np.sum(absolute_error * confidence) / np.maximum(np.sum(confidence), 1e-12)
    )
    board_error = absolute_error[:, archive.board_mask]
    start_residuals = []
    for frame_index in sorted({0, archive.n_frames // 2, archive.n_frames - 1}):
        _state, residual, decay_rate = model.split_initial_field(archive.theta_k[frame_index])
        field_values = archive.theta_k[frame_index, archive.board_mask]
        residual_values = residual[archive.board_mask]
        start_residuals.append({
            "frame_index": frame_index,
            "rms_k": float(np.sqrt(np.mean(residual_values**2))),
            "rms_fraction_of_field": float(
                np.sqrt(np.mean(residual_values**2))
                / max(np.sqrt(np.mean(field_values**2)), 1e-12)
            ),
            "energy_decay_rate_per_s": decay_rate,
            "note": "retained exactly at t=0, then decayed explicitly; never absorbed into source",
        })

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "pattern_library.npz",
        basis=library.basis.astype(np.float32),
        board_mask=library.board_mask,
        measured_board_mask=archive.measured_board_mask,
        candidate_names=np.asarray(library.candidate_names, dtype="U80"),
        candidate_coordinates=library.candidate_coordinates.astype(np.float64),
        source_names=np.asarray(library.source_names, dtype="U80"),
        source_shapes=library.source_shapes.astype(np.float32),
        rank_threshold=library.rank_threshold,
        reduced_operator=model.operator.astype(np.float64),
        reduced_source_drive=model.source_drive.astype(np.float64),
        spread_rate_px2_per_s=archive.spread_rate_px2_per_s,
        cooling_rate_per_s=archive.cooling_rate_per_s,
        pixel_size_mm=archive.pixel_size_mm,
    )
    report = {
        "stage": "3a fixed physical patterns and closed-form solver; no learning",
        "input": str(input_path),
        "input_sha256_16": _digest(input_path),
        "hotspots": str(hotspot_path),
        "source_geometry": source_evidence,
        "library": library.report,
        "active_model": {
            "spread_rate_px2_per_s": archive.spread_rate_px2_per_s,
            "cooling_rate_per_s": archive.cooling_rate_per_s,
            "settling_time_to_one_percent_s": model.time_to_settle(0.01),
            "largest_operator_real_eigenvalue_per_s": float(
                np.max(np.linalg.eigvals(model.operator).real)
            ),
            "solution": "matrix exponential for any positive time gap; direct linear solve at steady state",
        },
        "fine_grid_source_response_gate": {
            "threshold_k": args.gate_hotspot_error_k,
            "worst_peak_region_max_abs_error_k": worst_gate,
            "passed": bool(worst_gate < args.gate_hotspot_error_k),
            "sources": gates,
        },
        "archive_projection_diagnostic_not_a_fit": {
            "mae_k": float(board_error.mean()),
            "p95_absolute_error_k": float(np.percentile(board_error, 95.0)),
            "uncertainty_weighted_mae_k": weighted_error,
            "note": (
                "temperature frames do not construct the basis; this only measures the fixed "
                "library's truncation error before forecasting is fitted"
            ),
        },
        "arbitrary_start_residual": start_residuals,
        "uncertainty_use": (
            "Stage 2's model range is carried into an observation weight; rebuilt pixels "
            "remain estimates and never receive measurement weight one"
        ),
    }
    (outdir / "pattern_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"patterns      : {library.n_patterns} from {len(library.candidate_names)} candidates")
    print(f"sources       : {', '.join(source_names)}")
    print(f"source gate   : {worst_gate:.3e} K (limit {args.gate_hotspot_error_k:.2f} K)")
    print(f"projection    : {board_error.mean():.3f} K MAE; weighted {weighted_error:.3f} K")
    print(f"settling time : {model.time_to_settle(0.01):.1f} s to one percent")
    print(f"written       : {outdir / 'pattern_library.npz'}")
    print(f"                {outdir / 'pattern_report.json'}")
    if worst_gate >= args.gate_hotspot_error_k:
        raise SystemExit("source-response pattern gate failed; forecasting fit must not start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
