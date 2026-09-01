"""Stage 3b: freeze, test the source proposal, and fit the production forecast.

The fixed pattern library and time split are never changed.  The positive smooth
source-history proposal is fitted and retained as an ablation.  Production uses
closed-form decay toward a training-only settled field; validation selects only
bounded rate multipliers and the held-out test is opened once for final scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tdp.thermal import load_pattern_library, load_reconstruction_archive
from tdp.thermal.train_forecast import (
    fit_equilibrium_forecast,
    fit_source_histories,
    make_frozen_split,
)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction", default="outputs/thermal/case10/reconstructed_sequence.npz"
    )
    parser.add_argument("--patterns", default="outputs/thermal/case10/pattern_library.npz")
    parser.add_argument("--hotspots", default="outputs/thermal/case10/hotspots.csv")
    parser.add_argument("--reconstruction-report", default="outputs/thermal/case10/reconstruction_report.json")
    parser.add_argument("--pattern-report", default="outputs/thermal/case10/pattern_report.json")
    parser.add_argument("--outdir", default="outputs/thermal/case10")
    parser.add_argument("--knots", type=int, default=12)
    parser.add_argument("--smooth-strengths", default="0.01,0.03,0.1,0.3")
    parser.add_argument("--spread-multipliers", default="0.8,1,1.2")
    parser.add_argument("--cooling-multipliers", default="0.9,1,1.1")
    parser.add_argument("--residual-decay-multipliers", default="0.5,1,2")
    parser.add_argument("--source-rate-upper", type=float, default=None)
    parser.add_argument("--measurement-resolution-k", type=float, default=0.1)
    args = parser.parse_args()

    reconstruction_path = Path(args.reconstruction)
    pattern_path = Path(args.patterns)
    hotspot_path = Path(args.hotspots)
    reconstruction_report_path = Path(args.reconstruction_report)
    pattern_report_path = Path(args.pattern_report)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    archive = load_reconstruction_archive(reconstruction_path)
    library = load_pattern_library(pattern_path)
    split = make_frozen_split(archive.n_frames)
    coordinates = library.project(archive.theta_k)

    config = {
        "knots": int(args.knots),
        "smooth_strengths": [float(value) for value in args.smooth_strengths.split(",")],
        "spread_multipliers": [float(value) for value in args.spread_multipliers.split(",")],
        "cooling_multipliers": [float(value) for value in args.cooling_multipliers.split(",")],
        "residual_decay_multipliers": [
            float(value) for value in args.residual_decay_multipliers.split(",")
        ],
        "source_rate_upper_k_per_s": args.source_rate_upper,
        "measurement_resolution_k": float(args.measurement_resolution_k),
        "split": split.report,
    }

    # One file is sufficient to repeat the fit without the 3.1 GB image tree,
    # without rebuilding Stage 2, and without silently drawing a new split.
    export_path = outdir / "stage3_training_export.npz"
    np.savez_compressed(
        export_path,
        theta_k=archive.theta_k.astype(np.float32),
        measured_theta_k=archive.measured_theta_k.astype(np.float32),
        reconstruction_uncertainty_k=archive.uncertainty_k.astype(np.float32),
        times_s=archive.times_s,
        ambient_c=archive.ambient_c,
        board_mask=archive.board_mask,
        measured_board_mask=archive.measured_board_mask,
        rebuilt_mask=archive.rebuilt_mask,
        unknown_mask=archive.unknown_mask,
        heating_rate_k_per_s=archive.heating_rate_k_per_s.astype(np.float32),
        stage2_source_rate_by_frame=archive.source_rate_by_frame,
        stage2_region_ids=archive.region_ids,
        measured_spread_rate_px2_per_s=archive.spread_rate_px2_per_s,
        measured_cooling_rate_per_s=archive.cooling_rate_per_s,
        pixel_size_mm=archive.pixel_size_mm,
        pattern_basis=library.basis.astype(np.float32),
        pattern_coordinates=coordinates.astype(np.float32),
        candidate_names=np.asarray(library.candidate_names, dtype="U80"),
        candidate_coordinates=library.candidate_coordinates,
        source_names=np.asarray(library.source_names, dtype="U80"),
        source_shapes=library.source_shapes.astype(np.float32),
        split_roles=split.roles,
        split_json=np.asarray(json.dumps(split.report, default=_json_default)),
        fit_config_json=np.asarray(json.dumps(config, default=_json_default)),
        hotspots_csv=np.asarray(hotspot_path.read_text(encoding="utf-8")),
        reconstruction_report_json=np.asarray(
            reconstruction_report_path.read_text(encoding="utf-8")
        ),
        pattern_report_json=np.asarray(pattern_report_path.read_text(encoding="utf-8")),
        reconstruction_sha256_16=np.asarray(_digest(reconstruction_path)),
        pattern_library_sha256_16=np.asarray(_digest(pattern_path)),
    )

    fit = fit_source_histories(
        archive,
        library,
        split=split,
        n_knots=args.knots,
        smooth_strengths=config["smooth_strengths"],
        spread_multipliers=config["spread_multipliers"],
        cooling_multipliers=config["cooling_multipliers"],
        source_rate_upper_k_per_s=args.source_rate_upper,
        measurement_resolution_k=args.measurement_resolution_k,
    )
    equilibrium = fit_equilibrium_forecast(
        archive,
        library,
        split=split,
        spread_multipliers=config["spread_multipliers"],
        cooling_multipliers=config["cooling_multipliers"],
        residual_decay_multipliers=config["residual_decay_multipliers"],
        measurement_resolution_k=args.measurement_resolution_k,
    )

    source_predictions = fit.metrics["_test_predictions"]
    source_report = {
        key: value for key, value in fit.metrics.items() if key != "_test_predictions"
    }
    equilibrium_private = {
        key: value for key, value in equilibrium.metrics.items() if key.startswith("_")
    }
    equilibrium_report = {
        key: value for key, value in equilibrium.metrics.items() if not key.startswith("_")
    }
    report = {
        "stage": "3b frozen export + equilibrium-centred production forecast",
        "production_model": "closed-form decay of deviations toward a training-only settled field",
        "source_names": list(fit.source_names),
        "equilibrium_forecast": equilibrium_report,
        "positive_smooth_source_ablation": source_report,
        "artifacts": {
            "training_export": str(export_path),
            "source_fit": str(outdir / "source_history_fit.npz"),
            "equilibrium_fit": str(outdir / "equilibrium_forecast_fit.npz"),
            "heldout_predictions": str(outdir / "heldout_forward_predictions.npz"),
            "whole_phase_at_measured_times": str(
                outdir / "whole_phase_forward_at_measured_times.npz"
            ),
        },
        "limits": (
            "test rebuilt-area error is agreement with Stage 2 estimates; only directly measured "
            "pixels carry measurement weight one"
        ),
    }
    (outdir / "forecast_fit_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    np.savez_compressed(
        outdir / "source_history_fit.npz",
        source_names=np.asarray(fit.source_names, dtype="U80"),
        knot_times_s=fit.knot_times_s,
        knot_source_rate_k_per_s=fit.knot_source_rate_k_per_s,
        frame_source_rate_k_per_s=fit.frame_source_rate_k_per_s,
        interval_source_rate_k_per_s=fit.interval_source_rate_k_per_s,
        selected_spread_rate_px2_per_s=fit.model.spread_rate_px2_per_time,
        selected_cooling_rate_per_s=fit.model.cooling_rate_per_time,
        reduced_operator=fit.model.operator,
        reduced_source_drive=fit.model.source_drive,
        static_texture_pattern_k=fit.correction.pattern_k.astype(np.float32),
        static_texture_scale_intercept=fit.correction.scale_intercept,
        static_texture_scale_per_mean_temperature=fit.correction.scale_per_mean_temperature,
        static_texture_scale_bounds=np.array([
            fit.correction.minimum_scale, fit.correction.maximum_scale
        ]),
        split_roles=fit.split.roles,
        split_json=np.asarray(json.dumps(fit.split.report, default=_json_default)),
    )
    np.savez_compressed(
        outdir / "equilibrium_forecast_fit.npz",
        settled_theta_k=equilibrium.settled_theta_k.astype(np.float32),
        settled_coordinates=equilibrium.settled_coordinates,
        settled_model_range_k=equilibrium.settled_model_range_k.astype(np.float32),
        selected_spread_rate_px2_per_s=equilibrium.model.spread_rate_px2_per_time,
        selected_cooling_rate_per_s=equilibrium.model.cooling_rate_per_time,
        residual_decay_multiplier=equilibrium.residual_decay_multiplier,
        reduced_operator=equilibrium.model.operator,
        split_roles=equilibrium.split.roles,
        split_json=np.asarray(json.dumps(equilibrium.split.report, default=_json_default)),
    )
    prediction_payload = {}
    for model_name, blocks in source_predictions.items():
        for start, values in blocks.items():
            prediction_payload[f"source_ablation_{model_name}_start_{start}"] = values
    for prefix, private_key in (
        ("equilibrium_validation", "_validation_predictions"),
        ("equilibrium_test", "_test_predictions"),
        ("persistence_test", "_persistence_predictions"),
    ):
        for start, values in equilibrium_private[private_key].items():
            prediction_payload[f"{prefix}_start_{start}"] = values
    np.savez_compressed(
        outdir / "heldout_forward_predictions.npz",
        **prediction_payload,
        split_roles=fit.split.roles,
        times_s=archive.times_s,
    )
    np.savez_compressed(
        outdir / "whole_phase_forward_at_measured_times.npz",
        predicted_theta_k=equilibrium_private["_whole_phase_prediction"],
        measured_theta_k=archive.theta_k.astype(np.float32),
        times_s=archive.times_s,
        board_mask=archive.board_mask,
        rebuilt_mask=archive.rebuilt_mask,
        reconstruction_uncertainty_k=archive.uncertainty_k.astype(np.float32),
    )

    selected = equilibrium_report["selection"]
    test = equilibrium_report["test_forward"]
    persistence = equilibrium_report["test_persistence"]
    print(f"export        : {export_path}")
    print(
        f"selected rates: spread {selected['selected_spread_rate_px2_per_s']:.4f} px^2/s, "
        f"cooling {selected['selected_cooling_rate_per_s']:.6f} /s"
    )
    print(
        f"equilibrium   : training frames "
        f"{equilibrium_report['settled_field']['training_frame_start']}-"
        f"{equilibrium_report['settled_field']['training_frame_stop']} only"
    )
    print(f"test equilibrium          : {test['weighted_mae_k']:.4f} K weighted MAE")
    print(f"test persistence          : {persistence['weighted_mae_k']:.4f} K weighted MAE")
    print(
        "frame0 whole phase      : "
        f"{equilibrium_report['frame0_whole_powered_phase_weighted_mae_k']:.4f} K weighted MAE"
    )
    print(
        "source-only ablation    : "
        f"{source_report['test_forward_by_model']['fitted_smooth_source']['weighted_mae_k']:.4f} K"
    )
    print(
        "acceptance    : beats persistence="
        f"{equilibrium_report['acceptance']['beats_persistence_on_weighted_test_mae']}"
    )
    print(f"written       : {outdir / 'forecast_fit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
