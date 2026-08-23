"""Command line for the dynamic thermal inversion.

    python -m tdp.dynamic_inverse prepare      --config configs/dynamic_inverse.yaml
    python -m tdp.dynamic_inverse fit-thermal  --config configs/dynamic_inverse.yaml
    python -m tdp.dynamic_inverse fit-source   --config configs/dynamic_inverse.yaml \
        --checkpoint outputs/dynamic_inverse/<sequence>/latest/thermal_operator.pt
    python -m tdp.dynamic_inverse evaluate     --config configs/dynamic_inverse.yaml
    python -m tdp.dynamic_inverse report       --run-dir outputs/dynamic_inverse/<sequence>/latest
    python -m tdp.dynamic_inverse run          --config configs/dynamic_inverse.yaml   # all of it

Every subcommand takes ``--device``, ``--seed``, ``--output-dir`` and
``--dry-run``.  ``--dry-run`` reports what would happen and touches nothing, and
``--smoke`` shrinks the problem enough to run the whole chain on a laptop CPU in
about a minute, which is what the CPU smoke test uses.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tdp.dynamic_inverse import derived, visualize
from tdp.dynamic_inverse.config import (
    REPO_ROOT,
    AnalysisConfig,
    as_dict,
    load_analysis,
    load_measurements,
)
from tdp.dynamic_inverse.evaluate import compare_models, evaluate_fit, free_rollout
from tdp.dynamic_inverse.pipeline import Prepared, load_json, prepare, run_dir, save_json
from tdp.dynamic_inverse.regions import build_regions, expand_to_pixels
from tdp.dynamic_inverse.report import (
    write_relative_component_csv,
    write_report,
    write_run_metadata,
)
from tdp.dynamic_inverse.train_source import fit_source
from tdp.dynamic_inverse.train_thermal import fit_thermal_operator
from tdp.dynamic_inverse.uncertainty import partition_ensemble, seed_ensemble, window_bootstrap

DEFAULT_MEASUREMENTS = "configs/measurements_v3.yaml"
DEFAULT_ANALYSIS = "configs/dynamic_inverse.yaml"


# --------------------------------------------------------------------------
def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_ANALYSIS, help="analysis YAML")
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS, help="measurement YAML")
    parser.add_argument("--sequence", default=None, help="sequence name (default: config's)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default="latest", help="run directory name under the sequence")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--n-regions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny configuration: few frames, few regions, few epochs (CPU smoke test)",
    )


def _resolve(args, analysis: AnalysisConfig) -> AnalysisConfig:
    updates: dict[str, Any] = {}
    if args.device:
        updates["device"] = args.device
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.n_regions:
        updates["regions"] = dataclasses.replace(analysis.regions, n_regions=args.n_regions)
    if args.smoke:
        updates["regions"] = dataclasses.replace(
            updates.get("regions", analysis.regions), n_regions=args.n_regions or 24
        )
        updates["train"] = dataclasses.replace(
            analysis.train,
            epochs_thermal=25,
            epochs_source=25,
            epochs_joint=0,
            rollout_steps=6,
            batch_windows=4,
            patience=25,
        )
        updates["uncertainty"] = dataclasses.replace(
            analysis.uncertainty, seeds=(0, 1), bootstrap_windows=2, region_resolutions=(16, 24)
        )
    return dataclasses.replace(analysis, **updates) if updates else analysis


def _sequence_spec(measurements, name: str | None):
    if not measurements.sequences:
        raise SystemExit("measurement config defines no sequences")
    if name is None:
        name = next(iter(measurements.sequences))
    if name not in measurements.sequences:
        raise SystemExit(
            f"unknown sequence {name!r}; available: {sorted(measurements.sequences)}"
        )
    return measurements.sequences[name]


def _device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", file=sys.stderr)
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        print("mps requested but unavailable; falling back to cpu", file=sys.stderr)
        return "cpu"
    return name


def _prepare(args, measurements, analysis) -> tuple[Prepared, Path, Any]:
    spec = _sequence_spec(measurements, args.sequence)
    stride = args.frame_stride
    max_frames = args.max_frames
    if args.smoke:
        stride = max(stride, 6)
        max_frames = max_frames or 60
    prepared = prepare(
        spec,
        measurements,
        analysis,
        frame_stride=stride,
        max_frames=max_frames,
        n_regions=analysis.regions.n_regions,
        dataset_root=measurements.dataset_root,
    )
    directory = run_dir(analysis.output_dir, spec.name, args.tag)
    return prepared, directory, spec


# --------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    measurements = load_measurements(args.measurements)
    analysis = _resolve(args, load_analysis(args.config))
    spec = _sequence_spec(measurements, args.sequence)
    if args.dry_run:
        print(f"[dry-run] would load {spec.case_dir} from {measurements.dataset_root}")
        print(f"[dry-run] regions={analysis.regions.n_regions} device={analysis.device} seed={analysis.seed}")
        print(f"[dry-run] output -> {run_dir(analysis.output_dir, spec.name, args.tag)}")
        return 0

    started = time.time()
    prepared, directory, spec = _prepare(args, measurements, analysis)
    sequence = prepared.sequence

    payload = {
        "sequence": sequence.metadata,
        "phases": prepared.phase_payload(),
        "regions": prepared.partition.report,
        "elapsed_s": time.time() - started,
    }
    save_json(directory / "data_quality.json", payload)
    np.savez_compressed(
        directory / "temperature_corrected_sequence.npz",
        temperature_c=sequence.temperature_c,
        raw_temperature_c=sequence.raw_temperature_c,
        valid_mask=sequence.valid_mask,
        trust_weight=sequence.trust_weight,
        emissivity=sequence.emissivity,
        timestamps_s=sequence.timestamps_s,
        ambient_c=sequence.ambient_c,
        region_labels=prepared.partition.labels,
        phase_labels=prepared.phases.labels.astype("U32"),
        node_theta=prepared.series.theta,
    )
    visualize.plot_phase_detection(
        prepared.phases, sequence.timestamps_s, directory / "phase_detection.png", sequence.time_unit
    )
    visualize.plot_region_partition(sequence, prepared.partition, directory / "region_partition.png")
    visualize.plot_emissivity_repair(sequence, directory / "emissivity_repair.png")

    quality = sequence.metadata["quality"]
    print(f"sequence      : {spec.name}")
    print(f"frames        : {quality['n_frames_used']} of {quality['n_files_found']} "
          f"({quality['duration'] / 60:.1f} min, median dt {quality['median_dt']:.2f} {sequence.time_unit})")
    print(f"board window  : {quality['board_window']['size_hw']} px @ {quality['board_window']['angle_deg']:.2f} deg "
          f"-> {quality['board_window']['output_hw']}")
    print(f"pixel size    : {sequence.pixel_size_mm:.4f} mm (geometric estimate, not calibrated)")
    print(f"drift         : measured {quality['drift'].get('measured_max_abs_shift_px', 0):.1f} px, "
          f"applied={quality['drift']['applied']}")
    print(f"repair        : {quality['repair']['masked_fraction']:.3f} of pixels, "
          f"{quality['repair'].get('n_regions', 0)} footprints; valid fraction {quality['valid_fraction']:.3f}")
    print(f"nan/inf       : {quality['n_nan']} / {quality['n_inf']};  gaps: {len(quality['gap_indices'])}")
    print(f"phases        : " + ", ".join(
        f"{s['label']}[{s['start_index']}-{s['stop_index']}]" for s in prepared.phases.segments))
    print(f"regions       : {prepared.partition.n_regions} ({prepared.partition.graph.n_edges} edges)")
    print(f"written       : {directory}")
    return 0


def cmd_fit_thermal(args) -> int:
    measurements = load_measurements(args.measurements)
    analysis = _resolve(args, load_analysis(args.config))
    if args.dry_run:
        print(f"[dry-run] would fit the thermal operator for {analysis.train.epochs_thermal} epochs")
        return 0
    prepared, directory, spec = _prepare(args, measurements, analysis)
    device = _device(analysis.device)

    started = time.time()
    fit = fit_thermal_operator(
        prepared.partition.graph, prepared.series, analysis, seed=analysis.seed, device=device,
        progress=True,
    )
    torch.save(
        {
            "state_dict": fit.state_dict,
            "edge_rate": fit.edge_rate,
            "beta": fit.beta,
            "theta_idle": fit.theta_idle,
            "idle_source_rate": fit.idle_source_rate,
            "node_features": fit.node_features,
            "edge_features": fit.edge_features,
            "similarity": fit.similarity,
            "windows": fit.windows.as_dict(),
            "metrics": fit.metrics,
            "region_labels": prepared.partition.labels,
            "edge_index": prepared.partition.graph.edge_index,
            "time_unit": fit.time_unit,
        },
        directory / "thermal_operator.pt",
    )
    summary = derived.summarise(
        prepared.partition.graph, fit.edge_rate, fit.beta,
        prepared.series.time_unit, prepared.sequence.pixel_size_mm,
    )
    scores = evaluate_fit(prepared.partition.graph, prepared.series, fit)
    save_json(directory / "thermal_metrics.json", {
        "thermal_operator": fit.metrics, "derived": summary, "scores": scores,
        "history_tail": fit.history[-5:], "elapsed_s": time.time() - started,
    })
    _thermal_figures(prepared, fit, summary, directory)

    unit = prepared.series.time_unit
    print(f"val MAE       : {fit.metrics['best_val_mae']:.4f} K (epoch {fit.metrics['best_epoch']})")
    print(f"test MAE      : {scores['test']['temperature_mae_k']:.4f} K "
          f"(free rollout, {scores['test']['n_rollout_steps']} steps)")
    print(f"spectral bound: {fit.metrics['spectral_bound_per_unit_time']:.3e} 1/{unit} "
          f"(stable={fit.metrics['stable']})")
    print(f"a_ij          : median {summary['edge_rate_median']:.3e}, max {summary['edge_rate_max']:.3e} 1/{unit}")
    print(f"beta_i        : median {summary['beta_median']:.3e} 1/{unit}  "
          f"(tau median {summary['tau_median']:.0f} {unit})")
    print(f"alpha proxy   : median {summary['alpha_proxy_median']:.3e} px^2/{unit}"
          + (f" = {summary.get('alpha_proxy_median_mm2', float('nan')):.3e} mm^2/{unit}" if 'alpha_proxy_median_mm2' in summary else ""))
    print(f"length L      : median {summary['length_median_px']:.1f} px"
          + (f" = {summary.get('length_median_mm', float('nan')):.2f} mm" if 'length_median_mm' in summary else ""))
    print(f"residual idle : {fit.metrics['residual_idle_source_rate_mean']:.3e} K/{unit} "
          f"(load-off interval is a CANDIDATE, q=0 not asserted)")
    print(f"written       : {directory}")
    return 0


def _thermal_figures(prepared, fit, summary, directory: Path) -> dict[str, str]:
    unit = prepared.series.time_unit
    partition = prepared.partition
    values = derived.arrays(partition.graph, fit.edge_rate, fit.beta)
    px = prepared.sequence.pixel_size_mm
    out: dict[str, str] = {}
    out["thermal_coupling_graph"] = str(
        visualize.plot_coupling_graph(partition, fit.edge_rate, directory / "thermal_coupling_graph.png", unit)
    )
    out["effective_diffusivity_proxy"] = str(visualize.plot_parameter_map(
        values["alpha_proxy"], partition, directory / "effective_diffusivity_proxy.png",
        f"Effective diffusivity PROXY  [px^2/{unit}]\n(from graph coupling rates; not a conductivity)",
        f"alpha_proxy  [px^2/{unit}]", cmap="viridis", pixel_size_mm=px))
    out["cooling_rate_map"] = str(visualize.plot_parameter_map(
        values["beta"], partition, directory / "cooling_rate_map.png",
        f"Effective cooling rate beta_i = h/C  [1/{unit}]\n(region-resolution; not a heat-transfer coefficient)",
        f"beta  [1/{unit}]", cmap="magma", pixel_size_mm=px))
    out["time_constant_map"] = str(visualize.plot_parameter_map(
        values["tau"], partition, directory / "time_constant_map.png",
        f"Effective time constant tau_i = 1/beta_i  [{unit}]\n"
        f"({int(values['tau_low_confidence'].sum())} regions floored as low confidence)",
        f"tau  [{unit}]", cmap="cividis", pixel_size_mm=px))
    out["fin_length_map"] = str(visualize.plot_parameter_map(
        values["length_px"], partition, directory / "fin_length_map.png",
        "Effective thermal diffusion length L = sqrt(alpha_proxy/beta)  [px]"
        + (f"\n1 px = {px:.4f} mm, so median L = {summary.get('length_median_mm', float('nan')):.2f} mm" if px else ""),
        "L  [px]", cmap="plasma", pixel_size_mm=px))

    block = fit.windows.test[0]
    predicted = free_rollout(partition.graph, prepared.series, fit, block)
    out["held_out_rollout"] = str(visualize.plot_rollout_check(
        prepared.series, fit, predicted, block, directory / "held_out_rollout.png", unit))
    np.savez_compressed(directory / "thermal_maps.npz", **{
        k: (expand_to_pixels(v.astype(float), partition.labels) if v.ndim == 1 and len(v) == partition.n_regions else v)
        for k, v in values.items()
    }, region_labels=partition.labels)
    return out


def cmd_fit_source(args) -> int:
    measurements = load_measurements(args.measurements)
    analysis = _resolve(args, load_analysis(args.config))
    if args.dry_run:
        print(f"[dry-run] would invert the source for {analysis.train.epochs_source} epochs "
              "with the thermal operator frozen")
        return 0
    prepared, directory, spec = _prepare(args, measurements, analysis)
    device = _device(analysis.device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else directory / "thermal_operator.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"thermal checkpoint not found: {checkpoint_path}; run fit-thermal first")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    thermal = _rehydrate(checkpoint, prepared)

    started = time.time()
    fit = fit_source(prepared.partition.graph, prepared.series, thermal, analysis,
                     seed=analysis.seed, device=device, progress=True)
    torch.save({"state_dict": fit.state_dict, "source_rate": fit.source_rate,
                "relative_source": fit.relative_source, "powered": fit.powered,
                "metrics": fit.metrics, "windows": fit.windows.as_dict()},
               directory / "source_decoder.pt")
    save_json(directory / "source_metrics.json",
              {"source": fit.metrics, "history_tail": fit.history[-5:], "elapsed_s": time.time() - started})

    unit = prepared.series.time_unit
    visualize.plot_source_peak(fit.source_rate, prepared.partition,
                               directory / "normalized_source_peak.png", unit, fit.powered)
    animation = visualize.animate_source(
        fit.source_rate, prepared.partition, directory / "normalized_source_animation",
        prepared.series.times, unit)
    np.savez_compressed(directory / "normalized_source.npz",
                        source_rate=fit.source_rate, relative_source=fit.relative_source,
                        powered=fit.powered, times=prepared.series.times,
                        region_labels=prepared.partition.labels)
    write_relative_component_csv(directory / "relative_component_power.csv", prepared.partition,
                                 fit.relative_source, fit.source_rate, fit.powered, unit,
                                 prepared.sequence.pixel_size_mm)

    concentration = fit.metrics["relative_source_concentration"]
    print(f"val MAE       : {fit.metrics['best_val_mae']:.4f} K (epoch {fit.metrics['best_epoch']})")
    print(f"source rate   : peak {fit.metrics['source_rate_max']:.3e} K/{unit}  "
          f"(q/C, NOT power)")
    print(f"non-negative  : {fit.metrics['source_non_negative']};  zero when unpowered: "
          f"{fit.metrics['source_zero_outside_powered']}")
    print(f"concentration : top decile holds {concentration['top_decile_share']:.3f}; "
          f"effective regions {concentration['effective_regions']:.1f} of {concentration['n_regions']}")
    print(f"top regions   : " + ", ".join(
        f"{r['region']}({r['mean_relative_source']:.3f})" for r in fit.metrics['top_regions_by_relative_source'][:5]))
    if "animation" in animation:
        print(f"animation     : {animation['animation']}")
    else:
        print(f"animation     : unavailable ({animation.get('animation_error')}); montage + npz written")
    print(f"written       : {directory}")
    return 0


def _rehydrate(checkpoint: dict, prepared: Prepared):
    from tdp.dynamic_inverse.train_thermal import ThermalFit, Windows

    windows = checkpoint["windows"]
    return ThermalFit(
        edge_rate=np.asarray(checkpoint["edge_rate"]),
        beta=np.asarray(checkpoint["beta"]),
        theta_idle=np.asarray(checkpoint["theta_idle"]),
        idle_source_rate=np.asarray(checkpoint["idle_source_rate"]),
        state_dict=checkpoint["state_dict"],
        history=[],
        windows=Windows(
            train=[tuple(w) for w in windows["train"]],
            val=[tuple(w) for w in windows["val"]],
            test=[tuple(w) for w in windows["test"]],
        ),
        metrics=checkpoint["metrics"],
        node_features=np.asarray(checkpoint["node_features"]),
        edge_features=np.asarray(checkpoint["edge_features"]),
        similarity=np.asarray(checkpoint["similarity"]),
        model_kind=checkpoint["metrics"].get("model_kind", "graph_neural_ode"),
        time_unit=checkpoint["time_unit"],
    )


def cmd_evaluate(args) -> int:
    measurements = load_measurements(args.measurements)
    analysis = _resolve(args, load_analysis(args.config))
    if args.dry_run:
        print("[dry-run] would run the baseline ladder, the ablations and the uncertainty estimates")
        return 0
    prepared, directory, spec = _prepare(args, measurements, analysis)
    device = _device(analysis.device)
    started = time.time()

    comparison = compare_models(prepared.sequence, prepared.partition, prepared.series, analysis,
                               device=device, seed=analysis.seed,
                               include_source=not args.skip_source_ablation)

    uncertainty: dict[str, Any] = {}
    results = {}
    if not args.skip_uncertainty:
        try:
            ensemble = seed_ensemble(
                prepared.partition.graph, prepared.series, analysis, device=device
            )
            results["seed ensemble"] = ensemble
            uncertainty["seed_ensemble"] = {
                "report": ensemble.report, "members": ensemble.members,
                "beta_cv_median": float(np.median(ensemble.cv["beta"])),
                "edge_rate_cv_median": float(np.median(ensemble.cv["edge_rate"])),
            }
        except Exception as exc:
            uncertainty["seed_ensemble"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        try:
            boot = window_bootstrap(prepared.partition.graph, prepared.series, analysis, device=device)
            results["window bootstrap"] = boot
            uncertainty["window_bootstrap"] = {
                "report": boot.report, "members": boot.members,
                "beta_cv_median": float(np.median(boot.cv["beta"])),
            }
        except Exception as exc:
            # One uncertainty estimator failing is a missing estimate, not a
            # failed analysis: record why and carry on with the others.
            uncertainty["window_bootstrap"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        try:
            part = partition_ensemble(
                prepared.sequence, analysis, build_regions,
                lambda seq, partition: __import__(
                    "tdp.dynamic_inverse.pipeline", fromlist=["make_series"]
                ).make_series(seq, partition, prepared.phases),
                device=device,
            )
            results["partition ensemble"] = part
            uncertainty["partition_ensemble"] = {
                "report": part.report, "members": part.members,
                "beta_map_cv_median": float(np.nanmedian(part.cv["beta_map"])),
            }
        except Exception as exc:
            uncertainty["partition_ensemble"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        if results:
            visualize.plot_uncertainty(results, prepared.partition,
                                       directory / "uncertainty_maps.png", prepared.series.time_unit)

    save_json(directory / "evaluation.json",
              {"comparison": comparison, "uncertainty": uncertainty, "elapsed_s": time.time() - started})
    print("test MAE by model [K]:")
    for name, value in (comparison["test_mae_by_model_k"] or {}).items():
        print(f"   {name:22s} {value:.4f}" if value is not None else f"   {name:22s}   n/a")
    print(f"best on test  : {comparison['best_model_on_test']} (full model wins: {comparison['full_model_wins']})")
    for name, block in uncertainty.items():
        if "skipped" in block:
            print(f"{name:22s} skipped: {block['skipped']}")
        else:
            print(f"{name:22s} " + ", ".join(f"{k}={v:.3f}" for k, v in block.items()
                                             if isinstance(v, float)))
    print(f"written       : {directory}")
    return 0


def cmd_report(args) -> int:
    directory = Path(args.run_dir)
    if not directory.exists():
        raise SystemExit(f"run directory not found: {directory}")
    quality = load_json(directory / "data_quality.json")
    metrics: dict[str, Any] = {"phases": quality.get("phases", {})}
    for name, key in [("thermal_metrics.json", None), ("source_metrics.json", None),
                      ("evaluation.json", None)]:
        path = directory / name
        if path.exists():
            metrics.update(load_json(path))
    artefacts = {p.name: str(p.relative_to(directory)) for p in sorted(directory.iterdir())
                 if p.is_file() and p.suffix in {".png", ".gif", ".csv", ".npz", ".pt", ".json"}}
    sequence_meta = quality.get("sequence", {})
    unit = sequence_meta.get("time_unit", "s")
    pixel = sequence_meta.get("quality", {}).get("pixel_size", {}).get("value_mm")

    save_json(directory / "metrics.json", metrics)
    write_run_metadata(
        directory / "run_metadata.json", repo=REPO_ROOT,
        measurement_config=as_dict(load_measurements(args.measurements)),
        analysis_config=as_dict(load_analysis(args.config)),
        sequence_metadata=sequence_meta,
        seed=load_analysis(args.config).seed, device=load_analysis(args.config).device,
        time_unit=unit, pixel_size_mm=pixel,
    )
    path = write_report(directory / "report.md", sequence_name=sequence_meta.get("sequence", directory.parent.name),
                        metrics=metrics, metadata=sequence_meta, artefacts=artefacts,
                        time_unit=unit, pixel_size_mm=pixel)
    print(f"report        : {path}")
    print(f"metadata      : {directory / 'run_metadata.json'}")
    print(f"metrics       : {directory / 'metrics.json'}")
    return 0


def cmd_run(args) -> int:
    for step in (cmd_prepare, cmd_fit_thermal, cmd_fit_source, cmd_evaluate):
        print(f"\n=== {step.__name__.removeprefix('cmd_')} ===")
        code = step(args)
        if code:
            return code
    measurements = load_measurements(args.measurements)
    analysis = _resolve(args, load_analysis(args.config))
    spec = _sequence_spec(measurements, args.sequence)
    args.run_dir = str(run_dir(analysis.output_dir, spec.name, args.tag))
    print("\n=== report ===")
    return cmd_report(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tdp.dynamic_inverse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in [("prepare", cmd_prepare), ("fit-thermal", cmd_fit_thermal),
                          ("fit-source", cmd_fit_source), ("evaluate", cmd_evaluate),
                          ("run", cmd_run)]:
        p = sub.add_parser(name, help=handler.__doc__ or name)
        _common(p)
        if name in ("fit-source", "run"):
            p.add_argument("--checkpoint", default=None)
        if name in ("evaluate", "run"):
            p.add_argument("--skip-uncertainty", action="store_true")
            p.add_argument("--skip-source-ablation", action="store_true")
        if name == "run":
            p.add_argument("--run-dir", default=None)
        p.set_defaults(handler=handler)

    p = sub.add_parser("report", help="assemble report.md and run_metadata.json")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=DEFAULT_ANALYSIS)
    p.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    p.set_defaults(handler=cmd_report)

    args = parser.parse_args(argv)
    return args.handler(args)
