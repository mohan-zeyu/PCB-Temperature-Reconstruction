"""Assemble ``report.md`` and ``run_metadata.json`` for a run.

The report is generated from the run's own artefacts -- nothing is typed in by
hand -- so every number in it is traceable to the metrics file and the
checkpoint that produced it.  The identifiability section is emitted
unconditionally, because the one thing a reader must not be able to do is skim
this and come away thinking a conductivity was measured.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

NOT_IDENTIFIABLE = """\
## What this experiment cannot identify

Dividing the plate equation by the areal heat capacity `C = rho cp d` is what
makes the remaining parameters identifiable, and it is also what removes the
absolute scale:

    C d(theta)/dt = div(G grad theta) - h theta + q
      d(theta)/dt = div(alpha_eff grad theta) - beta theta + s

Multiplying `C`, `G`, `h` and `q` by one common constant leaves
`alpha_eff = G/C`, `beta = h/C` and `s = q/C` unchanged, and therefore leaves
every temperature trajectory unchanged.  A temperature-only recording cannot
distinguish those boards.  So this run does **not** report, and its outputs must
not be read as:

| Quantity | Unit | Why not |
|---|---|---|
| material thermal conductivity `k(x,y)` | W/(m K) | needs `rho cp d`, never measured here |
| areal conductance `G(x,y)` | W/K | same missing scale |
| absolute power density `q(x,y,t)` | W/m^2 | same missing scale |
| component power | W | same missing scale |
| hidden copper-layer geometry | - | the surface does not reveal what is buried under it |

What it does report is listed above: coupling and cooling *rates*, a diffusivity
*proxy*, a characteristic length, a *normalized* source rate in K per unit time,
and *relative* source shares.  Restoring watts would need one additional
measurement that fixes `C` -- a calibrated power step, a known heater, or a
measured board heat capacity.
"""


def git_commit(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    inside = run("rev-parse", "--is-inside-work-tree") == "true"
    reason = None
    if commit is None:
        reason = (
            "repository has no commits yet, so results cannot be pinned to one"
            if inside
            else "not a git repository"
        )
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "available": commit is not None,
        "unavailable_reason": reason,
    }


def write_run_metadata(
    path: Path,
    *,
    repo: Path,
    measurement_config: dict,
    analysis_config: dict,
    sequence_metadata: dict,
    seed: int,
    device: str,
    time_unit: str,
    pixel_size_mm: float | None,
    extra: dict | None = None,
) -> Path:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(repo),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "device": device,
        "units": {
            "time": time_unit,
            "rate": f"1/{time_unit}",
            "source_rate": f"K/{time_unit}",
            "diffusivity_proxy": f"px^2/{time_unit}",
            "length": "px",
            "pixel_size_mm": pixel_size_mm,
            "pixel_size_calibrated": False,
        },
        "measurement_config": measurement_config,
        "analysis_config": analysis_config,
        "sequence": sequence_metadata,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    sequence_name: str,
    metrics: dict,
    metadata: dict,
    artefacts: dict[str, str],
    time_unit: str,
    pixel_size_mm: float | None,
) -> Path:
    quality = metadata.get("quality", {})
    phases = metrics.get("phases", {})
    thermal = metrics.get("thermal_operator", {})
    source = metrics.get("source", {})
    derived = metrics.get("derived", {})
    comparison = metrics.get("comparison", {})
    uncertainty = metrics.get("uncertainty", {})

    lines: list[str] = []
    add = lines.append
    add(f"# Dynamic thermal inversion — `{sequence_name}`\n")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}\n")
    add(
        f"Time unit **{time_unit}**; spatial unit **px**"
        + (f" (1 px = {pixel_size_mm:.4f} mm, geometric estimate from the PCB outline, not a calibration)" if pixel_size_mm else "")
        + ".\n"
    )

    add("## Data\n")
    add(
        _table(
            [
                ["frames used", str(quality.get("n_frames_used"))],
                ["frames found", str(quality.get("n_files_found"))],
                ["duration", f"{quality.get('duration', 0) / 60:.1f} min"],
                ["median dt", f"{quality.get('median_dt', float('nan')):.2f} {time_unit}"],
                ["clock", str(quality.get("clock", {}).get("clock"))],
                ["gaps", str(len(quality.get("gap_indices", [])))],
                ["NaN / Inf", f"{quality.get('n_nan')} / {quality.get('n_inf')}"],
                ["drift corrected", str(quality.get("drift", {}).get("applied"))],
                ["repaired pixel fraction", f"{quality.get('repair', {}).get('masked_fraction', 0):.3f}"],
                ["valid pixel fraction", f"{quality.get('valid_fraction', 0):.3f}"],
            ],
            ["property", "value"],
        )
    )
    add("")

    if phases:
        add("## Phases\n")
        add(
            _table(
                [
                    [
                        s["label"],
                        f"{s['start_index']}–{s['stop_index']}",
                        f"{s['duration'] / 60:.1f} min" if time_unit == "s" else f"{s['duration']:.0f} frames",
                    ]
                    for s in phases.get("segments", [])
                ],
                ["phase", "frames", "duration"],
            )
        )
        add("")
        add(f"> {phases.get('report', {}).get('cooling_label_note', '')}\n")

    if thermal:
        add("## Identified thermal operator\n")
        add(
            _table(
                [
                    ["effective coupling rate a_ij", f"1/{time_unit}", f"median {derived.get('edge_rate_median', float('nan')):.3e}, max {derived.get('edge_rate_max', float('nan')):.3e}"],
                    ["effective cooling rate beta_i", f"1/{time_unit}", f"median {derived.get('beta_median', float('nan')):.3e}"],
                    ["time constant tau_i = 1/beta_i", time_unit, f"median {derived.get('tau_median', float('nan')):.1f}"],
                    ["effective diffusivity proxy", f"px^2/{time_unit}", f"median {derived.get('alpha_proxy_median', float('nan')):.3e}"],
                    ["characteristic length L_i", "px", f"median {derived.get('length_median_px', float('nan')):.1f}"],
                ]
                + (
                    [["characteristic length L_i", "mm", f"median {derived.get('length_median_mm', float('nan')):.2f}"]]
                    if pixel_size_mm
                    else []
                ),
                ["quantity", "unit", "value"],
            )
        )
        add("")
        add(
            f"- spectral bound: `{thermal.get('spectral_bound_per_unit_time', float('nan')):.3e}` "
            f"1/{time_unit} (must be ≤ 0; stable = `{thermal.get('stable')}`)\n"
        )
        add(
            f"- residual idle source over the load-off interval: mean "
            f"`{thermal.get('residual_idle_source_rate_mean', float('nan')):.3e}` K/{time_unit}. "
            f"{thermal.get('residual_idle_source_note', '')}\n"
        )

    audit = thermal.get("identifiability_audit", {}) if thermal else {}
    if audit:
        add("## What this decay can and cannot separate\n")
        collapse = audit.get("profile_collapse", {})
        rates = audit.get("region_rates", {})
        separability = audit.get("separability", {})
        if collapse.get("available"):
            rows = [
                ["initial profile spread", f"{collapse['initial_profile_spread']:.3f}"],
                ["settled profile spread", f"{collapse['settled_profile_spread']:.3f}"],
            ]
            if collapse.get("monotonic_collapse", True):
                rows += [
                    ["shape settle time", f"{collapse['shape_settle_time']:.0f} {time_unit}"],
                    [
                        "frames carrying conduction information",
                        f"{collapse['informative_frames']} "
                        f"({100 * collapse['informative_fraction']:.1f}% of the record)",
                    ],
                ]
            else:
                rows += [
                    ["peak profile spread", f"{collapse.get('peak_profile_spread', float('nan')):.3f}"],
                    ["monotonic collapse", "no -- diagnostic does not apply"],
                ]
            add(_table(rows, ["profile-collapse diagnostic", "value"]))
            add("")
            add(f"> {collapse['note']}\n")
        if rates.get("available"):
            add(
                f"- directly fitted **total** per-region relaxation rate: median "
                f"`{rates['median_rate']:.3e}` 1/{time_unit} "
                f"(tau = {rates['median_time_constant']:.0f} {time_unit}), "
                f"p90/p10 = {rates['p90_over_p10']:.2f}\n"
            )
            add(f"> {rates['note']}\n")
        if separability.get("available"):
            add(f"**Verdict:** {separability['verdict']}\n")
        contrast = audit.get("coupling_contrast", {})
        if contrast.get("available"):
            add(f"**Coupling contrast:** {contrast['verdict']}\n")
        add("Identifiable from this decay: " + ", ".join(
            f"`{x}`" for x in audit.get("identifiable_from_this_decay", [])) + "\n")
        add("Not identifiable from this decay: " + ", ".join(
            f"`{x}`" for x in audit.get("not_identifiable_from_this_decay", [])) + "\n")

    if source:
        add("## Normalized source\n")
        concentration = source.get("relative_source_concentration", {})
        add(
            _table(
                [
                    ["peak normalized source rate", f"K/{time_unit}", f"{source.get('source_rate_max', float('nan')):.3e}"],
                    ["mean over powered frames", f"K/{time_unit}", f"{source.get('source_rate_mean', float('nan')):.3e}"],
                    ["top-decile share of relative source", "-", f"{concentration.get('top_decile_share', float('nan')):.3f}"],
                    ["effective contributing regions", "-", f"{concentration.get('effective_regions', float('nan')):.1f} of {concentration.get('n_regions')}"],
                ],
                ["quantity", "unit", "value"],
            )
        )
        add("")
        add(f"> {source.get('units_note', '')}\n")

    if comparison:
        add("## Baselines and ablations\n")
        rows = []
        for name, value in (comparison.get("test_mae_by_model_k") or {}).items():
            rows.append([name, f"{value:.4f}" if value is not None else "n/a"])
        add(_table(rows, ["model", "held-out test MAE [K]"]))
        add("")
        add(f"- best on test: **{comparison.get('best_model_on_test')}**\n")
        add(f"- full model wins: **{comparison.get('full_model_wins')}**\n")
        add(f"> {comparison.get('verdict_note', '')}\n")

    if uncertainty:
        add("## Uncertainty\n")
        rows = []
        for name, block in uncertainty.items():
            report = block.get("report", {})
            rows.append([
                name,
                str(report.get("kind")),
                str(report.get("n_low_confidence_regions", report.get("low_confidence_pixel_fraction", "-"))),
            ])
        add(_table(rows, ["estimate", "kind", "low-confidence count / fraction"]))
        add("")

    if uncertainty:
        partition = uncertainty.get("partition_ensemble", {}).get("report", {})
        if partition.get("alpha_resolution_spread"):
            add("### Resolution dependence\n")
            add(
                _table(
                    [
                        [
                            "beta (per-region property)",
                            f"{partition['beta_resolution_spread']:.2f}x across "
                            f"{partition['resolutions']} regions",
                        ],
                        [
                            "alpha_proxy (graph-Laplacian construction)",
                            f"{partition['alpha_resolution_spread']:.2f}x across "
                            f"{partition['resolutions']} regions",
                        ],
                    ],
                    ["quantity", "spread across region counts"],
                )
            )
            add("")
            add(f"> {partition.get('resolution_note', '')}\n")

    add("## Artefacts\n")
    add(_table([[k, f"`{v}`"] for k, v in sorted(artefacts.items())], ["artefact", "path"]))
    add("")
    add(NOT_IDENTIFIABLE)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_relative_component_csv(
    path: Path,
    partition,
    relative_source: np.ndarray,
    source_rate: np.ndarray,
    powered: np.ndarray,
    time_unit: str,
    pixel_size_mm: float | None,
) -> Path:
    """Per-region relative contribution.  Every column carries its unit."""
    import csv

    graph = partition.graph
    mean_relative = relative_source[powered].mean(axis=0)
    peak_rate = source_rate[powered].max(axis=0)
    mean_rate = source_rate[powered].mean(axis=0)
    order = np.argsort(-mean_relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank",
            "region_id",
            "centroid_row_px",
            "centroid_col_px",
            "area_px",
            "area_mm2",
            f"mean_normalized_source_rate_K_per_{time_unit}",
            f"peak_normalized_source_rate_K_per_{time_unit}",
            "mean_relative_source_contribution_dimensionless",
            "note",
        ])
        area_scale = (pixel_size_mm**2) if pixel_size_mm else float("nan")
        for rank, region in enumerate(order, start=1):
            writer.writerow([
                rank,
                int(region),
                f"{graph.centroids_px[region, 0]:.2f}",
                f"{graph.centroids_px[region, 1]:.2f}",
                f"{graph.areas_px[region]:.0f}",
                f"{graph.areas_px[region] * area_scale:.2f}",
                f"{mean_rate[region]:.6e}",
                f"{peak_rate[region]:.6e}",
                f"{mean_relative[region]:.6f}",
                "normalized source rate q/C, not power; share assumes uniform areal heat capacity",
            ])
    return path
