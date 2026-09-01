"""Generate a forward-only 1-second temperature forecast from any case10 frame.

The forecast starts exactly at the selected reconstructed field and decays toward
the training-only settled field.  Ambient is an external boundary condition; by
default its last known value at the starting frame is held fixed and recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tdp-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tdp-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tdp.thermal import (
    ClosedFormPlateModel,
    EquilibriumForecastFit,
    load_pattern_library,
    load_reconstruction_archive,
    make_frozen_split,
    rollout_to_equilibrium,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction", default="outputs/thermal/case10/reconstructed_sequence.npz"
    )
    parser.add_argument("--patterns", default="outputs/thermal/case10/pattern_library.npz")
    parser.add_argument("--fit", default="outputs/thermal/case10/equilibrium_forecast_fit.npz")
    parser.add_argument("--report", default="outputs/thermal/case10/forecast_fit_report.json")
    parser.add_argument("--outdir", default="outputs/thermal/case10")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--remaining-fraction", type=float, default=0.01)
    args = parser.parse_args()

    archive = load_reconstruction_archive(args.reconstruction)
    library = load_pattern_library(args.patterns)
    start = int(args.start_frame)
    if not 0 <= start < archive.n_frames:
        raise SystemExit(f"start frame must lie in [0, {archive.n_frames - 1}]")
    if args.step_s <= 0.0:
        raise SystemExit("step-s must be positive")

    with np.load(args.fit) as data:
        model = ClosedFormPlateModel.build(
            library,
            spread_rate_px2_per_time=float(data["selected_spread_rate_px2_per_s"]),
            cooling_rate_per_time=float(data["selected_cooling_rate_per_s"]),
        )
        split = make_frozen_split(archive.n_frames)
        saved_roles = np.asarray(data["split_roles"])
        if not np.array_equal(saved_roles, split.roles):
            raise ValueError("saved equilibrium fit and current frozen split disagree")
        fit = EquilibriumForecastFit(
            model=model,
            split=split,
            settled_theta_k=np.asarray(data["settled_theta_k"], dtype=np.float64),
            settled_coordinates=np.asarray(data["settled_coordinates"], dtype=np.float64),
            settled_model_range_k=np.asarray(
                data["settled_model_range_k"], dtype=np.float64
            ),
            residual_decay_multiplier=float(data["residual_decay_multiplier"]),
            metrics={},
        )

    operator_duration = model.time_to_settle(args.remaining_fraction)
    residual_duration = (
        np.log(args.remaining_fraction)
        / (-fit.residual_decay_multiplier * model.cooling_rate_per_time)
    )
    duration = float(max(operator_duration, residual_duration))
    elapsed = np.arange(0.0, np.ceil(duration) + args.step_s, args.step_s)
    if elapsed[-1] > duration + args.step_s:
        elapsed = elapsed[:-1]
    theta = rollout_to_equilibrium(fit, archive.theta_k[start], elapsed)

    decay = np.exp(-model.cooling_rate_per_time * elapsed)[:, None, None]
    start_range = archive.uncertainty_k[start]
    with Path(args.report).open(encoding="utf-8") as handle:
        fit_report = json.load(handle)
    empirical_error = float(
        fit_report["equilibrium_forecast"]["test_forward"]["weighted_mae_k"]
    )
    # A declared model range, not a confidence interval: interpolate from the
    # starting reconstruction range to the training settled range, then add the
    # held-out forward discrepancy as the model takes over.
    model_range = (
        decay * start_range[None, :, :]
        + (1.0 - decay) * fit.settled_model_range_k[None, :, :]
        + (1.0 - decay) * empirical_error
    )
    ambient = np.full(len(elapsed), archive.ambient_c[start], dtype=np.float64)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_from_frame_{start:03d}"
    np.savez_compressed(
        outdir / f"{stem}.npz",
        theta_k=theta.astype(np.float32),
        elapsed_s=elapsed,
        absolute_time_s=archive.times_s[start] + elapsed,
        ambient_c=ambient,
        start_frame=start,
        board_mask=archive.board_mask,
        rebuilt_mask=archive.rebuilt_mask,
        settled_theta_k=fit.settled_theta_k.astype(np.float32),
        start_model_range_k=start_range.astype(np.float32),
        settled_model_range_k=fit.settled_model_range_k.astype(np.float32),
        heldout_forward_error_k=empirical_error,
        model_range_decay_rate_per_s=model.cooling_rate_per_time,
        remaining_fraction=args.remaining_fraction,
        ambient_scenario=np.asarray("hold starting-frame ambient constant"),
        reconstruction_note=np.asarray(
            "temperature_c = theta_k + ambient_c; model range follows the documented "
            "linear endpoint interpolation plus held-out discrepancy"
        ),
    )

    indices = [0, len(elapsed) // 2, len(elapsed) - 1]
    vmin = float(np.percentile(theta[:, archive.board_mask], 1.0))
    vmax = float(np.percentile(theta[:, archive.board_mask], 99.0))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    image = None
    for column, index in enumerate(indices):
        image = axes[0, column].imshow(theta[index], vmin=vmin, vmax=vmax, cmap="inferno")
        axes[0, column].set_title(f"excess temperature, +{elapsed[index]:.0f} s")
        axes[0, column].axis("off")
        uncertainty_image = axes[1, column].imshow(
            model_range[index], vmin=0.0, cmap="viridis"
        )
        axes[1, column].set_title("declared model range [K]")
        axes[1, column].axis("off")
    fig.colorbar(image, ax=axes[0], label="K above ambient", shrink=0.8)
    fig.colorbar(uncertainty_image, ax=axes[1], label="K", shrink=0.8)
    fig.suptitle(
        f"Forward-only equilibrium forecast from reconstructed frame {start}\n"
        f"ambient held at {ambient[0]:.2f} C; settled field uses training frames 147-188 only"
    )
    fig.savefig(outdir / f"{stem}_montage.png", dpi=130)
    plt.close(fig)

    print(f"start         : frame {start}, t={archive.times_s[start]:.1f} s")
    print(f"output        : {len(elapsed)} fields at {args.step_s:g}-second spacing")
    print(f"settling      : {elapsed[-1]:.1f} s to {args.remaining_fraction:.1%} remaining")
    print(f"ambient       : held at {ambient[0]:.2f} C (external scenario, not predicted)")
    print(f"written       : {outdir / f'{stem}.npz'}")
    print(f"                {outdir / f'{stem}_montage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
