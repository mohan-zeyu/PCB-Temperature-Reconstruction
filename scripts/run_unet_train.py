"""Train the residual correction U-Net on the whole case10<->case17 archive.

Usage:
    python scripts/run_unet_train.py [--epochs 200] [--device cpu]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet")

from tdp.unet.train import train_full_set

DEFAULT_ARCHIVE = "outputs/thermal/unet_case10/painted_pair_dataset.npz"
DEFAULT_OUT = "outputs/thermal/unet_case10/train_default"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mask-source", default="combined",
                        choices=["combined", "image-only"])
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    result = train_full_set(
        args.archive,
        args.outdir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        mask_source=args.mask_source,
        base_channels=args.base_channels,
        depth=args.depth,
    )
    print("best:", result["best"])
    print("final:", result["final"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
