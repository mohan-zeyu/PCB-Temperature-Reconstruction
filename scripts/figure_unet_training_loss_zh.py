"""用真实训练日志绘制树莓派 4B 数据的 U-Net 损失曲线。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_unet_loss_zh")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "outputs/thermal/unet_case10/train_default/train_metrics.json"
DEFAULT_OUTPUT = ROOT / "figures/unet_training_loss_rpi4b_zh.png"


def configure_chinese_font() -> None:
    candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path(
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
            "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"
        ),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_metrics(path: Path) -> tuple[dict, list[dict]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    history = [row for row in payload["history"] if "train_l1_k" in row]
    if not history:
        raise ValueError(f"训练日志中没有 train_l1_k 数据：{path}")
    return payload, history


def draw(metrics_path: Path, output_path: Path) -> None:
    configure_chinese_font()
    payload, history = load_metrics(metrics_path)
    config = payload["config"]

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_l1_k"] for row in history]
    board_mae = [row["board_mae_k"] for row in history]

    fig, ax = plt.subplots(figsize=(8.3, 4.2), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    loss_color = "#d94841"
    mae_color = "#277da1"
    ax.fill_between(epochs, train_loss, color=loss_color, alpha=0.10)
    ax.plot(
        epochs,
        train_loss,
        color=loss_color,
        linewidth=2.8,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label="训练 L1 损失",
        zorder=3,
    )
    ax.plot(
        epochs,
        board_mae,
        color=mae_color,
        linewidth=2.1,
        marker="s",
        markersize=3.8,
        alpha=0.92,
        label="全板平均绝对误差（同一训练集评估）",
        zorder=2,
    )

    ax.set_xlim(0, config["epochs"])
    ax.set_ylim(0, max(train_loss[0], board_mae[0]) * 1.13)
    ax.set_xticks(range(50, config["epochs"] + 1, 50))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_xlabel("训练轮次（Epoch）", fontsize=13, labelpad=10)
    ax.set_ylabel("误差 / 损失（K）", fontsize=13, labelpad=10)

    ax.annotate(
        f"第 {epochs[0]} 轮\n{train_loss[0]:.3f} K",
        xy=(epochs[0], train_loss[0]),
        xytext=(32, 1.27),
        fontsize=10.5,
        color=loss_color,
        arrowprops={"arrowstyle": "->", "color": loss_color, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#efb2ae"},
    )
    ax.annotate(
        f"第 {epochs[-1]} 轮\n{train_loss[-1]:.3f} K",
        xy=(epochs[-1], train_loss[-1]),
        xytext=(244, 0.31),
        fontsize=10.5,
        color=loss_color,
        arrowprops={"arrowstyle": "->", "color": loss_color, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#efb2ae"},
    )

    legend = ax.legend(
        loc="upper right",
        frameon=False,
        ncol=1,
        fontsize=10.5,
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black", labelsize=10.5, direction="out")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw(args.metrics.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
