"""Combine the reflectivity figure pairs side by side.

Group 1 (raw IR camera shots): bare board (case10) | painted board (case17).
Group 2 (matplotlib analysis): steady-state maps + difference | SoC radiometry.

Outputs land next to the sources in figures/reflectivity/.
"""

from pathlib import Path

from PIL import Image

REFLECTIVITY_DIR = Path(__file__).resolve().parents[1] / "figures" / "reflectivity"

RAW_PAIR = [
    ("raw_ir_bare_case10.jpeg", (0, 0, 0)),
    ("raw_ir_painted_case17.jpeg", (0, 0, 0)),
]
PROCESSED_PAIR = [
    ("processed_steady_maps_and_difference.png", (255, 255, 255)),
    ("processed_soc_and_radiometry.png", (255, 255, 255)),
]

GAP = 24


def combine_horizontally(items, out_path):
    images = [Image.open(REFLECTIVITY_DIR / name).convert("RGB") for name, _ in items]
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + GAP * (len(images) - 1)
    background = items[0][1]
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for im in images:
        y = (height - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + GAP
    canvas.save(out_path)
    print(f"{out_path.name}: {width}x{height} px")


def main():
    combine_horizontally(
        RAW_PAIR, REFLECTIVITY_DIR / "combined_raw_ir_bare_vs_painted.png"
    )
    combine_horizontally(
        PROCESSED_PAIR, REFLECTIVITY_DIR / "combined_processed_maps_and_radiometry.png"
    )


if __name__ == "__main__":
    main()
