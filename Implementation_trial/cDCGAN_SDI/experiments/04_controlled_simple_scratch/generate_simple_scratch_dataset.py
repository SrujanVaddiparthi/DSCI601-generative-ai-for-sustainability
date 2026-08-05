#!/usr/bin/env python3
"""
Generate a controlled 128x128 grayscale dataset for a diagnostic cDCGAN experiment.

Classes
-------
normal:
    Mildly varying grayscale background with low-amplitude noise.
scratch:
    The same background process plus one bright, anti-aliased straight scratch.

The scratch parameter space is deliberately covered across:
- angle
- spatial region
- length
- width

Outputs
-------
<out>/
    normal/
    scratches/
    metadata.csv
    dataset_preview.png
    dataset_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMG_SIZE = 128
DEFAULT_COUNT_PER_CLASS = 700
DEFAULT_SEED = 42

ANGLE_BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
LENGTH_BINS = [(28, 48), (49, 72), (73, 96)]
WIDTH_BINS = [(1, 1), (2, 2), (3, 4)]
POSITION_BINS = [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("simple_scratch_v1"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--count-per-class",
        type=int,
        default=DEFAULT_COUNT_PER_CLASS,
        help="Number of unique images in each class.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before generation.",
    )
    return parser.parse_args()


def make_background(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Create a simple but non-identical grayscale background."""
    base_gray = int(rng.integers(96, 161))
    noise_sigma = float(rng.uniform(0.3, 2.2))

    background = np.full((IMG_SIZE, IMG_SIZE), base_gray, dtype=np.float32)
    background += rng.normal(0.0, noise_sigma, size=background.shape)

    # Very small illumination variation. This keeps the task controlled while
    # preventing every normal image from being an identical constant matrix.
    gradient_axis = str(rng.choice(["horizontal", "vertical"]))
    gradient_amplitude = float(rng.uniform(-3.0, 3.0))
    gradient = np.linspace(
        -gradient_amplitude / 2.0,
        gradient_amplitude / 2.0,
        IMG_SIZE,
        dtype=np.float32,
    )
    if gradient_axis == "horizontal":
        background += gradient[None, :]
    else:
        background += gradient[:, None]

    background = np.clip(background, 0, 255).astype(np.uint8)
    metadata = {
        "base_gray": base_gray,
        "noise_sigma": round(noise_sigma, 4),
        "gradient_axis": gradient_axis,
        "gradient_amplitude": round(gradient_amplitude, 4),
    }
    return background, metadata


def make_coverage_plan(count: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    """Create broad, repeated coverage of angle/position/length/width bins."""
    combinations = list(
        product(
            range(len(ANGLE_BINS)),
            range(len(POSITION_BINS)),
            range(len(LENGTH_BINS)),
            range(len(WIDTH_BINS)),
        )
    )

    plan: list[tuple[int, int, int, int]] = []
    while len(plan) < count:
        shuffled = combinations.copy()
        rng.shuffle(shuffled)
        plan.extend(shuffled)
    return plan[:count]


def sample_center(
    rng: np.random.Generator,
    position_bin_index: int,
    half_dx: float,
    half_dy: float,
    margin: int = 5,
) -> tuple[float, float]:
    """Sample a center in a 3x3 region while keeping endpoints in bounds."""
    col, row = POSITION_BINS[position_bin_index]
    region_min_x = col * IMG_SIZE / 3.0
    region_max_x = (col + 1) * IMG_SIZE / 3.0
    region_min_y = row * IMG_SIZE / 3.0
    region_max_y = (row + 1) * IMG_SIZE / 3.0

    global_min_x = abs(half_dx) + margin
    global_max_x = IMG_SIZE - 1 - abs(half_dx) - margin
    global_min_y = abs(half_dy) + margin
    global_max_y = IMG_SIZE - 1 - abs(half_dy) - margin

    low_x = max(region_min_x, global_min_x)
    high_x = min(region_max_x, global_max_x)
    low_y = max(region_min_y, global_min_y)
    high_y = min(region_max_y, global_max_y)

    # Long lines near image edges can make a requested 3x3 region infeasible.
    # Fall back to the full feasible interval rather than clipping the line.
    if low_x >= high_x:
        low_x, high_x = global_min_x, global_max_x
    if low_y >= high_y:
        low_y, high_y = global_min_y, global_max_y

    return float(rng.uniform(low_x, high_x)), float(rng.uniform(low_y, high_y))


def draw_antialiased_scratch(
    background: np.ndarray,
    rng: np.random.Generator,
    angle_bin_index: int,
    position_bin_index: int,
    length_bin_index: int,
    width_bin_index: int,
) -> tuple[np.ndarray, dict]:
    scale = 4

    angle_low, angle_high = ANGLE_BINS[angle_bin_index]
    length_low, length_high = LENGTH_BINS[length_bin_index]
    width_low, width_high = WIDTH_BINS[width_bin_index]

    angle_deg = float(rng.uniform(angle_low, angle_high))
    length_px = float(rng.uniform(length_low, length_high))
    width_px = int(rng.integers(width_low, width_high + 1))

    theta = math.radians(angle_deg)
    half_dx = 0.5 * length_px * math.cos(theta)
    half_dy = 0.5 * length_px * math.sin(theta)

    center_x, center_y = sample_center(
        rng,
        position_bin_index,
        half_dx=half_dx,
        half_dy=half_dy,
    )

    x1, y1 = center_x - half_dx, center_y - half_dy
    x2, y2 = center_x + half_dx, center_y + half_dy

    mask_large = Image.new("L", (IMG_SIZE * scale, IMG_SIZE * scale), 0)
    draw = ImageDraw.Draw(mask_large)
    draw.line(
        [(x1 * scale, y1 * scale), (x2 * scale, y2 * scale)],
        fill=255,
        width=max(scale, width_px * scale),
    )
    mask = mask_large.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0

    background_mean = float(np.mean(background))
    intensity_delta = float(rng.uniform(35.0, 90.0))
    scratch_intensity = min(255.0, background_mean + intensity_delta)

    result = background.astype(np.float32)
    result = result * (1.0 - mask_array) + scratch_intensity * mask_array
    result = np.clip(result, 0, 255).astype(np.uint8)

    metadata = {
        "angle_deg": round(angle_deg, 4),
        "angle_bin": f"{angle_low}-{angle_high}",
        "position_bin": position_bin_index,
        "length_px": round(length_px, 4),
        "length_bin": f"{length_low}-{length_high}",
        "width_px": width_px,
        "width_bin": f"{width_low}-{width_high}",
        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),
        "x1": round(x1, 4),
        "y1": round(y1, 4),
        "x2": round(x2, 4),
        "y2": round(y2, 4),
        "scratch_intensity": round(scratch_intensity, 4),
        "intensity_delta": round(intensity_delta, 4),
    }
    return result, metadata


def save_preview(out_root: Path, seed: int) -> None:
    """Save a 2x8 dataset preview without requiring matplotlib."""
    rng = np.random.default_rng(seed + 100_000)
    normal_paths = sorted((out_root / "normal").glob("*.png"))
    scratch_paths = sorted((out_root / "scratches").glob("*.png"))
    chosen_normal = rng.choice(normal_paths, size=8, replace=False)
    chosen_scratch = rng.choice(scratch_paths, size=8, replace=False)

    tile_size = IMG_SIZE
    label_height = 22
    canvas = Image.new("L", (8 * tile_size, 2 * (tile_size + label_height)), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), "normal", fill=0)
    draw.text((4, tile_size + label_height + 3), "scratch", fill=0)

    for col, path in enumerate(chosen_normal):
        img = Image.open(path).convert("L")
        canvas.paste(img, (col * tile_size, label_height))

    second_y = tile_size + 2 * label_height
    for col, path in enumerate(chosen_scratch):
        img = Image.open(path).convert("L")
        canvas.paste(img, (col * tile_size, second_y))

    canvas.save(out_root / "dataset_preview.png")


def main() -> None:
    args = parse_args()
    if args.count_per_class <= 0:
        raise ValueError("--count-per-class must be positive")

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    normal_dir = args.out / "normal"
    scratch_dir = args.out / "scratches"
    normal_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    coverage_plan = make_coverage_plan(args.count_per_class, rng)

    rows: list[dict] = []

    for index in range(args.count_per_class):
        background, bg_metadata = make_background(rng)
        filename = f"normal_{index:05d}.png"
        Image.fromarray(background, mode="L").save(normal_dir / filename)

        rows.append(
            {
                "filename": filename,
                "class": "normal",
                "seed": args.seed,
                **bg_metadata,
                "angle_deg": "",
                "angle_bin": "",
                "position_bin": "",
                "length_px": "",
                "length_bin": "",
                "width_px": "",
                "width_bin": "",
                "center_x": "",
                "center_y": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "scratch_intensity": "",
                "intensity_delta": "",
            }
        )

    for index, coverage in enumerate(coverage_plan):
        angle_bin_index, position_bin_index, length_bin_index, width_bin_index = coverage
        background, bg_metadata = make_background(rng)
        scratch, scratch_metadata = draw_antialiased_scratch(
            background,
            rng,
            angle_bin_index=angle_bin_index,
            position_bin_index=position_bin_index,
            length_bin_index=length_bin_index,
            width_bin_index=width_bin_index,
        )
        filename = f"scratch_{index:05d}.png"
        Image.fromarray(scratch, mode="L").save(scratch_dir / filename)
        rows.append(
            {
                "filename": filename,
                "class": "scratch",
                "seed": args.seed,
                **bg_metadata,
                **scratch_metadata,
            }
        )

    metadata_path = args.out / "metadata.csv"
    fieldnames = list(rows[0].keys())
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_preview(args.out, args.seed)

    summary = {
        "image_size": [IMG_SIZE, IMG_SIZE],
        "grayscale": True,
        "seed": args.seed,
        "normal_count": args.count_per_class,
        "scratch_count": args.count_per_class,
        "one_scratch_per_scratch_image": True,
        "angle_bins": ANGLE_BINS,
        "length_bins": LENGTH_BINS,
        "width_bins": WIDTH_BINS,
        "position_regions": "3x3",
        "background": {
            "base_gray_range": [96, 160],
            "noise_sigma_range": [0.3, 2.2],
            "gradient_amplitude_range": [-3.0, 3.0],
        },
    }
    with (args.out / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Dataset created at: {args.out.resolve()}")
    print(f"Normal images: {args.count_per_class}")
    print(f"Scratch images: {args.count_per_class}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Preview: {(args.out / 'dataset_preview.png').resolve()}")


if __name__ == "__main__":
    main()
