#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from src.data import IMAGE_COLUMN, filter_rows_by_label_values, load_rgb_image, read_classification_csv
from src.frequency_transforms import (
    dct_coefficient_map,
    fft_magnitude_map,
    fft_magnitude_rgb_map,
    resize_rgb_image,
    rgb_to_float_array,
    rgb_to_luminance,
)
from src.utils import write_json


def normalize_channel_mode(channel_mode: str) -> str:
    normalized = channel_mode.strip().lower().replace("-", "_")
    if normalized in {"luminance", "luma", "y"}:
        return "luminance"
    if normalized in {"rgb", "per_channel", "per_channel_fft", "per_channel_rgb"}:
        return "rgb"
    raise ValueError("--channel-mode must be luminance or rgb.")


def compute_frequency_map(
    image_path: Path,
    mode: str,
    image_size: int,
    block_size: int,
    channel_mode: str,
) -> np.ndarray:
    image = resize_rgb_image(load_rgb_image(image_path), image_size)
    if mode == "fft_magnitude":
        if channel_mode == "rgb":
            return fft_magnitude_rgb_map(rgb_to_float_array(image))
        return fft_magnitude_map(rgb_to_luminance(image))
    if mode == "dct_map":
        if channel_mode != "luminance":
            raise ValueError("--channel-mode rgb is currently supported only for --mode fft_magnitude.")
        luminance = rgb_to_luminance(image)
        return dct_coefficient_map(luminance, block_size=block_size)
    raise ValueError(f"Unsupported frequency mode: {mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute fixed normalization stats for frequency-domain training.")
    parser.add_argument("--csv", required=True, help="Training CSV file or directory of CSVs.")
    parser.add_argument("--dataset-root", required=True, help="Root for relative image_path values.")
    parser.add_argument("--output-json", required=True, help="Where to write normalization stats JSON.")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--include-labels", nargs="+", default=None)
    parser.add_argument("--exclude-labels", nargs="+", default=None)
    parser.add_argument("--mode", choices=["fft_magnitude", "dct_map"], required=True)
    parser.add_argument(
        "--channel-mode",
        choices=["luminance", "rgb"],
        default="luminance",
        help="Use luminance FFT/DCT or per-channel RGB FFT. RGB is valid only with --mode fft_magnitude.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--sample-pixels-per-image", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channel_mode = normalize_channel_mode(args.channel_mode)
    rng = np.random.default_rng(args.seed)
    dataset_root = Path(args.dataset_root)
    rows = read_classification_csv(args.csv, label_column=args.label_column)
    if args.include_labels or args.exclude_labels:
        rows, _ = filter_rows_by_label_values(rows, include=args.include_labels, exclude=args.exclude_labels)
    if not rows:
        raise ValueError("No rows left after applying label filters.")
    if args.max_images > 0 and len(rows) > args.max_images:
        indices = rng.choice(len(rows), size=args.max_images, replace=False)
        rows = [rows[int(index)] for index in indices]

    sampled_values = []
    missing = 0
    for row in tqdm(rows, desc=f"Computing {args.mode} stats"):
        raw_path = Path(row[IMAGE_COLUMN])
        image_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
        if not image_path.is_file():
            missing += 1
            continue
        frequency_map = compute_frequency_map(
            image_path,
            mode=args.mode,
            image_size=args.image_size,
            block_size=args.block_size,
            channel_mode=channel_mode,
        )
        flat = frequency_map.reshape(-1)
        sample_size = min(args.sample_pixels_per_image, flat.shape[0])
        sample_indices = rng.choice(flat.shape[0], size=sample_size, replace=False)
        sampled_values.append(flat[sample_indices])

    if not sampled_values:
        raise ValueError("No usable images found for frequency-stat computation.")

    values = np.concatenate(sampled_values)
    min_value, max_value = np.percentile(values, [args.lower_percentile, args.upper_percentile])
    write_json(
        {
            "mode": args.mode,
            "channel_mode": channel_mode,
            "image_size": args.image_size,
            "block_size": args.block_size,
            "normalization_mode": "fixed_minmax",
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "min_value": float(min_value),
            "max_value": float(max_value),
            "images_used": len(rows) - missing,
            "missing_images": missing,
            "sampled_values": int(values.shape[0]),
        },
        args.output_json,
    )


if __name__ == "__main__":
    main()
