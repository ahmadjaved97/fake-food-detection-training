from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"

EXPECTED_CLASSES = [
    "real_good",
    "real_bad",
    "fake_good",
    "fake_bad",
    "manipulated_bad",
]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    base = read_yaml(BASE_CONFIG)
    if config_path.resolve() == BASE_CONFIG.resolve():
        return base
    override = read_yaml(config_path)
    return deep_update(base, override)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
    if args.per_device_batch_size is not None:
        cfg["training"]["per_device_batch_size"] = args.per_device_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.disable_effective_batch_size_check:
        cfg["training"]["enforce_effective_batch_size"] = False
    if args.enable_wandb:
        cfg["logging"]["wandb"]["enabled"] = True
    if args.enable_tensorboard:
        cfg["logging"]["tensorboard"]["enabled"] = True
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.mixed_precision:
        cfg["training"]["mixed_precision"] = args.mixed_precision
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train timm image classifiers with Accelerate.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Training CSV file or directory of CSVs with image_path and class_category/category columns.",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Validation CSV file or directory of CSVs with image_path and class_category/category columns.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Image root directory used to resolve relative image_path values.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for configs, logs, checkpoints.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16", "fp8"], default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--disable-effective-batch-size-check", action="store_true")
    return parser
