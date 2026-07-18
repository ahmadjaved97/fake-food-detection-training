#!/usr/bin/env python
from __future__ import annotations

import argparse

import timm


DEFAULT_PATTERNS = [
    "resnet*",
    "deit*",
    "vit*_patch16_224*",
    "*dinov2*",
    "mobilenetv3*",
    "mobilenetv4*",
    "swin*",
    "convnext*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="List timm model names for common families.")
    parser.add_argument("patterns", nargs="*", default=DEFAULT_PATTERNS)
    parser.add_argument("--pretrained", action="store_true", default=True)
    args = parser.parse_args()

    for pattern in args.patterns:
        print(f"\n# {pattern}")
        models = timm.list_models(pattern, pretrained=args.pretrained)
        if not models:
            print("  no pretrained matches")
            continue
        for model_name in models:
            print(f"  {model_name}")


if __name__ == "__main__":
    main()
