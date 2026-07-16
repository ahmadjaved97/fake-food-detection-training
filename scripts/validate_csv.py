#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPECTED_CLASSES
from src.data import infer_class_to_idx, read_classification_csv, validate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/validation CSV files or CSV directories.")
    parser.add_argument("--train-csv", required=True, help="Training CSV file or directory of CSVs.")
    parser.add_argument("--val-csv", required=True, help="Validation CSV file or directory of CSVs.")
    parser.add_argument("--dataset-root", required=True, help="Image root for relative image_path values.")
    parser.add_argument(
        "--label-column",
        default=None,
        help="Label column to use. Defaults to auto-detecting class_category, then category.",
    )
    parser.add_argument("--no-file-check", action="store_true")
    args = parser.parse_args()

    train_rows = read_classification_csv(args.train_csv, label_column=args.label_column)
    val_rows = read_classification_csv(args.val_csv, label_column=args.label_column)
    class_to_idx = infer_class_to_idx(train_rows, EXPECTED_CLASSES)
    validate_rows(
        train_rows,
        Path(args.dataset_root),
        class_to_idx,
        validate_files=not args.no_file_check,
        split_name="train",
    )
    validate_rows(
        val_rows,
        Path(args.dataset_root),
        class_to_idx,
        validate_files=not args.no_file_check,
        split_name="validation",
    )

    print("CSV validation passed.")
    print("Class mapping:")
    for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1]):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
