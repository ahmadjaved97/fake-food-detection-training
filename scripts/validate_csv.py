#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import filter_rows_by_label_values, infer_class_to_idx, read_classification_csv, validate_rows
from src.rich_utils import print_table


def apply_label_filters(
    rows: list[dict[str, str]],
    allowed_classes: list[str] | None,
    include_labels: list[str] | None,
    exclude_labels: list[str] | None,
) -> tuple[list[dict[str, str]], int]:
    include = include_labels
    if allowed_classes:
        allowed_set = set(allowed_classes)
        include = [label for label in include if label in allowed_set] if include else allowed_classes
    filtered_rows, dropped_rows = filter_rows_by_label_values(rows, include=include, exclude=exclude_labels)
    return filtered_rows, len(dropped_rows)


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
    parser.add_argument(
        "--missing-files-policy",
        choices=["error", "skip"],
        default="error",
        help="Use skip to validate class labels while allowing missing images.",
    )
    parser.add_argument("--allowed-classes", nargs="+", default=None, help="Allowed output labels/classes.")
    parser.add_argument("--train-include-labels", nargs="+", default=None, help="Keep only these train labels.")
    parser.add_argument("--train-exclude-labels", nargs="+", default=None, help="Drop these train labels.")
    parser.add_argument("--val-include-labels", nargs="+", default=None, help="Keep only these validation labels.")
    parser.add_argument("--val-exclude-labels", nargs="+", default=None, help="Drop these validation labels.")
    parser.add_argument("--no-file-check", action="store_true")
    args = parser.parse_args()

    train_rows = read_classification_csv(args.train_csv, label_column=args.label_column)
    val_rows = read_classification_csv(args.val_csv, label_column=args.label_column)
    raw_train_rows = len(train_rows)
    raw_val_rows = len(val_rows)
    train_rows, train_filtered_rows = apply_label_filters(
        train_rows,
        args.allowed_classes,
        args.train_include_labels,
        args.train_exclude_labels,
    )
    val_rows, val_filtered_rows = apply_label_filters(
        val_rows,
        args.allowed_classes,
        args.val_include_labels,
        args.val_exclude_labels,
    )
    class_to_idx = infer_class_to_idx(train_rows, args.allowed_classes)
    validate_rows(
        train_rows,
        Path(args.dataset_root),
        class_to_idx,
        validate_files=not args.no_file_check,
        split_name="train",
        missing_files_policy=args.missing_files_policy,
    )
    validate_rows(
        val_rows,
        Path(args.dataset_root),
        class_to_idx,
        validate_files=not args.no_file_check,
        split_name="validation",
        missing_files_policy=args.missing_files_policy,
    )

    print_table(
        "CSV Validation Passed",
        ["item", "value"],
        [
            ["raw train rows", raw_train_rows],
            ["filtered train rows", train_filtered_rows],
            ["selected train rows", len(train_rows)],
            ["raw validation rows", raw_val_rows],
            ["filtered validation rows", val_filtered_rows],
            ["selected validation rows", len(val_rows)],
            ["classes", len(class_to_idx)],
        ],
    )
    print_table(
        "Class Mapping",
        ["index", "class"],
        [[idx, name] for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1])],
    )


if __name__ == "__main__":
    main()
