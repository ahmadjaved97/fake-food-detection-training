#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import (
    IMAGE_COLUMN,
    filter_rows_by_column_values,
    filter_rows_by_label_values,
    load_rgb_image,
    resolve_csv_files,
    select_label_column,
)
from src.metrics import compute_classification_metrics
from src.models import create_model, create_transforms
from src.rich_utils import print_panel, print_table
from src.utils import write_json


class EvaluationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        dataset_root: str | Path,
        transform: Any,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.rows = rows
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, str, int]:
        row = self.rows[index]
        raw_path = Path(row[IMAGE_COLUMN])
        image_path = raw_path if raw_path.is_absolute() else self.dataset_root / raw_path
        image = self.transform(load_rgb_image(image_path))
        label = -1
        if self.class_to_idx is not None and row.get("category") is not None:
            label = self.class_to_idx[row["category"]]
        return image, str(image_path), label


def read_eval_csv(
    csv_source: str | Path,
    label_column: str | None,
    class_to_idx: dict[str, int],
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    include_column_values: dict[str, list[str]] | None = None,
    exclude_column_values: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    has_labels = True
    for csv_path in resolve_csv_files(csv_source):
        with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or IMAGE_COLUMN not in reader.fieldnames:
                raise ValueError(f"{csv_path} must contain an {IMAGE_COLUMN!r} column.")
            try:
                selected_label_column = select_label_column(reader.fieldnames, Path(csv_path), label_column)
            except ValueError:
                if label_column:
                    raise
                selected_label_column = None
                has_labels = False
            for csv_row in reader:
                row = {
                    str(key): (value or "").strip()
                    for key, value in csv_row.items()
                    if key is not None
                }
                image_path = row.get(IMAGE_COLUMN, "")
                if not image_path:
                    continue
                row[IMAGE_COLUMN] = image_path
                row["source_csv"] = str(csv_path)
                if selected_label_column is not None:
                    label = row.get(selected_label_column, "")
                    row["category"] = label
                rows.append(row)
    if not rows:
        raise ValueError(f"No images found in CSV source: {csv_source}")
    filter_report = {"loaded": len(rows), "dropped_by_label_filters": 0, "dropped_by_column_filters": 0}
    if include_labels or exclude_labels:
        if not has_labels:
            raise ValueError("--include-labels/--exclude-labels require a CSV label column.")
        rows, dropped_rows = filter_rows_by_label_values(rows, include=include_labels, exclude=exclude_labels)
        filter_report["dropped_by_label_filters"] = len(dropped_rows)
    if not rows:
        raise ValueError(f"No images left after applying label filters for CSV source: {csv_source}")
    if include_column_values or exclude_column_values:
        rows, dropped_rows = filter_rows_by_column_values(
            rows,
            include=include_column_values,
            exclude=exclude_column_values,
            split_name="evaluation CSV",
        )
        filter_report["dropped_by_column_filters"] = len(dropped_rows)
    if not rows:
        raise ValueError(f"No images left after applying column filters for CSV source: {csv_source}")
    filter_report["kept"] = len(rows)
    if has_labels:
        unknown_labels = sorted({row["category"] for row in rows} - set(class_to_idx))
        if unknown_labels:
            raise ValueError(f"CSV contains labels not found in checkpoint classes: {unknown_labels}")
    return rows, has_labels, filter_report


def parse_column_value_filters(raw_filters: list[list[str]] | None) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    if not raw_filters:
        return filters
    for raw_filter in raw_filters:
        if len(raw_filter) < 2:
            raise ValueError("Column filters must be passed as: COLUMN VALUE [VALUE ...]")
        column = raw_filter[0].strip()
        values = [value.strip() for value in raw_filter[1:] if value.strip()]
        if not column or not values:
            raise ValueError("Column filters require a non-empty column name and at least one non-empty value.")
        filters.setdefault(column, []).extend(values)
    return filters


def load_checkpoint_model(model_path: str | Path, device: torch.device) -> tuple[Any, dict[str, Any], dict[str, int]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' and 'class_to_idx'. Use a training checkpoint.")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    config["model"]["pretrained"] = False
    model = create_model(config, num_classes=len(class_to_idx))
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, class_to_idx


@torch.no_grad()
def predict(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for images, paths, labels in dataloader:
        images = images.to(device)
        probs = torch.softmax(model(images).float(), dim=1).cpu()
        pred_ids = probs.argmax(dim=1)
        for idx, image_path in enumerate(paths):
            pred_id = int(pred_ids[idx].item())
            label_id = int(labels[idx].item())
            predictions.append(
                {
                    "image_path": image_path,
                    "true_label": class_names[label_id] if label_id >= 0 else None,
                    "predicted_label": class_names[pred_id],
                    "confidence": float(probs[idx, pred_id].item()),
                    "probabilities": {
                        class_name: float(probs[idx, class_idx].item())
                        for class_idx, class_name in enumerate(class_names)
                    },
                }
            )
    return predictions


def write_predictions_csv(predictions: list[dict[str, Any]], output_path: str | Path, class_names: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "true_label", "predicted_label", "confidence"] + [
        f"prob_{class_name}" for class_name in class_names
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            row = {
                "image_path": pred["image_path"],
                "true_label": pred["true_label"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
            }
            row.update({f"prob_{name}": pred["probabilities"][name] for name in class_names})
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained timm checkpoint.")
    parser.add_argument("--model-path", required=True, help="Path to a training checkpoint .pt file.")
    parser.add_argument("--dataset-root", default=".", help="Root for relative image_path values.")
    parser.add_argument("--csv", default=None, help="CSV file or directory of CSVs to evaluate.")
    parser.add_argument("--image-path", default=None, help="Single image path to classify.")
    parser.add_argument("--label-column", default=None, help="CSV label column for metrics.")
    parser.add_argument("--include-labels", nargs="+", default=None, help="Evaluate only these label values.")
    parser.add_argument("--exclude-labels", nargs="+", default=None, help="Exclude these label values from evaluation.")
    parser.add_argument(
        "--include-column-values",
        nargs="+",
        action="append",
        default=None,
        metavar=("COLUMN", "VALUE"),
        help="Keep rows where COLUMN equals any listed VALUE. Repeat for multiple columns.",
    )
    parser.add_argument(
        "--exclude-column-values",
        nargs="+",
        action="append",
        default=None,
        metavar=("COLUMN", "VALUE"),
        help="Drop rows where COLUMN equals any listed VALUE. Repeat for multiple columns.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--enable-auc", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if bool(args.csv) == bool(args.image_path):
        raise ValueError("Pass exactly one of --csv or --image-path.")

    device = torch.device(args.device)
    model, config, class_to_idx = load_checkpoint_model(args.model_path, device)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    _, val_transform, _ = create_transforms(model, config)

    if args.image_path:
        row = {IMAGE_COLUMN: args.image_path}
        dataset = EvaluationDataset([row], args.dataset_root, val_transform)
        predictions = predict(model, DataLoader(dataset, batch_size=1), device, class_names)
        pred = predictions[0]
        top_items = sorted(pred["probabilities"].items(), key=lambda item: item[1], reverse=True)[: args.top_k]
        print_panel(
            f"Image: {pred['image_path']}\n"
            f"Prediction: {pred['predicted_label']}\n"
            f"Confidence: {pred['confidence']:.4f}",
            title="Prediction",
        )
        print_table("Top Classes", ["class", "probability"], [[name, f"{prob:.4f}"] for name, prob in top_items])
    else:
        include_column_values = parse_column_value_filters(args.include_column_values)
        exclude_column_values = parse_column_value_filters(args.exclude_column_values)
        rows, has_labels, filter_report = read_eval_csv(
            args.csv,
            args.label_column,
            class_to_idx,
            include_labels=args.include_labels,
            exclude_labels=args.exclude_labels,
            include_column_values=include_column_values,
            exclude_column_values=exclude_column_values,
        )
        dataset = EvaluationDataset(rows, args.dataset_root, val_transform, class_to_idx if has_labels else None)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        predictions = predict(model, dataloader, device, class_names)
        print_table(
            "Evaluation Input",
            ["loaded", "kept", "dropped labels", "dropped columns", "labels available"],
            [
                [
                    filter_report["loaded"],
                    filter_report["kept"],
                    filter_report["dropped_by_label_filters"],
                    filter_report["dropped_by_column_filters"],
                    "yes" if has_labels else "no",
                ]
            ],
        )
        if has_labels:
            labels = torch.tensor([class_to_idx[pred["true_label"]] for pred in predictions]).numpy()
            probabilities = torch.tensor(
                [[pred["probabilities"][class_name] for class_name in class_names] for pred in predictions]
            ).numpy()
            metrics = compute_classification_metrics(labels, probabilities, class_names, args.enable_auc)
            metrics["num_images"] = len(predictions)
            metrics["filter_report"] = filter_report
            print_table(
                "Metrics",
                ["metric", "value"],
                [
                    ["accuracy", f"{metrics['accuracy']:.4f}"],
                    ["macro_f1", f"{metrics['macro_f1']:.4f}"],
                    ["weighted_f1", f"{metrics['weighted_f1']:.4f}"],
                    ["macro_precision", f"{metrics['macro_precision']:.4f}"],
                    ["macro_recall", f"{metrics['macro_recall']:.4f}"],
                ],
            )
            if args.output_json:
                write_json(metrics, args.output_json)
        else:
            print_panel("No label column found, so metrics were skipped and predictions were produced.", title="Note")

    if args.predictions_csv:
        write_predictions_csv(predictions, args.predictions_csv, class_names)


if __name__ == "__main__":
    main()
