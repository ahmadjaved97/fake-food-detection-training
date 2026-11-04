#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from scripts.evaluate import (
    EvaluationDataset,
    build_arg_parser,
    load_model_state_dict,
    normalize_checkpoint_state_dict,
    parse_column_value_filters,
    read_eval_csv,
    write_predictions_csv,
)
from src.data import IMAGE_COLUMN
from src.fusion_models import create_rgb_fft_fusion_model, create_rgb_fft_fusion_transforms
from src.metrics import compute_classification_metrics
from src.rich_utils import print_panel, print_table
from src.utils import write_json


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def load_checkpoint_model(model_path: str | Path, device: torch.device) -> tuple[Any, dict[str, Any], dict[str, int]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' and 'class_to_idx'. Use a fusion training checkpoint.")
    config = checkpoint["config"]
    if config.get("model", {}).get("family") != "rgb_fft_fusion_timm":
        raise ValueError("Fusion evaluator requires a checkpoint with model.family=rgb_fft_fusion_timm.")
    class_to_idx = checkpoint["class_to_idx"]
    config["model"]["pretrained"] = False
    model = create_rgb_fft_fusion_model(config, num_classes=len(class_to_idx))
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    normalized_state_dict = normalize_checkpoint_state_dict(state_dict)
    has_fusion_keys = any(
        key.startswith(("rgb_backbone.", "fft_backbone."))
        for key in normalized_state_dict
    )
    if not has_fusion_keys:
        first_keys = list(state_dict.keys())[:10]
        raise ValueError(
            "This checkpoint does not look like an RGB+FFT fusion checkpoint. "
            "Use scripts/evaluate.py for RGB checkpoints or scripts/evaluate_frequency.py for FFT/DCT checkpoints. "
            f"First checkpoint keys: {first_keys}"
        )
    load_model_state_dict(model, state_dict, model_path, "RGB+FFT fusion")
    model.to(device)
    model.eval()
    return model, config, class_to_idx


@torch.no_grad()
def predict_fusion(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for inputs, paths, labels in dataloader:
        inputs = move_to_device(inputs, device)
        probs = torch.softmax(model(inputs).float(), dim=1).cpu()
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


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Evaluate a trained RGB+FFT late-fusion checkpoint."
    args = parser.parse_args()
    if bool(args.csv) == bool(args.image_path):
        raise ValueError("Pass exactly one of --csv or --image-path.")

    device = torch.device(args.device)
    model, config, class_to_idx = load_checkpoint_model(args.model_path, device)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    _, val_transform, fusion_data_config = create_rgb_fft_fusion_transforms(model, config)

    if args.image_path:
        row = {IMAGE_COLUMN: args.image_path}
        dataset = EvaluationDataset([row], args.dataset_root, val_transform)
        predictions = predict_fusion(model, DataLoader(dataset, batch_size=1), device, class_names)
        pred = predictions[0]
        top_items = sorted(pred["probabilities"].items(), key=lambda item: item[1], reverse=True)[: args.top_k]
        print_panel(
            f"Image: {pred['image_path']}\n"
            "Input domain: RGB + FFT late fusion\n"
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
        predictions = predict_fusion(model, dataloader, device, class_names)
        print_table(
            "Evaluation Input",
            ["domain", "loaded", "kept", "dropped labels", "dropped columns", "labels available"],
            [
                [
                    "rgb_fft_fusion",
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
            metrics["fusion"] = fusion_data_config["fusion"]
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
