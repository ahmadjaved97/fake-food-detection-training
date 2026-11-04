#!/usr/bin/env python
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import nn
from torch.utils.data import DataLoader

from src.checkpointing import load_checkpoint, save_checkpoint
from src.config import apply_cli_overrides, build_arg_parser, load_config
from src.data import (
    ImageCsvDataset,
    apply_configured_row_filters,
    class_counts,
    infer_class_to_idx,
    make_class_weights,
    make_weighted_sampler,
    read_classification_csv,
)
from src.frequency_transforms import create_frequency_transforms
from src.logging_utils import RunLoggers
from src.metrics import compute_classification_metrics
from src.models import create_model
from src.optim import create_optimizer
from src.rich_utils import print_panel, print_table
from src.scheduler import create_cosine_scheduler
from src.utils import (
    append_jsonl,
    ensure_dir,
    get_env_info,
    seed_everything,
    setup_logging,
    timestamp,
    write_json,
    write_yaml,
)


LOGGER = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce = -log_pt
        if self.weight is not None:
            ce = ce * self.weight.to(logits.device).gather(0, targets)
        loss = (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch[0], batch[1]
    return images, labels


def batch_paths(batch: Any) -> list[str] | None:
    if len(batch) < 3:
        return None
    return [str(path) for path in batch[2]]


def build_criteria(
    config: dict[str, Any],
    class_weight_tensor: torch.Tensor | None,
    mixup_enabled: bool,
) -> tuple[nn.Module, nn.Module]:
    aug_cfg = config["augmentations"]
    imbalance_cfg = config["imbalance"]
    label_smoothing = float(aug_cfg.get("label_smoothing") or 0.0)

    if mixup_enabled:
        if imbalance_cfg["class_weights"]["enabled"] or imbalance_cfg["focal_loss"]["enabled"]:
            raise ValueError("Mixup/CutMix cannot currently be combined with class weights or focal loss.")
        train_criterion = SoftTargetCrossEntropy()
    elif imbalance_cfg["focal_loss"]["enabled"]:
        train_criterion = FocalLoss(
            gamma=float(imbalance_cfg["focal_loss"].get("gamma", 2.0)),
            weight=class_weight_tensor,
        )
    elif label_smoothing > 0:
        train_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        train_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    val_criterion = nn.CrossEntropyLoss()
    return train_criterion, val_criterion


def validate_effective_batch_size(config: dict[str, Any], accelerator: Accelerator) -> None:
    train_cfg = config["training"]
    actual = (
        int(train_cfg["per_device_batch_size"])
        * accelerator.num_processes
        * int(train_cfg["gradient_accumulation_steps"])
    )
    expected = int(train_cfg.get("effective_batch_size", 256))
    msg = (
        f"Effective batch size is {actual}; expected {expected}. "
        "Formula: per_device_batch_size * num_processes * gradient_accumulation_steps."
    )
    if actual != expected and train_cfg.get("enforce_effective_batch_size", True):
        raise ValueError(msg + " Pass --disable-effective-batch-size-check to allow this run.")
    if actual != expected:
        LOGGER.warning(msg)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
    class_names: list[str],
    enable_auc: bool,
    max_wrong_examples: int = 0,
) -> dict[str, Any]:
    model.eval()
    loss_values: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    local_wrong_examples: list[dict[str, Any]] = []

    for batch in dataloader:
        images, labels = unpack_batch(batch)
        paths = batch_paths(batch)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        if paths is not None and max_wrong_examples > 0 and len(local_wrong_examples) < max_wrong_examples:
            wrong_indices = (preds != labels).nonzero(as_tuple=False).flatten().tolist()
            for idx in wrong_indices:
                pred_idx = int(preds[idx].item())
                true_idx = int(labels[idx].item())
                local_wrong_examples.append(
                    {
                        "image_path": paths[idx],
                        "true_label": class_names[true_idx],
                        "predicted_label": class_names[pred_idx],
                        "confidence": float(probs[idx, pred_idx].item()),
                    }
                )
                if len(local_wrong_examples) >= max_wrong_examples:
                    break

        gathered_probs, gathered_labels = accelerator.gather_for_metrics((probs, labels))
        gathered_loss = accelerator.gather_for_metrics(loss.detach().float().repeat(labels.shape[0]))
        all_probs.append(gathered_probs.cpu())
        all_labels.append(gathered_labels.cpu())
        loss_values.append(gathered_loss.cpu())

    probabilities = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    metrics = compute_classification_metrics(
        labels=labels_np,
        probabilities=probabilities,
        class_names=class_names,
        enable_auc=enable_auc,
    )
    metrics["loss"] = float(torch.cat(loss_values).mean().item())
    if max_wrong_examples > 0:
        gathered_wrong_examples = gather_object(local_wrong_examples)
        wrong_examples: list[dict[str, Any]] = []
        for item in gathered_wrong_examples:
            if isinstance(item, list):
                wrong_examples.extend(item)
            else:
                wrong_examples.append(item)
        metrics["wrong_examples"] = wrong_examples[:max_wrong_examples]
    return metrics


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_config = load_config(args.config)
    config = apply_cli_overrides(raw_config, args)
    frequency_mode = config.get("frequency", {}).get("mode")
    if frequency_mode not in {"fft_magnitude", "dct_map"}:
        raise ValueError("Frequency training requires frequency.mode to be fft_magnitude or dct_map.")

    accelerator = Accelerator(
        mixed_precision=config["training"].get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
    )
    setup_logging(accelerator.is_main_process)
    validate_effective_batch_size(config, accelerator)
    seed_everything(int(config["seed"]) + accelerator.process_index, bool(config.get("deterministic", False)))

    output_dir = ensure_dir(args.output_dir)
    run_name = config.get("run_name") or f"{config['model']['name']}_{frequency_mode}_{timestamp()}"

    label_column = config["data"].get("label_column")
    missing_files_policy = config["data"].get("missing_files_policy", "skip")
    wrong_examples_cfg = config["logging"].get("wrong_examples", {})
    max_wrong_examples = (
        int(wrong_examples_cfg.get("max_images", 16))
        if bool(wrong_examples_cfg.get("enabled", True))
        else 0
    )
    train_rows = read_classification_csv(args.train_csv, label_column=label_column)
    val_rows = read_classification_csv(args.val_csv, label_column=label_column)
    train_rows, train_filter_report = apply_configured_row_filters(train_rows, config["data"], "train")
    val_rows, val_filter_report = apply_configured_row_filters(val_rows, config["data"], "validation")
    class_to_idx = infer_class_to_idx(train_rows, config["data"].get("allowed_classes"))
    num_classes = len(class_to_idx)
    expected_num_classes = config["model"].get("num_classes")
    if expected_num_classes is not None and num_classes != int(expected_num_classes):
        raise ValueError(
            f"Training CSV produced {num_classes} classes but config expects {expected_num_classes}. "
            "Set model.num_classes to null to infer this from the selected label column."
        )
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    if accelerator.is_main_process:
        print_panel(
            f"Model: {config['model']['name']}\n"
            f"Input domain: {frequency_mode}\n"
            f"Label column: {label_column or 'auto-detect class_category/category'}\n"
            f"Classes: {num_classes}\n"
            f"Class labels: {', '.join(class_names)}",
            title="Frequency Training Setup",
        )
        write_json(class_to_idx, output_dir / "class_to_idx.json")
        write_yaml(config, output_dir / "resolved_config.yaml")
        write_json(get_env_info(), output_dir / "env_info.json")

    model = create_model(config, num_classes=num_classes)
    train_transform, val_transform, data_config = create_frequency_transforms(model, config)
    if accelerator.is_main_process:
        write_json(data_config, output_dir / "frequency_data_config.json")

    train_dataset = ImageCsvDataset(
        args.train_csv,
        args.dataset_root,
        class_to_idx,
        transform=train_transform,
        validate_files=bool(config["data"].get("validate_files", True)),
        return_paths=bool(config["data"].get("return_paths", False)),
        split_name="train",
        label_column=label_column,
        missing_files_policy=missing_files_policy,
        missing_files_log_path=output_dir / "missing_files_train.csv"
        if accelerator.is_main_process
        else None,
        rows=train_rows,
        row_filter_report=train_filter_report,
    )
    val_dataset = ImageCsvDataset(
        args.val_csv,
        args.dataset_root,
        class_to_idx,
        transform=val_transform,
        validate_files=bool(config["data"].get("validate_files", True)),
        return_paths=bool(config["data"].get("return_paths", False)) or max_wrong_examples > 0,
        split_name="validation",
        label_column=label_column,
        missing_files_policy=missing_files_policy,
        missing_files_log_path=output_dir / "missing_files_validation.csv"
        if accelerator.is_main_process
        else None,
        rows=val_rows,
        row_filter_report=val_filter_report,
    )

    counts = class_counts(train_dataset.targets, num_classes)
    if accelerator.is_main_process:
        write_json({class_names[i]: int(counts[i]) for i in range(num_classes)}, output_dir / "class_counts.json")
        print_table(
            "Dataset Summary",
            [
                "split",
                "raw rows",
                "label-filtered",
                "metadata-filtered",
                "selected rows",
                "usable images",
                "missing images",
            ],
            [
                [
                    "train",
                    train_dataset.total_rows,
                    train_dataset.filtered_label_rows,
                    train_dataset.filtered_metadata_rows,
                    train_dataset.selected_rows,
                    len(train_dataset),
                    len(train_dataset.missing_rows),
                ],
                [
                    "validation",
                    val_dataset.total_rows,
                    val_dataset.filtered_label_rows,
                    val_dataset.filtered_metadata_rows,
                    val_dataset.selected_rows,
                    len(val_dataset),
                    len(val_dataset.missing_rows),
                ],
            ],
        )
        if train_dataset.missing_rows:
            LOGGER.warning(
                "Skipped %d missing train images. See %s",
                len(train_dataset.missing_rows),
                output_dir / "missing_files_train.csv",
            )
        else:
            LOGGER.info("Skipped 0 missing train images.")
        if val_dataset.missing_rows:
            LOGGER.warning(
                "Skipped %d missing validation images. See %s",
                len(val_dataset.missing_rows),
                output_dir / "missing_files_validation.csv",
            )
        else:
            LOGGER.info("Skipped 0 missing validation images.")
        zero_count_classes = [class_names[i] for i, count in enumerate(counts.tolist()) if count == 0]
        if zero_count_classes:
            LOGGER.warning("Classes with zero usable training images after filtering: %s", zero_count_classes)

    sampler = None
    if config["imbalance"]["weighted_sampler"]["enabled"]:
        sampler_seed = int(config["imbalance"]["weighted_sampler"].get("seed") or config["seed"])
        LOGGER.info(
            "Weighted sampler is enabled with shared sampler seed %d. "
            "All ranks build the same weighted stream and Accelerate shards it.",
            sampler_seed,
        )
        sampler = make_weighted_sampler(
            train_dataset.targets,
            counts,
            replacement=bool(config["imbalance"]["weighted_sampler"].get("replacement", True)),
            seed=sampler_seed,
        )

    batch_size = int(config["training"]["per_device_batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        drop_last=False,
    )

    class_weight_tensor = None
    if config["imbalance"]["class_weights"]["enabled"]:
        class_weight_tensor = make_class_weights(
            counts,
            mode=config["imbalance"]["class_weights"].get("mode", "inverse_frequency"),
        ).to(accelerator.device)

    aug_cfg = config["augmentations"]
    mixup_enabled = float(aug_cfg.get("mixup_alpha") or 0.0) > 0 or float(aug_cfg.get("cutmix_alpha") or 0.0) > 0
    mixup_fn = None
    if mixup_enabled:
        mixup_fn = Mixup(
            mixup_alpha=float(aug_cfg.get("mixup_alpha") or 0.0),
            cutmix_alpha=float(aug_cfg.get("cutmix_alpha") or 0.0),
            label_smoothing=float(aug_cfg.get("label_smoothing") or 0.0),
            num_classes=num_classes,
        )
    train_criterion, val_criterion = build_criteria(config, class_weight_tensor, mixup_enabled)

    if config["training"].get("compile", False):
        model = torch.compile(model)

    optimizer = create_optimizer(model, config)
    local_batches = math.ceil(len(train_dataset) / (batch_size * accelerator.num_processes))
    update_steps_per_epoch = max(
        1,
        math.ceil(local_batches / int(config["training"]["gradient_accumulation_steps"])),
    )
    scheduler = create_cosine_scheduler(optimizer, config, update_steps_per_epoch)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    start_epoch = 0
    global_step = 0
    monitor = config["checkpoint"].get("monitor", "accuracy")
    monitor_mode = config["checkpoint"].get("mode", "max")
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"checkpoint.mode must be 'max' or 'min', got: {monitor_mode}")
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    if args.resume:
        start_epoch, global_step, best_metric = load_checkpoint(
            args.resume,
            accelerator,
            model,
            optimizer,
            scheduler,
            expected_class_to_idx=class_to_idx,
        )
        accelerator.print(f"Resumed from {args.resume} at epoch {start_epoch}")

    run_loggers = RunLoggers(config, output_dir, run_name)
    if accelerator.is_main_process:
        run_loggers.start()

    epochs = int(config["training"]["epochs"])
    max_grad_norm = float(config["training"].get("max_grad_norm") or 0.0)

    for epoch in range(start_epoch, epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        train_loss_values: list[torch.Tensor] = []
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images, labels = unpack_batch(batch)
            with accelerator.accumulate(model):
                if mixup_fn is not None:
                    images, targets = mixup_fn(images, labels)
                    logits = model(images)
                    loss = train_criterion(logits, targets)
                else:
                    logits = model(images)
                    loss = train_criterion(logits, labels)

                train_preds = logits.detach().argmax(dim=1)
                gathered_train_correct = accelerator.gather_for_metrics((train_preds == labels).long())
                train_correct += int(gathered_train_correct.sum().item())
                train_total += int(gathered_train_correct.numel())

                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)

            gathered_loss = accelerator.gather_for_metrics(loss.detach().float().repeat(labels.shape[0]))
            train_loss_values.append(gathered_loss.cpu())

        epoch_time = time.time() - epoch_start
        train_loss = float(torch.cat(train_loss_values).mean().item())
        val_metrics = evaluate(
            model,
            val_loader,
            val_criterion,
            accelerator,
            class_names,
            enable_auc=bool(config["metrics"].get("enable_auc", False)),
            max_wrong_examples=max_wrong_examples,
        )

        train_metrics = {
            "loss": train_loss,
            "accuracy": float(train_correct / max(1, train_total)),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_seconds": epoch_time,
            "throughput_images_per_second": float(len(train_dataset) / max(epoch_time, 1e-9)),
        }

        if accelerator.is_main_process:
            run_loggers.log(train_metrics, step=epoch + 1, prefix="train")
            run_loggers.log(val_metrics, step=epoch + 1, prefix="validation")
            run_loggers.log_epoch_comparisons(train_metrics, val_metrics, step=epoch + 1)
            if "confusion_matrix" in val_metrics:
                run_loggers.log_confusion_matrix(val_metrics["confusion_matrix"], class_names, step=epoch + 1)
            run_loggers.log_wrong_examples(val_metrics.get("wrong_examples", []), step=epoch + 1)
            LOGGER.info(
                "epoch=%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f lr=%.6g",
                epoch + 1,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                train_metrics["lr"],
            )
            append_jsonl(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train": train_metrics,
                    "validation": val_metrics,
                },
                output_dir / "metrics.jsonl",
            )

        if monitor not in val_metrics or val_metrics[monitor] is None:
            raise KeyError(f"checkpoint.monitor={monitor!r} is not available in validation metrics.")
        current_metric = float(val_metrics[monitor])
        is_better = current_metric > best_metric if monitor_mode == "max" else current_metric < best_metric
        if accelerator.is_main_process and is_better:
            best_metric = current_metric
            save_checkpoint(
                accelerator,
                output_dir / f"best_{monitor}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_metric,
                class_to_idx,
                config,
            )

        save_epoch = bool(config["checkpoint"].get("save_epoch_checkpoints", False))
        every_n = int(config["checkpoint"].get("save_every_n_epochs", 1))
        if save_epoch and (epoch + 1) % every_n == 0:
            save_checkpoint(
                accelerator,
                output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_metric,
                class_to_idx,
                config,
            )
        accelerator.wait_for_everyone()

    save_checkpoint(
        accelerator,
        output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        epochs - 1,
        global_step,
        best_metric,
        class_to_idx,
        config,
    )
    if accelerator.is_main_process:
        run_loggers.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
