from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn


def checkpoint_payload(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    unwrapped = accelerator.unwrap_model(model)
    payload = {
        "model": accelerator.get_state_dict(unwrapped),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "config": config,
    }
    if accelerator.scaler is not None:
        payload["scaler"] = accelerator.scaler.state_dict()
    return payload


def save_checkpoint(
    accelerator: Accelerator,
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return
    payload = checkpoint_payload(
        accelerator,
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        best_metric,
        class_to_idx,
        config,
    )
    accelerator.save(payload, str(path))


def load_checkpoint(
    checkpoint_path: str | Path,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_class_to_idx: dict[str, int] | None = None,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_class_to_idx is not None and checkpoint.get("class_to_idx") != expected_class_to_idx:
        raise ValueError(
            "Checkpoint class_to_idx does not match the current training CSV. "
            f"Checkpoint: {checkpoint.get('class_to_idx')}; current: {expected_class_to_idx}"
        )
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if accelerator.scaler is not None and "scaler" in checkpoint:
        accelerator.scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    return start_epoch, global_step, best_metric
