from __future__ import annotations

import math
from typing import Any

import torch


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    epochs = int(config["training"]["epochs"])
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    sched_cfg = config["scheduler"]
    if sched_cfg.get("warmup_ratio") is not None:
        warmup_steps = int(total_steps * float(sched_cfg["warmup_ratio"]))
    else:
        warmup_steps = int(float(sched_cfg.get("warmup_epochs", 0)) * max(1, steps_per_epoch))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
