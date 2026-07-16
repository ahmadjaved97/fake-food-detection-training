from __future__ import annotations

from typing import Any

import torch
from torch import nn


def parameter_groups(
    model: nn.Module,
    weight_decay: float,
    no_weight_decay_for_bias_norm: bool = True,
) -> list[dict[str, Any]]:
    if not no_weight_decay_for_bias_norm:
        return [{"params": model.parameters(), "weight_decay": weight_decay}]

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config["optimizer"]
    return torch.optim.AdamW(
        parameter_groups(
            model,
            weight_decay=float(opt_cfg["weight_decay"]),
            no_weight_decay_for_bias_norm=bool(opt_cfg.get("no_weight_decay_for_bias_norm", True)),
        ),
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
