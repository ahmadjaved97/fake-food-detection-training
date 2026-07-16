from __future__ import annotations

from typing import Any

import timm
from timm.data import create_transform, resolve_model_data_config
from torch import nn


def create_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = config["model"]
    model = timm.create_model(
        model_cfg["name"],
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_classes=num_classes,
    )
    for param in model.parameters():
        param.requires_grad = True
    return model


def create_transforms(model: nn.Module, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    data_cfg = resolve_model_data_config(model)
    aug_cfg = config.get("augmentations", {})

    train_kwargs = dict(data_cfg)
    train_kwargs.update(
        {
            "is_training": True,
            "re_prob": float(aug_cfg.get("random_erasing_prob") or 0.0),
            "auto_augment": aug_cfg.get("rand_augment") or aug_cfg.get("auto_augment"),
        }
    )
    val_kwargs = dict(data_cfg)
    val_kwargs.update({"is_training": False})

    train_transform = create_transform(**train_kwargs)
    val_transform = create_transform(**val_kwargs)
    return train_transform, val_transform, data_cfg
