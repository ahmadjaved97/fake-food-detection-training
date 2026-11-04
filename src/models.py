from __future__ import annotations

from typing import Any

import timm
from timm.data import create_transform, resolve_model_data_config
from torch import nn
from torchvision import transforms


def _is_dinov2_model(model_name: str) -> bool:
    return "dinov2" in model_name.lower()

def _is_dinov3_model(model_name: str) -> bool:
    return "dinov3" in model_name.lower()


def _is_dino_model(model_name: str) -> bool:
    model_name = model_name.lower()
    return "dinov2" in model_name or "dinov3" in model_name

def _classifier_parameter_ids(model: nn.Module) -> set[int]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        return {id(param) for param in classifier.parameters()}

    classifier_ids: set[int] = set()
    for module_name in ("head", "fc", "classifier"):
        module = getattr(model, module_name, None)
        if isinstance(module, nn.Module):
            classifier_ids.update(id(param) for param in module.parameters())
    return classifier_ids


def freeze_backbone_except_classifier(model: nn.Module) -> None:
    classifier_ids = _classifier_parameter_ids(model)
    if not classifier_ids:
        raise ValueError("Could not identify classifier head parameters to keep trainable.")
    for param in model.parameters():
        param.requires_grad = id(param) in classifier_ids


def create_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = config["model"]
    model = timm.create_model(
        model_cfg["name"],
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_classes=num_classes,
    )
    for param in model.parameters():
        param.requires_grad = True
    if bool(model_cfg.get("freeze_backbone", False)):
        if not _is_dino_model(model_cfg["name"]):
            print(model_cfg["name"])
            print("\n\n")
            raise ValueError("model.freeze_backbone is currently supported only for DINOv2 and DINOv3 timm models.")
        freeze_backbone_except_classifier(model)
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
    simple_transforms = []
    horizontal_flip_prob = float(aug_cfg.get("horizontal_flip_prob") or 0.0)
    vertical_flip_prob = float(aug_cfg.get("vertical_flip_prob") or 0.0)
    rotation_degrees = float(aug_cfg.get("rotation_degrees") or 0.0)
    color_jitter_cfg = aug_cfg.get("color_jitter") or {}

    if horizontal_flip_prob > 0:
        simple_transforms.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))
    if vertical_flip_prob > 0:
        simple_transforms.append(transforms.RandomVerticalFlip(p=vertical_flip_prob))
    if rotation_degrees > 0:
        simple_transforms.append(transforms.RandomRotation(degrees=rotation_degrees))
    if color_jitter_cfg.get("enabled", False):
        simple_transforms.append(
            transforms.ColorJitter(
                brightness=float(color_jitter_cfg.get("brightness") or 0.0),
                contrast=float(color_jitter_cfg.get("contrast") or 0.0),
                saturation=float(color_jitter_cfg.get("saturation") or 0.0),
                hue=float(color_jitter_cfg.get("hue") or 0.0),
            )
        )
    if simple_transforms:
        train_transform = transforms.Compose([*simple_transforms, train_transform])
    val_transform = create_transform(**val_kwargs)
    return train_transform, val_transform, data_cfg
