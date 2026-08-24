from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

from src.frequency_transforms import FrequencyImageTransform


def _hf_kwargs(cache_dir: str | None = None, local_files_only: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": bool(local_files_only)}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return kwargs


def _processor_image_config(processor: Any) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None) or {}
    crop_size = getattr(image_processor, "crop_size", None) or {}
    height = crop_size.get("height") or size.get("height") or size.get("shortest_edge")
    width = crop_size.get("width") or size.get("width") or size.get("shortest_edge")
    if height is None or width is None:
        raise ValueError("Could not infer MetaCLIP-2 image size from the Hugging Face processor.")
    if int(height) != int(width):
        raise ValueError(f"Only square MetaCLIP-2 inputs are supported, got {height}x{width}.")
    return {
        "input_size": (3, int(height), int(width)),
        "mean": tuple(float(value) for value in getattr(image_processor, "image_mean")),
        "std": tuple(float(value) for value in getattr(image_processor, "image_std")),
        "size": size,
        "crop_size": crop_size,
    }


def _vision_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        outputs = model.get_image_features(pixel_values=pixel_values)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs

    vision_model = getattr(model, "vision_model", None)
    visual_projection = getattr(model, "visual_projection", None)
    if vision_model is None:
        raise ValueError("Could not find MetaCLIP-2 vision encoder.")
    outputs = vision_model(pixel_values=pixel_values, return_dict=True)
    pooled_output = outputs.pooler_output
    if visual_projection is None:
        return pooled_output
    if isinstance(visual_projection, nn.Module):
        return visual_projection(pooled_output)
    return pooled_output @ visual_projection


class MetaClip2ImageTransform:
    def __init__(self, processor: Any, augment_cfg: dict[str, Any], is_training: bool) -> None:
        self.processor = processor
        simple_transforms = []
        if is_training:
            horizontal_flip_prob = float(augment_cfg.get("horizontal_flip_prob") or 0.0)
            vertical_flip_prob = float(augment_cfg.get("vertical_flip_prob") or 0.0)
            rotation_degrees = float(augment_cfg.get("rotation_degrees") or 0.0)
            color_jitter_cfg = augment_cfg.get("color_jitter") or {}
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
        self.preprocess = transforms.Compose(simple_transforms) if simple_transforms else None

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.preprocess is not None:
            image = self.preprocess(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)


class MetaClip2FusionTransform:
    def __init__(self, rgb_transform: Any, frequency_transform: Any) -> None:
        self.rgb_transform = rgb_transform
        self.frequency_transform = frequency_transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_transform(image), self.frequency_transform(image)


class MetaClip2Classifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kwargs = _hf_kwargs(cache_dir, local_files_only)
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        self.clip_model = AutoModel.from_pretrained(model_name, **kwargs)
        feature_dim = int(getattr(self.clip_model.config, "projection_dim", 0) or 0)
        if feature_dim <= 0:
            feature_dim = int(getattr(self.clip_model.config.vision_config, "hidden_size"))
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(feature_dim, num_classes))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = _vision_features(self.clip_model, pixel_values)
        return self.classifier(features)


class MetaClip2FusionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        kwargs = _hf_kwargs(cache_dir, local_files_only)
        self.model_name = model_name
        self.rgb_processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        self.rgb_model = AutoModel.from_pretrained(model_name, **kwargs)
        self.fft_model = AutoModel.from_pretrained(model_name, **kwargs)
        feature_dim = int(getattr(self.rgb_model.config, "projection_dim", 0) or 0)
        if feature_dim <= 0:
            feature_dim = int(getattr(self.rgb_model.config.vision_config, "hidden_size"))
        fusion_dim = feature_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), num_classes),
        )

    @property
    def processor(self) -> Any:
        return self.rgb_processor

    def forward(self, inputs: Any) -> torch.Tensor:
        rgb, frequency = inputs
        rgb_features = _vision_features(self.rgb_model, rgb)
        frequency_features = _vision_features(self.fft_model, frequency)
        return self.classifier(torch.cat([rgb_features, frequency_features], dim=1))


def create_metaclip2_ft_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = config["model"]
    ft_cfg = config.get("metaclip2_ft", {})
    input_domain = str(ft_cfg.get("input_domain", "rgb"))
    cache_dir = model_cfg.get("cache_dir")
    local_files_only = bool(model_cfg.get("local_files_only", False))
    if input_domain == "fusion":
        fusion_cfg = config.get("fusion", {})
        model = MetaClip2FusionClassifier(
            model_name=model_cfg["name"],
            num_classes=num_classes,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            hidden_dim=int(fusion_cfg.get("hidden_dim", 512)),
            dropout=float(fusion_cfg.get("dropout", 0.2)),
        )
    else:
        model = MetaClip2Classifier(
            model_name=model_cfg["name"],
            num_classes=num_classes,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            dropout=float(ft_cfg.get("classifier_dropout", 0.0)),
        )
    for param in model.parameters():
        param.requires_grad = True
    return model


def create_metaclip2_ft_transforms(model: nn.Module, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    ft_cfg = config.get("metaclip2_ft", {})
    input_domain = str(ft_cfg.get("input_domain", "rgb"))
    processor = model.processor
    data_config = _processor_image_config(processor)

    if input_domain == "rgb":
        train_transform = MetaClip2ImageTransform(processor, config.get("augmentations", {}), is_training=True)
        val_transform = MetaClip2ImageTransform(processor, config.get("augmentations", {}), is_training=False)
        return train_transform, val_transform, {"input_domain": "rgb", **data_config}

    frequency_cfg = config.get("frequency", {})
    mode = frequency_cfg.get("mode")
    if mode not in {"fft_magnitude", "dct_map"}:
        raise ValueError("MetaCLIP-2 frequency/fusion training requires frequency.mode=fft_magnitude or dct_map.")
    frequency_transform = FrequencyImageTransform(mode, data_config, frequency_cfg)
    frequency_info = {
        "mode": mode,
        "channel_mode": frequency_transform.channel_mode,
        "normalization": frequency_transform.normalization,
    }

    if input_domain == "frequency":
        return frequency_transform, frequency_transform, {
            "input_domain": "frequency",
            **data_config,
            "frequency": frequency_info,
        }
    if input_domain == "fusion":
        if mode != "fft_magnitude":
            raise ValueError("MetaCLIP-2 RGB+frequency fusion currently supports FFT only.")
        rgb_transform = MetaClip2ImageTransform(processor, {}, is_training=False)
        fusion_transform = MetaClip2FusionTransform(rgb_transform, frequency_transform)
        return fusion_transform, fusion_transform, {
            "input_domain": "fusion",
            **data_config,
            "frequency": frequency_info,
        }
    raise ValueError("metaclip2_ft.input_domain must be rgb, frequency, or fusion.")
