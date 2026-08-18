from __future__ import annotations

from typing import Any

import timm
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from torch import nn
from torchvision import transforms

from src.frequency_transforms import FrequencyImageTransform


class RGBFFTFusionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rgb_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fft_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        rgb_dim = self._feature_dim(self.rgb_backbone, "RGB")
        fft_dim = self._feature_dim(self.fft_backbone, "FFT")
        fusion_dim = rgb_dim + fft_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _feature_dim(backbone: nn.Module, branch_name: str) -> int:
        num_features = getattr(backbone, "num_features", None)
        if num_features is None:
            raise ValueError(f"Could not infer {branch_name} backbone feature dimension from timm model.")
        return int(num_features)

    def forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            rgb = inputs["rgb"]
            fft = inputs["fft"]
        else:
            rgb, fft = inputs
        rgb_features = self.rgb_backbone(rgb)
        fft_features = self.fft_backbone(fft)
        features = torch.cat([rgb_features, fft_features], dim=1)
        return self.classifier(features)


class DeterministicRGBTransform:
    def __init__(self, data_config: dict[str, Any]) -> None:
        input_size = data_config.get("input_size", (3, 224, 224))
        image_size = int(input_size[-1])
        mean = tuple(float(value) for value in data_config.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(value) for value in data_config.get("std", (0.229, 0.224, 0.225)))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))


class RGBFFTFusionTransform:
    def __init__(self, rgb_transform: Any, fft_transform: Any) -> None:
        self.rgb_transform = rgb_transform
        self.fft_transform = fft_transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_transform(image), self.fft_transform(image)


def create_rgb_fft_fusion_model(config: dict[str, Any], num_classes: int) -> RGBFFTFusionModel:
    model_cfg = config["model"]
    fusion_cfg = config.get("fusion", {})
    model = RGBFFTFusionModel(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        hidden_dim=int(fusion_cfg.get("hidden_dim", 512)),
        dropout=float(fusion_cfg.get("dropout", 0.2)),
    )
    for param in model.parameters():
        param.requires_grad = True
    return model


def create_rgb_fft_fusion_transforms(
    model: RGBFFTFusionModel,
    config: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    data_config = resolve_model_data_config(model.rgb_backbone)
    frequency_cfg = config.get("frequency", {})
    mode = frequency_cfg.get("mode", "fft_magnitude")
    if mode != "fft_magnitude":
        raise ValueError("RGB+FFT fusion training currently requires frequency.mode=fft_magnitude.")

    rgb_transform = DeterministicRGBTransform(data_config)
    fft_transform = FrequencyImageTransform(mode, data_config, frequency_cfg)
    fusion_transform = RGBFFTFusionTransform(rgb_transform, fft_transform)
    output_config = dict(data_config)
    output_config["fusion"] = {
        "rgb": "deterministic_resize_normalize",
        "frequency": "fft_magnitude",
        "augmentation": "disabled",
    }
    return fusion_transform, fusion_transform, output_config
