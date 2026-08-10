from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from timm.data import resolve_model_data_config


def resize_rgb_image(image: Image.Image, image_size: int) -> Image.Image:
    return image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)


def rgb_to_luminance(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def percentile_normalize(
    array: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    low, high = np.percentile(array, [lower_percentile, upper_percentile])
    if high <= low:
        low = float(array.min())
        high = float(array.max())
    normalized = (array - low) / max(high - low, eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def fixed_minmax_normalize(array: np.ndarray, min_value: float, max_value: float, eps: float = 1e-8) -> np.ndarray:
    normalized = (array - float(min_value)) / max(float(max_value) - float(min_value), eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def default_fixed_bounds(mode: str, image_size: int, block_size: int) -> tuple[float, float]:
    if mode == "fft_magnitude":
        return 0.0, float(np.log1p(image_size * image_size))
    if mode == "dct_map":
        max_abs = float(np.log1p(block_size))
        return -max_abs, max_abs
    raise ValueError(f"Unsupported frequency mode: {mode}")


def resolve_normalization_config(frequency_cfg: dict[str, Any]) -> dict[str, Any]:
    normalization = dict(frequency_cfg.get("normalization") or {})
    stats_path = normalization.get("stats_path")
    if stats_path:
        with Path(stats_path).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        for key in ("min_value", "max_value", "lower_percentile", "upper_percentile"):
            if key in stats and key not in normalization:
                normalization[key] = stats[key]
    return normalization


def normalize_frequency_map(
    array: np.ndarray,
    mode: str,
    image_size: int,
    block_size: int,
    normalization: dict[str, Any],
) -> np.ndarray:
    normalization_mode = normalization.get("mode", "fixed_minmax")
    if normalization_mode == "percentile":
        return percentile_normalize(
            array,
            lower_percentile=float(normalization.get("lower_percentile", 1.0)),
            upper_percentile=float(normalization.get("upper_percentile", 99.0)),
        )
    if normalization_mode != "fixed_minmax":
        raise ValueError("frequency.normalization.mode must be fixed_minmax or percentile")
    default_min, default_max = default_fixed_bounds(mode, image_size, block_size)
    min_value = float(normalization.get("min_value", default_min))
    max_value = float(normalization.get("max_value", default_max))
    return fixed_minmax_normalize(array, min_value, max_value)


def fft_magnitude_map(luminance: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(luminance)
    magnitude = np.log1p(np.abs(spectrum))
    return np.fft.fftshift(magnitude).astype(np.float32)


def dct_coefficient_map(
    luminance: np.ndarray,
    block_size: int,
) -> np.ndarray:
    try:
        from scipy.fft import dctn
    except ImportError as exc:
        raise ImportError("DCT frequency training requires scipy. Install scipy>=1.10.0.") from exc

    height, width = luminance.shape
    pad_h = (-height) % block_size
    pad_w = (-width) % block_size
    if pad_h or pad_w:
        luminance = np.pad(luminance, ((0, pad_h), (0, pad_w)), mode="reflect")

    padded_height, padded_width = luminance.shape
    blocks = luminance.reshape(
        padded_height // block_size,
        block_size,
        padded_width // block_size,
        block_size,
    )
    coeffs = dctn(blocks, type=2, norm="ortho", axes=(1, 3))
    signed_log = np.sign(coeffs) * np.log1p(np.abs(coeffs))
    out = signed_log.reshape(padded_height, padded_width).astype(np.float32)
    out = out[:height, :width]
    return out


class FrequencyImageTransform:
    def __init__(self, mode: str, data_config: dict[str, Any], frequency_cfg: dict[str, Any]) -> None:
        self.mode = mode
        input_size = data_config.get("input_size", (3, 224, 224))
        self.image_size = int(input_size[-1])
        self.frequency_cfg = frequency_cfg
        self.normalization = resolve_normalization_config(frequency_cfg)
        self.block_size = int(frequency_cfg.get("block_size", 8))
        self.mean = torch.tensor(
            tuple(float(value) for value in data_config.get("mean", (0.485, 0.456, 0.406))),
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.std = torch.tensor(
            tuple(float(value) for value in data_config.get("std", (0.229, 0.224, 0.225))),
            dtype=torch.float32,
        ).view(3, 1, 1)

    def _normalize(self, frequency_map: np.ndarray) -> np.ndarray:
        return normalize_frequency_map(
            frequency_map,
            mode=self.mode,
            image_size=self.image_size,
            block_size=self.block_size,
            normalization=self.normalization,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized_image = resize_rgb_image(image, self.image_size)
        luminance = rgb_to_luminance(resized_image)
        if self.mode == "fft_magnitude":
            frequency_map = fft_magnitude_map(luminance)
        elif self.mode == "dct_map":
            frequency_map = dct_coefficient_map(luminance, block_size=self.block_size)
        else:
            raise ValueError(f"Unsupported frequency mode: {self.mode}")

        frequency_map = self._normalize(frequency_map)
        tensor = torch.from_numpy(frequency_map).unsqueeze(0).repeat(3, 1, 1).float()
        return (tensor - self.mean) / self.std


def create_frequency_transforms(model: Any, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    data_config = resolve_model_data_config(model)
    frequency_cfg = config.get("frequency", {})
    mode = frequency_cfg.get("mode")
    if mode not in {"fft_magnitude", "dct_map"}:
        raise ValueError("frequency.mode must be one of: fft_magnitude, dct_map")
    train_transform = FrequencyImageTransform(mode, data_config, frequency_cfg)
    val_transform = FrequencyImageTransform(mode, data_config, frequency_cfg)
    output_config = dict(data_config)
    normalization = resolve_normalization_config(frequency_cfg)
    normalization_mode = normalization.get("mode", "fixed_minmax")
    default_min, default_max = default_fixed_bounds(
        mode,
        image_size=int(data_config.get("input_size", (3, 224, 224))[-1]),
        block_size=int(frequency_cfg.get("block_size", 8)),
    )
    output_config["frequency"] = {
        "mode": mode,
        "block_size": int(frequency_cfg.get("block_size", 8)),
        "image_size": int(data_config.get("input_size", (3, 224, 224))[-1]),
        "normalization": {
            "mode": normalization_mode,
            "min_value": float(normalization.get("min_value", default_min)),
            "max_value": float(normalization.get("max_value", default_max)),
            "lower_percentile": float(
                normalization.get("lower_percentile", frequency_cfg.get("lower_percentile", 1.0))
            ),
            "upper_percentile": float(
                normalization.get("upper_percentile", frequency_cfg.get("upper_percentile", 99.0))
            ),
        },
    }
    return train_transform, val_transform, output_config
