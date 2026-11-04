from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModel, AutoProcessor


LABEL_TO_TEXT = {
    "real_good": "a photo of a real, clean dish of food with no foreign objects",
    "real_bad": (
        "a photo of a real dish of food containing a foreign object such as a hair, "
        "stone, or insect"
    ),
    "fake_good": "a photo of a fake or AI-generated clean dish of food",
    "fake_bad": (
        "a photo of a fake or AI-generated dish of food containing a foreign object "
        "such as a hair, stone, or insect"
    ),
    "manipulated_bad": (
        "a photo of a dish of food digitally manipulated to add a foreign object such "
        "as a hair, stone, or insect"
    ),
}


def validate_label_texts(class_names: list[str]) -> None:
    missing = [class_name for class_name in class_names if class_name not in LABEL_TO_TEXT]
    if missing:
        raise ValueError(
            "MetaCLIP CoOp is configured for class_category labels only. "
            f"Missing LABEL_TO_TEXT entries for CSV categories: {missing}"
        )


def label_texts_for_classes(class_names: list[str]) -> list[str]:
    validate_label_texts(class_names)
    return [LABEL_TO_TEXT[class_name] for class_name in class_names]


def _text_projection(model: nn.Module, pooled_output: torch.Tensor) -> torch.Tensor:
    projection = getattr(model, "text_projection", None)
    if isinstance(projection, nn.Module):
        return projection(pooled_output)
    if isinstance(projection, nn.Parameter) or torch.is_tensor(projection):
        return pooled_output @ projection
    raise ValueError("Could not find MetaCLIP text projection layer.")


def _image_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        outputs = model.get_image_features(pixel_values=pixel_values)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs

    vision_model = getattr(model, "vision_model", None)
    visual_projection = getattr(model, "visual_projection", None)
    if vision_model is None or visual_projection is None:
        raise ValueError("Could not find MetaCLIP vision encoder/projection.")
    outputs = vision_model(pixel_values=pixel_values, return_dict=True)
    pooled_output = outputs.pooler_output
    if isinstance(visual_projection, nn.Module):
        return visual_projection(pooled_output)
    return pooled_output @ visual_projection


class MetaClipImageTransform:
    def __init__(self, processor: Any, augment_cfg: dict[str, Any], is_training: bool) -> None:
        self.processor = processor
        self.is_training = is_training
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

    def __call__(self, image: Any) -> torch.Tensor:
        if self.preprocess is not None:
            image = self.preprocess(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)


class MetaClipCoOpModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        class_names: list[str],
        n_ctx: int = 16,
        context_std: float = 0.02,
    ) -> None:
        super().__init__()
        validate_label_texts(class_names)
        self.model_name = model_name
        self.class_names = class_names
        self.n_ctx = int(n_ctx)
        self.clip_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        for param in self.clip_model.parameters():
            param.requires_grad = False

        token_embedding = self.clip_model.text_model.embeddings.token_embedding
        embedding_dim = token_embedding.embedding_dim
        self.context = nn.Parameter(torch.empty(self.n_ctx, embedding_dim))
        nn.init.normal_(self.context, std=float(context_std))

        phrase_inputs = self._tokenize_label_texts(label_texts_for_classes(class_names))
        self.register_buffer("phrase_input_ids", phrase_inputs["input_ids"], persistent=False)
        self.register_buffer("phrase_attention_mask", phrase_inputs["attention_mask"], persistent=False)
        self.register_buffer(
            "phrase_eos_positions",
            phrase_inputs["attention_mask"].sum(dim=1).long() - 1,
            persistent=False,
        )

    def _context_length(self) -> int:
        text_config = getattr(self.clip_model.config, "text_config", None)
        context_length = getattr(text_config, "max_position_embeddings", None)
        if context_length is not None:
            return int(context_length)
        model_max_length = getattr(self.tokenizer, "model_max_length", 77)
        return int(min(model_max_length, 77 if model_max_length > 10000 else model_max_length))

    def _tokenize_label_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        max_phrase_length = self._context_length() - self.n_ctx
        if max_phrase_length < 4:
            raise ValueError(
                f"n_ctx={self.n_ctx} leaves only {max_phrase_length} text positions for class phrases."
            )
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_phrase_length,
            return_tensors="pt",
        )
        if torch.any(inputs["attention_mask"].sum(dim=1) >= max_phrase_length):
            raise ValueError(
                "At least one LABEL_TO_TEXT phrase was truncated. "
                "Use fewer context tokens or shorter label phrases."
            )
        return inputs

    def encode_text_prompts(self) -> torch.Tensor:
        token_embedding = self.clip_model.text_model.embeddings.token_embedding
        phrase_embeds = token_embedding(self.phrase_input_ids)
        num_classes = phrase_embeds.shape[0]
        context = self.context.unsqueeze(0).expand(num_classes, -1, -1)

        prompt_embeds = torch.cat(
            [
                phrase_embeds[:, :1, :],
                context,
                phrase_embeds[:, 1:, :],
            ],
            dim=1,
        )
        context_mask = torch.ones(
            num_classes,
            self.n_ctx,
            dtype=self.phrase_attention_mask.dtype,
            device=self.phrase_attention_mask.device,
        )
        attention_mask = torch.cat(
            [
                self.phrase_attention_mask[:, :1],
                context_mask,
                self.phrase_attention_mask[:, 1:],
            ],
            dim=1,
        )
        text_model = self.clip_model.text_model
        hidden_states = text_model.embeddings(inputs_embeds=prompt_embeds)
        encoder_attention_mask = self._create_text_attention_mask(hidden_states, attention_mask)
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            is_causal=True,
        )
        last_hidden_state = text_model.final_layer_norm(encoder_outputs.last_hidden_state)
        eos_positions = self.phrase_eos_positions + self.n_ctx
        pooled_output = last_hidden_state[
            torch.arange(num_classes, device=eos_positions.device),
            eos_positions,
        ]
        text_features = _text_projection(self.clip_model, pooled_output)
        return torch.nn.functional.normalize(text_features, dim=-1)

    def _create_text_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from transformers.masking_utils import create_causal_mask

            return create_causal_mask(
                config=self.clip_model.text_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
            )
        except Exception:
            batch_size, seq_length = attention_mask.shape
            causal_mask = torch.full(
                (seq_length, seq_length),
                torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            padding_mask = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(
                hidden_states.dtype
            ).min
            return causal_mask + padding_mask

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = _image_features(self.clip_model, pixel_values)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = self.encode_text_prompts()
        logit_scale = self.clip_model.logit_scale.exp() if hasattr(self.clip_model, "logit_scale") else 100.0
        return logit_scale * image_features @ text_features.t()

    def prompt_state_dict(self) -> dict[str, torch.Tensor]:
        return {"context": self.context.detach().cpu()}

    def load_prompt_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if "context" not in state_dict:
            raise ValueError("MetaCLIP CoOp checkpoint is missing prompt context weights.")
        self.context.data.copy_(state_dict["context"].to(self.context.device, dtype=self.context.dtype))


def create_metaclip_coop_model(config: dict[str, Any], class_names: list[str]) -> MetaClipCoOpModel:
    model_cfg = config["model"]
    coop_cfg = config.get("coop", {})
    if bool(coop_cfg.get("class_specific_context", False)):
        raise ValueError("MetaCLIP CoOp currently supports only shared context tokens.")
    return MetaClipCoOpModel(
        model_name=model_cfg["name"],
        class_names=class_names,
        n_ctx=int(coop_cfg.get("n_ctx", 16)),
        context_std=float(coop_cfg.get("context_std", 0.02)),
    )


def create_metaclip_transforms(
    processor: Any,
    config: dict[str, Any],
) -> tuple[MetaClipImageTransform, MetaClipImageTransform, dict[str, Any]]:
    train_transform = MetaClipImageTransform(processor, config.get("augmentations", {}), is_training=True)
    val_transform = MetaClipImageTransform(processor, config.get("augmentations", {}), is_training=False)
    image_processor = getattr(processor, "image_processor", None)
    data_config = {}
    if image_processor is not None:
        data_config = {
            "size": getattr(image_processor, "size", None),
            "crop_size": getattr(image_processor, "crop_size", None),
            "image_mean": getattr(image_processor, "image_mean", None),
            "image_std": getattr(image_processor, "image_std", None),
        }
    return train_transform, val_transform, data_config


def save_prompt_checkpoint(
    accelerator: Any,
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
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(
        {
            "checkpoint_type": "metaclip2_coop_prompt",
            "prompt_state": unwrapped.prompt_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "class_to_idx": class_to_idx,
            "class_names": unwrapped.class_names,
            "label_to_text": LABEL_TO_TEXT,
            "base_model_name": unwrapped.model_name,
            "config": config,
        },
        str(path),
    )


def load_prompt_checkpoint(
    checkpoint_path: str | Path,
    accelerator: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_class_to_idx: dict[str, int] | None = None,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "metaclip2_coop_prompt":
        raise ValueError(f"Expected a MetaCLIP CoOp prompt checkpoint, got: {checkpoint.get('checkpoint_type')}")
    if expected_class_to_idx is not None and checkpoint.get("class_to_idx") != expected_class_to_idx:
        raise ValueError(
            "Checkpoint class_to_idx does not match the current training CSV. "
            f"Checkpoint: {checkpoint.get('class_to_idx')}; current: {expected_class_to_idx}"
        )
    accelerator.unwrap_model(model).load_prompt_state_dict(checkpoint["prompt_state"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    return start_epoch, global_step, best_metric
