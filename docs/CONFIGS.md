# Configs

`configs/base.yaml` supplies defaults. Model configs in `configs/models/` override only the model-specific fields.

Important fields:

- `data.label_column`: label column to read. `null` auto-detects `class_category`, then `category`.
- `model.name`: timm model name.
- `model.pretrained`: use pretrained weights.
- `training.epochs`: full fine-tuning epochs.
- `training.per_device_batch_size`: batch size per GPU/process.
- `training.gradient_accumulation_steps`: accumulation steps.
- `training.effective_batch_size`: expected global effective batch size.
- `optimizer.lr`: AdamW learning rate.
- `optimizer.weight_decay`: AdamW weight decay.
- `scheduler.warmup_epochs` or `scheduler.warmup_ratio`: warmup length.
- `augmentations.random_erasing_prob`: timm random erasing probability.
- `augmentations.auto_augment` or `augmentations.rand_augment`: timm policy string.
- `augmentations.mixup_alpha`: Mixup alpha, disabled at `0.0`.
- `augmentations.cutmix_alpha`: CutMix alpha, disabled at `0.0`.
- `augmentations.label_smoothing`: label smoothing.
- `imbalance.class_weights.enabled`: inverse-frequency class weighting.
- `imbalance.weighted_sampler.enabled`: weighted sampling.
- `imbalance.focal_loss.enabled`: focal loss.
- `logging.wandb.enabled`: W&B logging.
- `logging.wandb.mode`: set to `offline` on compute nodes without internet.
- `logging.tensorboard.enabled`: TensorBoard logging.
- `checkpoint.save_epoch_checkpoints`: save every configured epoch interval.

CLI overrides are available for epochs, LR, per-device batch size, gradient accumulation, mixed precision, run name, W&B, TensorBoard, and resume path.
