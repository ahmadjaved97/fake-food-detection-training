# Configs

`configs/base.yaml` supplies defaults. Model configs in `configs/models/` override only the model-specific fields.

Important fields:

- `data.label_column`: label column to read. `null` auto-detects `class_category`, then `category`.
- `data.missing_files_policy`: `skip` logs missing images and continues; `error` fails fast.
- `data.allowed_classes`: optional target-label allow-list and class order. Rows with other target labels are filtered out. `null` infers labels from filtered training data.
- `data.label_filters`: optional include/exclude filters for values in the selected target label column, separately for `train` and `validation`.
- `data.row_filters`: optional include/exclude filters on any CSV metadata column, separately for `train` and `validation`.
- `model.name`: timm model name.
- `model.family`: optional model family marker. `metaclip2_coop` configs use Hugging Face MetaCLIP 2 instead of `timm`.
- `model.pretrained`: use pretrained weights.
- `model.num_classes`: optional class-count assertion. `null` infers from training labels.
- `model.freeze_backbone`: for DINOv2 configs, freeze the pretrained backbone and train only the classifier head.
- `training.epochs`: full fine-tuning epochs.
- `training.per_device_batch_size`: batch size per GPU/process.
- `training.gradient_accumulation_steps`: accumulation steps.
- `training.effective_batch_size`: expected global effective batch size.
- `optimizer.lr`: AdamW learning rate.
- `optimizer.weight_decay`: AdamW weight decay.
- `scheduler.warmup_epochs` or `scheduler.warmup_ratio`: warmup length.
- `augmentations.random_erasing_prob`: timm random erasing probability.
- `augmentations.horizontal_flip_prob`: simple random horizontal flip probability.
- `augmentations.vertical_flip_prob`: simple random vertical flip probability.
- `augmentations.rotation_degrees`: simple random rotation range in degrees.
- `augmentations.color_jitter`: optional simple brightness/contrast/saturation/hue jitter.
- `augmentations.auto_augment` or `augmentations.rand_augment`: timm policy string.
- `augmentations.mixup_alpha`: Mixup alpha, disabled at `0.0`.
- `augmentations.cutmix_alpha`: CutMix alpha, disabled at `0.0`.
- `augmentations.label_smoothing`: label smoothing.
- `imbalance.class_weights.enabled`: inverse-frequency class weighting.
- `imbalance.weighted_sampler.enabled`: weighted sampling.
- `imbalance.weighted_sampler.seed`: shared sampler seed for coordinated multi-GPU weighted sampling. `null` uses top-level `seed`.
- `imbalance.focal_loss.enabled`: focal loss.
- `logging.wandb.enabled`: W&B logging.
- `logging.wandb.mode`: set to `offline` on compute nodes without internet.
- `logging.tensorboard.enabled`: TensorBoard logging.
- `logging.wrong_examples.enabled`: log wrongly classified validation examples to W&B.
- `logging.wrong_examples.max_images`: maximum wrong examples to log per epoch.
- `checkpoint.monitor`: validation metric used for best checkpoint, default `macro_f1`.
- `checkpoint.mode`: `max` for metrics like F1/accuracy, `min` for loss.
- `checkpoint.save_epoch_checkpoints`: save every configured epoch interval.

CLI overrides are available for epochs, LR, per-device batch size, gradient accumulation, mixed precision, run name, W&B, TensorBoard, label column, allowed classes, split label filters, and resume path.

## Choosing Label Values

To train on only 3 target labels from a 5-label CSV, set `allowed_classes`:

```yaml
data:
  label_column: class_category
  allowed_classes: [real_good, fake_good, fake_bad]
```

The classifier head will have exactly those 3 outputs, in that order. Training and validation rows with other target labels are filtered out.

For split-specific target-label filtering:

```yaml
data:
  label_filters:
    train:
      include: [real_good, fake_good, fake_bad]
      exclude: null
    validation:
      include: [real_good, fake_good, fake_bad]
      exclude: null
```

Validation labels must still exist in the filtered training labels, because a classifier cannot predict a target class it was never given as an output.

To hold out a generator/model domain while predicting another label column, use `row_filters` on a metadata column:

```yaml
data:
  label_column: class_category
  row_filters:
    train:
      exclude:
        generator_model: [FLUX1-dev]
    validation:
      include:
        generator_model: [FLUX1-dev]
```

Example configs are available in `configs/examples/`.

## MetaCLIP 2 CoOp Configs

MetaCLIP 2 CoOp configs live in `configs/metaclip2_coop/` and should be launched with `src/train_metaclip_coop.py`.

Important fields:

- `model.name`: Hugging Face model id, for example `facebook/metaclip-2-worldwide-s16`.
- `coop.n_ctx`: number of shared learnable context tokens. The default CoOp configs use `16`.
- `coop.class_specific_context`: currently `false`; one shared context is used for all classes.
- `coop.context_std`: random initialization standard deviation for the context vectors.

The configs intentionally use `data.label_filters` rather than `data.allowed_classes`, so class indices are still inferred with the existing sorted-label behavior after filtering to the five supported `class_category` values.
