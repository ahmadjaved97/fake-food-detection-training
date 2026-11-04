# Training

Use Accelerate, not raw PyTorch DDP:

```bash
accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config configs/models/resnet50.yaml \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --output-dir outputs/resnet50
```

Recommended 4x A100 default:

```text
per_device_batch_size: 32
gradient_accumulation_steps: 2
num_processes: 4
effective batch size: 256
mixed precision: bf16
```

Run all ResNets:

```bash
TRAIN_CSV=/path/to/metadata/train \
VAL_CSV=/path/to/metadata/val \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_resnets.sh
```

Run all MobileNetV3/V4 configs:

```bash
TRAIN_CSV=/path/to/metadata/train \
VAL_CSV=/path/to/metadata/val \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_mobilenets.sh
```

Run transformer configs, including DeiT, ViT, DINOv2, and Swin:

```bash
TRAIN_CSV=/path/to/metadata/train \
VAL_CSV=/path/to/metadata/val \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_transformers.sh
```

`--train-csv` and `--val-csv` may point to one CSV file or to a directory containing multiple CSVs.
Each CSV must include `image_path` and a label column. Extra columns can be used for split filtering.

Heavy augmentations are disabled by default. Start with the baseline, then try RandomErasing, RandAugment, Mixup, or CutMix only after you have a clean reference run.

Simple augmentations can be controlled in YAML:

```yaml
augmentations:
  horizontal_flip_prob: 0.5
  vertical_flip_prob: 0.0
  rotation_degrees: 10
  color_jitter:
    enabled: false
```

To train on a different label column, such as the generative model name:

```bash
scripts/train.sh --label-column generator_model
```

The model infers classes from the selected and filtered training rows unless `data.allowed_classes` sets an explicit target label subset/order.

To train only selected target labels:

```yaml
data:
  label_column: class_category
  allowed_classes: [real_good, fake_good, fake_bad]
```

To hold out one generator output as a validation domain while still predicting `class_category`:

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

If the held-out value is itself the target label, the validation label must still exist in training. Otherwise the classifier has no output for that class.

Evaluate a saved checkpoint:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50/best_macro_f1.pt \
  --csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --label-column class_category
```

## MetaCLIP 2 CoOp Training

MetaCLIP 2 CoOp prompt tuning uses a separate script and configs:

```bash
accelerate launch --config_file accelerate_config.yaml src/train_metaclip_coop.py \
  --config configs/metaclip2_coop/s16.yaml \
  --train-csv /path/to/metadata/train.csv \
  --val-csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --output-dir outputs/metaclip2_coop_s16
```

Run all configured MetaCLIP 2 CoOp variants:

```bash
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_metaclip2_coop.sh
```

The first CoOp implementation freezes the MetaCLIP 2 image and text encoders and trains only 16 shared learnable context tokens. Prompts use `LABEL_TO_TEXT` natural-language phrases, while metrics still use raw CSV labels.

## Frequency-Domain Training

Train timm models on FFT or DCT frequency images with `src/train_frequency.py`:

```bash
accelerate launch --config_file accelerate_config.yaml src/train_frequency.py \
  --config configs/frequency/resnet50_fft.yaml \
  --train-csv /path/to/metadata/train.csv \
  --val-csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --output-dir outputs/frequency_resnet50_fft
```

Run the starter grid:

```bash
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_frequency.sh
```

Frequency transforms are computed on the fly after resizing each RGB image to the model input size. The first version uses float32 frequency maps as 3-channel images for existing pretrained timm backbones; DCT-token transformers are intentionally left for a later architecture-specific experiment.

## Offline W&B on a Cluster

For compute nodes without internet, set W&B to offline mode in the YAML:

```yaml
logging:
  wandb:
    enabled: true
    mode: offline
```

Train into a shared filesystem path, then run this on a login node that has internet:

```bash
scripts/sync_wandb_loop.sh --watch-dir outputs --interval 300
```

The sync script scans for `wandb/offline-run-*` directories and retries failed syncs on the next interval.
