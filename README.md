# Fake Food Detection Training

Production-ready PyTorch image classification training using `timm` models and Hugging Face Accelerate.
The target task is a 5-class fake food detection dataset:

```text
real_good
real_bad
fake_good
fully_fake_bad
fake_bad_manipulated
```

The code infers `class_to_idx` from the training CSV by sorting category names alphabetically, validates the validation CSV against that mapping, and saves `class_to_idx.json` in the output directory.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

Use either single train/validation CSV files or split directories of CSVs. Your current layout can be:

```text
metadata/
  train/
    real_good.csv
    real_bad.csv
    fake_good.csv
    fully_fake_bad.csv
    fake_bad_manipulated.csv
  val/
    real_good.csv
    real_bad.csv
    fake_good.csv
    fully_fake_bad.csv
    fake_bad_manipulated.csv
```

Each CSV must include `image_path` and a label column named `class_category`. Extra columns are allowed and ignored.
The older label column name `category` is also supported.

```csv
image_path,class_category,source,anything_else
relative/path/to/image.jpg,real_good,my_source,...
relative/path/to/image2.jpg,real_bad,my_source,...
```

`image_path` is resolved relative to `--dataset-root`, which should point at the root where all image folders are available.

Validate before training:

```bash
python3 scripts/validate_csv.py \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root
```

## Train One Model

```bash
accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config configs/models/resnet50.yaml \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --output-dir outputs/resnet50_run
```

The default 4x A100 setup uses:

```text
effective_batch_size = 32 per GPU * 4 GPUs * 2 accumulation steps = 256
```

Override common settings:

```bash
accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config configs/models/vit_base.yaml \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --output-dir outputs/vit_base \
  --epochs 100 \
  --lr 1e-4 \
  --per-device-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --run-name vit_base_experiment
```

If your hardware does not produce an effective batch size of 256, pass:

```bash
--disable-effective-batch-size-check
```

## Logging

TensorBoard and W&B are disabled by default.

```bash
--enable-tensorboard
--enable-wandb
```

W&B login is only required when W&B is enabled.

For compute nodes without internet, run W&B in offline mode:

```yaml
logging:
  wandb:
    enabled: true
    mode: offline
```

Then sync from an internet-connected login node that can see the shared output directory:

```bash
scripts/sync_wandb_loop.sh --watch-dir outputs --interval 300
```

## Resume

```bash
accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config configs/models/resnet50.yaml \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --output-dir outputs/resnet50_run \
  --resume outputs/resnet50_run/checkpoint_epoch_010.pt
```

## Outputs

Each run writes:

- `resolved_config.yaml`
- `class_to_idx.json`
- `class_counts.json`
- `timm_data_config.json`
- `env_info.json`
- `best_accuracy.pt`
- `final.pt`
- optional `checkpoint_epoch_XXX.pt`

Checkpoints contain model, optimizer, scheduler, epoch, global step, best metric, config, and class mapping.

## Imbalanced Data

The default loss is standard cross entropy, which is the safest baseline. The config includes disabled hooks for:

- class weights
- weighted sampler
- focal loss

Because `real_bad` is the smallest class, watch macro F1 and per-class recall, not only accuracy. Enable imbalance strategies one at a time and compare validation macro F1 and minority-class recall.

## Models

Configs live in `configs/models/`. Check available pretrained timm names with:

```bash
python scripts/list_timm_models.py "*dinov2*" "deit*"
```

See [docs/MODELS.md](docs/MODELS.md) for mappings and caveats.

## Troubleshooting

- Missing images: run `scripts/validate_csv.py`.
- Unknown validation class: ensure the class appears in the training CSV.
- Batch size error: adjust `--per-device-batch-size` or `--gradient-accumulation-steps`.
- Model name error: verify installed timm names with `scripts/list_timm_models.py`.
- CUDA OOM: lower per-device batch size and increase gradient accumulation to keep effective batch size near 256.
