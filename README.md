# Fake Food Detection Training

Production-ready PyTorch image classification training using `timm` models and Hugging Face Accelerate.
The default target task is fake food classification, but labels are configurable through the CSV label column. A common 5-class setup is:

```text
real_good
real_bad
fake_good
fake_bad
manipulated_bad
```

These labels are examples, not hard-coded. The code infers `class_to_idx` from the selected and filtered training CSV label column, validates the validation CSV against that mapping, and saves `class_to_idx.json` in the output directory. If `data.allowed_classes` is set, that list controls both the target label subset and class order.

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
    fake_bad.csv
    manipulated_bad.csv
  val/
    real_good.csv
    real_bad.csv
    fake_good.csv
    fake_bad.csv
    manipulated_bad.csv
```

Each training CSV must include `image_path` and a label column. By default the code auto-detects `class_category`, then `category`. Extra columns are allowed and can be used for split filtering. To train on a different task, such as generative model family, set `data.label_column` or pass `--label-column`.

For example, if your CSV has:

```csv
image_path,generator_model
relative/path/to/image.jpg,flux
relative/path/to/image2.jpg,sdxl
```

then this command trains a model whose output labels are `flux`, `sdxl`, and any other values found in `generator_model`:

```bash
scripts/train.sh --label-column generator_model
```

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

Training skips missing image files by default and logs them to:

```text
outputs/<run>/missing_files_train.csv
outputs/<run>/missing_files_validation.csv
```

To fail fast instead, set:

```yaml
data:
  missing_files_policy: error
```

You can optionally choose which target labels are used in YAML:

```yaml
data:
  allowed_classes:
    - real_good
    - fake_good
    - fake_bad
```

This trains a 3-class model and filters out rows with other target labels from both train and validation. For one-off runs, the same can be passed on the command line:

```bash
accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config configs/models/resnet50.yaml \
  --train-csv /path/to/metadata/train \
  --val-csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --output-dir outputs/resnet50_three_class \
  --allowed-classes real_good fake_good fake_bad
```

For a domain holdout, keep the prediction label fixed and filter using another CSV column:

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

If the held-out value is itself the target label, it still has to appear in training; otherwise the classifier has no output for that class. For true domain holdouts, use `row_filters` on a metadata column while keeping `label_column` as the prediction target.

Example configs:

```text
configs/examples/resnet50_three_class.yaml
configs/examples/resnet50_generator_holdout.yaml
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
- `frequency_data_config.json` for frequency-domain runs
- `env_info.json`
- `best_macro_f1.pt` by default
- `final.pt`
- optional `checkpoint_epoch_XXX.pt`

Checkpoints contain model, optimizer, scheduler, epoch, global step, best metric, config, and class mapping.

MetaCLIP 2 CoOp checkpoints are prompt checkpoints: they save the learned context vectors plus optimizer/scheduler/config/class mapping, and reload the frozen Hugging Face MetaCLIP 2 base model by name.

## Evaluation

Evaluate a checkpoint on a labeled CSV:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50_run/best_macro_f1.pt \
  --csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --label-column class_category \
  --predictions-csv outputs/resnet50_run/eval_predictions.csv \
  --output-json outputs/resnet50_run/eval_metrics.json
```

Evaluate only selected labels from a larger CSV:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50_three_class/best_macro_f1.pt \
  --csv /path/to/metadata/val \
  --dataset-root /path/to/image_root \
  --label-column class_category \
  --include-labels real_good fake_good fake_bad
```

Evaluate in-distribution rows by excluding held-out generator models:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50_run/best_macro_f1.pt \
  --csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --label-column class_category \
  --exclude-column-values generator_model gemini gpt \
  --output-json outputs/resnet50_run/eval_id_metrics.json
```

Evaluate out-of-distribution rows by keeping only held-out generator models:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50_run/best_macro_f1.pt \
  --csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --label-column class_category \
  --include-column-values generator_model gemini gpt \
  --output-json outputs/resnet50_run/eval_ood_metrics.json
```

Column filters use exact CSV values and can be repeated for multiple metadata columns.

Classify one image:

```bash
python3 scripts/evaluate.py \
  --model-path outputs/resnet50_run/best_macro_f1.pt \
  --image-path /path/to/image.jpg
```

## Imbalanced Data

The default loss is standard cross entropy, which is the safest baseline. The config includes disabled hooks for:

- class weights
- weighted sampler
- focal loss

Because `real_bad` is the smallest class, watch macro F1 and per-class recall, not only accuracy. Enable imbalance strategies one at a time and compare validation macro F1 and minority-class recall.

When `imbalance.weighted_sampler.enabled` is true in multi-GPU training, all ranks use the same sampler seed and Accelerate shards that shared weighted stream. This avoids each GPU independently drawing overlapping weighted samples in the same epoch.

## Models

Configs live in `configs/models/`. Check available pretrained timm names with:

```bash
python scripts/list_timm_models.py "*dinov2*" "deit*" "mobilenetv4*" "swin*"
```

See [docs/MODELS.md](docs/MODELS.md) for mappings and caveats.

New configs include MobileNetV4 and Swin Transformer variants:

```text
configs/models/mobilenetv4_conv_small.yaml
configs/models/mobilenetv4_conv_medium.yaml
configs/models/mobilenetv4_conv_large.yaml
configs/models/mobilenetv4_hybrid_medium.yaml
configs/models/mobilenetv4_hybrid_large.yaml
configs/models/swin_tiny.yaml
configs/models/swin_small.yaml
configs/models/swin_base.yaml
configs/models/swin_base_22k.yaml
configs/models/swin_large_22k.yaml
```

## MetaCLIP 2 CoOp

MetaCLIP 2 CoOp prompt tuning uses a separate training entrypoint and configs so it does not mix with the normal `timm` classifier path.

Available configs:

```text
configs/metaclip2_coop/s16.yaml
configs/metaclip2_coop/m16.yaml
configs/metaclip2_coop/b16.yaml
```

Run one MetaCLIP 2 CoOp model:

```bash
accelerate launch --config_file accelerate_config.yaml src/train_metaclip_coop.py \
  --config configs/metaclip2_coop/s16.yaml \
  --train-csv /path/to/metadata/train.csv \
  --val-csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --output-dir outputs/metaclip2_coop_s16
```

Or run all three:

```bash
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_metaclip2_coop.sh
```

The CLIP path only supports `class_category` labels matching `LABEL_TO_TEXT`: `real_good`, `real_bad`, `fake_good`, `fake_bad`, and `manipulated_bad`. It keeps `class_to_idx` based on raw CSV labels for metrics, but builds prompts from natural-language phrases.

## Frequency-Domain Training

Frequency-domain training uses a separate timm-only entrypoint and configs. RGB images are converted on the fly into either FFT magnitude maps or block-DCT coefficient maps, then replicated to 3 channels for pretrained backbones.

Available starter configs:

```text
configs/frequency/resnet18_fft.yaml
configs/frequency/resnet18_dct.yaml
configs/frequency/resnet34_fft.yaml
configs/frequency/resnet34_dct.yaml
configs/frequency/resnet50_fft.yaml
configs/frequency/resnet50_dct.yaml
configs/frequency/mobilenetv3_small_100_fft.yaml
configs/frequency/mobilenetv3_small_100_dct.yaml
configs/frequency/mobilenetv3_large_075_fft.yaml
configs/frequency/mobilenetv3_large_075_dct.yaml
configs/frequency/mobilenetv3_large_100_fft.yaml
configs/frequency/mobilenetv3_large_100_dct.yaml
configs/frequency/convnext_tiny_fft.yaml
configs/frequency/convnext_tiny_dct.yaml
configs/frequency/swin_tiny_fft.yaml
configs/frequency/swin_tiny_dct.yaml
configs/frequency/vit_tiny_fft.yaml
configs/frequency/vit_tiny_dct.yaml
configs/frequency/vit_small_fft.yaml
configs/frequency/vit_small_dct.yaml
configs/frequency/vit_base_fft.yaml
configs/frequency/vit_base_dct.yaml
configs/frequency/vit_large_fft.yaml
configs/frequency/vit_large_dct.yaml
configs/frequency/dinov3_small_fft.yaml
configs/frequency/dinov3_small_dct.yaml
configs/frequency/dinov3_small_plus_fft.yaml
configs/frequency/dinov3_small_plus_dct.yaml
configs/frequency/dinov3_base_fft.yaml
configs/frequency/dinov3_base_dct.yaml
configs/frequency/dinov3_large_fft.yaml
configs/frequency/dinov3_large_dct.yaml
```

Run one frequency model:

```bash
accelerate launch --config_file accelerate_config.yaml src/train_frequency.py \
  --config configs/frequency/resnet50_fft.yaml \
  --train-csv /path/to/metadata/train.csv \
  --val-csv /path/to/metadata/val.csv \
  --dataset-root /path/to/image_root \
  --output-dir outputs/frequency_resnet50_fft
```

Or run the starter grid:

```bash
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_frequency.sh
```

The first version uses frequency images only. It does not replace ViT/Swin patch embedding with DCT tokens yet. Images are resized before FFT/DCT, frequency maps remain float32, and DCT is computed with vectorized 8x8 blocks.

To compute fixed normalization bounds from training images:

```bash
python3 scripts/compute_frequency_stats.py \
  --csv /path/to/metadata/train.csv \
  --dataset-root /path/to/image_root \
  --mode fft_magnitude \
  --include-labels real_good real_bad \
  --output-json outputs/frequency_fft_real_stats.json
```

Then set:

```yaml
frequency:
  normalization:
    mode: fixed_minmax
    stats_path: outputs/frequency_fft_real_stats.json
```

## RGB + FFT Fusion Training

Fusion training uses a separate entrypoint and configs. It creates two independent copies of the same timm backbone: one branch receives deterministic RGB tensors, the other receives the existing FFT magnitude tensors. Their pooled feature vectors are concatenated and passed through a small MLP classification head.

Available fusion configs:

```text
configs/fusion/resnet18_rgb_fft.yaml
configs/fusion/resnet34_rgb_fft.yaml
configs/fusion/resnet50_rgb_fft.yaml
configs/fusion/vit_small_rgb_fft.yaml
configs/fusion/dinov3_small_rgb_fft.yaml
configs/fusion/convnext_tiny_rgb_fft.yaml
```

Run one fusion model:

```bash
CONFIG=configs/fusion/resnet50_rgb_fft.yaml \
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_fusion.sh
```

Run all fusion configs:

```bash
TRAIN_CSV=/path/to/metadata/train.csv \
VAL_CSV=/path/to/metadata/val.csv \
DATASET_ROOT=/path/to/image_root \
scripts/train_all_fusion.sh
```

The first fusion baseline disables augmentation deliberately. The RGB and FFT branches should describe the same image evidence; random crop, rotation, interpolation, blur, color jitter, Mixup, or CutMix can change the spectral signal itself. A deterministic resize keeps both branches paired cleanly and makes the result easier to interpret before running augmentation ablations.

## Troubleshooting

- Missing images: run `scripts/validate_csv.py`.
- Unknown validation class: ensure the class appears in the training CSV.
- Batch size error: adjust `--per-device-batch-size` or `--gradient-accumulation-steps`.
- Model name error: verify installed timm names with `scripts/list_timm_models.py`.
- CUDA OOM: lower per-device batch size and increase gradient accumulation to keep effective batch size near 256.
