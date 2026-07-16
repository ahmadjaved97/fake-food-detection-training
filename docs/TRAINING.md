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

`--train-csv` and `--val-csv` may point to one CSV file or to a directory containing multiple CSVs.
Each CSV must include `image_path` and `class_category`; extra columns are ignored.

Strong augmentations are disabled by default. Start with the baseline, then try RandomErasing, RandAugment, Mixup, or CutMix only after you have a clean reference run.

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
