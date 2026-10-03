#!/bin/bash

set -e

CONFIGS=(
    "resnet18"
    "resnet34"
    "resnet50"
    "mobilenetv3_large_100"
    "mobilenetv3_small_100"
    "convnext_base"
    "convnext_small"
    "convnext_tiny"
    "dinov3_base"
    "dinov3_small"
    "dinov3_small_plus"
    "vit_base"
    "vit_small"
    "vit_large"
)

for NAME in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Training: $NAME"
    echo "========================================"

    accelerate launch --config_file accelerate_config.yaml src/train.py \
      --config "configs/models/${NAME}.yaml" \
      --train-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/csv_files/train.csv \
      --val-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/csv_files/val.csv \
      --dataset-root /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/normalized/fake_food_dataset \
      --output-dir "/lustre/mlnvme/data/s63ajave_hpc-datasets/rgb_models_10epoch/${NAME}_output"

    echo "Finished: $NAME"
done

echo "All training runs completed."