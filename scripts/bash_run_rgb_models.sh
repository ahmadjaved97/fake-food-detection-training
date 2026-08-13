#!/bin/bash

set -e

CONFIGS=(
    "convnext_base"
    "convnext_small"
    "convnext_tiny"
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
      --output-dir "/lustre/mlnvme/data/s63ajave_hpc-datasets/model_outputs_v2/${NAME}_output"

    echo "Finished: $NAME"
done

echo "All training runs completed."