#!/bin/bash

set -e

CONFIGS=(
    "dinov3_small_fft"
    "dinov3_small_dct"
    "vit_small_fft"
    "vit_small_dct"
)

for NAME in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Training: $NAME"
    echo "========================================"

    accelerate launch --config_file accelerate_config.yaml src/train_frequency.py \
      --config "configs/frequency/${NAME}.yaml" \
      --train-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/csv_files/train.csv \
      --val-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/csv_files/val.csv \
      --dataset-root /lustre/mlnvme/data/s63ajave_hpc-datasets/food_dataset/normalized/fake_food_dataset \
      --output-dir "/lustre/mlnvme/data/s63ajave_hpc-datasets/model_outputs_v2/frequency_${NAME}_output"

    echo "Finished: $NAME"
done

echo "All training runs completed."