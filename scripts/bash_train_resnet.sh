#!/bin/bash


mkdir -p logs

LOG_FILE="logs/resnet50_$(date +%Y%m%d_%H%M%S).log"

# Send stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate thesis_training



echo "Starting training at: $(date)"

accelerate launch \
    --config_file ../accelerate_config.yaml \
    ../src/train.py \
    --config ../configs/models/resnet50.yaml \
    --train-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection/model_csvs/train/ \
    --val-csv /lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection/model_csvs/val \
    --dataset-root /lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection/ \
    --output-dir outputs/resnet50_run

echo "Training finished at: $(date)"