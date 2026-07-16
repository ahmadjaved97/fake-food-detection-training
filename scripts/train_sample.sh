#!/bin/bash
#SBATCH --job-name=resnet50_train
#SBATCH --account=EUHPC_R05_218
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s63ajave@uni-bonn.de
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err



mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

# ------------------------------------------------------------------
# Activate environment
# ------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate food_detection

# ------------------------------------------------------------------
# Cache directories
# ------------------------------------------------------------------
export HF_HOME=/leonardo_work/EUHPC_R05_218/ajaved01/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export DIFFUSERS_CACHE="$HF_HOME"
export TMPDIR=/leonardo_work/EUHPC_R05_218/ajaved01/hf_cache/tmp


# ------------------------------------------------------------------
# Move to project directory
# ------------------------------------------------------------------

echo "Starting training at $(date)"

accelerate launch \
    --config_file accelerate_config.yaml \
    src/train.py \
    --config configs/models/resnet50.yaml \
    --train-csv /leonardo_work/EUHPC_R05_218/ajaved01/fake_food_detection/model_csvs/train \
    --val-csv /leonardo_work/EUHPC_R05_218/ajaved01/fake_food_detection/model_csvs/val \
    --dataset-root /leonardo_work/EUHPC_R05_218/ajaved01/fake_food_detection \
    --output-dir outputs/resnet50_run

echo "Training finished at: $(date)"