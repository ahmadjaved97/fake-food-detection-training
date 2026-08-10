#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-configs/frequency/resnet50_fft.yaml}"

TRAIN_CSV="${TRAIN_CSV:-/lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection/model_csvs/train.csv}"
VAL_CSV="${VAL_CSV:-/lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection/model_csvs/val.csv}"
DATASET_ROOT="${DATASET_ROOT:-/lustre/mlnvme/data/s63ajave_hpc-datasets/fake_food_detection}"

OUTPUT_DIR="${OUTPUT_DIR:-/lustre/mlnvme/data/s63ajave_hpc-datasets/outputs/$(basename -- "${CONFIG}" .yaml)}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

accelerate launch \
  --config_file "${REPO_ROOT}/accelerate_config.yaml" \
  "${REPO_ROOT}/src/train_frequency.py" \
  --config "${CONFIG}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

