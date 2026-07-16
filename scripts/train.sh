#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/models/resnet50.yaml}"
TRAIN_CSV="${TRAIN_CSV:?Set TRAIN_CSV=/path/to/train.csv or /path/to/metadata/train}"
VAL_CSV="${VAL_CSV:?Set VAL_CSV=/path/to/val.csv or /path/to/metadata/val}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT=/path/to/image_root}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(basename "${CONFIG}" .yaml)}"

accelerate launch --config_file accelerate_config.yaml src/train.py \
  --config "${CONFIG}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
