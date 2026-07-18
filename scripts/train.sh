#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-configs/models/resnet50.yaml}"
TRAIN_CSV="${TRAIN_CSV:?Set TRAIN_CSV=/path/to/train.csv or /path/to/metadata/train}"
VAL_CSV="${VAL_CSV:?Set VAL_CSV=/path/to/val.csv or /path/to/metadata/val}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT=/path/to/image_root}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(basename "${CONFIG}" .yaml)}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

accelerate launch --config_file "${REPO_ROOT}/accelerate_config.yaml" "${REPO_ROOT}/src/train.py" \
  --config "${CONFIG}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
