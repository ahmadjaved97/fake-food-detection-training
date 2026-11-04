#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_BASE="${OUTPUT_BASE:-/lustre/mlnvme/data/s63ajave_hpc-datasets/model_outputs}"

for config in \
  configs/models/resnet101.yaml
do
  name="$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE}/${name}" "${REPO_ROOT}/scripts/train.sh" "$@"
done
