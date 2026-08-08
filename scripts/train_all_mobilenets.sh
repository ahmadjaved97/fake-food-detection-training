#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_BASE="${OUTPUT_BASE:-/lustre/mlnvme/data/s63ajave_hpc-datasets/model_outputs}"

for config in \
  configs/models/mobilenetv3_small_100.yaml \
  configs/models/mobilenetv4_conv_small.yaml \
  configs/models/mobilenetv3_large_075.yaml \
  configs/models/mobilenetv3_large_100.yaml \
  configs/models/mobilenetv4_conv_medium.yaml \
  configs/models/mobilenetv4_conv_large.yaml \
  configs/models/mobilenetv4_hybrid_medium.yaml \
  configs/models/mobilenetv4_hybrid_large.yaml
do
  name="$(basename -- "${config}" .yaml)"

  CONFIG="${config}" \
  OUTPUT_DIR="${OUTPUT_BASE}/${name}" \
    "${REPO_ROOT}/scripts/train.sh" "$@"
done
