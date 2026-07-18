#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/models/deit_tiny.yaml \
  configs/models/deit_small.yaml \
  configs/models/deit_base.yaml \
  configs/models/deit_large.yaml \
  configs/models/vit_tiny.yaml \
  configs/models/vit_small.yaml \
  configs/models/vit_base.yaml \
  configs/models/vit_large.yaml \
  configs/models/dinov2_tiny.yaml \
  configs/models/dinov2_small.yaml \
  configs/models/dinov2_base.yaml \
  configs/models/dinov2_large.yaml
do
  name="$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train.sh" "$@"
done
