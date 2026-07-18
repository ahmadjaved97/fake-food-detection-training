#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/models/resnet18.yaml \
  configs/models/resnet34.yaml \
  configs/models/resnet50.yaml \
  configs/models/resnet101.yaml
do
  name="$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train.sh" "$@"
done
