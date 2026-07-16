#!/usr/bin/env bash
set -euo pipefail

for config in \
  configs/models/resnet18.yaml \
  configs/models/resnet34.yaml \
  configs/models/resnet50.yaml \
  configs/models/resnet101.yaml
do
  name="$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" scripts/train.sh "$@"
done
