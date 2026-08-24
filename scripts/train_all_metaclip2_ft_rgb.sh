#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/metaclip2_ft/rgb/s16.yaml \
  configs/metaclip2_ft/rgb/b16.yaml \
  configs/metaclip2_ft/rgb/m16.yaml
do
  name="metaclip2_ft_rgb_$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train_metaclip2_ft.sh" "$@"
done
