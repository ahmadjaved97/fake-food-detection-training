#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/metaclip2_ft/fusion/s16_rgb_fft.yaml \
  configs/metaclip2_ft/fusion/s16_rgb_fft_rgb_channels.yaml \
  configs/metaclip2_ft/fusion/b16_rgb_fft.yaml \
  configs/metaclip2_ft/fusion/b16_rgb_fft_rgb_channels.yaml \
  configs/metaclip2_ft/fusion/m16_rgb_fft.yaml \
  configs/metaclip2_ft/fusion/m16_rgb_fft_rgb_channels.yaml
do
  name="metaclip2_ft_fusion_$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train_metaclip2_ft.sh" "$@"
done
