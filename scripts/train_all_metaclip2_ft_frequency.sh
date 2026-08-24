#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/metaclip2_ft/frequency/s16_fft.yaml \
  configs/metaclip2_ft/frequency/s16_fft_rgb_channels.yaml \
  configs/metaclip2_ft/frequency/s16_dct.yaml \
  configs/metaclip2_ft/frequency/b16_fft.yaml \
  configs/metaclip2_ft/frequency/b16_fft_rgb_channels.yaml \
  configs/metaclip2_ft/frequency/b16_dct.yaml \
  configs/metaclip2_ft/frequency/m16_fft.yaml \
  configs/metaclip2_ft/frequency/m16_fft_rgb_channels.yaml \
  configs/metaclip2_ft/frequency/m16_dct.yaml
do
  name="metaclip2_ft_frequency_$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train_metaclip2_ft.sh" "$@"
done
