#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for config in \
  configs/frequency/resnet18_fft.yaml \
  configs/frequency/resnet18_dct.yaml \
  configs/frequency/resnet34_fft.yaml \
  configs/frequency/resnet34_dct.yaml \
  configs/frequency/resnet50_fft.yaml \
  configs/frequency/resnet50_dct.yaml \
  configs/frequency/mobilenetv3_small_100_fft.yaml \
  configs/frequency/mobilenetv3_small_100_dct.yaml \
  configs/frequency/mobilenetv3_large_075_fft.yaml \
  configs/frequency/mobilenetv3_large_075_dct.yaml \
  configs/frequency/mobilenetv3_large_100_fft.yaml \
  configs/frequency/mobilenetv3_large_100_dct.yaml \
  configs/frequency/convnext_tiny_fft.yaml \
  configs/frequency/convnext_tiny_dct.yaml \
  configs/frequency/swin_tiny_fft.yaml \
  configs/frequency/swin_tiny_dct.yaml \
  configs/frequency/vit_tiny_fft.yaml \
  configs/frequency/vit_tiny_dct.yaml \
  configs/frequency/vit_small_fft.yaml \
  configs/frequency/vit_small_dct.yaml \
  configs/frequency/vit_base_fft.yaml \
  configs/frequency/vit_base_dct.yaml \
  configs/frequency/vit_large_fft.yaml \
  configs/frequency/vit_large_dct.yaml \
  configs/frequency/dinov3_small_fft.yaml \
  configs/frequency/dinov3_small_dct.yaml \
  configs/frequency/dinov3_small_plus_fft.yaml \
  configs/frequency/dinov3_small_plus_dct.yaml \
  configs/frequency/dinov3_base_fft.yaml \
  configs/frequency/dinov3_base_dct.yaml \
  configs/frequency/dinov3_large_fft.yaml \
  configs/frequency/dinov3_large_dct.yaml
do
  name="frequency_$(basename "${config}" .yaml)"
  CONFIG="${config}" OUTPUT_DIR="${OUTPUT_BASE:-outputs}/${name}" "${REPO_ROOT}/scripts/train_frequency.sh" "$@"
done
