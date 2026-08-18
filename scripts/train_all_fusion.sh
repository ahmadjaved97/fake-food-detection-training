#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIGS=(
  configs/fusion/resnet18_rgb_fft.yaml
  configs/fusion/resnet34_rgb_fft.yaml
  configs/fusion/resnet50_rgb_fft.yaml
  configs/fusion/vit_small_rgb_fft.yaml
  configs/fusion/dinov3_small_rgb_fft.yaml
  configs/fusion/convnext_tiny_rgb_fft.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "Starting RGB+FFT fusion training for ${config}"
  CONFIG="${config}" "${REPO_ROOT}/scripts/train_fusion.sh" "$@"
done
