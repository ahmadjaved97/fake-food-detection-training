#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIGS=(
  configs/fusion/resnet18_rgb_fft_rgb_channels.yaml
  configs/fusion/resnet34_rgb_fft_rgb_channels.yaml
  configs/fusion/resnet50_rgb_fft_rgb_channels.yaml
  configs/fusion/vit_small_rgb_fft_rgb_channels.yaml
  configs/fusion/vit_base_rgb_fft_rgb_channels.yaml
  configs/fusion/vit_large_rgb_fft_rgb_channels.yaml
  configs/fusion/dinov3_small_rgb_fft_rgb_channels.yaml
  configs/fusion/dinov3_small_plus_rgb_fft_rgb_channels.yaml
  configs/fusion/dinov3_base_rgb_fft_rgb_channels.yaml
  configs/fusion/convnext_tiny_rgb_fft_rgb_channels.yaml
  configs/fusion/convnext_small_rgb_fft_rgb_channels.yaml
  configs/fusion/convnext_large_rgb_fft_rgb_channels.yaml
  configs/fusion/mobilenetv3_large_rgb_fft_rgb_channels.yaml
  configs/fusion/mobilenetv3_small_rgb_fft_rgb_channels.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "Starting RGB+per-channel-FFT fusion training for ${config}"
  CONFIG="${config}" "${REPO_ROOT}/scripts/train_fusion.sh" "$@"
done
