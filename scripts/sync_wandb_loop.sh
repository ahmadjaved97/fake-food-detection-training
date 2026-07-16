#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_wandb_loop.sh --watch-dir /shared/outputs [options]

Sync W&B offline runs from a login node at regular intervals. Point --watch-dir
at the shared output directory that contains run folders with wandb/offline-run-*.

Options:
  --watch-dir PATH    Shared output directory to scan. Required.
  --interval SECONDS  Seconds between sync attempts. Default: 300.
  --once              Run one sync pass and exit.
  --dry-run           Print runs that would be synced without calling wandb.
  -h, --help          Show this help.

Example:
  scripts/sync_wandb_loop.sh --watch-dir outputs --interval 300
EOF
}

WATCH_DIR=""
INTERVAL=300
ONCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir)
      WATCH_DIR="${2:?--watch-dir requires a path}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:?--interval requires seconds}"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${WATCH_DIR}" ]]; then
  echo "--watch-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${WATCH_DIR}" ]]; then
  echo "Watch directory does not exist: ${WATCH_DIR}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" -eq 0 ]] && ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI is not available. Install wandb on the login node first." >&2
  exit 127
fi

sync_once() {
  local found=0
  while IFS= read -r -d '' run_dir; do
    found=1
    echo "[$(date --iso-8601=seconds)] Syncing ${run_dir}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      continue
    fi
    if ! wandb sync "${run_dir}"; then
      echo "[$(date --iso-8601=seconds)] Warning: failed to sync ${run_dir}; will retry later" >&2
    fi
  done < <(find "${WATCH_DIR}" -type d -path '*/wandb/offline-run-*' -print0 2>/dev/null | sort -z)

  if [[ "${found}" -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] No W&B offline runs found under ${WATCH_DIR}"
  fi
}

while true; do
  sync_once
  if [[ "${ONCE}" -eq 1 ]]; then
    exit 0
  fi
  sleep "${INTERVAL}"
done
