#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${1:-custom_scene_0001}"
RESULT_DIR="${2:-${ROOT_DIR}/deploy_host/exp/custom/ptv3-2080-custom/result}"
DATA_ROOT="${3:-${ROOT_DIR}/deploy_host/data/custom_scannet}"
OUTPUT_DIR="${4:-${ROOT_DIR}/deploy_host/exp/vis_custom}"

python3 "${ROOT_DIR}/tools/visualize_compare.py" \
  --data-root "${DATA_ROOT}" \
  --result-dir "${RESULT_DIR}" \
  --scene "${SCENE}" \
  --output-dir "${OUTPUT_DIR}"
