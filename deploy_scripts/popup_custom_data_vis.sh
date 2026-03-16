#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${1:-custom_scene_0001}"
MODE="${2:-compare_input_pred}"
RESULT_DIR="${3:-${ROOT_DIR}/deploy_host/exp/custom/ptv3-2080-custom/result}"

python3 "${ROOT_DIR}/tools/show_popup.py" \
  --scene "${SCENE}" \
  --mode "${MODE}" \
  --data-root "${ROOT_DIR}/deploy_host/data/custom_scannet" \
  --result-dir "${RESULT_DIR}" \
  --vis-root "${ROOT_DIR}/deploy_host/exp/vis_custom" \
  --auto-generate
