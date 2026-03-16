#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${ROOT_DIR}/deploy_host/data/my_data}"
OUTPUT_ROOT="${2:-${ROOT_DIR}/deploy_host/data/custom_scannet}"
SPLIT="${3:-val}"
SCENE_PREFIX="${4:-custom_scene}"

python3 "${ROOT_DIR}/tools/prepare_my_data_for_ptv3.py" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --scene-prefix "${SCENE_PREFIX}" \
  --write-dummy-segment
