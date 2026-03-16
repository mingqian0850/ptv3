#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-ptv3-deploy}"
WEIGHT_NAME="${1:-scannet-pt-v3m1-0-base-model_best.pth}"
CUSTOM_ROOT_IN_CONTAINER="${2:-/datasets/custom_scannet}"
TEST_SPLIT="${3:-val}"
SAVE_PATH_IN_CONTAINER="${4:-/workspace/exp/custom/ptv3-2080-custom}"
WEIGHT_IN_CONTAINER="/weights/${WEIGHT_NAME}"

"${ROOT_DIR}/deploy_scripts/start_ptv3.sh"

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
cd /opt/Pointcept

if [ ! -d '${CUSTOM_ROOT_IN_CONTAINER}' ]; then
  echo '[ERROR] Custom data root not found: ${CUSTOM_ROOT_IN_CONTAINER}'
  exit 1
fi

if [ ! -d '${CUSTOM_ROOT_IN_CONTAINER}/${TEST_SPLIT}' ] && [ ! -f '${CUSTOM_ROOT_IN_CONTAINER}/${TEST_SPLIT}' ]; then
  echo '[ERROR] Test split not found as directory or json file: ${CUSTOM_ROOT_IN_CONTAINER}/${TEST_SPLIT}'
  exit 1
fi

if [ ! -f '${WEIGHT_IN_CONTAINER}' ]; then
  echo '[ERROR] Weight not found: ${WEIGHT_IN_CONTAINER}'
  echo '[INFO] Put weight file in deploy_host/weights.'
  exit 1
fi

cp configs/scannet/semseg-pt-v3m1-0-base.py configs/scannet/semseg-pt-v3m1-0-base-custom-2080.py
export PT_DATA_ROOT='${CUSTOM_ROOT_IN_CONTAINER}'
export PT_TEST_SPLIT='${TEST_SPLIT}'
python - <<'PY'
import os
import re
from pathlib import Path

cfg = Path('configs/scannet/semseg-pt-v3m1-0-base-custom-2080.py')
text = cfg.read_text()
text = text.replace('enable_flash=True', 'enable_flash=False')
text = text.replace(
    'enc_patch_size=(1024, 1024, 1024, 1024, 1024)',
    'enc_patch_size=(128, 128, 128, 128, 128)',
)
text = text.replace(
    'dec_patch_size=(1024, 1024, 1024, 1024)',
    'dec_patch_size=(128, 128, 128, 128)',
)
text = re.sub(
    r'data_root\\s*=\\s*\"[^\"]+\"',
    f'data_root = \"{os.environ[\"PT_DATA_ROOT\"]}\"',
    text,
    count=1,
)
text = re.sub(
    r'(data\\s*=\\s*dict\\(.*?test\\s*=\\s*dict\\(.*?split\\s*=\\s*)\"[^\"]+\"',
    r'\\1\"' + os.environ['PT_TEST_SPLIT'] + '\"',
    text,
    count=1,
    flags=re.S,
)
text += '\\n'
text += (
    'data[\"test\"][\"test_cfg\"][\"aug_transform\"] = '
    '[[dict(type=\"RandomRotateTargetAngle\", angle=[0], axis=\"z\", center=[0, 0, 0], p=1)]]\\n'
)
cfg.write_text(text)
print('[OK] Prepared custom config:', cfg)
PY

export PYTHONPATH=./
python tools/test.py \
  --config-file configs/scannet/semseg-pt-v3m1-0-base-custom-2080.py \
  --num-gpus 1 \
  --options save_path=${SAVE_PATH_IN_CONTAINER} weight=${WEIGHT_IN_CONTAINER}
"

echo "[OK] Custom inference finished."
echo "[INFO] Outputs in host path:"
echo "       ${ROOT_DIR}/deploy_host/exp/custom/ptv3-2080-custom"
