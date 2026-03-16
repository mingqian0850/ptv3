#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-ptv3:2080}"
CONTAINER_NAME="${CONTAINER_NAME:-ptv3-deploy}"
HOST_DATA_DIR="${HOST_DATA_DIR:-${ROOT_DIR}/deploy_host/data}"
HOST_WEIGHT_DIR="${HOST_WEIGHT_DIR:-${ROOT_DIR}/deploy_host/weights}"
HOST_EXP_DIR="${HOST_EXP_DIR:-${ROOT_DIR}/deploy_host/exp}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/deploy_host/logs}"

mkdir -p "${HOST_DATA_DIR}" "${HOST_WEIGHT_DIR}" "${HOST_EXP_DIR}" "${HOST_LOG_DIR}"

RUNNING=0
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  RUNNING=1
  echo "[INFO] Container ${CONTAINER_NAME} is already running."
fi

if [ "${RUNNING}" -eq 0 ] && docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[INFO] Starting existing container ${CONTAINER_NAME} ..."
  docker start "${CONTAINER_NAME}" >/dev/null
elif [ "${RUNNING}" -eq 0 ]; then
  echo "[INFO] Creating container ${CONTAINER_NAME} from image ${IMAGE} ..."
  docker run --gpus all -d \
    --name "${CONTAINER_NAME}" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HOST_DATA_DIR}:/datasets" \
    -v "${HOST_WEIGHT_DIR}:/weights" \
    -v "${HOST_EXP_DIR}:/workspace/exp" \
    -v "${HOST_LOG_DIR}:/workspace/logs" \
    "${IMAGE}" bash -lc "sleep infinity" >/dev/null
fi

docker exec "${CONTAINER_NAME}" bash -lc '
set -e
mkdir -p /opt/Pointcept/data
mkdir -p /workspace/exp /workspace/logs
'

echo "[OK] ${CONTAINER_NAME} is ready."
echo "[INFO] Host data dir:    ${HOST_DATA_DIR}"
echo "[INFO] Host weight dir:  ${HOST_WEIGHT_DIR}"
echo "[INFO] Host output dir:  ${HOST_EXP_DIR}"
