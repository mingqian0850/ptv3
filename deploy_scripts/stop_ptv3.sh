#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ptv3-deploy}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker stop "${CONTAINER_NAME}" >/dev/null
  echo "[OK] Stopped ${CONTAINER_NAME}"
else
  echo "[INFO] ${CONTAINER_NAME} is not running."
fi
