#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-ptv3-deploy}"

"${ROOT_DIR}/deploy_scripts/start_ptv3.sh"
exec docker exec -it "${CONTAINER_NAME}" bash
