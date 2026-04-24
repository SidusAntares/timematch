#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

REMOTE_USER="${REMOTE_USER:-user}"
REMOTE_HOST="${REMOTE_HOST:-10.150.10.38}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/user}"
REMOTE_PROJECT_DIR="${REMOTE_BASE_DIR}/${PROJECT_NAME}"
ARCHIVE_PATH="$(mktemp "/tmp/${PROJECT_NAME}_sync_XXXXXX.tar.gz")"
ARCHIVE_NAME="$(basename "$ARCHIVE_PATH")"

echo "[INFO] Project directory: ${PROJECT_DIR}"
echo "[INFO] Remote target: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"

cd "$(dirname "$PROJECT_DIR")"

tar \
    --exclude="${PROJECT_NAME}/.git" \
    --exclude="${PROJECT_NAME}/.idea" \
    --exclude="${PROJECT_NAME}/__pycache__" \
    --exclude="${PROJECT_NAME}/**/__pycache__" \
    --exclude="${PROJECT_NAME}/outputs" \
    --exclude="${PROJECT_NAME}/runs" \
    --exclude="${PROJECT_NAME}/result" \
    --exclude="${PROJECT_NAME}/*.log" \
    -czf "$ARCHIVE_PATH" \
    "$PROJECT_NAME"

echo "[INFO] Archive created: ${ARCHIVE_PATH}"

scp "$ARCHIVE_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    set -euo pipefail
    mkdir -p '${REMOTE_PROJECT_DIR}'
    tar -xzf '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}' -C '${REMOTE_BASE_DIR}'
    rm -f '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}'
"

rm -f "$ARCHIVE_PATH"

echo "[SUCCESS] Project synced to ${REMOTE_PROJECT_DIR}"
