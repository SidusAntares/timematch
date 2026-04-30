#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

REMOTE_USER="${REMOTE_USER:-user}"
REMOTE_HOST="${REMOTE_HOST:-10.150.10.38}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/user}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_BASE_DIR}/${PROJECT_NAME}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_DEST_DIR="${LOCAL_DEST_DIR:-${PROJECT_DIR}/server_artifacts/${STAMP}}"
REMOTE_ARCHIVE_NAME="${PROJECT_NAME}_artifacts_${STAMP}.tar.gz"
REMOTE_ARCHIVE_PATH="/tmp/${REMOTE_ARCHIVE_NAME}"
LOCAL_ARCHIVE_PATH="${LOCAL_DEST_DIR}/${REMOTE_ARCHIVE_NAME}"

mkdir -p "$LOCAL_DEST_DIR"

echo "[INFO] Local project: ${PROJECT_DIR}"
echo "[INFO] Remote project: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
echo "[INFO] Local artifact destination: ${LOCAL_DEST_DIR}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    set -euo pipefail
    cd '${REMOTE_PROJECT_DIR}'
    TMP_LIST=\$(mktemp)

    for path in logs outputs result runs; do
        if [ -e \"\$path\" ]; then
            printf '%s\n' \"\$path\" >> \"\$TMP_LIST\"
        fi
    done

    find . -maxdepth 1 -type f \\( -name '*.log' -o -name '*.txt' -o -name '*.csv' -o -name '*.json' \\) -print >> \"\$TMP_LIST\"

    tar \
        --ignore-failed-read \
        --warning=no-file-changed \
        --exclude='outputs/**/*.pt' \
        --exclude='outputs/**/*.pth' \
        --exclude='outputs/**/*.ckpt' \
        --exclude='runs/**/events.out.tfevents*' \
        -czf '${REMOTE_ARCHIVE_PATH}' \
        -T \"\$TMP_LIST\"

    rm -f \"\$TMP_LIST\"
"

scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARCHIVE_PATH}" "${LOCAL_ARCHIVE_PATH}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "rm -f '${REMOTE_ARCHIVE_PATH}'"

tar -xzf "${LOCAL_ARCHIVE_PATH}" -C "${LOCAL_DEST_DIR}"

cat > "${LOCAL_DEST_DIR}/README_pull.txt" <<EOF
Pulled from: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}
Pulled at: ${STAMP}
Contents:
- logs/
- outputs/ (excluding checkpoints)
- result/
- runs/ (excluding tensorboard event files)
- root-level *.log, *.txt, *.csv, *.json
EOF

echo "[SUCCESS] Server artifacts downloaded to:"
echo "  ${LOCAL_DEST_DIR}"
echo "[INFO] Archive kept at:"
echo "  ${LOCAL_ARCHIVE_PATH}"
