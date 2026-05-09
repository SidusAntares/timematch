#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

REMOTE_USER="${REMOTE_USER:-user}"
REMOTE_HOST="${REMOTE_HOST:-10.150.10.38}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/user}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_BASE_DIR}/${PROJECT_NAME}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
INPLACE_SYNC="${INPLACE_SYNC:-True}"
LOCAL_DEST_DIR="${LOCAL_DEST_DIR:-${PROJECT_DIR}}"
REMOTE_ARCHIVE_NAME="${PROJECT_NAME}_artifacts_${STAMP}.tar.gz"
REMOTE_ARCHIVE_PATH="/tmp/${REMOTE_ARCHIVE_NAME}"
LOCAL_ARCHIVE_PATH="${PROJECT_DIR}/${REMOTE_ARCHIVE_NAME}"

mkdir -p "$PROJECT_DIR"

echo "[INFO] Local project: ${PROJECT_DIR}"
echo "[INFO] Remote project: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
echo "[INFO] Local extract destination: ${LOCAL_DEST_DIR}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    set -euo pipefail
    cd '${REMOTE_PROJECT_DIR}'
    TMP_LIST=\$(mktemp)

    for path in logs; do
        if [ -e \"\$path\" ]; then
            printf '%s\n' \"\$path\" >> \"\$TMP_LIST\"
        fi
    done

    if [ -d result ]; then
        find result -type f -name '*.csv' -print >> \"\$TMP_LIST\"
    fi

    tar \
        --ignore-failed-read \
        --warning=no-file-changed \
        -czf '${REMOTE_ARCHIVE_PATH}' \
        -T \"\$TMP_LIST\"

    rm -f \"\$TMP_LIST\"
"

scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARCHIVE_PATH}" "${LOCAL_ARCHIVE_PATH}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "rm -f '${REMOTE_ARCHIVE_PATH}'"

tar -xzf "${LOCAL_ARCHIVE_PATH}" -C "${LOCAL_DEST_DIR}"
rm -f "${LOCAL_ARCHIVE_PATH}"

if [[ "${INPLACE_SYNC,,}" != "true" ]]; then
    cat > "${LOCAL_DEST_DIR}/README_pull.txt" <<EOF
Pulled from: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}
Pulled at: ${STAMP}
Contents:
- logs/
- result/ (*.csv only)
EOF
fi

echo "[SUCCESS] Server artifacts downloaded to:"
echo "  ${LOCAL_DEST_DIR}"
echo "[INFO] Local archive removed after extraction."
