#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_COMMIT="${OLD_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
SMOOTH_COMMIT="${SMOOTH_COMMIT:-89d9df4e52744cb955168b0d203a2ddd61c3199e}"
REMOTE_USER="${REMOTE_USER:?REMOTE_USER is required}"
REMOTE_HOST="${REMOTE_HOST:?REMOTE_HOST is required}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/user}"
REMOTE_OLD_ROOT="${REMOTE_OLD_ROOT:-${REMOTE_BASE_DIR}/timematch_old_da_f04e1e0}"
REMOTE_SOURCE_ROOT="${REMOTE_SOURCE_ROOT:-${REMOTE_BASE_DIR}/timematch_old_source_89d9df4}"
ARCHIVE_PATH="$(mktemp /tmp/timematch_old_da_XXXXXX.tar.gz)"
ARCHIVE_NAME="$(basename "${ARCHIVE_PATH}")"
SOURCE_ARCHIVE_PATH="$(mktemp /tmp/timematch_old_source_XXXXXX.tar.gz)"
SOURCE_ARCHIVE_NAME="$(basename "${SOURCE_ARCHIVE_PATH}")"

git -C "${PROJECT_DIR}" cat-file -e "${OLD_COMMIT}^{commit}"
git -C "${PROJECT_DIR}" cat-file -e "${SMOOTH_COMMIT}^{commit}"
git -C "${PROJECT_DIR}" archive --format=tar "${OLD_COMMIT}" | gzip > "${ARCHIVE_PATH}"
git -C "${PROJECT_DIR}" archive --format=tar "${SMOOTH_COMMIT}" | gzip > "${SOURCE_ARCHIVE_PATH}"

scp "${ARCHIVE_PATH}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/${ARCHIVE_NAME}"
scp "${SOURCE_ARCHIVE_PATH}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/${SOURCE_ARCHIVE_NAME}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
  set -euo pipefail
  if [ -e '${REMOTE_OLD_ROOT}' ]; then
    if [ ! -f '${REMOTE_OLD_ROOT}/.v28_old_commit' ] || [ \"\$(cat '${REMOTE_OLD_ROOT}/.v28_old_commit')\" != '${OLD_COMMIT}' ]; then
      echo 'ERROR: existing old root has no matching commit marker: ${REMOTE_OLD_ROOT}' >&2
      exit 2
    fi
  else
    mkdir -p '${REMOTE_OLD_ROOT}'
    tar -xzf '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}' -C '${REMOTE_OLD_ROOT}'
    printf '%s\n' '${OLD_COMMIT}' > '${REMOTE_OLD_ROOT}/.v28_old_commit'
  fi
  if [ -e '${REMOTE_SOURCE_ROOT}' ]; then
    if [ ! -f '${REMOTE_SOURCE_ROOT}/.v28_source_commit' ] || [ \"\$(cat '${REMOTE_SOURCE_ROOT}/.v28_source_commit')\" != '${SMOOTH_COMMIT}' ]; then
      echo 'ERROR: existing source root has no matching commit marker: ${REMOTE_SOURCE_ROOT}' >&2
      exit 2
    fi
  else
    mkdir -p '${REMOTE_SOURCE_ROOT}'
    tar -xzf '${REMOTE_BASE_DIR}/${SOURCE_ARCHIVE_NAME}' -C '${REMOTE_SOURCE_ROOT}'
    printf '%s\n' '${SMOOTH_COMMIT}' > '${REMOTE_SOURCE_ROOT}/.v28_source_commit'
  fi
  cp '${REMOTE_SOURCE_ROOT}/ideas/source_raw_compactness.py' '${REMOTE_OLD_ROOT}/v276_source_raw_compactness.py'
  printf '%s\n' '${SMOOTH_COMMIT}' > '${REMOTE_OLD_ROOT}/.v276_smooth_commit'
  rm -f '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}' '${REMOTE_BASE_DIR}/${SOURCE_ARCHIVE_NAME}'
"
rm -f "${ARCHIVE_PATH}" "${SOURCE_ARCHIVE_PATH}"
echo "[SUCCESS] Exported DA ${OLD_COMMIT} to ${REMOTE_OLD_ROOT} and source ${SMOOTH_COMMIT} to ${REMOTE_SOURCE_ROOT}"
