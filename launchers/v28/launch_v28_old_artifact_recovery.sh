#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/user/timematch}"
RUN_TAG="${RUN_TAG:-v28_old_artifact_recovery_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/logs/${RUN_TAG}}"
MAX_DEPTH="${MAX_DEPTH:-7}"
mkdir -p "${OUTPUT_DIR}"

python -B "${ROOT}/tools/recover_v28_old_artifacts.py" \
  --search_root /data/user \
  --search_root /data \
  --max_depth "${MAX_DEPTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --repo_root "${ROOT}"
