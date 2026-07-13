#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-${ROOT}/logs/v28_source_checkpoint_first_round_20260711_230756}"
CROSS_RUN_TAG="${CROSS_RUN_TAG:-v28_source_da_cross_audit_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "FINAL_CAUSAL_AUDIT_START|first_round_root=${FIRST_ROUND_ROOT}|cross_run_tag=${CROSS_RUN_TAG}"

env \
  ROOT="${ROOT}" \
  LOG_ROOT="${FIRST_ROUND_ROOT}" \
  RUN_TAG="$(basename "${FIRST_ROUND_ROOT}")" \
  GPUS="${GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  REUSE_EXISTING_RESULTS=True \
  EXISTING_DA_ONLY=True \
  bash "${SCRIPT_DIR}/launch_v28_source_checkpoint_first_round_4gpu.sh"
existing_status=$?

if [[ "${existing_status}" -ne 0 || ! -f "${FIRST_ROUND_ROOT}/FIRST_ROUND_PASSED" ]]; then
  echo "ERROR existing DA common-evaluation gate failed; cross jobs were not launched" >&2
  exit 2
fi

env \
  ROOT="${ROOT}" \
  FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT}" \
  RUN_TAG="${CROSS_RUN_TAG}" \
  GPUS="${GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  bash "${SCRIPT_DIR}/launch_v28_source_da_cross_audit_4gpu.sh"
cross_status=$?

echo "FINAL_CAUSAL_AUDIT_DONE|status=${cross_status}|first_round_root=${FIRST_ROUND_ROOT}|cross_log_root=${ROOT}/logs/${CROSS_RUN_TAG}"
exit "${cross_status}"
