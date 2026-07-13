#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANED_ROOT="${CLEANED_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_TAG="${RUN_TAG:-v28_followup_causal_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${CLEANED_ROOT}/logs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}"
phase_status="${LOG_ROOT}/phase_status.tsv"
printf "phase\tstatus\toutput\n" > "${phase_status}"

run_phase() {
  local phase="$1" output="$2"
  shift 2
  echo "PHASE_START|phase=${phase}|output=${output}"
  "$@"
  local status=$?
  printf "%s\t%s\t%s\n" "${phase}" "${status}" "${output}" >> "${phase_status}"
  echo "PHASE_DONE|phase=${phase}|status=${status}|output=${output}"
  return "${status}"
}

run_phase artifact_recovery "${LOG_ROOT}/artifact_recovery" \
  env ROOT="${CLEANED_ROOT}" OUTPUT_DIR="${LOG_ROOT}/artifact_recovery" \
  bash "${SCRIPT_DIR}/launch_v28_old_artifact_recovery.sh" || exit $?

run_phase source_task_smoke "${LOG_ROOT}/source_task_smoke" \
  env CLEANED_ROOT="${CLEANED_ROOT}" LOG_ROOT="${LOG_ROOT}/source_task_smoke" \
  bash "${SCRIPT_DIR}/launch_v28_source_task_specific_smoke_4gpu.sh" || exit $?

run_phase at1_fr2_full_da "${LOG_ROOT}/at1_fr2_full_da" \
  env CLEANED_ROOT="${CLEANED_ROOT}" LOG_ROOT="${LOG_ROOT}/at1_fr2_full_da" \
  bash "${SCRIPT_DIR}/launch_v28_at1_fr2_full_da_audit_4gpu.sh" || exit $?

echo "AUDIT_DONE|run_tag=${RUN_TAG}|log_root=${LOG_ROOT}|phase_status=${phase_status}"
