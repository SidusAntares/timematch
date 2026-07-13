#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OLD_DA_ROOT="${OLD_DA_ROOT:-/data/user/timematch_old_da_f04e1e0}"
OLD_DA_COMMIT="${OLD_DA_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${ROOT}/outputs}"
RUN_TAG="${RUN_TAG:-v28_source_checkpoint_first_round_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_TAG}}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REUSE_EXISTING_RESULTS="${REUSE_EXISTING_RESULTS:-True}"
EXISTING_DA_ONLY="${EXISTING_DA_ONLY:-False}"
CLEANED_RUN="${CLEANED_RUN:-${ROOT}/logs/v28_cleaned_full12_20260710_120037}"
RECOVERY_RESULTS="${RECOVERY_RESULTS:-${ROOT}/logs/v28_followup_causal_audit_20260711_125438/artifact_recovery/recovered_results.tsv}"
OLD_BASE_ROWS="${OLD_BASE_ROWS:-${ROOT}/logs/v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121/raw_strength_rows.tsv}"
OLD_SMOOTH_ROWS="${OLD_SMOOTH_ROWS:-${ROOT}/logs/v276_smoothed_lambda12_half_20260619_215420/w1p0/raw_strength_rows.tsv}"

mkdir -p "${LOG_ROOT}/checkpoint_audit" "${LOG_ROOT}/eval_results" "${LOG_ROOT}/eval_logs" "${LOG_ROOT}/eval_status"
for path in \
  "${CLEANED_RUN}/source_checkpoint_inventory.tsv" \
  "${CLEANED_RUN}/summary.tsv" \
  "${RECOVERY_RESULTS}" \
  "${OLD_BASE_ROWS}" \
  "${OLD_SMOOTH_ROWS}"; do
  [[ -f "${path}" ]] || { echo "ERROR missing required input: ${path}" >&2; exit 2; }
done
if [[ -f "${OLD_DA_ROOT}/.v28_old_commit" ]]; then
  [[ "$(cat "${OLD_DA_ROOT}/.v28_old_commit")" == "${OLD_DA_COMMIT}" ]] || {
    echo "ERROR old DA commit marker mismatch" >&2; exit 2;
  }
elif git -C "${OLD_DA_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "${OLD_DA_ROOT}" rev-parse HEAD)" == "${OLD_DA_COMMIT}" ]] || {
    echo "ERROR old DA worktree commit mismatch" >&2; exit 2;
  }
else
  echo "ERROR missing old DA worktree/export: ${OLD_DA_ROOT}" >&2
  exit 2
fi

SOURCE_MANIFEST="${LOG_ROOT}/checkpoint_audit/source_checkpoint_manifest.tsv"
EXISTING_DA_MANIFEST="${LOG_ROOT}/checkpoint_audit/existing_da_manifest.tsv"
SOURCE_COMPARISON="${LOG_ROOT}/source_checkpoint_comparison.tsv"
SOURCE_COMPARISON_SUMMARY="${LOG_ROOT}/source_checkpoint_comparison_summary.tsv"
EVAL_JOBS="${LOG_ROOT}/eval_jobs.tsv"

python -B "${ROOT}/tools/v28_source_da_cross_audit.py" build-manifests \
  --outputs_root "${OUTPUTS_ROOT}" \
  --cleaned_source_inventory "${CLEANED_RUN}/source_checkpoint_inventory.tsv" \
  --recovered_results "${RECOVERY_RESULTS}" \
  --cleaned_summary "${CLEANED_RUN}/summary.tsv" \
  --source_manifest "${SOURCE_MANIFEST}" \
  --existing_da_manifest "${EXISTING_DA_MANIFEST}" || exit $?

python -B "${ROOT}/tools/v28_source_da_cross_audit.py" compare-sources \
  --source_manifest "${SOURCE_MANIFEST}" \
  --output "${SOURCE_COMPARISON}" \
  --summary "${SOURCE_COMPARISON_SUMMARY}" || exit $?

python -B "${ROOT}/tools/v28_source_da_cross_audit.py" build-eval-jobs \
  --source_manifest "${SOURCE_MANIFEST}" \
  --existing_da_manifest "${EXISTING_DA_MANIFEST}" \
  --old_base_rows "${OLD_BASE_ROWS}" \
  --old_smooth_rows "${OLD_SMOOTH_ROWS}" \
  --cleaned_summary "${CLEANED_RUN}/summary.tsv" \
  --output "${EVAL_JOBS}" || exit $?

if [[ "${EXISTING_DA_ONLY}" == "True" ]]; then
  SOURCE_RESULT_COUNT="$(find "${LOG_ROOT}/eval_results" -maxdepth 1 -type f -name 'job_*.json' \
    | awk -F/ '{name=$NF; sub(/^job_/, "", name); sub(/\.json$/, "", name); if ((name + 0) < 144) count++} END {print count + 0}')"
  [[ "${SOURCE_RESULT_COUNT}" -eq 144 ]] || {
    echo "ERROR existing-DA-only resume requires all 144 source results; found ${SOURCE_RESULT_COUNT}" >&2
    exit 2
  }
fi

run_eval() {
  local gpu="$1" job_id="$2" kind="$3" task="$4" source="$5" target="$6" config="$7"
  local seed="$8" checkpoint_version="$9" source_version="${10}" da_version="${11}"
  local checkpoint_path="${12}" native_test_f1="${13}"
  local result="${LOG_ROOT}/eval_results/job_$(printf '%03d' "${job_id}").json"
  local log="${LOG_ROOT}/eval_logs/job_$(printf '%03d' "${job_id}").log"
  local status_path="${LOG_ROOT}/eval_status/job_$(printf '%03d' "${job_id}").tsv"
  local start status runtime
  job_id="${job_id%$'\r'}"
  start="$(date +%s)"
  if [[ "${REUSE_EXISTING_RESULTS}" == "True" && -f "${result}" ]]; then
    status=0
    echo "REUSED_RESULT|${result}" > "${log}"
  elif [[ ! -f "${checkpoint_path}" ]]; then
    status=2
    echo "MISSING_CHECKPOINT|${checkpoint_path}" > "${log}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -B "${ROOT}/tools/evaluate_common_timematch_checkpoint.py" \
      --checkpoint "${checkpoint_path}" --data_root "${DATA_ROOT}" \
      --source "${source}" --target "${target}" --seed "${seed}" \
      --closed_set True --num_workers "${NUM_WORKERS}" --batch_size 128 \
      --output "${result}" > "${log}" 2>&1
    status=$?
  fi
  runtime=$(( $(date +%s) - start ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" "${kind}" "${task}" "${config}" "${seed}" "${checkpoint_version}" \
    "${source_version}" "${da_version}" "${gpu}" "${BASHPID}" "${status}" "${runtime}" "${log}" "${checkpoint_path}" \
    > "${status_path}"
  return "${status}"
}

mapfile -t jobs < <(tail -n +2 "${EVAL_JOBS}")
if [[ "${EXISTING_DA_ONLY}" == "True" ]]; then
  jobs=("${jobs[@]:144:8}")
  [[ "${#jobs[@]}" -eq 8 ]] || { echo "ERROR expected 8 existing DA jobs" >&2; exit 2; }
fi
read -r -a gpu_ids <<< "${GPUS}"
overall=0
for ((offset=0; offset<${#jobs[@]}; offset+=${#gpu_ids[@]})); do
  pids=()
  for ((slot=0; slot<${#gpu_ids[@]} && offset+slot<${#jobs[@]}; slot++)); do
    jobs[$((offset+slot))]="${jobs[$((offset+slot))]%$'\r'}"
    IFS=$'\t' read -r kind task source target config seed checkpoint_version source_version da_version checkpoint_path native_test_f1 job_id \
      <<< "${jobs[$((offset+slot))]}"
    run_eval "${gpu_ids[slot]}" "${job_id}" "${kind}" "${task}" "${source}" "${target}" "${config}" \
      "${seed}" "${checkpoint_version}" "${source_version}" "${da_version}" "${checkpoint_path}" "${native_test_f1}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
done

JOB_STATUS="${LOG_ROOT}/job_status.tsv"
printf "job_id\tkind\ttask\tconfig\tseed\tcheckpoint_version\tsource_version\tda_version\tgpu\tpid\tstatus\truntime_s\tlog_path\tcheckpoint_path\n" > "${JOB_STATUS}"
find "${LOG_ROOT}/eval_status" -name 'job_*.tsv' -print0 | sort -z | while IFS= read -r -d '' file; do cat "${file}" >> "${JOB_STATUS}"; done

python -B "${ROOT}/tools/v28_source_da_cross_audit.py" summarize-evals \
  --jobs "${EVAL_JOBS}" --results_root "${LOG_ROOT}/eval_results" \
  --source_output "${LOG_ROOT}/source_common_eval.tsv" \
  --paired_output "${LOG_ROOT}/source_common_eval_paired.tsv" \
  --da_output "${LOG_ROOT}/existing_da_common_eval.tsv" || overall=1

python -B "${ROOT}/tools/check_v28_old_da_checkpoint_load.py" \
  --old_root "${OLD_DA_ROOT}" --source_manifest "${SOURCE_MANIFEST}" \
  --output "${LOG_ROOT}/old_da_cleaned_source_load_smoke.tsv" || overall=1

python -B "${ROOT}/tools/v28_source_da_cross_audit.py" gate-first-round \
  --source_comparison "${SOURCE_COMPARISON}" \
  --source_eval "${LOG_ROOT}/source_common_eval.tsv" \
  --da_eval "${LOG_ROOT}/existing_da_common_eval.tsv" \
  --load_smoke "${LOG_ROOT}/old_da_cleaned_source_load_smoke.tsv" \
  --output "${LOG_ROOT}/audit_gate_summary.tsv" \
  --marker "${LOG_ROOT}/FIRST_ROUND_PASSED" || overall=1
cp "${LOG_ROOT}/audit_gate_summary.tsv" "${LOG_ROOT}/first_round_gate.tsv"

echo "FIRST_ROUND_DONE|status=${overall}|log_root=${LOG_ROOT}|gate=${LOG_ROOT}/audit_gate_summary.tsv"
exit "${overall}"
