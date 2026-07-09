#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/data/user/timematch}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
MASTER_TAG="${MASTER_TAG:-v321_stage3d_residual_safety_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${MASTER_TAG}}"
REF_DIR="${REF_DIR:-${LOG_DIR}/references}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-AT1_to_DK1,FR2_to_FR1}"
CONFIGS="${CONFIGS:-global_only,residual_raw,residual_zero_mean,residual_scaled_zero_mean_alpha05,residual_gated_scaled_zero_mean_alpha05_top065}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}"
SOURCE_CHECKPOINT_MANIFEST="${SOURCE_CHECKPOINT_MANIFEST:-}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "${LOG_DIR}" "${REF_DIR}"
STATUS="${LOG_DIR}/job_status.tsv"
SUMMARY="${LOG_DIR}/summary.tsv"
REPORT="${ROOT}/analysis/v321_stage3d_residual_safety_report.md"
printf "task\tconfig\tgpu\tpid\tlog_path\tlocal_shift_log\tstatus\truntime_seconds\tsource_dir\treference_path\n" > "${STATUS}"

read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

task_spec() {
  case "$1" in
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 33UVP AT1" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 31TCJ FR2" ;;
    *) echo "ERROR unknown task: $1" >&2; return 1 ;;
  esac
}

manifest_source_dir() {
  local task="$1"
  if [[ -z "${SOURCE_CHECKPOINT_MANIFEST}" || ! -f "${SOURCE_CHECKPOINT_MANIFEST}" ]]; then
    return 1
  fi
  awk -F '\t' -v task="${task}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    $idx["task"] == task {
      if (idx["smooth_source"]) print $idx["smooth_source"]
    }
  ' "${SOURCE_CHECKPOINT_MANIFEST}" | head -1
}

latest_matching_dir() {
  local pattern="$1"
  find "${ROOT}/outputs" -maxdepth 1 -type d -name "${pattern}" -printf "%T@\t%p\n" 2>/dev/null \
    | sort -nr \
    | awk -F '\t' 'NR==1 {print $2}'
}

find_smooth_source_dir() {
  local task="$1"
  local tile="$2"
  local from_manifest
  from_manifest="$(manifest_source_dir "${task}" || true)"
  if [[ -n "${from_manifest}" && -f "${from_manifest}/fold_0/model.pt" ]]; then
    echo "${from_manifest}"
    return 0
  fi
  latest_matching_dir "pseltae_${tile}_closedset_noshift_*_${task}_seed${SEED}_v276_smooth_k3_w1_source"
}

with_extra_for_source() {
  local source_dir="$1"
  python -c "import json; print(json.load(open('${source_dir}/train_config.json')).get('with_extra', False))" 2>/dev/null || echo "False"
}

build_reference() {
  local source_alias="$1"
  local source_dataset="$2"
  local source_dir="$3"
  local ref_path="${REF_DIR}/source_stage_reference_${source_alias}_smooth.pt"
  if [[ -f "${ref_path}" ]]; then
    echo "${ref_path}"
    return 0
  fi
  local with_extra
  with_extra="$(with_extra_for_source "${source_dir}")"
  echo "BUILD_REFERENCE|source=${source_alias}|source_dir=${source_dir}|output=${ref_path}" >&2
  if [[ "${DRY_RUN}" == "True" ]]; then
    echo "${ref_path}"
    return 0
  fi
  (
    cd "${ROOT}" && \
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python tools/build_source_stage_reference.py \
      --weights "${source_dir}/fold_0/model.pt" \
      --source "${source_dataset}" \
      --output "${ref_path}" \
      --with_extra "${with_extra}" \
      --closed_set True \
      --device cuda \
      --kmax 8 \
      --min_stage_len 3 \
      --change_quantile 0.75
  ) > "${LOG_DIR}/build_reference_${source_alias}_smooth.log" 2>&1
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "REFERENCE_FAILED|source=${source_alias}|status=${status}|log=${LOG_DIR}/build_reference_${source_alias}_smooth.log" >&2
    return "${status}"
  fi
  echo "${ref_path}"
}

declare -A TASK_SOURCE
declare -A TASK_TARGET
declare -A TASK_SOURCE_DIR
declare -A TASK_REFERENCE
declare -A TASK_SOURCE_ALIAS

IFS=',' read -r -a TASK_NAMES <<< "${TASKS}"
for raw_task in "${TASK_NAMES[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  read -r source_dataset target_dataset source_tile source_alias <<< "$(task_spec "${task}")" || exit 2
  source_dir="$(find_smooth_source_dir "${task}" "${source_tile}" || true)"
  if [[ -z "${source_dir}" || ! -f "${source_dir}/fold_0/model.pt" ]]; then
    echo "ERROR: missing smooth source checkpoint for ${task}" >&2
    exit 3
  fi
  ref_path="$(build_reference "${source_alias}" "${source_dataset}" "${source_dir}")" || exit 4
  TASK_SOURCE["${task}"]="${source_dataset}"
  TASK_TARGET["${task}"]="${target_dataset}"
  TASK_SOURCE_DIR["${task}"]="${source_dir}"
  TASK_REFERENCE["${task}"]="${ref_path}"
  TASK_SOURCE_ALIAS["${task}"]="${source_alias}"
done

run_one_job() {
  local task="$1"
  local config="$2"
  local gpu="$3"
  local source_dataset="${TASK_SOURCE[${task}]}"
  local target_dataset="${TASK_TARGET[${task}]}"
  local source_dir="${TASK_SOURCE_DIR[${task}]}"
  local ref_path="${TASK_REFERENCE[${task}]}"
  local job_dir="${LOG_DIR}/${task}/${config}"
  local log_path="${job_dir}/train.log"
  local local_shift_log="${job_dir}/local_shift.tsv"
  local exp_name="v321_stage3d_${task}_${config}_seed${SEED}"
  mkdir -p "${job_dir}"

  local start_time end_time status
  start_time="$(date +%s)"
  echo "START|task=${task}|config=${config}|gpu=${gpu}|log=${log_path}"
  if [[ "${DRY_RUN}" == "True" ]]; then
    (
      sleep 1
      echo "DRY_RUN ${task} ${config}"
    ) > "${log_path}" 2>&1 &
  else
    (
      cd "${ROOT}" && \
      CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
        --data_root "${DATA_ROOT}" \
        --source "${source_dataset}" \
        --target "${target_dataset}" \
        --closed_set True \
        --with_shift_aug False \
        --output_dir "${ROOT}/outputs" \
        --tensorboard_log_dir "${ROOT}/runs/${exp_name}" \
        --device cuda:0 \
        --num_workers "${NUM_WORKERS}" \
        --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
        --seed "${SEED}" \
        -e "${exp_name}" \
        timematch_local_shift \
        --weights "${source_dir}" \
        --epochs "${EPOCHS}" \
        --steps_per_epoch "${STEPS_PER_EPOCH}" \
        --sample_size "${SAMPLE_SIZE}" \
        --source_stage_reference_path "${ref_path}" \
        --local_shift_mode "${config}" \
        --local_shift_log_path "${local_shift_log}" \
        --balance_source True
    ) > "${log_path}" 2>&1 &
  fi
  local pid=$!
  wait "${pid}"
  status=$?
  end_time="$(date +%s)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${config}" "${gpu}" "${pid}" "${log_path}" "${local_shift_log}" "${status}" "$((end_time - start_time))" "${source_dir}" "${ref_path}" \
    >> "${STATUS}"
  echo "DONE|task=${task}|config=${config}|gpu=${gpu}|status=${status}|seconds=$((end_time - start_time))"
  return "${status}"
}

run_batch() {
  local -a pids=()
  local batch_status=0
  for item in "$@"; do
    IFS='|' read -r task config gpu <<< "${item}"
    run_one_job "${task}" "${config}" "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || batch_status=1
  done
  return "${batch_status}"
}

echo "MASTER_TAG=${MASTER_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "TASKS=${TASKS}"
echo "CONFIGS=${CONFIGS}"
echo "GPUS=${GPUS}"
echo "REFERENCES_REUSED=True"

run_start="$(date +%s)"
overall_status=0
job_index=0
batch=()
IFS=',' read -r -a CONFIG_NAMES <<< "${CONFIGS}"
for raw_task in "${TASK_NAMES[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  for raw_config in "${CONFIG_NAMES[@]}"; do
    config="$(echo "${raw_config}" | xargs)"
    case "${config}" in
      global_only|residual_raw|residual_zero_mean|residual_scaled_zero_mean_alpha05|residual_gated_scaled_zero_mean_alpha05_top065) ;;
      *) echo "ERROR: unsupported Stage 3d config ${config}" >&2; exit 5 ;;
    esac
    gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
    batch+=("${task}|${config}|${gpu}")
    job_index=$((job_index + 1))
    if [[ "${#batch[@]}" -ge "${#GPU_IDS[@]}" ]]; then
      run_batch "${batch[@]}" || overall_status=1
      batch=()
    fi
  done
done
if [[ "${#batch[@]}" -gt 0 ]]; then
  run_batch "${batch[@]}" || overall_status=1
fi
run_end="$(date +%s)"
wall_seconds=$((run_end - run_start))

python "${ROOT}/tools/summarize_local_shift_logs.py" \
  --log_root "${LOG_DIR}" \
  --output "${SUMMARY}" || overall_status=1

failed_jobs="$(awk -F '\t' 'NR > 1 && $7 != "0" {print $1 ":" $2 ":status=" $7}' "${STATUS}" | paste -sd ', ' -)"
{
  echo "# v3.2.1 Stage 3d Residual Safety Report"
  echo
  echo "- jobs: ${job_index}"
  echo "- gpus: ${GPUS}"
  echo "- gpu_allocation: round-robin batches, max ${#GPU_IDS[@]} concurrent single-GPU jobs"
  echo "- wall_clock_seconds: ${wall_seconds}"
  echo "- references_reused: true"
  echo "- reference_dir: ${REF_DIR}"
  echo "- summary: ${SUMMARY}"
  if [[ -n "${failed_jobs}" ]]; then
    echo "- failed_jobs: ${failed_jobs}"
  else
    echo "- failed_jobs: none"
  fi
  echo
  echo "## Per-job Runtime"
  echo
  echo "| task | config | gpu | status | runtime_seconds | log |"
  echo "|---|---|---:|---:|---:|---|"
  awk -F '\t' 'NR > 1 {printf "| %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $7, $8, $5}' "${STATUS}"
} > "${REPORT}"

echo "STATUS=${STATUS}"
echo "SUMMARY=${SUMMARY}"
echo "REPORT=${REPORT}"
exit "${overall_status}"
