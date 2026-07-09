#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/data/user/timematch}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
MASTER_TAG="${MASTER_TAG:-v321_stage3b_equivalence_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${MASTER_TAG}}"
REF_ROOT="${REF_ROOT:-${ROOT}/outputs/${MASTER_TAG}_refs}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-AT1_to_DK1,FR2_to_FR1}"
CONFIGS="${CONFIGS:-smooth_base,base_equiv,global_forward,global_only,residual}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}"
SOURCE_CHECKPOINT_MANIFEST="${SOURCE_CHECKPOINT_MANIFEST:-}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "${LOG_DIR}" "${REF_ROOT}"
JOBS="${LOG_DIR}/jobs.tsv"
STATUS="${LOG_DIR}/job_status.tsv"
SUMMARY="${LOG_DIR}/summary.tsv"
REPORT="${ROOT}/analysis/v321_stage3b_equivalence_audit.md"
: > "${JOBS}"
printf "task\tconfig\tsource_kind\tgpu\tstatus\tduration_seconds\tlog\tdiagnostic_log\tlocal_shift_log\tsource_dir\treference_path\n" > "${STATUS}"

read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

task_spec() {
  case "$1" in
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 30TXT" ;;
    FR2_to_AT1) echo "france/31TCJ/2017 austria/33UVP/2017 31TCJ" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 31TCJ" ;;
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 33UVP" ;;
    *) echo "ERROR unknown task: $1" >&2; return 1 ;;
  esac
}

manifest_source_dir() {
  local task="$1"
  local kind="$2"
  if [[ -z "${SOURCE_CHECKPOINT_MANIFEST}" || ! -f "${SOURCE_CHECKPOINT_MANIFEST}" ]]; then
    return 1
  fi
  awk -F '\t' -v task="${task}" -v kind="${kind}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    $idx["task"] == task {
      if (kind == "smooth" && idx["smooth_source"]) print $idx["smooth_source"]
      if (kind == "plain" && idx["plain_source"]) print $idx["plain_source"]
    }
  ' "${SOURCE_CHECKPOINT_MANIFEST}" | head -1
}

latest_matching_dir() {
  local pattern="$1"
  find "${ROOT}/outputs" -maxdepth 1 -type d -name "${pattern}" -printf "%T@\t%p\n" 2>/dev/null \
    | sort -nr \
    | awk -F '\t' 'NR==1 {print $2}'
}

find_source_dir() {
  local task="$1"
  local tile="$2"
  local kind="$3"
  local from_manifest
  from_manifest="$(manifest_source_dir "${task}" "${kind}" || true)"
  if [[ -n "${from_manifest}" && -f "${from_manifest}/fold_0/model.pt" ]]; then
    echo "${from_manifest}"
    return 0
  fi
  if [[ "${kind}" == "smooth" ]]; then
    latest_matching_dir "pseltae_${tile}_closedset_noshift_*_${task}_seed${SEED}_v276_smooth_k3_w1_source"
  else
    latest_matching_dir "pseltae_${tile}_closedset_noshift_*_${task}_seed${SEED}_plain_source"
  fi
}

with_extra_for_source() {
  local source_dir="$1"
  python -c "import json; print(json.load(open('${source_dir}/train_config.json')).get('with_extra', False))" 2>/dev/null || echo "False"
}

build_reference_if_needed() {
  local task="$1"
  local source_dataset="$2"
  local source_kind="$3"
  local source_dir="$4"
  local ref_dir="${REF_ROOT}/${task}_${source_kind}"
  local ref_path="${ref_dir}/source_stage_reference.pt"
  mkdir -p "${ref_dir}"
  if [[ -f "${ref_path}" ]]; then
    echo "${ref_path}"
    return 0
  fi
  local with_extra
  with_extra="$(with_extra_for_source "${source_dir}")"
  echo "BUILD_REFERENCE|task=${task}|source_kind=${source_kind}|source_dir=${source_dir}|output=${ref_path}" >&2
  if [[ "${DRY_RUN}" == "True" ]]; then
    echo "${ref_path}"
    return 0
  fi
  (cd "${ROOT}" && python tools/build_source_stage_reference.py \
    --weights "${source_dir}/fold_0/model.pt" \
    --source "${source_dataset}" \
    --output "${ref_path}" \
    --with_extra "${with_extra}" \
    --closed_set True \
    --device cuda \
    --kmax 8 \
    --min_stage_len 3 \
    --change_quantile 0.75) \
    > "${LOG_DIR}/build_reference_${task}_${source_kind}.log" 2>&1
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "REFERENCE_FAILED|task=${task}|source_kind=${source_kind}|log=${LOG_DIR}/build_reference_${task}_${source_kind}.log" >&2
    return "${status}"
  fi
  echo "${ref_path}"
}

append_job() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${JOBS}"
}

IFS=',' read -r -a TASK_NAMES <<< "${TASKS}"
IFS=',' read -r -a CONFIG_NAMES <<< "${CONFIGS}"

for raw_task in "${TASK_NAMES[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  read -r source_dataset target_dataset source_tile <<< "$(task_spec "${task}")" || exit 2
  smooth_dir="$(find_source_dir "${task}" "${source_tile}" smooth || true)"
  if [[ -z "${smooth_dir}" || ! -f "${smooth_dir}/fold_0/model.pt" ]]; then
    echo "SKIP_TASK|task=${task}|reason=missing_smooth_source"
    continue
  fi
  for raw_config in "${CONFIG_NAMES[@]}"; do
    config="$(echo "${raw_config}" | xargs)"
    case "${config}" in
      smooth_base|base_equiv|global_forward)
        append_job "${task}" "${config}" "${source_dataset}" "${target_dataset}" "smooth" "${smooth_dir}" ""
        ;;
      global_only|residual)
        ref_path="$(build_reference_if_needed "${task}" "${source_dataset}" "smooth" "${smooth_dir}")" || continue
        append_job "${task}" "${config}" "${source_dataset}" "${target_dataset}" "smooth" "${smooth_dir}" "${ref_path}"
        ;;
      *)
        echo "ERROR unknown config: ${config}" >&2
        exit 2
        ;;
    esac
  done
done

for gpu in "${GPU_IDS[@]}"; do
  : > "${LOG_DIR}/queue_gpu${gpu}.tsv"
done
job_index=0
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "${line}" >> "${LOG_DIR}/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "${JOBS}"

run_job() {
  local gpu="$1"
  local task="$2"
  local config="$3"
  local source_dataset="$4"
  local target_dataset="$5"
  local source_kind="$6"
  local source_dir="$7"
  local reference_path="$8"
  local run_tag="${MASTER_TAG}_${task}_seed${SEED}_${config}"
  local out_dir="${ROOT}/outputs/${run_tag}"
  local run_log="${LOG_DIR}/gpu${gpu}_${task}_${config}.log"
  local diagnostic_log="${LOG_DIR}/diagnostic_${task}_${config}.tsv"
  local local_shift_log="${LOG_DIR}/local_shift_${task}_${config}.tsv"
  local start_time end_time status
  start_time="$(date +%s)"
  echo "START|task=${task}|config=${config}|gpu=${gpu}|log=${run_log}"

  if [[ "${DRY_RUN}" == "True" ]]; then
    status=0
  elif [[ "${config}" == "smooth_base" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/train.py" \
      --data_root "${DATA_ROOT}" \
      --source "${source_dataset}" \
      --target "${target_dataset}" \
      --closed_set True \
      --with_shift_aug False \
      --output_dir "${out_dir}" \
      --tensorboard_log_dir "${ROOT}/runs/${run_tag}" \
      --device cuda:0 \
      --num_workers "${NUM_WORKERS}" \
      --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
      --seed "${SEED}" \
      -e "${run_tag}" \
      timematch \
      --weights "${source_dir}" \
      --epochs "${EPOCHS}" \
      --steps_per_epoch "${STEPS_PER_EPOCH}" \
      --sample_size "${SAMPLE_SIZE}" \
      --timematch_diagnostic_log_path "${diagnostic_log}" \
      --balance_source True \
      > "${run_log}" 2>&1
    status=$?
    local_shift_log=""
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/train.py" \
      --data_root "${DATA_ROOT}" \
      --source "${source_dataset}" \
      --target "${target_dataset}" \
      --closed_set True \
      --with_shift_aug False \
      --output_dir "${out_dir}" \
      --tensorboard_log_dir "${ROOT}/runs/${run_tag}" \
      --device cuda:0 \
      --num_workers "${NUM_WORKERS}" \
      --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
      --seed "${SEED}" \
      -e "${run_tag}" \
      timematch_local_shift \
      --weights "${source_dir}" \
      --epochs "${EPOCHS}" \
      --steps_per_epoch "${STEPS_PER_EPOCH}" \
      --sample_size "${SAMPLE_SIZE}" \
      --source_stage_reference_path "${reference_path}" \
      --local_shift_mode "${config}" \
      --local_shift_log_path "${local_shift_log}" \
      --balance_source True \
      > "${run_log}" 2>&1
    status=$?
    diagnostic_log=""
  fi
  end_time="$(date +%s)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${config}" "${source_kind}" "${gpu}" "${status}" "$((end_time - start_time))" "${run_log}" "${diagnostic_log}" "${local_shift_log}" "${source_dir}" "${reference_path}" \
    >> "${STATUS}"
  echo "DONE|task=${task}|config=${config}|status=${status}|seconds=$((end_time - start_time))"
}

run_queue() {
  local gpu="$1"
  local queue="${LOG_DIR}/queue_gpu${gpu}.tsv"
  while IFS=$'\t' read -r task config source_dataset target_dataset source_kind source_dir reference_path; do
    [[ -z "${task}" ]] && continue
    run_job "${gpu}" "${task}" "${config}" "${source_dataset}" "${target_dataset}" "${source_kind}" "${source_dir}" "${reference_path}"
  done < "${queue}"
}

echo "MASTER_TAG=${MASTER_TAG}"
echo "TASKS=${TASKS}"
echo "CONFIGS=${CONFIGS}"
echo "LOG_DIR=${LOG_DIR}"
echo "JOBS=$(wc -l < "${JOBS}")"

for gpu in "${GPU_IDS[@]}"; do
  run_queue "${gpu}" &
done
wait

python "${ROOT}/analysis/summarize_v321_stage3b_equivalence_audit.py" \
  --log_dir "${LOG_DIR}" \
  --summary "${SUMMARY}" \
  --report "${REPORT}" || true

echo "SUMMARY=${SUMMARY}"
echo "REPORT=${REPORT}"
