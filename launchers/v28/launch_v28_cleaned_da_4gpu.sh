#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_cleaned_repro_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_TAG}}"
SOURCE_INVENTORY="${SOURCE_INVENTORY:-${LOG_DIR}/source_checkpoint_inventory.tsv}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-AT1_to_DK1,FR1_to_FR2,FR2_to_AT1,FR2_to_FR1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-base,raw_global,smooth_k3,elastic_r2}"
CLOSED_SET="${CLOSED_SET:-True}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "${LOG_DIR}"
DA_JOBS="${LOG_DIR}/da_jobs.tsv"
DA_STATUS="${LOG_DIR}/da_job_status.tsv"
SUMMARY="${LOG_DIR}/summary.tsv"

if [[ ! -f "${SOURCE_INVENTORY}" ]]; then
  echo "ERROR: missing source inventory: ${SOURCE_INVENTORY}" >&2
  exit 2
fi

read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

task_spec() {
  case "$1" in
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 AT1 DK1" ;;
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 FR1 FR2" ;;
    FR2_to_AT1) echo "france/31TCJ/2017 austria/33UVP/2017 FR2 AT1" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 FR2 FR1" ;;
    *) echo "ERROR unknown task: $1" >&2; return 1 ;;
  esac
}

checkpoint_for() {
  local task="$1"
  local seed="$2"
  local config="$3"
  awk -F '\t' -v task="${task}" -v seed="${seed}" -v config="${config}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    $idx["task"] == task && $idx["seed"] == seed && $idx["config"] == config {
      print $idx["checkpoint_path"]
      exit
    }
  ' "${SOURCE_INVENTORY}"
}

build_jobs() {
  : > "${DA_JOBS}"
  printf "task\tsource\ttarget\tseed\tconfig\tsource_checkpoint\tstatus\truntime_s\tsource_eval_log\tda_log\tdiag_log\n" > "${DA_STATUS}"
  IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
  IFS=',' read -r -a CONFIG_ARRAY <<< "${CONFIGS}"
  for raw_task in "${TASK_ARRAY[@]}"; do
    task="$(echo "${raw_task}" | xargs)"
    read -r source_dataset target_dataset _source_alias _target_alias <<< "$(task_spec "${task}")" || exit 2
    for seed in ${SEEDS}; do
      for raw_config in "${CONFIG_ARRAY[@]}"; do
        config="$(echo "${raw_config}" | xargs)"
        case "${config}" in base|raw_global|smooth_k3|elastic_r2) ;; *) echo "ERROR unknown config ${config}" >&2; exit 2 ;; esac
        checkpoint="$(checkpoint_for "${task}" "${seed}" "${config}")"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${task}" "${source_dataset}" "${target_dataset}" "${seed}" "${config}" "${checkpoint}" >> "${DA_JOBS}"
      done
    done
  done
}

run_da_job() {
  local gpu="$1"
  local task="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local seed="$5"
  local config="$6"
  local checkpoint_path="$7"
  local source_dir
  source_dir="$(dirname "$(dirname "${checkpoint_path}")")"
  local safe_task="${task/_to_/_}"
  local exp_name="v28_cleaned_${safe_task}_${config}_seed${seed}"
  local job_dir="${LOG_DIR}/${task}/${config}/seed${seed}"
  local source_eval_log="${job_dir}/source_eval.log"
  local da_log="${job_dir}/train.log"
  local diag_log="${job_dir}/timematch_diag.tsv"
  mkdir -p "${job_dir}"
  local start_time end_time status
  start_time="$(date +%s)"
  echo "DA_START|task=${task}|config=${config}|seed=${seed}|gpu=${gpu}|log=${da_log}"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "MISSING_CHECKPOINT|task=${task}|seed=${seed}|config=${config}|path=${checkpoint_path}" > "${da_log}"
    end_time="$(date +%s)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task}" "${source_dataset}" "${target_dataset}" "${seed}" "${config}" "${checkpoint_path}" "missing_checkpoint" "$((end_time - start_time))" "${source_eval_log}" "${da_log}" "${diag_log}" \
      >> "${DA_STATUS}"
    return 1
  fi
  if [[ "${DRY_RUN}" == "True" ]]; then
    echo "DRY_RUN da ${task} ${config} seed${seed}" > "${da_log}"
    end_time="$(date +%s)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task}" "${source_dataset}" "${target_dataset}" "${seed}" "${config}" "${checkpoint_path}" "dry_run" "$((end_time - start_time))" "${source_eval_log}" "${da_log}" "${diag_log}" \
      >> "${DA_STATUS}"
    return 0
  fi
  (
    cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --data_root "${DATA_ROOT}" \
      --closed_set "${CLOSED_SET}" \
      --with_shift_aug False \
      --num_workers "${NUM_WORKERS}" \
      --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
      --seed "${seed}" \
      -e "$(basename "${source_dir}")" \
      --source "${source_dataset}" \
      --target "${target_dataset}" \
      --eval
  ) > "${source_eval_log}" 2>&1
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    (
      cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
        --data_root "${DATA_ROOT}" \
        --closed_set "${CLOSED_SET}" \
        --with_shift_aug False \
        --num_workers "${NUM_WORKERS}" \
        --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
        --seed "${seed}" \
        -e "${exp_name}" \
        --source "${source_dataset}" \
        --target "${target_dataset}" \
        timematch \
        --weights "${source_dir}" \
        --epochs "${DA_EPOCHS}" \
        --steps_per_epoch "${STEPS_PER_EPOCH}" \
        --timematch_diagnostic_task "${task}" \
        --timematch_diagnostic_log_path "${diag_log}"
    ) > "${da_log}" 2>&1
    status=$?
  else
    echo "SOURCE_EVAL_FAILED|status=${status}|log=${source_eval_log}" > "${da_log}"
  fi
  end_time="$(date +%s)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${source_dataset}" "${target_dataset}" "${seed}" "${config}" "${checkpoint_path}" "${status}" "$((end_time - start_time))" "${source_eval_log}" "${da_log}" "${diag_log}" \
    >> "${DA_STATUS}"
  echo "DA_DONE|task=${task}|config=${config}|seed=${seed}|status=${status}|seconds=$((end_time - start_time))"
  return "${status}"
}

run_batch() {
  local -a pids=()
  local batch_status=0
  local gpu_idx=0
  for item in "$@"; do
    IFS=$'\t' read -r task source_dataset target_dataset seed config checkpoint_path <<< "${item}"
    gpu="${GPU_IDS[$((gpu_idx % ${#GPU_IDS[@]}))]}"
    gpu_idx=$((gpu_idx + 1))
    run_da_job "${gpu}" "${task}" "${source_dataset}" "${target_dataset}" "${seed}" "${config}" "${checkpoint_path}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || batch_status=1
  done
  return "${batch_status}"
}

build_jobs
echo "RUN_TAG=${RUN_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "SOURCE_INVENTORY=${SOURCE_INVENTORY}"
echo "DA_JOBS=$(wc -l < "${DA_JOBS}")"

overall_status=0
batch=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  batch+=("${line}")
  if [[ "${#batch[@]}" -ge "${#GPU_IDS[@]}" ]]; then
    run_batch "${batch[@]}" || overall_status=1
    batch=()
  fi
done < "${DA_JOBS}"
if [[ "${#batch[@]}" -gt 0 ]]; then
  run_batch "${batch[@]}" || overall_status=1
fi

python "${ROOT}/tools/summarize_v28_cleaned_repro.py" \
  --log_root "${LOG_DIR}" \
  --output "${SUMMARY}" \
  --report "${ROOT}/analysis/v28_cleaned_code_reproduction_report.md" || overall_status=1

echo "DA_STATUS=${DA_STATUS}"
echo "SUMMARY=${SUMMARY}"
exit "${overall_status}"
