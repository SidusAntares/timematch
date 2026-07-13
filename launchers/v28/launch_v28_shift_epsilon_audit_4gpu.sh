#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_da_causal_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_TAG}}"
SOURCE_INVENTORY="${SOURCE_INVENTORY:-${ROOT}/logs/v28_cleaned_full12_20260710_120037/source_checkpoint_inventory.tsv}"
TASKS="${TASKS:-AT1_to_FR2,FR2_to_FR1}"
METHODS="${METHODS:-base,smooth_k3}"
EPSILONS="${EPSILONS:-1e-5,1e-12}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${LOG_ROOT}/epsilon"
STATUS="${LOG_ROOT}/job_status.tsv"
printf "phase\ttask\tmethod\tvariant\tda_impl\tepsilon\tgpu\tpid\tstatus\truntime_s\tcheckpoint_path\tcheckpoint_file_sha256\tcheckpoint_state_dict_sha256\tlog_path\tdiag_path\tcommon_eval_path\n" > "${STATUS}"

dataset_for() {
  case "$1" in
    AT1) echo "austria/33UVP/2017" ;;
    FR1) echo "france/30TXT/2017" ;;
    FR2) echo "france/31TCJ/2017" ;;
    *) echo "ERROR unknown domain: $1" >&2; return 1 ;;
  esac
}

checkpoint_for() {
  local source_domain="$1" method="$2"
  awk -F '\t' -v source_domain="${source_domain}" -v method="${method}" '
    NR == 1 { for (i=1; i<=NF; i++) idx[$i]=i; next }
    $idx["source_domain"] == source_domain && $idx["seed"] == "1" && $idx["source_config"] == method {
      print $idx["checkpoint_path"]; exit
    }
  ' "${SOURCE_INVENTORY}"
}

run_job() {
  local gpu="$1" task="$2" method="$3" epsilon="$4"
  local source_domain="${task%%_to_*}" target_domain="${task##*_to_}"
  local source_dataset target_dataset checkpoint source_dir job_dir log_path diag_path hash_json
  local file_hash state_hash exp_name start status runtime
  source_dataset="$(dataset_for "${source_domain}")" || return 2
  target_dataset="$(dataset_for "${target_domain}")" || return 2
  checkpoint="$(checkpoint_for "${source_domain}" "${method}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "MISSING_CHECKPOINT|task=${task}|method=${method}|path=${checkpoint}" >&2
    return 2
  fi
  source_dir="$(dirname "$(dirname "${checkpoint}")")"
  job_dir="${LOG_ROOT}/epsilon/${task}/${method}/eps_${epsilon}"
  log_path="${job_dir}/train.log"
  diag_path="${job_dir}/timematch_diag.tsv"
  mkdir -p "${job_dir}"
  hash_json="$(cd "${ROOT}" && python -B tools/hash_checkpoint.py "${checkpoint}")" || return 2
  file_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["file_sha256"])' "${hash_json}")"
  state_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["state_dict_sha256"])' "${hash_json}")"
  exp_name="v28_epsilon_${task}_${method}_eps${epsilon}_seed1"
  start="$(date +%s)"
  (
    cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python -B train.py \
      --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False \
      --num_workers "${NUM_WORKERS}" --seed 1 \
      --output_dir "${job_dir}/outputs" --tensorboard_log_dir "${job_dir}/runs" \
      -e "${exp_name}" --source "${source_dataset}" --target "${target_dataset}" \
      timematch --weights "${source_dir}" --epochs 20 --steps_per_epoch 500 \
      --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 \
      --timematch_topk_shifts 5 \
      --timematch_shift_score_epsilon "${epsilon}" \
      --timematch_diagnostic_task "${task}" --timematch_diagnostic_log_path "${diag_path}"
  ) > "${log_path}" 2>&1
  status=$?
  runtime=$(( $(date +%s) - start ))
  printf "epsilon\t%s\t%s\t%s\tcleaned\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n" \
    "${task}" "${method}" "eps_${epsilon}" "${epsilon}" "${gpu}" "${BASHPID}" "${status}" "${runtime}" \
    "${checkpoint}" "${file_hash}" "${state_hash}" "${log_path}" "${diag_path}" >> "${STATUS}"
  return "${status}"
}

read -r -a GPU_IDS <<< "${GPUS}"
IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
IFS=',' read -r -a EPS_ARRAY <<< "${EPSILONS}"
jobs=()
for task in "${TASK_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    source_domain="${task%%_to_*}"
    checkpoint="$(checkpoint_for "${source_domain}" "${method}")"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "ERROR preflight missing checkpoint: task=${task} method=${method} path=${checkpoint}" >&2
      exit 2
    fi
    for epsilon in "${EPS_ARRAY[@]}"; do
      jobs+=("${task}|${method}|${epsilon}")
    done
  done
done

overall=0
for ((offset=0; offset<${#jobs[@]}; offset+=${#GPU_IDS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPU_IDS[@]} && offset+slot<${#jobs[@]}; slot++)); do
    IFS='|' read -r task method epsilon <<< "${jobs[$((offset+slot))]}"
    run_job "${GPU_IDS[slot]}" "${task}" "${method}" "${epsilon}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
done

python -B "${ROOT}/tools/summarize_v28_da_causal_audit.py" \
  --log_root "${LOG_ROOT}" --output "${LOG_ROOT}/epsilon_summary.tsv" || overall=1
echo "DONE|phase=epsilon|status=${overall}|log_root=${LOG_ROOT}"
exit "${overall}"
