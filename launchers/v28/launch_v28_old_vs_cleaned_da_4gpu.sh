#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANED_ROOT="${CLEANED_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OLD_ROOT="${OLD_ROOT:-/data/user/timematch_old_da_f04e1e0}"
OLD_COMMIT="${OLD_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_da_causal_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${CLEANED_ROOT}/logs/${RUN_TAG}}"
SOURCE_INVENTORY="${SOURCE_INVENTORY:-${CLEANED_ROOT}/logs/v28_cleaned_full12_20260710_120037/source_checkpoint_inventory.tsv}"
TASKS="${TASKS:-AT1_to_FR2,FR2_to_FR1}"
METHODS="${METHODS:-base,smooth_k3}"
IMPLEMENTATIONS="${IMPLEMENTATIONS:-old,cleaned}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ -f "${OLD_ROOT}/.v28_old_commit" ]]; then
  actual_old_commit="$(cat "${OLD_ROOT}/.v28_old_commit")"
elif git -C "${OLD_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  actual_old_commit="$(git -C "${OLD_ROOT}" rev-parse HEAD)"
else
  echo "ERROR missing exported old source: ${OLD_ROOT}" >&2
  echo "Run bash ./sync_v28_old_da_to_server.sh locally before starting this audit." >&2
  exit 2
fi
if [[ "${actual_old_commit}" != "${OLD_COMMIT}" ]]; then
  echo "ERROR old source commit mismatch: ${actual_old_commit}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}/old_vs_cleaned" "${LOG_ROOT}/common_eval"
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
  local gpu="$1" task="$2" method="$3" implementation="$4"
  local source_domain="${task%%_to_*}" target_domain="${task##*_to_}"
  local source_dataset target_dataset checkpoint source_dir job_dir log_path diag_path common_path
  local hash_json file_hash state_hash repo exp_name output_base final_checkpoint start status runtime
  source_dataset="$(dataset_for "${source_domain}")" || return 2
  target_dataset="$(dataset_for "${target_domain}")" || return 2
  checkpoint="$(checkpoint_for "${source_domain}" "${method}")"
  [[ -f "${checkpoint}" ]] || { echo "MISSING_CHECKPOINT|${checkpoint}" >&2; return 2; }
  source_dir="$(dirname "$(dirname "${checkpoint}")")"
  repo="${CLEANED_ROOT}"
  [[ "${implementation}" == "old" ]] && repo="${OLD_ROOT}"
  job_dir="${LOG_ROOT}/old_vs_cleaned/${task}/${method}/${implementation}"
  log_path="${job_dir}/train.log"
  diag_path="${job_dir}/timematch_diag.tsv"
  common_path="${LOG_ROOT}/common_eval/${task}_${method}_${implementation}.json"
  output_base="${job_dir}/outputs"
  exp_name="v28_sameckpt_${task}_${method}_${implementation}_seed1"
  mkdir -p "${job_dir}"
  hash_json="$(cd "${CLEANED_ROOT}" && python -B tools/hash_checkpoint.py "${checkpoint}")" || return 2
  file_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["file_sha256"])' "${hash_json}")"
  state_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["state_dict_sha256"])' "${hash_json}")"
  start="$(date +%s)"
  common_args=(
    --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False
    --num_workers "${NUM_WORKERS}" --seed 1
    --output_dir "${output_base}" --tensorboard_log_dir "${job_dir}/runs"
    -e "${exp_name}" --source "${source_dataset}" --target "${target_dataset}"
    timematch --weights "${source_dir}" --epochs 20 --steps_per_epoch 500
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0
  )
  if [[ "${implementation}" == "cleaned" ]]; then
    common_args+=(--timematch_topk_shifts 5 --timematch_shift_score_epsilon 1e-5 --timematch_diagnostic_task "${task}" --timematch_diagnostic_log_path "${diag_path}")
  fi
  (cd "${repo}" && CUDA_VISIBLE_DEVICES="${gpu}" python -B train.py "${common_args[@]}") > "${log_path}" 2>&1
  status=$?
  final_checkpoint="${output_base}/${exp_name}/fold_0/model.pt"
  if [[ "${status}" -eq 0 && -f "${final_checkpoint}" ]]; then
    (cd "${CLEANED_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python -B tools/evaluate_common_timematch_checkpoint.py \
      --checkpoint "${final_checkpoint}" --data_root "${DATA_ROOT}" \
      --source "${source_dataset}" --target "${target_dataset}" --seed 1 \
      --closed_set True --num_workers "${NUM_WORKERS}" --output "${common_path}") \
      >> "${log_path}" 2>&1 || status=$?
  else
    status=1
  fi
  runtime=$(( $(date +%s) - start ))
  printf "old_vs_cleaned\t%s\t%s\t%s\t%s\t1e-5\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${method}" "${implementation}" "${implementation}" "${gpu}" "${BASHPID}" "${status}" "${runtime}" \
    "${checkpoint}" "${file_hash}" "${state_hash}" "${log_path}" "${diag_path}" "${common_path}" >> "${STATUS}"
  return "${status}"
}

read -r -a GPU_IDS <<< "${GPUS}"
IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
IFS=',' read -r -a IMPL_ARRAY <<< "${IMPLEMENTATIONS}"
jobs=()
for task in "${TASK_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    source_domain="${task%%_to_*}"
    checkpoint="$(checkpoint_for "${source_domain}" "${method}")"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "ERROR preflight missing checkpoint: task=${task} method=${method} path=${checkpoint}" >&2
      exit 2
    fi
    for implementation in "${IMPL_ARRAY[@]}"; do
      jobs+=("${task}|${method}|${implementation}")
    done
  done
done

overall=0
for ((offset=0; offset<${#jobs[@]}; offset+=${#GPU_IDS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPU_IDS[@]} && offset+slot<${#jobs[@]}; slot++)); do
    IFS='|' read -r task method implementation <<< "${jobs[$((offset+slot))]}"
    run_job "${GPU_IDS[slot]}" "${task}" "${method}" "${implementation}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
done

python -B "${CLEANED_ROOT}/tools/summarize_v28_da_causal_audit.py" \
  --log_root "${LOG_ROOT}" --output "${LOG_ROOT}/old_vs_cleaned_summary.tsv" || overall=1
echo "DONE|phase=old_vs_cleaned|status=${overall}|log_root=${LOG_ROOT}"
exit "${overall}"
