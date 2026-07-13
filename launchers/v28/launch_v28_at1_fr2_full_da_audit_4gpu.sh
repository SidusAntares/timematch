#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANED_ROOT="${CLEANED_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OLD_ROOT="${OLD_ROOT:-/data/user/timematch_old_da_f04e1e0}"
OLD_COMMIT="${OLD_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE_INVENTORY="${SOURCE_INVENTORY:-${CLEANED_ROOT}/logs/v28_cleaned_full12_20260710_120037/source_checkpoint_inventory.tsv}"
RUN_TAG="${RUN_TAG:-v28_at1_fr2_full_da_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${CLEANED_ROOT}/logs/${RUN_TAG}}"
TASK="${TASK:-AT1_to_FR2}"
METHODS="${METHODS:-base,smooth_k3}"
IMPLEMENTATIONS="${IMPLEMENTATIONS:-old,cleaned}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

[[ "${TASK}" == "AT1_to_FR2" ]] || { echo "ERROR Stage 2 is restricted to AT1_to_FR2" >&2; exit 2; }
[[ -f "${OLD_ROOT}/.v28_old_commit" && "$(cat "${OLD_ROOT}/.v28_old_commit")" == "${OLD_COMMIT}" ]] || {
  echo "ERROR old DA export missing or mismatched: ${OLD_ROOT}" >&2
  exit 2
}
mkdir -p "${LOG_ROOT}/jobs" "${LOG_ROOT}/common_eval"
status_file="${LOG_ROOT}/job_status.tsv"
printf "phase\ttask\tmethod\tvariant\tda_impl\tepsilon\tgpu\tpid\tstatus\truntime_s\tcheckpoint_path\tcheckpoint_file_sha256\tcheckpoint_state_dict_sha256\tlog_path\tdiag_path\tcommon_eval_path\trun_summary_path\n" > "${status_file}"

checkpoint_for() {
  local method="$1"
  awk -F '\t' -v method="${method}" '
    NR == 1 { for (i=1; i<=NF; i++) idx[$i]=i; next }
    $idx["source_domain"] == "AT1" && $idx["seed"] == "1" && $idx["source_config"] == method {
      print $idx["checkpoint_path"]; exit
    }
  ' "${SOURCE_INVENTORY}"
}

run_job() {
  local gpu="$1" method="$2" implementation="$3"
  local checkpoint source_dir repo job_dir output_dir experiment log diag common run_summary
  local hash_json file_hash state_hash final_checkpoint start status runtime
  checkpoint="$(checkpoint_for "${method}")"
  [[ -f "${checkpoint}" ]] || { echo "ERROR missing checkpoint: ${checkpoint}" >&2; return 2; }
  source_dir="$(dirname "$(dirname "${checkpoint}")")"
  repo="${CLEANED_ROOT}"
  [[ "${implementation}" == "old" ]] && repo="${OLD_ROOT}"
  job_dir="${LOG_ROOT}/jobs/${method}/${implementation}"
  output_dir="${job_dir}/outputs"
  experiment="v28_at1_fr2_${method}_${implementation}_seed1"
  log="${job_dir}/train.log"
  diag="${job_dir}/epoch_trace.tsv"
  common="${LOG_ROOT}/common_eval/${method}_${implementation}.json"
  run_summary="${job_dir}/run_summary.json"
  mkdir -p "${job_dir}"
  hash_json="$(cd "${CLEANED_ROOT}" && python -B tools/hash_checkpoint.py "${checkpoint}")" || return 2
  file_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["file_sha256"])' "${hash_json}")"
  state_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["state_dict_sha256"])' "${hash_json}")"
  args=(
    --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False
    --batch_size 128 --num_workers "${NUM_WORKERS}" --seed 1
    --output_dir "${output_dir}" --tensorboard_log_dir "${job_dir}/runs"
    -e "${experiment}" --source "austria/33UVP/2017" --target "france/31TCJ/2017"
    timematch --weights "${source_dir}" --epochs 20 --steps_per_epoch 500
    --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0
  )
  [[ "${implementation}" == "cleaned" ]] && args+=(--timematch_shift_score_epsilon 1e-5)
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" python -B "${CLEANED_ROOT}/tools/run_v28_full_da_audit.py" \
    --implementation "${implementation}" --repo_root "${repo}" \
    --diag_output "${diag}" --run_summary "${run_summary}" -- "${args[@]}" > "${log}" 2>&1
  status=$?
  final_checkpoint="${output_dir}/${experiment}/fold_0/model.pt"
  if [[ "${status}" -eq 0 && -f "${final_checkpoint}" ]]; then
    (cd "${CLEANED_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python -B tools/evaluate_common_timematch_checkpoint.py \
      --checkpoint "${final_checkpoint}" --data_root "${DATA_ROOT}" \
      --source "austria/33UVP/2017" --target "france/31TCJ/2017" \
      --seed 1 --closed_set True --num_workers "${NUM_WORKERS}" --output "${common}") >> "${log}" 2>&1 || status=$?
  else
    status=1
  fi
  runtime=$(( $(date +%s) - start ))
  printf "full_da\t%s\t%s\t%s\t%s\t1e-5\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${TASK}" "${method}" "${implementation}" "${implementation}" "${gpu}" "${BASHPID}" "${status}" "${runtime}" \
    "${checkpoint}" "${file_hash}" "${state_hash}" "${log}" "${diag}" "${common}" "${run_summary}" >> "${status_file}"
  return "${status}"
}

IFS=',' read -r -a method_array <<< "${METHODS}"
IFS=',' read -r -a implementation_array <<< "${IMPLEMENTATIONS}"
read -r -a gpu_array <<< "${GPUS}"
jobs=()
for method in "${method_array[@]}"; do
  checkpoint="$(checkpoint_for "${method}")"
  [[ -f "${checkpoint}" ]] || { echo "ERROR preflight missing checkpoint: ${checkpoint}" >&2; exit 2; }
  for implementation in "${implementation_array[@]}"; do jobs+=("${method}|${implementation}"); done
done
[[ "${#jobs[@]}" -eq 4 ]] || { echo "ERROR expected exactly four full DA jobs" >&2; exit 2; }
[[ "${#gpu_array[@]}" -ge 4 ]] || { echo "ERROR four GPUs are required" >&2; exit 2; }

pids=()
for index in "${!jobs[@]}"; do
  IFS='|' read -r method implementation <<< "${jobs[index]}"
  run_job "${gpu_array[index]}" "${method}" "${implementation}" &
  pids+=("$!")
done
overall=0
for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
python -B "${CLEANED_ROOT}/tools/summarize_v28_da_causal_audit.py" \
  --log_root "${LOG_ROOT}" --output "${LOG_ROOT}/full_da_summary.tsv" || overall=1
echo "DONE|status=${overall}|summary=${LOG_ROOT}/full_da_summary.tsv|log_root=${LOG_ROOT}"
exit "${overall}"
