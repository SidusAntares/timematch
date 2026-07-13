#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OLD_DA_ROOT="${OLD_DA_ROOT:-/data/user/timematch_old_da_f04e1e0}"
OLD_DA_COMMIT="${OLD_DA_COMMIT:-f04e1e06805270d4e98db688ae869fbdeb6493b6}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:?Set FIRST_ROUND_ROOT to the completed first-round log directory}"
RUN_TAG="${RUN_TAG:-v28_source_da_cross_audit_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_TAG}}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REUSE_EXISTING_CROSS="${REUSE_EXISTING_CROSS:-True}"

GATE="${FIRST_ROUND_ROOT}/FIRST_ROUND_PASSED"
SOURCE_MANIFEST="${FIRST_ROUND_ROOT}/checkpoint_audit/source_checkpoint_manifest.tsv"
EXISTING_DA_EVAL="${FIRST_ROUND_ROOT}/existing_da_common_eval.tsv"
[[ -f "${GATE}" ]] || { echo "ERROR first-round gate has not passed: ${GATE}" >&2; exit 2; }
[[ -f "${SOURCE_MANIFEST}" && -f "${EXISTING_DA_EVAL}" ]] || { echo "ERROR missing first-round artifacts" >&2; exit 2; }
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
mkdir -p "${LOG_ROOT}/cross_jobs" "${LOG_ROOT}/common_eval" "${LOG_ROOT}/cross_status"

dataset_for() {
  case "$1" in
    AT1) echo "austria/33UVP/2017" ;;
    FR1) echo "france/30TXT/2017" ;;
    FR2) echo "france/31TCJ/2017" ;;
    *) echo "ERROR unknown domain $1" >&2; return 1 ;;
  esac
}

checkpoint_for() {
  local source="$1" method="$2" version="$3"
  python -B "${ROOT}/tools/v28_source_da_cross_audit.py" lookup-source-checkpoint \
    --manifest "${SOURCE_MANIFEST}" --source "${source}" --config "${method}" \
    --version "${version}" --seed 1
}

jobs=()
for task in AT1_to_FR2 FR2_to_FR1; do
  source="${task%%_to_*}"
  for method in base smooth_k3; do
    jobs+=("${task}|${method}|old|cleaned")
    jobs+=("${task}|${method}|cleaned|old")
  done
done
[[ "${#jobs[@]}" -eq 8 ]] || exit 2

for job in "${jobs[@]}"; do
  IFS='|' read -r task method source_version da_version <<< "${job}"
  source="${task%%_to_*}"
  checkpoint="$(checkpoint_for "${source}" "${method}" "${source_version}")" || exit 2
  [[ -f "${checkpoint}" ]] || {
    echo "ERROR cross preflight missing source checkpoint: ${checkpoint}" >&2
    exit 2
  }
done
echo "CROSS_PREFLIGHT_OK|jobs=8"

status_file="${LOG_ROOT}/job_status.tsv"
status_header="job_id\ttask\tmethod\tsource_version\tda_version\tseed\tgpu\tpid\tstatus\treturn_code\truntime_s\tlog_path\tcheckpoint_path\tnative_test_f1\tcommon_macro_f1\terror_message\tstart_time\tend_time\tcommand\tgit_commit\tpython_version\ttorch_version\tcuda_version\tsource_file_sha256\tsource_state_dict_sha256\tdiag_path\tcommon_eval_path\trun_summary_path\tfinal_student_checkpoint_path\tvariant\tda_impl\tepsilon\tcheckpoint_file_sha256\tcheckpoint_state_dict_sha256"

refresh_status() {
  printf "%b\n" "${status_header}" > "${status_file}.tmp"
  find "${LOG_ROOT}/cross_status" -maxdepth 1 -type f -name '*.tsv' -print0 \
    | sort -z | while IFS= read -r -d '' file; do cat "${file}" >> "${status_file}.tmp"; done
  mv "${status_file}.tmp" "${status_file}"
}

run_job() {
  local gpu="$1" job_id="$2" task="$3" method="$4" source_version="$5" da_version="$6"
  local source target source_dataset target_dataset checkpoint source_dir repo job_dir output_dir experiment
  local log diag common run_summary hash_json file_hash state_hash start status runtime final_checkpoint status_path
  local start_time end_time command git_commit python_version torch_version cuda_version native_f1 common_f1 error_message status_name
  source="${task%%_to_*}"; target="${task##*_to_}"
  source_dataset="$(dataset_for "${source}")" || return 2
  target_dataset="$(dataset_for "${target}")" || return 2
  checkpoint="$(checkpoint_for "${source}" "${method}" "${source_version}")"
  [[ -f "${checkpoint}" ]] || { echo "ERROR missing source checkpoint ${checkpoint}" >&2; return 2; }
  source_dir="$(dirname "$(dirname "${checkpoint}")")"
  repo="${ROOT}"; [[ "${da_version}" == "old" ]] && repo="${OLD_DA_ROOT}"
  job_dir="${LOG_ROOT}/cross_jobs/${task}/${method}/${source_version}_source_${da_version}_da"
  output_dir="${job_dir}/outputs"
  experiment="v28_cross_${task}_${method}_${source_version}_source_${da_version}_da_seed1"
  log="${job_dir}/train.log"; diag="${job_dir}/epoch_trace.tsv"
  common="${LOG_ROOT}/common_eval/${task}_${method}_${source_version}_source_${da_version}_da.json"
  run_summary="${job_dir}/run_summary.json"
  status_path="${LOG_ROOT}/cross_status/${task}_${method}_${source_version}_source_${da_version}_da.tsv"
  final_checkpoint="${output_dir}/${experiment}/fold_0/model.pt"
  mkdir -p "${job_dir}"
  if [[ "${REUSE_EXISTING_CROSS}" == "True" && -f "${status_path}" && -f "${final_checkpoint}" \
    && -f "${common}" && -f "${run_summary}" \
    && "$(cut -f9 "${status_path}")" == "success" ]]; then
    echo "REUSED_CROSS|task=${task}|method=${method}|source=${source_version}|da=${da_version}"
    return 0
  fi
  hash_json="$(cd "${ROOT}" && python -B tools/hash_checkpoint.py "${checkpoint}")" || return 2
  file_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["file_sha256"])' "${hash_json}")"
  state_hash="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["state_dict_sha256"])' "${hash_json}")"
  git_commit="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || cat "${repo}/.v28_old_commit" 2>/dev/null || echo unknown)"
  python_version="$(python -c 'import platform; print(platform.python_version())')"
  torch_version="$(python -c 'import torch; print(torch.__version__)')"
  cuda_version="$(python -c 'import torch; print(torch.version.cuda or "none")')"
  args=(
    --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False
    --batch_size 128 --num_workers "${NUM_WORKERS}" --seed 1 --weight_decay 1e-4
    --output_dir "${output_dir}" --tensorboard_log_dir "${job_dir}/runs"
    -e "${experiment}" --source "${source_dataset}" --target "${target_dataset}"
    timematch --weights "${source_dir}" --epochs 20 --steps_per_epoch 500
    --lr 0.0001 --pseudo_threshold 0.9 --ema_decay 0.9999 --trade_off 2.0 --output_student True
    --domain_specific_bn True --shift_estimator AM --sample_size 100 --max_temporal_shift 60
    --balance_source True --use_focal_loss True --shift_source True
  )
  [[ "${da_version}" == "cleaned" ]] && args+=(--timematch_shift_score_epsilon 1e-5)
  command="CUDA_VISIBLE_DEVICES=${gpu} python -B ${ROOT}/tools/run_v28_full_da_audit.py --implementation ${da_version} --repo_root ${repo} -- ${args[*]}"
  start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\trunning\t\t0\t%s\t%s\t\t\t\t%s\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t1e-5\t%s\t%s\n" \
    "${job_id}" "${task}" "${method}" "${source_version}" "${da_version}" "${gpu}" "${BASHPID}" \
    "${log}" "${checkpoint}" "${start_time}" "${command}" "${git_commit}" "${python_version}" "${torch_version}" "${cuda_version}" \
    "${file_hash}" "${state_hash}" "${diag}" "${common}" "${run_summary}" "${final_checkpoint}" \
    "${source_version}_source_${da_version}_da" "${da_version}" "${checkpoint}" "${state_hash}" > "${status_path}"
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" python -B "${ROOT}/tools/run_v28_full_da_audit.py" \
    --implementation "${da_version}" --repo_root "${repo}" \
    --diag_output "${diag}" --run_summary "${run_summary}" -- "${args[@]}" > "${log}" 2>&1
  status=$?
  if [[ "${status}" -eq 0 && -f "${final_checkpoint}" ]]; then
    (cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python -B tools/evaluate_common_timematch_checkpoint.py \
      --checkpoint "${final_checkpoint}" --data_root "${DATA_ROOT}" \
      --source "${source_dataset}" --target "${target_dataset}" --seed 1 \
      --closed_set True --num_workers "${NUM_WORKERS}" --batch_size 128 --output "${common}") >> "${log}" 2>&1 || status=$?
  else
    status=1
  fi
  runtime=$(( $(date +%s) - start ))
  end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  native_f1="$(python -c 'import pathlib,re,sys; x=re.findall(r"Test result for [^:]+:\s*accuracy=[-+0-9.eE]+,\s*f1=([-+0-9.eE]+)", pathlib.Path(sys.argv[1]).read_text(errors="replace")); print(x[-1] if x else "")' "${log}")"
  common_f1="$(python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("macro_f1", "") if p.is_file() else "")' "${common}")"
  status_name="success"; error_message=""
  if [[ "${status}" -ne 0 ]]; then
    status_name="failed"
    error_message="$(tail -n 1 "${log}" 2>/dev/null | tr '\t\r\n' '   ')"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" "${task}" "${method}" "${source_version}" "${da_version}" "${gpu}" "${BASHPID}" \
    "${status_name}" "${status}" "${runtime}" "${log}" "${checkpoint}" "${native_f1}" "${common_f1}" "${error_message}" \
    "${start_time}" "${end_time}" "${command}" "${git_commit}" "${python_version}" "${torch_version}" "${cuda_version}" \
    "${file_hash}" "${state_hash}" "${diag}" "${common}" "${run_summary}" "${final_checkpoint}" \
    "${source_version}_source_${da_version}_da" "${da_version}" "1e-5" "${checkpoint}" "${state_hash}" > "${status_path}"
  return "${status}"
}

read -r -a gpu_ids <<< "${GPUS}"
overall=0
for ((offset=0; offset<${#jobs[@]}; offset+=${#gpu_ids[@]})); do
  pids=()
  for ((slot=0; slot<${#gpu_ids[@]} && offset+slot<${#jobs[@]}; slot++)); do
    IFS='|' read -r task method source_version da_version <<< "${jobs[$((offset+slot))]}"
    run_job "${gpu_ids[slot]}" "$((offset+slot))" "${task}" "${method}" "${source_version}" "${da_version}" &
    pids+=("$!")
  done
  sleep 1
  refresh_status
  for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done
  refresh_status
done

refresh_status
[[ "$(($(wc -l < "${status_file}") - 1))" -eq 8 ]] || {
  echo "ERROR expected 8 cross-job status rows" >&2
  overall=1
}

python -B "${ROOT}/tools/summarize_v28_da_causal_audit.py" \
  --log_root "${LOG_ROOT}" --output "${LOG_ROOT}/cross_summary.tsv" || overall=1
python -B "${ROOT}/tools/v28_source_da_cross_audit.py" write-cross-common \
  --cross_summary "${LOG_ROOT}/cross_summary.tsv" \
  --output "${LOG_ROOT}/cross_da_common_eval.tsv" || overall=1
python -B "${ROOT}/tools/v28_source_da_cross_audit.py" summarize-cross \
  --existing_da_eval "${EXISTING_DA_EVAL}" --cross_summary "${LOG_ROOT}/cross_summary.tsv" \
  --results_output "${LOG_ROOT}/source_da_2x2_results.tsv" \
  --output "${LOG_ROOT}/source_da_2x2_effects.tsv" || overall=1
python -B "${ROOT}/tools/v28_source_da_cross_audit.py" write-final-report \
  --gate "${FIRST_ROUND_ROOT}/audit_gate_summary.tsv" \
  --job_status "${LOG_ROOT}/job_status.tsv" \
  --cross_common "${LOG_ROOT}/cross_da_common_eval.tsv" \
  --results "${LOG_ROOT}/source_da_2x2_results.tsv" \
  --effects "${LOG_ROOT}/source_da_2x2_effects.tsv" \
  --output "${ROOT}/analysis/v28_source_checkpoint_da_cross_audit.md" || overall=1
echo "CROSS_AUDIT_DONE|status=${overall}|log_root=${LOG_ROOT}|table=${LOG_ROOT}/source_da_2x2_effects.tsv"
exit "${overall}"
