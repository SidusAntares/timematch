#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANED_ROOT="${CLEANED_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/user/timematch_old_source_89d9df4}"
SOURCE_COMMIT="${SOURCE_COMMIT:-89d9df4e52744cb955168b0d203a2ddd61c3199e}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_source_task_specific_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${CLEANED_ROOT}/logs/${RUN_TAG}}"
TASKS="${TASKS:-FR1_to_FR2,FR1_to_DK1}"
CONFIGS="${CONFIGS:-base,smooth_k3}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-1}"
GPUS="${GPUS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-16}"

marker="${SOURCE_ROOT}/.v28_source_commit"
[[ -f "${marker}" && "$(cat "${marker}")" == "${SOURCE_COMMIT}" ]] || {
  echo "ERROR source export missing or mismatched: ${SOURCE_ROOT}" >&2
  exit 2
}

mkdir -p "${LOG_ROOT}/jobs"
status_file="${LOG_ROOT}/job_status.tsv"
summary_file="${LOG_ROOT}/source_task_specific_smoke.tsv"
printf "task\tconfig\tgpu\tpid\tstatus\tlog_path\tresult_path\n" > "${status_file}"

run_job() {
  local gpu="$1" task="$2" config="$3"
  local job_dir="${LOG_ROOT}/jobs/${task}/${config}"
  local result="${job_dir}/result.tsv" log="${job_dir}/train.log"
  local experiment="v28_source_task_smoke_${task}_${config}_seed1"
  local output_dir="${job_dir}/outputs" status
  mkdir -p "${job_dir}"
  args=(
    --data_root "${DATA_ROOT}" --closed_set True --with_shift_aug False
    --epochs "${SOURCE_EPOCHS}" --num_workers "${NUM_WORKERS}" --seed 1
    --output_dir "${output_dir}" --tensorboard_log_dir "${job_dir}/runs"
    -e "${experiment}" --source "france/30TXT/2017" --target "france/30TXT/2017"
  )
  if [[ "${config}" == "smooth_k3" ]]; then
    args+=(
      --source_phase_partition_mode uniform --source_segment_partition_mode uniform
      --source_phase_count 1 --source_segment_count 1
      --source_structure_loss_version v276_raw_smoothed_timepoint_compactness
      --source_structure_feature_target raw --source_structure_detach_features False
      --source_structure_intra_trade_off 1.0
      --source_structure_amplitude_trade_off 0.0 --source_structure_interphase_trade_off 0.0
      --source_structure_shape_trade_off 0.0 --source_structure_trend_trade_off 0.0
      --source_structure_season_trade_off 0.0 --source_structure_segment_inter_trade_off 0.0
      --source_structure_boundary_window_trade_off 0.0 --source_structure_compact_distance mse
      sourcephasecompact
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python -B "${CLEANED_ROOT}/tools/trace_v28_source_task_smoke.py" \
    --repo_root "${SOURCE_ROOT}" --task "${task}" --config "${config}" --output "${result}" -- \
    "${args[@]}" > "${log}" 2>&1
  status=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${config}" "${gpu}" "${BASHPID}" "${status}" "${log}" "${result}" >> "${status_file}"
  return "${status}"
}

IFS=',' read -r -a task_array <<< "${TASKS}"
IFS=',' read -r -a config_array <<< "${CONFIGS}"
read -r -a gpu_array <<< "${GPUS}"
jobs=()
for task in "${task_array[@]}"; do
  for config in "${config_array[@]}"; do jobs+=("${task}|${config}"); done
done
[[ "${#jobs[@]}" -eq 4 ]] || { echo "ERROR expected exactly four smoke cells" >&2; exit 2; }

pids=()
for index in "${!jobs[@]}"; do
  IFS='|' read -r task config <<< "${jobs[index]}"
  run_job "${gpu_array[index]}" "${task}" "${config}" &
  pids+=("$!")
done
overall=0
for pid in "${pids[@]}"; do wait "${pid}" || overall=1; done

first_result="${LOG_ROOT}/jobs/${task_array[0]}/${config_array[0]}/result.tsv"
if [[ -f "${first_result}" ]]; then
  head -n 1 "${first_result}" > "${summary_file}"
  find "${LOG_ROOT}/jobs" -name result.tsv -print0 | sort -z | while IFS= read -r -d '' result; do
    tail -n 1 "${result}" >> "${summary_file}"
  done
else
  overall=1
fi
if [[ "${overall}" -eq 0 ]]; then
  python -B "${CLEANED_ROOT}/tools/check_v28_source_task_smoke.py" \
    --input "${summary_file}" --output "${LOG_ROOT}/source_task_specific_comparison.tsv" || overall=1
fi
echo "DONE|status=${overall}|summary=${summary_file}|log_root=${LOG_ROOT}"
exit "${overall}"
