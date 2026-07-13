#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
RUN_TAG="${RUN_TAG:-v28_cleaned_repro_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_TAG}}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-AT1_to_DK1,FR1_to_FR2,FR2_to_AT1,FR2_to_FR1}"
SEEDS="${SEEDS:-1 2 3}"
CONFIGS="${CONFIGS:-base,raw_global,smooth_k3,elastic_r2}"
CLOSED_SET="${CLOSED_SET:-True}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_LOADER_TIMEOUT="${DATA_LOADER_TIMEOUT:-60}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "${LOG_DIR}/source_train"
INVENTORY="${LOG_DIR}/source_checkpoint_inventory.tsv"
SOURCE_JOBS="${LOG_DIR}/source_jobs.tsv"
SOURCE_STATUS="${LOG_DIR}/source_train_status.tsv"

read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

task_spec() {
  case "$1" in
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 AT1" ;;
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 FR1" ;;
    FR2_to_AT1) echo "france/31TCJ/2017 austria/33UVP/2017 FR2" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 FR2" ;;
    *) echo "ERROR unknown task: $1" >&2; return 1 ;;
  esac
}

source_experiment_name() {
  local source_alias="$1"
  local config="$2"
  local seed="$3"
  echo "v28_cleaned_source_${source_alias}_${config}_seed${seed}"
}

source_loss_args() {
  local config="$1"
  case "${config}" in
    raw_global)
      echo "--source_structure_loss_version v275_raw_global_compactness --source_structure_time_smooth_kernel_size 3 --source_structure_elastic_radius 0"
      ;;
    smooth_k3)
      echo "--source_structure_loss_version v276_raw_smoothed_timepoint_compactness --source_structure_time_smooth_kernel_size 3 --source_structure_elastic_radius 0"
      ;;
    elastic_r2)
      echo "--source_structure_loss_version v284_elastic_smoothed_timepoint_compactness --source_structure_time_smooth_kernel_size 3 --source_structure_elastic_radius 2 --source_structure_elastic_eta 0.1 --source_structure_elastic_softmin_tau 0.1 --source_structure_elastic_detach_center False"
      ;;
    *)
      echo ""
      ;;
  esac
}

write_inventory() {
  local status_mode="$1"
  printf "task\tseed\tconfig\tsource_region\tcheckpoint_path\tcheckpoint_exists\tsource_training_needed\tsource_train_status\n" > "${INVENTORY}"
  IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
  IFS=',' read -r -a CONFIG_ARRAY <<< "${CONFIGS}"
  for raw_task in "${TASK_ARRAY[@]}"; do
    task="$(echo "${raw_task}" | xargs)"
    read -r source_dataset _target_dataset source_alias <<< "$(task_spec "${task}")" || exit 2
    for seed in ${SEEDS}; do
      for raw_config in "${CONFIG_ARRAY[@]}"; do
        config="$(echo "${raw_config}" | xargs)"
        source_name="$(source_experiment_name "${source_alias}" "${config}" "${seed}")"
        checkpoint="${ROOT}/outputs/${source_name}/fold_0/model.pt"
        exists="False"
        needed="True"
        train_status="missing"
        recorded_status=""
        if [[ "${status_mode}" == "after" && -f "${SOURCE_STATUS}" ]]; then
          recorded_status="$(awk -F '\t' -v source_alias="${source_alias}" -v seed="${seed}" -v config="${config}" '
            NR == 1 {
              for (i = 1; i <= NF; i++) idx[$i] = i
              next
            }
            $idx["source_region"] == source_alias && $idx["seed"] == seed && $idx["config"] == config {
              print $idx["status"]
              exit
            }
          ' "${SOURCE_STATUS}")"
        fi
        if [[ -f "${checkpoint}" ]]; then
          exists="True"
          if [[ "${status_mode}" == "after" && -n "${recorded_status}" ]]; then
            train_status="${recorded_status}"
            if [[ "${recorded_status}" == "reused" ]]; then
              needed="False"
            else
              needed="True"
            fi
          else
            train_status="reused"
            needed="False"
          fi
        elif [[ "${status_mode}" == "after" ]]; then
          if [[ -n "${recorded_status}" ]]; then
            train_status="${recorded_status}"
          else
            train_status="failed"
          fi
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${task}" "${seed}" "${config}" "${source_alias}" "${checkpoint}" "${exists}" "${needed}" "${train_status}" \
          >> "${INVENTORY}"
      done
    done
  done
}

build_source_jobs() {
  : > "${SOURCE_JOBS}"
  printf "source_region\tseed\tconfig\tsource_dataset\texperiment_name\tcheckpoint_path\tstatus\n" > "${SOURCE_STATUS}"
  declare -A seen=()
  IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
  IFS=',' read -r -a CONFIG_ARRAY <<< "${CONFIGS}"
  for raw_task in "${TASK_ARRAY[@]}"; do
    task="$(echo "${raw_task}" | xargs)"
    read -r source_dataset _target_dataset source_alias <<< "$(task_spec "${task}")" || exit 2
    for seed in ${SEEDS}; do
      for raw_config in "${CONFIG_ARRAY[@]}"; do
        config="$(echo "${raw_config}" | xargs)"
        case "${config}" in base|raw_global|smooth_k3|elastic_r2) ;; *) echo "ERROR unknown config ${config}" >&2; exit 2 ;; esac
        key="${source_alias}|${seed}|${config}"
        [[ -n "${seen[${key}]:-}" ]] && continue
        seen["${key}"]=1
        source_name="$(source_experiment_name "${source_alias}" "${config}" "${seed}")"
        checkpoint="${ROOT}/outputs/${source_name}/fold_0/model.pt"
        if [[ -f "${checkpoint}" ]]; then
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${source_alias}" "${seed}" "${config}" "${source_dataset}" "${source_name}" "${checkpoint}" "reused" >> "${SOURCE_STATUS}"
          continue
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${source_alias}" "${seed}" "${config}" "${source_dataset}" "${source_name}" "${checkpoint}" >> "${SOURCE_JOBS}"
      done
    done
  done
}

run_source_job() {
  local gpu="$1"
  local source_alias="$2"
  local seed="$3"
  local config="$4"
  local source_dataset="$5"
  local source_name="$6"
  local checkpoint="$7"
  local log_dir="${LOG_DIR}/source_train/${source_alias}/${config}/seed${seed}"
  local log_path="${log_dir}/train.log"
  mkdir -p "${log_dir}"
  echo "SOURCE_START|source=${source_alias}|config=${config}|seed=${seed}|gpu=${gpu}|log=${log_path}"
  if [[ "${DRY_RUN}" == "True" ]]; then
    echo "DRY_RUN source ${source_alias} ${config} seed${seed}" > "${log_path}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${source_alias}" "${seed}" "${config}" "${source_dataset}" "${source_name}" "${checkpoint}" "dry_run" >> "${SOURCE_STATUS}"
    return 0
  fi
  if [[ "${config}" == "base" ]]; then
    (
      cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
        --data_root "${DATA_ROOT}" \
        --closed_set "${CLOSED_SET}" \
        --with_shift_aug False \
        --epochs "${SOURCE_EPOCHS}" \
        --num_workers "${NUM_WORKERS}" \
        --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
        --seed "${seed}" \
        -e "${source_name}" \
        --source "${source_dataset}" \
        --target "${source_dataset}"
    ) > "${log_path}" 2>&1
  else
    # shellcheck disable=SC2206
    extra_args=($(source_loss_args "${config}"))
    (
      cd "${ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
        --data_root "${DATA_ROOT}" \
        --closed_set "${CLOSED_SET}" \
        --with_shift_aug False \
        --source_phase_partition_mode uniform \
        --source_segment_partition_mode uniform \
        --source_phase_count 1 \
        --source_segment_count 1 \
        --source_structure_feature_target raw \
        --source_structure_detach_features False \
        --source_structure_intra_trade_off 1.0 \
        --source_structure_amplitude_trade_off 0.0 \
        --source_structure_interphase_trade_off 0.0 \
        --source_structure_shape_trade_off 0.0 \
        --source_structure_trend_trade_off 0.0 \
        --source_structure_season_trade_off 0.0 \
        --source_structure_segment_inter_trade_off 0.0 \
        --source_structure_boundary_window_trade_off 0.0 \
        --source_structure_compact_distance mse \
        "${extra_args[@]}" \
        --epochs "${SOURCE_EPOCHS}" \
        --num_workers "${NUM_WORKERS}" \
        --data_loader_timeout "${DATA_LOADER_TIMEOUT}" \
        --seed "${seed}" \
        -e "${source_name}" \
        --source "${source_dataset}" \
        --target "${source_dataset}" \
        sourcephasecompact
    ) > "${log_path}" 2>&1
  fi
  status=$?
  final_status="trained"
  [[ "${status}" -ne 0 || ! -f "${checkpoint}" ]] && final_status="failed"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${source_alias}" "${seed}" "${config}" "${source_dataset}" "${source_name}" "${checkpoint}" "${final_status}" >> "${SOURCE_STATUS}"
  echo "SOURCE_DONE|source=${source_alias}|config=${config}|seed=${seed}|status=${status}|checkpoint_exists=$(test -f "${checkpoint}" && echo True || echo False)"
  return "${status}"
}

run_batch() {
  local -a pids=()
  local batch_status=0
  for item in "$@"; do
    IFS=$'\t' read -r source_alias seed config source_dataset source_name checkpoint <<< "${item}"
    gpu="${GPU_IDS[$(( ${#pids[@]} % ${#GPU_IDS[@]} ))]}"
    run_source_job "${gpu}" "${source_alias}" "${seed}" "${config}" "${source_dataset}" "${source_name}" "${checkpoint}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || batch_status=1
  done
  return "${batch_status}"
}

write_inventory "before"
build_source_jobs

echo "RUN_TAG=${RUN_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "SOURCE_JOBS=$(wc -l < "${SOURCE_JOBS}")"
echo "SOURCE_INVENTORY=${INVENTORY}"

overall_status=0
batch=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  batch+=("${line}")
  if [[ "${#batch[@]}" -ge "${#GPU_IDS[@]}" ]]; then
    run_batch "${batch[@]}" || overall_status=1
    batch=()
  fi
done < "${SOURCE_JOBS}"
if [[ "${#batch[@]}" -gt 0 ]]; then
  run_batch "${batch[@]}" || overall_status=1
fi

write_inventory "after"
echo "SOURCE_INVENTORY=${INVENTORY}"
echo "SOURCE_STATUS=${SOURCE_STATUS}"
exit "${overall_status}"
