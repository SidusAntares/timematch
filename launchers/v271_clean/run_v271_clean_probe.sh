#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v271_clean_probe_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
DA_EPOCHS="${DA_EPOCHS:-20}"
DA_STEPS_PER_EPOCH="${DA_STEPS_PER_EPOCH:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
ADAPTIVE_SUPPORT_ROOT="${ADAPTIVE_SUPPORT_ROOT:-}"
ADAPTIVE_TRADE_OFF="${ADAPTIVE_TRADE_OFF:-0.0}"
ADAPTIVE_WARMUP_EPOCHS="${ADAPTIVE_WARMUP_EPOCHS:-3}"
ADAPTIVE_RAMP_EPOCHS="${ADAPTIVE_RAMP_EPOCHS:-3}"
ADAPTIVE_MIN_SCORE="${ADAPTIVE_MIN_SCORE:-0.0}"
ADAPTIVE_MIN_GATE="${ADAPTIVE_MIN_GATE:-0.0}"
ADAPTIVE_MIN_POINTS="${ADAPTIVE_MIN_POINTS:-2}"

TREND_KERNEL_SIZE="${TREND_KERNEL_SIZE:-5}"
TREND_SMOOTHING_MODE="${TREND_SMOOTHING_MODE:-time}"
TREND_BANDWIDTH="${TREND_BANDWIDTH:-0.0}"
TREND_KERNEL="${TREND_KERNEL:-gaussian}"
TREND_DYNAMICS_TRADE_OFF="${TREND_DYNAMICS_TRADE_OFF:-0.05}"
RESIDUAL_VARIANCE_TRADE_OFF="${RESIDUAL_VARIANCE_TRADE_OFF:-0.10}"
RESIDUAL_ENERGY_TRADE_OFF="${RESIDUAL_ENERGY_TRADE_OFF:-0.05}"
RESIDUAL_ENERGY_MARGIN="${RESIDUAL_ENERGY_MARGIN:-1.0}"

TASKS=(
  "FR2 DK1 france/31TCJ/2017 denmark/32VNH/2017"
  "DK1 FR1 denmark/32VNH/2017 france/30TXT/2017"
  "FR1 DK1 france/30TXT/2017 denmark/32VNH/2017"
  "AT1 DK1 austria/33UVP/2017 denmark/32VNH/2017"
)

TASK_FILTER="${TASK_FILTER:-all}"

should_run_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local filter="${TASK_FILTER}"
  if [ "${filter}" = "all" ] || [ -z "${filter}" ]; then
    return 0
  fi
  local key_arrow="${src_alias}->${tgt_alias}"
  local key_to="${src_alias}_to_${tgt_alias}"
  IFS=',' read -ra items <<< "${filter}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "${item}" = "${key_arrow}" ] || [ "${item}" = "${key_to}" ]; then
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "${running}" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 20
  done
}

support_file_for_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  if [ -z "${ADAPTIVE_SUPPORT_ROOT}" ]; then
    return 1
  fi
  local key="${src_alias}_to_${tgt_alias}"
  local candidates=(
    "${ADAPTIVE_SUPPORT_ROOT}/${key}/adaptive_supports.json"
    "${ADAPTIVE_SUPPORT_ROOT}/${key}/supports.json"
    "${ADAPTIVE_SUPPORT_ROOT}/${key}.json"
  )
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

common_structure_args=(
  --source_feature_reshaper residual_temporal_conv
  --source_feature_reshaper_strength 0.10
  --source_feature_reshaper_kernel_size 3
  --source_feature_reshaper_reg_trade_off 0.05
  --source_feature_dual_path True
  --source_feature_dual_cls_trade_off 1.0
  --source_feature_dual_relation_trade_off 0.03
  --source_phase_partition_mode uniform
  --source_segment_partition_mode uniform
  --source_phase_count 1
  --source_segment_count 1
  --source_phase_min_sample_points 2
  --source_structure_loss_version v271_global
  --source_structure_intra_trade_off 1.0
  --source_structure_v271_trend_kernel_size "${TREND_KERNEL_SIZE}"
  --source_structure_v271_trend_smoothing_mode "${TREND_SMOOTHING_MODE}"
  --source_structure_v271_trend_bandwidth "${TREND_BANDWIDTH}"
  --source_structure_v271_trend_kernel "${TREND_KERNEL}"
  --source_structure_v271_trend_dynamics_trade_off "${TREND_DYNAMICS_TRADE_OFF}"
  --source_structure_v271_residual_variance_trade_off "${RESIDUAL_VARIANCE_TRADE_OFF}"
  --source_structure_v271_residual_energy_trade_off "${RESIDUAL_ENERGY_TRADE_OFF}"
  --source_structure_v271_residual_energy_margin "${RESIDUAL_ENERGY_MARGIN}"
)

run_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local gpu="$5"

  local task_key="${src_alias}_to_${tgt_alias}"
  local source_exp="remote_${task_key}_v271clean_source_${STAMP}"
  local da_exp="remote_${task_key}_v271clean_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_weights="${OUT_ROOT}/${source_exp}"

  echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} source GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${SOURCE_EPOCHS}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${source_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  local adaptive_args=(--timematch_source_structure_trade_off 0.0)
  local support_file=""
  if support_file="$(support_file_for_task "${src_alias}" "${tgt_alias}")"; then
    adaptive_args+=(
      --timematch_v271_adaptive_support_file "${support_file}"
      --timematch_v271_adaptive_trade_off "${ADAPTIVE_TRADE_OFF}"
      --timematch_v271_adaptive_warmup_epochs "${ADAPTIVE_WARMUP_EPOCHS}"
      --timematch_v271_adaptive_ramp_epochs "${ADAPTIVE_RAMP_EPOCHS}"
      --timematch_v271_adaptive_min_score "${ADAPTIVE_MIN_SCORE}"
      --timematch_v271_adaptive_min_gate "${ADAPTIVE_MIN_GATE}"
      --timematch_v271_adaptive_min_points "${ADAPTIVE_MIN_POINTS}"
    )
    echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} adaptive_support=${support_file}"
  else
    echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} adaptive support missing; DA adaptive loss disabled"
  fi

  echo "[v2.7.1-clean] ${src_alias}->${tgt_alias} timematch GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${da_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    timematch \
    --weights "${source_weights}" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift 60 \
    --shift_source True \
    --balance_source True \
    "${adaptive_args[@]}" \
    > "${da_log}" 2>&1

  echo -e "${task_key}\t${source_log}\t${da_log}\t${support_file:-none}" >> "${LOG_ROOT}/task_logs.tsv"
}

gpu_idx=0
for task in "${TASKS[@]}"; do
  read -r src_alias tgt_alias source_dataset target_dataset <<< "${task}"
  if ! should_run_task "${src_alias}" "${tgt_alias}"; then
    continue
  fi
  wait_for_slot
  gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  gpu_idx=$((gpu_idx + 1))
  run_task "${src_alias}" "${tgt_alias}" "${source_dataset}" "${target_dataset}" "${gpu}" &
done

wait
echo "[v2.7.1-clean] done. logs=${LOG_ROOT}"
