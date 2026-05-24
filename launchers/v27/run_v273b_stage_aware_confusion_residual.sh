#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v273b_stage_aware_confusion_residual_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${ROOT_DIR}/outputs/v271_basis_view_probe_20260522_162224}"
GATE_ROOT="${GATE_ROOT:-${ROOT_DIR}/outputs/v273a2_confusion_gate_diagnostic_20260523_193733}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
DA_EPOCHS="${DA_EPOCHS:-20}"
REMOTE_BATCH_SIZE="${REMOTE_BATCH_SIZE:-128}"
HAR_BATCH_SIZE="${HAR_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/user/DBL/timematch_data}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"
REMOTE_DA_STEPS_PER_EPOCH="${REMOTE_DA_STEPS_PER_EPOCH:-300}"
HAR_DA_STEPS_PER_EPOCH="${HAR_DA_STEPS_PER_EPOCH:-0}"
HAR_SEQ_LENGTH="${HAR_SEQ_LENGTH:-128}"
REMOTE_MAX_TEMPORAL_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT:-60}"
HAR_MAX_TEMPORAL_SHIFT="${HAR_MAX_TEMPORAL_SHIFT:-16}"
CONFUSION_TRADE_OFF="${CONFUSION_TRADE_OFF:-0.20}"
CONFUSION_WARMUP_EPOCHS="${CONFUSION_WARMUP_EPOCHS:-2}"
CONFUSION_RAMP_EPOCHS="${CONFUSION_RAMP_EPOCHS:-3}"
TASK_FILTER="${TASK_FILTER:-all}"

# dataset src tgt source_dataset target_dataset source_run gate_key base_name global segment trend boundary dynamics trajectory
TASKS=(
  "remote DK1 FR1 denmark/32VNH/2017 france/30TXT/2017 remote_DK1_to_FR1_soft_bank_v271_source_20260522_162227 remote_DK1_to_FR1 soft_bank 0.20 0.50 0.02 0.01 0.02 0.20"
  "remote FR2 DK1 france/31TCJ/2017 denmark/32VNH/2017 remote_FR2_to_DK1_dynamics_v271_source_20260522_162227 remote_FR2_to_DK1 dynamics 0.0 0.0 0.0 0.0 0.05 0.0"
  "HHAR_SA 1 6 HHAR_SA HHAR_SA hhar_sa_1_to_6_soft_bank_v271_source_20260522_162227 HHAR_SA_1_to_6 soft_bank 0.20 0.50 0.02 0.01 0.02 0.20"
)

VARIANTS=(${VARIANTS:-no_conf:0 gated:1})

should_run_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local filter="${TASK_FILTER}"
  if [ "${filter}" = "all" ] || [ -z "${filter}" ]; then
    return 0
  fi
  local task_key="${src}->${tgt}"
  local task_key_alt="${src}_to_${tgt}"
  local dataset_task_key="${dataset}:${src}->${tgt}"
  local dataset_task_key_alt="${dataset}:${src}_to_${tgt}"
  IFS=',' read -ra filter_items <<< "${filter}"
  for item in "${filter_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "${item}" = "${task_key}" ] || [ "${item}" = "${task_key_alt}" ]; then
      return 0
    fi
    if [ "${item}" = "${dataset_task_key}" ] || [ "${item}" = "${dataset_task_key_alt}" ]; then
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

common_structure_args=(
  --source_feature_reshaper residual_temporal_conv
  --source_feature_reshaper_strength 0.10
  --source_feature_reshaper_kernel_size 3
  --source_feature_reshaper_reg_trade_off 0.05
  --source_feature_dual_path True
  --source_feature_dual_cls_trade_off 1.0
  --source_feature_dual_relation_trade_off 0.03
  --source_structure_loss_version basis_view
  --source_structure_intra_trade_off 1.0
  --source_structure_amplitude_trade_off 0.0
  --source_structure_interphase_trade_off 0.0
  --source_structure_shape_trade_off 0.0
  --source_structure_trend_trade_off 0.0
  --source_structure_season_trade_off 0.0
  --source_structure_segment_inter_trade_off 0.0
  --source_structure_boundary_window_trade_off 0.20
  --source_structure_boundary_window_size 2
  --source_phase_grid_trade_off 0.0
  --source_structure_adaptive_weights False
  --source_structure_adaptivity_mode none
  --source_structure_temporal_window_mode full
  --source_structure_temporal_window_min_weight 1.00
)

remote_partition_args=(
  --source_phase_partition_mode "${REMOTE_PARTITION_MODE:-doy_gap}"
  --source_segment_partition_mode "${REMOTE_PARTITION_MODE:-doy_gap}"
  --source_phase_count "${REMOTE_PHASE_COUNT:-5}"
  --source_segment_count "${REMOTE_PHASE_COUNT:-5}"
  --source_phase_gap_threshold "${REMOTE_PHASE_GAP_THRESHOLD:-45}"
  --source_phase_min_points "${REMOTE_PHASE_MIN_POINTS:-3}"
  --source_phase_max_points "${REMOTE_PHASE_MAX_POINTS:-8}"
  --source_phase_max_span "${REMOTE_PHASE_MAX_SPAN:-120}"
  --source_phase_min_sample_points "${REMOTE_MIN_SAMPLE_POINTS:-2}"
)

har_partition_args=(
  --source_phase_partition_mode "${HAR_PARTITION_MODE:-uniform}"
  --source_segment_partition_mode "${HAR_PARTITION_MODE:-uniform}"
  --source_phase_count "${HAR_PHASE_COUNT:-5}"
  --source_segment_count "${HAR_PHASE_COUNT:-5}"
  --source_phase_min_sample_points "${HAR_MIN_SAMPLE_POINTS:-1}"
)

run_remote_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local source_run="$5"
  local gate_key="$6"
  local base_name="$7"
  local global_weight="$8"
  local segment_weight="$9"
  local trend_weight="${10}"
  local boundary_weight="${11}"
  local dynamics_weight="${12}"
  local trajectory_weight="${13}"
  local variant_name="${14}"
  local use_conf="${15}"
  local gpu="${16}"
  local source_out="${SOURCE_RUN_ROOT}/${source_run}"
  local gate_file="${GATE_ROOT}/${gate_key}/confusion_gate_diagnostic.json"
  local trade_off="0.0"
  if [ "${use_conf}" = "1" ]; then
    trade_off="${CONFUSION_TRADE_OFF}"
  fi
  local exp="remote_${src_alias}_to_${tgt_alias}_${base_name}_${variant_name}_v273b_timematch_${STAMP}"
  local log_file="${LOG_ROOT}/${exp}.log"

  echo "[v2.7.3b] remote ${src_alias}->${tgt_alias} ${variant_name} gpu=${gpu} gate=${gate_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${REMOTE_DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${REMOTE_BATCH_SIZE}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    "${remote_partition_args[@]}" \
    --source_structure_basis_global_trade_off "${global_weight}" \
    --source_structure_basis_segment_trade_off "${segment_weight}" \
    --source_structure_basis_trend_trade_off "${trend_weight}" \
    --source_structure_basis_boundary_trade_off "${boundary_weight}" \
    --source_structure_basis_dynamics_trade_off "${dynamics_weight}" \
    --source_structure_basis_trajectory_trade_off "${trajectory_weight}" \
    timematch \
    --weights "${source_out}" \
    --weights_checkpoint "checkpoints/epoch_20.pt" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${REMOTE_DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift "${REMOTE_MAX_TEMPORAL_SHIFT}" \
    --shift_source True \
    --balance_source True \
    --timematch_confusion_gate_file "${gate_file}" \
    --timematch_confusion_trade_off "${trade_off}" \
    --timematch_confusion_warmup_epochs "${CONFUSION_WARMUP_EPOCHS}" \
    --timematch_confusion_ramp_epochs "${CONFUSION_RAMP_EPOCHS}" \
    > "${log_file}" 2>&1

  echo -e "remote\t${src_alias}->${tgt_alias}\t${base_name}\t${variant_name}\t${trade_off}\t${gate_file}\t${log_file}" >> "${LOG_ROOT}/task_logs.tsv"
}

run_hhar_task() {
  local src="$1"
  local tgt="$2"
  local source_run="$3"
  local gate_key="$4"
  local base_name="$5"
  local global_weight="$6"
  local segment_weight="$7"
  local trend_weight="$8"
  local boundary_weight="$9"
  local dynamics_weight="${10}"
  local trajectory_weight="${11}"
  local variant_name="${12}"
  local use_conf="${13}"
  local gpu="${14}"
  local source_out="${SOURCE_RUN_ROOT}/${source_run}"
  local gate_file="${GATE_ROOT}/${gate_key}/confusion_gate_diagnostic.json"
  local trade_off="0.0"
  if [ "${use_conf}" = "1" ]; then
    trade_off="${CONFUSION_TRADE_OFF}"
  fi
  local exp="hhar_sa_${src}_to_${tgt}_${base_name}_${variant_name}_v273b_timematch_${STAMP}"
  local log_file="${LOG_ROOT}/${exp}.log"

  echo "[v2.7.3b] HHAR_SA ${src}->${tgt} ${variant_name} gpu=${gpu} gate=${gate_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --dataset_type har \
    --har_dataset_name HHAR_SA \
    --data_root "${HHAR_DATA_ROOT}" \
    --source "${src}" \
    --target "${tgt}" \
    --closed_set True \
    --num_folds 1 \
    --seed "${SEED}" \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --batch_size "${HAR_BATCH_SIZE}" \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --input_dim 3 \
    --num_pixels 1 \
    --seq_length "${HAR_SEQ_LENGTH}" \
    --model pseltae \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    --experiment_name "${exp}" \
    "${common_structure_args[@]}" \
    "${har_partition_args[@]}" \
    --source_structure_basis_global_trade_off "${global_weight}" \
    --source_structure_basis_segment_trade_off "${segment_weight}" \
    --source_structure_basis_trend_trade_off "${trend_weight}" \
    --source_structure_basis_boundary_trade_off "${boundary_weight}" \
    --source_structure_basis_dynamics_trade_off "${dynamics_weight}" \
    --source_structure_basis_trajectory_trade_off "${trajectory_weight}" \
    timematch \
    --weights "${source_out}" \
    --weights_checkpoint "checkpoints/epoch_20.pt" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${HAR_DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift "${HAR_MAX_TEMPORAL_SHIFT}" \
    --sample_size 100 \
    --shift_source True \
    --balance_source True \
    --timematch_confusion_gate_file "${gate_file}" \
    --timematch_confusion_trade_off "${trade_off}" \
    --timematch_confusion_warmup_epochs "${CONFUSION_WARMUP_EPOCHS}" \
    --timematch_confusion_ramp_epochs "${CONFUSION_RAMP_EPOCHS}" \
    > "${log_file}" 2>&1

  echo -e "HHAR_SA\t${src}->${tgt}\t${base_name}\t${variant_name}\t${trade_off}\t${gate_file}\t${log_file}" >> "${LOG_ROOT}/task_logs.tsv"
}

echo -e "dataset\ttask\tbase\tvariant\tconfusion_trade_off\tgate_file\tlog_file" > "${LOG_ROOT}/task_logs.tsv"
echo "v2.7.3b stage-aware confusion residual"
echo "SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT}"
echo "GATE_ROOT=${GATE_ROOT}"
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "VARIANTS=${VARIANTS[*]} TASK_FILTER=${TASK_FILTER} GPUS=${GPUS[*]}"

job_idx=0
for task in "${TASKS[@]}"; do
  read -r dataset src tgt source_dataset target_dataset source_run gate_key base_name global_weight segment_weight trend_weight boundary_weight dynamics_weight trajectory_weight <<< "${task}"
  if ! should_run_task "${dataset}" "${src}" "${tgt}"; then
    continue
  fi
  for variant in "${VARIANTS[@]}"; do
    IFS=':' read -r variant_name use_conf <<< "${variant}"
    wait_for_slot
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    if [ "${dataset}" = "remote" ]; then
      run_remote_task "${src}" "${tgt}" "${source_dataset}" "${target_dataset}" "${source_run}" "${gate_key}" "${base_name}" "${global_weight}" "${segment_weight}" "${trend_weight}" "${boundary_weight}" "${dynamics_weight}" "${trajectory_weight}" "${variant_name}" "${use_conf}" "${gpu}" &
    else
      run_hhar_task "${src}" "${tgt}" "${source_run}" "${gate_key}" "${base_name}" "${global_weight}" "${segment_weight}" "${trend_weight}" "${boundary_weight}" "${dynamics_weight}" "${trajectory_weight}" "${variant_name}" "${use_conf}" "${gpu}" &
    fi
    job_idx=$((job_idx + 1))
  done
done

wait
echo "v2.7.3b finished."
echo "Task log index: ${LOG_ROOT}/task_logs.tsv"
