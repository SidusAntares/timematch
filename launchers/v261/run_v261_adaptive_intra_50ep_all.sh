#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v261_adaptive_intra_50ep_all_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SOURCE_EPOCHS="${SOURCE_EPOCHS:-50}"
DA_EPOCHS="${DA_EPOCHS:-50}"
REMOTE_BATCH_SIZE="${REMOTE_BATCH_SIZE:-128}"
HAR_BATCH_SIZE="${HAR_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/user/DBL/timematch_data}"
HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"

REMOTE_DA_STEPS_PER_EPOCH="${REMOTE_DA_STEPS_PER_EPOCH:-500}"
HAR_DA_STEPS_PER_EPOCH="${HAR_DA_STEPS_PER_EPOCH:-0}"
HAR_SEQ_LENGTH="${HAR_SEQ_LENGTH:-128}"
HAR_MAX_TEMPORAL_SHIFT="${HAR_MAX_TEMPORAL_SHIFT:-16}"
REMOTE_MAX_TEMPORAL_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT:-60}"

ADAPT_MIN_FACTOR="${ADAPT_MIN_FACTOR:-0.75}"
ADAPT_MAX_FACTOR="${ADAPT_MAX_FACTOR:-1.00}"

REMOTE_TASKS=(
  "remote FR1 FR2 france/30TXT/2017 france/31TCJ/2017"
  "remote FR1 DK1 france/30TXT/2017 denmark/32VNH/2017"
  "remote FR1 AT1 france/30TXT/2017 austria/33UVP/2017"
  "remote FR2 FR1 france/31TCJ/2017 france/30TXT/2017"
  "remote FR2 DK1 france/31TCJ/2017 denmark/32VNH/2017"
  "remote FR2 AT1 france/31TCJ/2017 austria/33UVP/2017"
  "remote DK1 FR1 denmark/32VNH/2017 france/30TXT/2017"
  "remote DK1 FR2 denmark/32VNH/2017 france/31TCJ/2017"
  "remote DK1 AT1 denmark/32VNH/2017 austria/33UVP/2017"
  "remote AT1 FR1 austria/33UVP/2017 france/30TXT/2017"
  "remote AT1 FR2 austria/33UVP/2017 france/31TCJ/2017"
  "remote AT1 DK1 austria/33UVP/2017 denmark/32VNH/2017"
)

HAR_TASKS=(
  "HAR 2 11"
  "HAR 6 23"
  "HAR 7 13"
  "HAR 9 18"
  "HAR 12 16"
)

HHAR_TASKS=(
  "HHAR_SA 0 6"
  "HHAR_SA 1 6"
  "HHAR_SA 2 7"
  "HHAR_SA 3 8"
  "HHAR_SA 4 5"
)

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "${running}" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 30
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
  --source_structure_loss_version segment_boundary_window_residual
  --source_structure_intra_trade_off 1.0
  --source_structure_amplitude_trade_off 0.0
  --source_structure_interphase_trade_off 0.0
  --source_structure_shape_trade_off 0.0
  --source_structure_trend_trade_off 0.05
  --source_structure_season_trade_off 0.0
  --source_structure_segment_inter_trade_off 0.02
  --source_structure_boundary_window_trade_off 0.20
  --source_structure_boundary_window_size 2
  --source_phase_grid_trade_off 0.0
  --source_structure_adaptive_weights True
  --source_structure_adaptivity_mode target_margin
  --source_structure_reliability_min_factor "${ADAPT_MIN_FACTOR}"
  --source_structure_reliability_max_factor "${ADAPT_MAX_FACTOR}"
)

remote_partition_args=(
  --source_phase_partition_mode doy_gap
  --source_segment_partition_mode doy_gap
  --source_phase_count 5
  --source_segment_count 5
  --source_phase_gap_threshold 45
  --source_phase_min_points 3
  --source_phase_max_points 8
  --source_phase_max_span 120
  --source_phase_min_sample_points 2
)

har_partition_args=(
  --source_phase_partition_mode uniform
  --source_segment_partition_mode uniform
  --source_phase_count 5
  --source_segment_count 5
  --source_phase_min_sample_points 1
)

run_remote_task() {
  local src_alias="$1"
  local tgt_alias="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local gpu="$5"

  local source_tile
  local target_tile
  source_tile="$(echo "${source_dataset}" | cut -d'/' -f2)"
  target_tile="$(echo "${target_dataset}" | cut -d'/' -f2)"
  local tag="remote_${src_alias}_to_${tgt_alias}"
  local source_exp="${tag}_v261_adaptive_source_${STAMP}"
  local da_exp="${tag}_v261_adaptive_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_out="${OUT_ROOT}/${source_exp}"

  echo "[v2.6.1] remote ${src_alias}->${tgt_alias} source on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${REMOTE_DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${REMOTE_BATCH_SIZE}" \
    --epochs "${SOURCE_EPOCHS}" \
    --source_checkpoint_epochs "${SOURCE_EPOCHS}" \
    --source_checkpoint_dirname checkpoints \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${source_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    "${remote_partition_args[@]}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  echo "[v2.6.1] remote ${src_alias}->${tgt_alias} timematch on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --data_root "${REMOTE_DATA_ROOT}" \
    --closed_set True \
    --with_shift_aug False \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${REMOTE_BATCH_SIZE}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    -e "${da_exp}" \
    --source "${source_dataset}" \
    --target "${target_dataset}" \
    "${common_structure_args[@]}" \
    "${remote_partition_args[@]}" \
    timematch \
    --weights "${source_out}" \
    --weights_checkpoint "checkpoints/epoch_${SOURCE_EPOCHS}.pt" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${REMOTE_DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift "${REMOTE_MAX_TEMPORAL_SHIFT}" \
    --shift_source True \
    --balance_source True \
    > "${da_log}" 2>&1

  echo -e "remote\t${src_alias}->${tgt_alias}\t${source_tile}->${target_tile}\t${source_log}\t${da_log}" >> "${LOG_ROOT}/task_logs.tsv"
}

run_har_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local gpu="$4"

  local data_root
  local input_dim
  local ds_lc
  ds_lc="${dataset,,}"
  if [ "${dataset}" = "HAR" ]; then
    data_root="${HAR_DATA_ROOT}"
    input_dim=9
  else
    data_root="${HHAR_DATA_ROOT}"
    input_dim=3
  fi

  local tag="${ds_lc}_${src}_to_${tgt}"
  local source_exp="${tag}_v261_adaptive_source_${STAMP}"
  local da_exp="${tag}_v261_adaptive_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_out="${OUT_ROOT}/${source_exp}"

  echo "[v2.6.1] ${dataset} ${src}->${tgt} source on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --dataset_type har \
    --har_dataset_name "${dataset}" \
    --data_root "${data_root}" \
    --source "${src}" \
    --target "${tgt}" \
    --closed_set True \
    --num_folds 1 \
    --seed "${SEED}" \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --epochs "${SOURCE_EPOCHS}" \
    --source_checkpoint_epochs "${SOURCE_EPOCHS}" \
    --source_checkpoint_dirname checkpoints \
    --batch_size "${HAR_BATCH_SIZE}" \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --input_dim "${input_dim}" \
    --num_pixels 1 \
    --seq_length "${HAR_SEQ_LENGTH}" \
    --model pseltae \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    --experiment_name "${source_exp}" \
    "${common_structure_args[@]}" \
    "${har_partition_args[@]}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  echo "[v2.6.1] ${dataset} ${src}->${tgt} timematch on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --dataset_type har \
    --har_dataset_name "${dataset}" \
    --data_root "${data_root}" \
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
    --input_dim "${input_dim}" \
    --num_pixels 1 \
    --seq_length "${HAR_SEQ_LENGTH}" \
    --model pseltae \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    --experiment_name "${da_exp}" \
    "${common_structure_args[@]}" \
    "${har_partition_args[@]}" \
    timematch \
    --weights "${source_out}" \
    --weights_checkpoint "checkpoints/epoch_${SOURCE_EPOCHS}.pt" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${HAR_DA_STEPS_PER_EPOCH}" \
    --estimate_shift True \
    --max_temporal_shift "${HAR_MAX_TEMPORAL_SHIFT}" \
    --sample_size 100 \
    --shift_source True \
    --balance_source True \
    > "${da_log}" 2>&1

  echo -e "${dataset}\t${src}->${tgt}\t${src}->${tgt}\t${source_log}\t${da_log}" >> "${LOG_ROOT}/task_logs.tsv"
}

echo -e "dataset\ttask\tids\tsource_log\tda_log" > "${LOG_ROOT}/task_logs.tsv"
echo "v2.6.1 adaptive intra run"
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "Runs: ${RUN_ROOT}"
echo "SOURCE_EPOCHS=${SOURCE_EPOCHS} DA_EPOCHS=${DA_EPOCHS} GPUS=${GPUS[*]}"
echo "v2.6.1 factor: conservative target_margin_ratio safety gate; it may weaken intra but does not strengthen it by default"

job_idx=0
for task in "${REMOTE_TASKS[@]}"; do
  wait_for_slot
  read -r _ src_alias tgt_alias source_dataset target_dataset <<< "${task}"
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  run_remote_task "${src_alias}" "${tgt_alias}" "${source_dataset}" "${target_dataset}" "${gpu}" &
  job_idx=$((job_idx + 1))
done

for task in "${HAR_TASKS[@]}"; do
  wait_for_slot
  read -r dataset src tgt <<< "${task}"
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  run_har_task "${dataset}" "${src}" "${tgt}" "${gpu}" &
  job_idx=$((job_idx + 1))
done

for task in "${HHAR_TASKS[@]}"; do
  wait_for_slot
  read -r dataset src tgt <<< "${task}"
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  run_har_task "${dataset}" "${src}" "${tgt}" "${gpu}" &
  job_idx=$((job_idx + 1))
done

wait

echo "v2.6.1 adaptive intra all-task run finished."
echo "Logs: ${LOG_ROOT}"
echo "Task log index: ${LOG_ROOT}/task_logs.tsv"
