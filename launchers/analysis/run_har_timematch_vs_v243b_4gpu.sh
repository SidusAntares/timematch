#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

ADATIME_DATASET="${ADATIME_DATASET:-HAR}"
if [ "${ADATIME_DATASET}" = "HAR" ]; then
  HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
  INPUT_DIM="${INPUT_DIM:-9}"
  TASKS=(
    "2 11"
    "6 23"
    "7 13"
    "9 18"
    "12 16"
  )
elif [ "${ADATIME_DATASET}" = "HHAR" ] || [ "${ADATIME_DATASET}" = "HHAR_SA" ]; then
  HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"
  INPUT_DIM="${INPUT_DIM:-3}"
  TASKS=(
    "0 6"
    "1 6"
    "2 7"
    "3 8"
    "4 5"
  )
else
  echo "Unsupported ADATIME_DATASET=${ADATIME_DATASET}. Use HAR or HHAR_SA." >&2
  exit 1
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${ADATIME_DATASET,,}_timematch_vs_v243b_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${ADATIME_DATASET,,}_timematch_vs_v243b_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${ADATIME_DATASET,,}_timematch_vs_v243b_${STAMP}}"

SOURCE_EPOCHS="${SOURCE_EPOCHS:-50}"
DA_EPOCHS="${DA_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
MAX_TEMPORAL_SHIFT="${MAX_TEMPORAL_SHIFT:-16}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(0 1 2 3)

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

run_one_pair() {
  local src="$1"
  local tgt="$2"
  local variant="$3"
  local gpu="$4"

  local tag="${ADATIME_DATASET,,}_${variant}_${src}_to_${tgt}"
  local source_exp="${tag}_source_${STAMP}"
  local da_exp="${tag}_timematch_${STAMP}"
  local source_out="${OUT_ROOT}/${source_exp}"
  local da_out="${OUT_ROOT}/${da_exp}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"

  echo "[${tag}] source training on GPU ${gpu}"
  if [ "${variant}" = "baseline" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --dataset_type har \
      --har_dataset_name "${ADATIME_DATASET}" \
      --data_root "${HAR_DATA_ROOT}" \
      --source "${src}" \
      --target "${tgt}" \
      --closed_set true \
      --num_folds 1 \
      --val_ratio 0.1 \
      --test_ratio 0.0 \
      --epochs "${SOURCE_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --input_dim "${INPUT_DIM}" \
      --num_pixels 1 \
      --seq_length "${SEQ_LENGTH}" \
      --model pseltae \
      --source_feature_reshaper none \
      --output_dir "${OUT_ROOT}" \
      --tensorboard_log_dir "${RUN_ROOT}" \
      --experiment_name "${source_exp}" \
      > "${source_log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --dataset_type har \
      --har_dataset_name "${ADATIME_DATASET}" \
      --data_root "${HAR_DATA_ROOT}" \
      --source "${src}" \
      --target "${tgt}" \
      --closed_set true \
      --num_folds 1 \
      --val_ratio 0.1 \
      --test_ratio 0.0 \
      --epochs "${SOURCE_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --input_dim "${INPUT_DIM}" \
      --num_pixels 1 \
      --seq_length "${SEQ_LENGTH}" \
      --model pseltae \
      --source_feature_reshaper residual_temporal_conv \
      --source_feature_reshaper_strength 0.1 \
      --source_feature_reshaper_kernel_size 3 \
      --source_feature_reshaper_reg_trade_off 0.05 \
      --source_feature_dual_path true \
      --source_feature_dual_cls_trade_off 1.0 \
      --source_feature_dual_relation_trade_off 0.03 \
      --source_phase_partition_mode uniform \
      --source_segment_partition_mode uniform \
      --source_phase_count 5 \
      --source_segment_count 5 \
      --source_phase_min_sample_points 1 \
      --source_structure_loss_version segment_boundary_window_residual \
      --source_structure_intra_trade_off 1.0 \
      --source_structure_trend_trade_off 0.05 \
      --source_structure_segment_inter_trade_off 0.02 \
      --source_structure_boundary_window_trade_off 0.2 \
      --source_structure_boundary_window_size 2 \
      --output_dir "${OUT_ROOT}" \
      --tensorboard_log_dir "${RUN_ROOT}" \
      --experiment_name "${source_exp}" \
      sourcephasecompact \
      > "${source_log}" 2>&1
  fi

  echo "[${tag}] TimeMatch DA on GPU ${gpu}"
  local reshaper_args=()
  if [ "${variant}" = "v243b" ]; then
    reshaper_args=(
      --source_feature_reshaper residual_temporal_conv
      --source_feature_reshaper_strength 0.1
      --source_feature_reshaper_kernel_size 3
      --source_feature_reshaper_reg_trade_off 0.05
      --source_feature_dual_path true
      --source_feature_dual_cls_trade_off 1.0
      --source_feature_dual_relation_trade_off 0.03
      --source_phase_partition_mode uniform
      --source_segment_partition_mode uniform
      --source_phase_count 5
      --source_segment_count 5
      --source_phase_min_sample_points 1
      --source_structure_loss_version segment_boundary_window_residual
      --source_structure_intra_trade_off 1.0
      --source_structure_trend_trade_off 0.05
      --source_structure_segment_inter_trade_off 0.02
      --source_structure_boundary_window_trade_off 0.2
      --source_structure_boundary_window_size 2
    )
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --dataset_type har \
    --har_dataset_name "${ADATIME_DATASET}" \
    --data_root "${HAR_DATA_ROOT}" \
    --source "${src}" \
    --target "${tgt}" \
    --closed_set true \
    --num_folds 1 \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --epochs "${SOURCE_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --input_dim "${INPUT_DIM}" \
    --num_pixels 1 \
    --seq_length "${SEQ_LENGTH}" \
    --model pseltae \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    --experiment_name "${da_exp}" \
    "${reshaper_args[@]}" \
    timematch \
    --weights "${source_out}" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --estimate_shift true \
    --max_temporal_shift "${MAX_TEMPORAL_SHIFT}" \
    --sample_size 20 \
    --shift_source true \
    --balance_source true \
    > "${da_log}" 2>&1
}

job_idx=0
for variant in baseline v243b; do
  for task in "${TASKS[@]}"; do
    read -r src tgt <<< "${task}"
    wait_for_slot
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    run_one_pair "${src}" "${tgt}" "${variant}" "${gpu}" &
    job_idx=$((job_idx + 1))
  done
done

wait
echo "${ADATIME_DATASET} TimeMatch vs v2.4.3b run finished."
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
