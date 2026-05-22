#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v262a_soft_window_probe_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
DA_EPOCHS="${DA_EPOCHS:-20}"
REMOTE_BATCH_SIZE="${REMOTE_BATCH_SIZE:-128}"
HAR_BATCH_SIZE="${HAR_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/user/DBL/timematch_data}"
HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"

REMOTE_DA_STEPS_PER_EPOCH="${REMOTE_DA_STEPS_PER_EPOCH:-300}"
HAR_DA_STEPS_PER_EPOCH="${HAR_DA_STEPS_PER_EPOCH:-0}"
HAR_SEQ_LENGTH="${HAR_SEQ_LENGTH:-128}"
REMOTE_MAX_TEMPORAL_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT:-60}"
HAR_MAX_TEMPORAL_SHIFT="${HAR_MAX_TEMPORAL_SHIFT:-16}"

WINDOWS=(${WINDOWS:-full early middle late})
WINDOW_WIDTH="${WINDOW_WIDTH:-0.28}"
WINDOW_MIN_WEIGHT="${WINDOW_MIN_WEIGHT:-0.15}"
STATIC_MASK_WARMUP_EPOCHS="${STATIC_MASK_WARMUP_EPOCHS:-0}"
STATIC_MASK_MAX_BATCHES="${STATIC_MASK_MAX_BATCHES:-64}"
TEMPORAL_MASK_RELIABILITY_GATE="${TEMPORAL_MASK_RELIABILITY_GATE:-true}"
TEMPORAL_MASK_RELIABILITY_LOW="${TEMPORAL_MASK_RELIABILITY_LOW:-0.0005}"
TEMPORAL_MASK_RELIABILITY_HIGH="${TEMPORAL_MASK_RELIABILITY_HIGH:-0.04}"
TEMPORAL_SUPPORT_SMOOTH_KERNEL="${TEMPORAL_SUPPORT_SMOOTH_KERNEL:-5}"
ADAPT_MIN_FACTOR="${ADAPT_MIN_FACTOR:-0.75}"
ADAPT_MAX_FACTOR="${ADAPT_MAX_FACTOR:-1.00}"

REMOTE_TASKS=(
  "remote FR2 DK1 france/31TCJ/2017 denmark/32VNH/2017"
  "remote DK1 FR1 denmark/32VNH/2017 france/30TXT/2017"
)

HAR_TASKS=(
  "HAR 7 13"
  "HHAR_SA 1 6"
)

TASK_FILTER="${TASK_FILTER:-all}"

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

window_center() {
  case "$1" in
    early) echo "0.20" ;;
    middle|mid|full) echo "0.50" ;;
    late) echo "0.80" ;;
    *) echo "0.50" ;;
  esac
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
  --source_structure_static_mask_warmup_epochs "${STATIC_MASK_WARMUP_EPOCHS}"
  --source_structure_static_mask_max_batches "${STATIC_MASK_MAX_BATCHES}"
  --source_structure_temporal_window_reliability_gate "${TEMPORAL_MASK_RELIABILITY_GATE}"
  --source_structure_temporal_window_reliability_low "${TEMPORAL_MASK_RELIABILITY_LOW}"
  --source_structure_temporal_window_reliability_high "${TEMPORAL_MASK_RELIABILITY_HIGH}"
  --source_structure_temporal_support_smooth_kernel_size "${TEMPORAL_SUPPORT_SMOOTH_KERNEL}"
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
  local window="$5"
  local gpu="$6"
  local center
  center="$(window_center "${window}")"
  local tag="remote_${src_alias}_to_${tgt_alias}_${window}"
  local source_exp="${tag}_v262a_source_${STAMP}"
  local da_exp="${tag}_v262a_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_out="${OUT_ROOT}/${source_exp}"

  echo "[v2.6.2a] remote ${src_alias}->${tgt_alias} window=${window} source on GPU ${gpu}"
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
    --source_structure_temporal_window_mode "${window}" \
    --source_structure_temporal_window_center "${center}" \
    --source_structure_temporal_window_width "${WINDOW_WIDTH}" \
    --source_structure_temporal_window_min_weight "${WINDOW_MIN_WEIGHT}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  echo "[v2.6.2a] remote ${src_alias}->${tgt_alias} window=${window} timematch on GPU ${gpu}"
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
    --source_structure_temporal_window_mode "${window}" \
    --source_structure_temporal_window_center "${center}" \
    --source_structure_temporal_window_width "${WINDOW_WIDTH}" \
    --source_structure_temporal_window_min_weight "${WINDOW_MIN_WEIGHT}" \
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

  echo -e "remote\t${src_alias}->${tgt_alias}\t${window}\t${source_log}\t${da_log}" >> "${LOG_ROOT}/task_logs.tsv"
}

run_har_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local window="$4"
  local gpu="$5"
  local center
  center="$(window_center "${window}")"
  local data_root input_dim ds_lc
  ds_lc="${dataset,,}"
  if [ "${dataset}" = "HAR" ]; then
    data_root="${HAR_DATA_ROOT}"
    input_dim=9
  else
    data_root="${HHAR_DATA_ROOT}"
    input_dim=3
  fi
  local tag="${ds_lc}_${src}_to_${tgt}_${window}"
  local source_exp="${tag}_v262a_source_${STAMP}"
  local da_exp="${tag}_v262a_timematch_${STAMP}"
  local source_log="${LOG_ROOT}/${source_exp}.log"
  local da_log="${LOG_ROOT}/${da_exp}.log"
  local source_out="${OUT_ROOT}/${source_exp}"

  echo "[v2.6.2a] ${dataset} ${src}->${tgt} window=${window} source on GPU ${gpu}"
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
    --source_structure_temporal_window_mode "${window}" \
    --source_structure_temporal_window_center "${center}" \
    --source_structure_temporal_window_width "${WINDOW_WIDTH}" \
    --source_structure_temporal_window_min_weight "${WINDOW_MIN_WEIGHT}" \
    sourcephasecompact \
    > "${source_log}" 2>&1

  echo "[v2.6.2a] ${dataset} ${src}->${tgt} window=${window} timematch on GPU ${gpu}"
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
    --source_structure_temporal_window_mode "${window}" \
    --source_structure_temporal_window_center "${center}" \
    --source_structure_temporal_window_width "${WINDOW_WIDTH}" \
    --source_structure_temporal_window_min_weight "${WINDOW_MIN_WEIGHT}" \
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

  echo -e "${dataset}\t${src}->${tgt}\t${window}\t${source_log}\t${da_log}" >> "${LOG_ROOT}/task_logs.tsv"
}

echo -e "dataset\ttask\twindow\tsource_log\tda_log" > "${LOG_ROOT}/task_logs.tsv"
echo "v2.6.2a soft temporal window probe"
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "SOURCE_EPOCHS=${SOURCE_EPOCHS} DA_EPOCHS=${DA_EPOCHS} WINDOWS=${WINDOWS[*]} GPUS=${GPUS[*]}"
echo "Conservative strength clamp: ADAPT_MIN_FACTOR=${ADAPT_MIN_FACTOR}, ADAPT_MAX_FACTOR=${ADAPT_MAX_FACTOR}"
echo "Static mask: WARMUP_EPOCHS=${STATIC_MASK_WARMUP_EPOCHS}, MAX_BATCHES=${STATIC_MASK_MAX_BATCHES}"
echo "Temporal mask reliability gate: ENABLED=${TEMPORAL_MASK_RELIABILITY_GATE}, LOW=${TEMPORAL_MASK_RELIABILITY_LOW}, HIGH=${TEMPORAL_MASK_RELIABILITY_HIGH}"
echo "Temporal support smooth kernel: ${TEMPORAL_SUPPORT_SMOOTH_KERNEL}"
echo "TASK_FILTER=${TASK_FILTER}"

job_idx=0
for window in "${WINDOWS[@]}"; do
  for task in "${REMOTE_TASKS[@]}"; do
    wait_for_slot
    read -r _ src_alias tgt_alias source_dataset target_dataset <<< "${task}"
    if ! should_run_task "remote" "${src_alias}" "${tgt_alias}"; then
      continue
    fi
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    run_remote_task "${src_alias}" "${tgt_alias}" "${source_dataset}" "${target_dataset}" "${window}" "${gpu}" &
    job_idx=$((job_idx + 1))
  done
  for task in "${HAR_TASKS[@]}"; do
    wait_for_slot
    read -r dataset src tgt <<< "${task}"
    if ! should_run_task "${dataset}" "${src}" "${tgt}"; then
      continue
    fi
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    run_har_task "${dataset}" "${src}" "${tgt}" "${window}" "${gpu}" &
    job_idx=$((job_idx + 1))
  done
done

wait
echo "v2.6.2a soft-window probe finished."
echo "Logs: ${LOG_ROOT}"
echo "Task log index: ${LOG_ROOT}/task_logs.tsv"
