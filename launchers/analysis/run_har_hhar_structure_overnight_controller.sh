#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/har_hhar_structure_overnight_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/har_hhar_structure_overnight_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/har_hhar_structure_overnight_${STAMP}}"

DATASETS="${DATASETS:-HAR HHAR_SA}"
GPUS=(${GPUS:-0 1 2 3})
RUN_SEED="${SEED:-1}"

# AdaTime-style defaults for HAR/HHAR.
SOURCE_EPOCHS="${SOURCE_EPOCHS:-40}"
DA_EPOCHS="${DA_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SOURCE_LR="${SOURCE_LR:-0.001}"
DA_LR="${DA_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

# 0 means TimeMatch uses max(len(source_loader), len(target_loader)).
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
MAX_TEMPORAL_SHIFT="${MAX_TEMPORAL_SHIFT:-16}"
SHIFT_SAMPLE_SIZE="${SHIFT_SAMPLE_SIZE:-100}"

STOP_ON_GAIN="${STOP_ON_GAIN:-true}"
GAIN_EPS="${GAIN_EPS:-0.0000}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${RUN_ROOT}"

parse_test_f1() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8", errors="ignore").read()
matches = re.findall(r"Test result for .*?: accuracy=[0-9.]+, f1=([0-9.]+)", text)
if matches:
    print(matches[-1])
PY
}

is_gain() {
  local candidate="$1"
  local baseline="$2"
  python - "$candidate" "$baseline" "$GAIN_EPS" <<'PY'
import sys

candidate = float(sys.argv[1])
baseline = float(sys.argv[2])
eps = float(sys.argv[3])
raise SystemExit(0 if candidate > baseline + eps else 1)
PY
}

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

dataset_config() {
  local dataset="$1"
  case "${dataset}" in
    HAR)
      DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
      INPUT_DIM=9
      TASKS=(
        "2 11"
        "6 23"
        "7 13"
        "9 18"
        "12 16"
      )
      ;;
    HHAR|HHAR_SA)
      DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"
      INPUT_DIM=3
      TASKS=(
        "0 6"
        "1 6"
        "2 7"
        "3 8"
        "4 5"
      )
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

run_source() {
  local dataset="$1"
  local data_root="$2"
  local input_dim="$3"
  local src="$4"
  local tgt="$5"
  local variant="$6"
  local source_exp="$7"
  local log_file="$8"
  local gpu="$9"

  local common_args=(
    --dataset_type har
    --har_dataset_name "${dataset}"
    --data_root "${data_root}"
    --source "${src}"
    --target "${tgt}"
    --closed_set true
    --num_folds 1
    --seed "${RUN_SEED}"
    --val_ratio "${VAL_RATIO}"
    --test_ratio 0.0
    --epochs "${SOURCE_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${SOURCE_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --input_dim "${input_dim}"
    --num_pixels 1
    --seq_length "${SEQ_LENGTH}"
    --model pseltae
    --output_dir "${OUT_ROOT}"
    --tensorboard_log_dir "${RUN_ROOT}"
    --experiment_name "${source_exp}"
  )

  if [ "${variant}" = "baseline" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      "${common_args[@]}" \
      --source_feature_reshaper none \
      > "${log_file}" 2>&1
    return
  fi

  local params
  params="$(variant_params "${variant}")"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    "${common_args[@]}" \
    --source_feature_reshaper residual_temporal_conv \
    --source_feature_reshaper_strength 0.1 \
    --source_feature_reshaper_kernel_size 3 \
    --source_feature_reshaper_reg_trade_off 0.05 \
    --source_feature_dual_path true \
    --source_feature_dual_cls_trade_off 1.0 \
    --source_feature_dual_relation_trade_off 0.03 \
    ${params} \
    sourcephasecompact \
    > "${log_file}" 2>&1
}

run_da() {
  local dataset="$1"
  local data_root="$2"
  local input_dim="$3"
  local src="$4"
  local tgt="$5"
  local variant="$6"
  local source_out="$7"
  local da_exp="$8"
  local log_file="$9"
  local gpu="${10}"

  local reshaper_args=()
  if [ "${variant}" != "baseline" ]; then
    local params
    params="$(variant_params "${variant}")"
    # Store as a string because these are simple flag/value pairs by construction.
    reshaper_args=(
      --source_feature_reshaper residual_temporal_conv
      --source_feature_reshaper_strength 0.1
      --source_feature_reshaper_kernel_size 3
      --source_feature_reshaper_reg_trade_off 0.05
      --source_feature_dual_path true
      --source_feature_dual_cls_trade_off 1.0
      --source_feature_dual_relation_trade_off 0.03
    )
  fi

  if [ "${variant}" = "baseline" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --dataset_type har \
      --har_dataset_name "${dataset}" \
      --data_root "${data_root}" \
      --source "${src}" \
      --target "${tgt}" \
      --closed_set true \
      --num_folds 1 \
      --seed "${RUN_SEED}" \
      --val_ratio "${VAL_RATIO}" \
      --test_ratio 0.0 \
      --epochs "${SOURCE_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${SOURCE_LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --input_dim "${input_dim}" \
      --num_pixels 1 \
      --seq_length "${SEQ_LENGTH}" \
      --model pseltae \
      --output_dir "${OUT_ROOT}" \
      --tensorboard_log_dir "${RUN_ROOT}" \
      --experiment_name "${da_exp}" \
      timematch \
      --weights "${source_out}" \
      --lr "${DA_LR}" \
      --epochs "${DA_EPOCHS}" \
      --steps_per_epoch "${STEPS_PER_EPOCH}" \
      --estimate_shift true \
      --max_temporal_shift "${MAX_TEMPORAL_SHIFT}" \
      --sample_size "${SHIFT_SAMPLE_SIZE}" \
      --shift_source true \
      --balance_source true \
      > "${log_file}" 2>&1
    return
  fi

  local params
  params="$(variant_params "${variant}")"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --dataset_type har \
    --har_dataset_name "${dataset}" \
    --data_root "${data_root}" \
    --source "${src}" \
    --target "${tgt}" \
    --closed_set true \
    --num_folds 1 \
    --seed "${RUN_SEED}" \
    --val_ratio "${VAL_RATIO}" \
    --test_ratio 0.0 \
    --epochs "${SOURCE_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${SOURCE_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --input_dim "${input_dim}" \
    --num_pixels 1 \
    --seq_length "${SEQ_LENGTH}" \
    --model pseltae \
    --output_dir "${OUT_ROOT}" \
    --tensorboard_log_dir "${RUN_ROOT}" \
    --experiment_name "${da_exp}" \
    "${reshaper_args[@]}" \
    ${params} \
    timematch \
    --weights "${source_out}" \
    --lr "${DA_LR}" \
    --epochs "${DA_EPOCHS}" \
    --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --estimate_shift true \
    --max_temporal_shift "${MAX_TEMPORAL_SHIFT}" \
    --sample_size "${SHIFT_SAMPLE_SIZE}" \
    --shift_source true \
    --balance_source true \
    > "${log_file}" 2>&1
}

variant_params() {
  local variant="$1"
  case "${variant}" in
    compact_k5)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 5 --source_segment_count 5 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 1.0 --source_structure_trend_trade_off 0.05 --source_structure_segment_inter_trade_off 0.02 --source_structure_boundary_window_trade_off 0.2 --source_structure_boundary_window_size 2"
      ;;
    compact_k3)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 3 --source_segment_count 3 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 1.0 --source_structure_trend_trade_off 0.03 --source_structure_segment_inter_trade_off 0.01 --source_structure_boundary_window_trade_off 0.1 --source_structure_boundary_window_size 2"
      ;;
    compact_k8)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 8 --source_segment_count 8 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 1.0 --source_structure_trend_trade_off 0.03 --source_structure_segment_inter_trade_off 0.01 --source_structure_boundary_window_trade_off 0.1 --source_structure_boundary_window_size 2"
      ;;
    intra_light_k5)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 5 --source_segment_count 5 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 0.5 --source_structure_trend_trade_off 0.02 --source_structure_segment_inter_trade_off 0.01 --source_structure_boundary_window_trade_off 0.05 --source_structure_boundary_window_size 2"
      ;;
    intra_strong_k5)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 5 --source_segment_count 5 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 2.0 --source_structure_trend_trade_off 0.02 --source_structure_segment_inter_trade_off 0.01 --source_structure_boundary_window_trade_off 0.05 --source_structure_boundary_window_size 2"
      ;;
    noseg_global)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 1 --source_segment_count 1 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 5.0 --source_structure_trend_trade_off 0.0 --source_structure_segment_inter_trade_off 0.0 --source_structure_boundary_window_trade_off 0.0 --source_structure_boundary_window_size 2"
      ;;
    dynamics_cosine)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 1 --source_segment_count 1 --source_phase_min_sample_points 1 --source_structure_loss_version trajectory_prototype_dynamics_v244b --source_structure_intra_trade_off 5.0 --source_structure_trend_trade_off 0.0 --source_structure_segment_inter_trade_off 0.0 --source_structure_boundary_window_trade_off 0.0 --source_structure_prototype_dynamics_trade_off 0.01 --source_structure_trajectory_pooling meanmax --source_structure_prototype_dynamics_mode cosine"
      ;;
    dynamics_mse)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 1 --source_segment_count 1 --source_phase_min_sample_points 1 --source_structure_loss_version trajectory_prototype_dynamics_v244b --source_structure_intra_trade_off 5.0 --source_structure_trend_trade_off 0.0 --source_structure_segment_inter_trade_off 0.0 --source_structure_boundary_window_trade_off 0.0 --source_structure_prototype_dynamics_trade_off 0.005 --source_structure_trajectory_pooling meanmax --source_structure_prototype_dynamics_mode mse"
      ;;
    compact_boundary_light)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 5 --source_segment_count 5 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 1.0 --source_structure_trend_trade_off 0.0 --source_structure_segment_inter_trade_off 0.005 --source_structure_boundary_window_trade_off 0.05 --source_structure_boundary_window_size 2"
      ;;
    compact_boundary_strong)
      echo "--source_phase_partition_mode uniform --source_segment_partition_mode uniform --source_phase_count 5 --source_segment_count 5 --source_phase_min_sample_points 1 --source_structure_loss_version segment_boundary_window_residual --source_structure_intra_trade_off 1.0 --source_structure_trend_trade_off 0.0 --source_structure_segment_inter_trade_off 0.02 --source_structure_boundary_window_trade_off 0.3 --source_structure_boundary_window_size 2"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

if [ -n "${VARIANTS:-}" ]; then
  read -r -a STRUCTURE_VARIANTS <<< "${VARIANTS}"
else
  STRUCTURE_VARIANTS=(
    compact_k5
    compact_k3
    compact_k8
    intra_light_k5
    intra_strong_k5
    noseg_global
    compact_boundary_light
    compact_boundary_strong
    dynamics_cosine
    dynamics_mse
  )
fi

run_task_pipeline() {
  local dataset="$1"
  local data_root="$2"
  local input_dim="$3"
  local src="$4"
  local tgt="$5"
  local gpu="$6"

  local ds_lc="${dataset,,}"
  local task_tag="${ds_lc}_${src}_to_${tgt}"
  local summary_file="${LOG_ROOT}/${task_tag}_summary.tsv"
  echo -e "dataset\ttask\tvariant\tf1\tstatus\tlog" > "${summary_file}"

  echo "[${task_tag}] baseline on GPU ${gpu}"
  local base_source_exp="${task_tag}_baseline_source_${STAMP}"
  local base_da_exp="${task_tag}_baseline_timematch_${STAMP}"
  local base_source_out="${OUT_ROOT}/${base_source_exp}"
  local base_source_log="${LOG_ROOT}/${base_source_exp}.log"
  local base_da_log="${LOG_ROOT}/${base_da_exp}.log"

  run_source "${dataset}" "${data_root}" "${input_dim}" "${src}" "${tgt}" baseline "${base_source_exp}" "${base_source_log}" "${gpu}" || true
  run_da "${dataset}" "${data_root}" "${input_dim}" "${src}" "${tgt}" baseline "${base_source_out}" "${base_da_exp}" "${base_da_log}" "${gpu}" || true

  local baseline_f1
  baseline_f1="$(parse_test_f1 "${base_da_log}")"
  if [ -z "${baseline_f1}" ]; then
    echo -e "${dataset}\t${src}->${tgt}\tbaseline\tNA\tfailed\t${base_da_log}" >> "${summary_file}"
    echo "[${task_tag}] baseline failed; still running all structure variants for diagnostics"
  else
    echo -e "${dataset}\t${src}->${tgt}\tbaseline\t${baseline_f1}\tok\t${base_da_log}" >> "${summary_file}"
    echo "[${task_tag}] baseline f1=${baseline_f1}"
  fi

  for variant in "${STRUCTURE_VARIANTS[@]}"; do
    echo "[${task_tag}] variant=${variant} on GPU ${gpu}"
    local source_exp="${task_tag}_${variant}_source_${STAMP}"
    local da_exp="${task_tag}_${variant}_timematch_${STAMP}"
    local source_out="${OUT_ROOT}/${source_exp}"
    local source_log="${LOG_ROOT}/${source_exp}.log"
    local da_log="${LOG_ROOT}/${da_exp}.log"

    run_source "${dataset}" "${data_root}" "${input_dim}" "${src}" "${tgt}" "${variant}" "${source_exp}" "${source_log}" "${gpu}" || true
    run_da "${dataset}" "${data_root}" "${input_dim}" "${src}" "${tgt}" "${variant}" "${source_out}" "${da_exp}" "${da_log}" "${gpu}" || true

    local f1
    f1="$(parse_test_f1 "${da_log}")"
    if [ -z "${f1}" ]; then
      echo -e "${dataset}\t${src}->${tgt}\t${variant}\tNA\tfailed\t${da_log}" >> "${summary_file}"
      continue
    fi

    local status="ok"
    if [ -n "${baseline_f1}" ] && is_gain "${f1}" "${baseline_f1}"; then
      status="gain"
    fi
    echo -e "${dataset}\t${src}->${tgt}\t${variant}\t${f1}\t${status}\t${da_log}" >> "${summary_file}"

    if [ "${status}" = "gain" ] && [ "${STOP_ON_GAIN}" = "true" ]; then
      echo "[${task_tag}] early stop: ${variant} f1=${f1} > baseline=${baseline_f1}"
      break
    fi
  done
}

job_idx=0
for dataset in ${DATASETS}; do
  dataset_config "${dataset}"
  for task in "${TASKS[@]}"; do
    read -r src tgt <<< "${task}"
    wait_for_slot
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    run_task_pipeline "${dataset}" "${DATA_ROOT}" "${INPUT_DIM}" "${src}" "${tgt}" "${gpu}" &
    job_idx=$((job_idx + 1))
  done
done

wait

cat "${LOG_ROOT}"/*_summary.tsv > "${LOG_ROOT}/all_task_summaries.tsv" 2>/dev/null || true
echo "HAR/HHAR overnight controller finished."
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "Summary: ${LOG_ROOT}/all_task_summaries.tsv"
