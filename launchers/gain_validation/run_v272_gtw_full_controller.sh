#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v272_gtw_full_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-$LOG_ROOT/summary.tsv}"

DATASETS="${DATASETS:-REMOTE HAR HHAR_SA}"
MODES="${MODES:-G_global G_gtw_r1 G_gtw_r2 G_gtw_r4}"
GPUS=(${GPUS:-0 1 2 3})
TASK_FILTER="${TASK_FILTER:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-${DATA_ROOT:-/data/user/DBL/timematch_data}}"
HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"

SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

REMOTE_SOURCE_EPOCHS="${REMOTE_SOURCE_EPOCHS:-50}"
REMOTE_DA_EPOCHS="${REMOTE_DA_EPOCHS:-20}"
REMOTE_STEPS_PER_EPOCH="${REMOTE_STEPS_PER_EPOCH:-500}"
REMOTE_BATCH_SIZE="${REMOTE_BATCH_SIZE:-128}"
REMOTE_LR="${REMOTE_LR:-0.001}"
REMOTE_DA_LR="${REMOTE_DA_LR:-0.0001}"
REMOTE_WEIGHT_DECAY="${REMOTE_WEIGHT_DECAY:-0.0001}"
REMOTE_MAX_TEMPORAL_SHIFT="${REMOTE_MAX_TEMPORAL_SHIFT:-60}"
REMOTE_SHIFT_SAMPLE_SIZE="${REMOTE_SHIFT_SAMPLE_SIZE:-100}"
REMOTE_VAL_RATIO="${REMOTE_VAL_RATIO:-0.1}"
REMOTE_TEST_RATIO="${REMOTE_TEST_RATIO:-0.2}"

HAR_SOURCE_EPOCHS="${HAR_SOURCE_EPOCHS:-40}"
HAR_DA_EPOCHS="${HAR_DA_EPOCHS:-40}"
HAR_STEPS_PER_EPOCH="${HAR_STEPS_PER_EPOCH:-0}"
HAR_BATCH_SIZE="${HAR_BATCH_SIZE:-32}"
HAR_LR="${HAR_LR:-0.001}"
HAR_DA_LR="${HAR_DA_LR:-0.001}"
HAR_WEIGHT_DECAY="${HAR_WEIGHT_DECAY:-0.0001}"
HAR_MAX_TEMPORAL_SHIFT="${HAR_MAX_TEMPORAL_SHIFT:-16}"
HAR_SHIFT_SAMPLE_SIZE="${HAR_SHIFT_SAMPLE_SIZE:-100}"
HAR_VAL_RATIO="${HAR_VAL_RATIO:-0.1}"

SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_RELATION_TRADE_OFF="${DUAL_RELATION_TRADE_OFF:-0.03}"

TREND_KERNEL_SIZE="${TREND_KERNEL_SIZE:-5}"
TREND_SMOOTHING_MODE="${TREND_SMOOTHING_MODE:-time}"
TREND_BANDWIDTH="${TREND_BANDWIDTH:-0.0}"
TREND_KERNEL="${TREND_KERNEL:-gaussian}"
TREND_DYNAMICS_TRADE_OFF="${TREND_DYNAMICS_TRADE_OFF:-0.05}"
RESIDUAL_VARIANCE_TRADE_OFF="${RESIDUAL_VARIANCE_TRADE_OFF:-0.10}"
RESIDUAL_ENERGY_TRADE_OFF="${RESIDUAL_ENERGY_TRADE_OFF:-0.05}"
RESIDUAL_ENERGY_MARGIN="${RESIDUAL_ENERGY_MARGIN:-1.0}"

GTW_SHIFT_COUNT="${GTW_SHIFT_COUNT:-5}"
GTW_TEMPERATURE="${GTW_TEMPERATURE:-0.05}"
TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF:-0.0}"
EVAL_SOURCE_ON_TARGET="${EVAL_SOURCE_ON_TARGET:-true}"

mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$RUN_ROOT"
printf "dataset\ttask\tmode\tstage\tstatus\tf1\tlog\toutput\n" > "$SUMMARY_FILE"

REMOTE_TASKS=(
  "FR1|FR2|france/30TXT/2017|france/31TCJ/2017|0.00"
  "FR1|DK1|france/30TXT/2017|denmark/32VNH/2017|0.02"
  "FR1|AT1|france/30TXT/2017|austria/33UVP/2017|0.03"
  "FR2|FR1|france/31TCJ/2017|france/30TXT/2017|0.00"
  "FR2|DK1|france/31TCJ/2017|denmark/32VNH/2017|0.05"
  "FR2|AT1|france/31TCJ/2017|austria/33UVP/2017|0.05"
  "DK1|FR1|denmark/32VNH/2017|france/30TXT/2017|0.00"
  "DK1|FR2|denmark/32VNH/2017|france/31TCJ/2017|0.05"
  "DK1|AT1|denmark/32VNH/2017|austria/33UVP/2017|0.05"
  "AT1|FR1|austria/33UVP/2017|france/30TXT/2017|0.05"
  "AT1|FR2|austria/33UVP/2017|france/31TCJ/2017|0.02"
  "AT1|DK1|austria/33UVP/2017|denmark/32VNH/2017|0.05"
)

HAR_TASKS=(
  "2|11"
  "6|23"
  "7|13"
  "9|18"
  "12|16"
)

HHAR_TASKS=(
  "0|6"
  "1|6"
  "2|7"
  "3|8"
  "4|5"
)

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "$running" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 20
  done
}

parse_test_f1() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
except OSError:
    print("")
    raise SystemExit(0)
matches = re.findall(r"Test result for .*?: accuracy=[0-9.]+, f1=([0-9.]+)", text)
print(matches[-1] if matches else "")
PY
}

append_summary() {
  local dataset="$1"
  local task="$2"
  local mode="$3"
  local stage="$4"
  local status="$5"
  local f1="$6"
  local log_file="$7"
  local out_dir="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset" "$task" "$mode" "$stage" "$status" "$f1" "$log_file" "$out_dir" >> "$SUMMARY_FILE"
}

should_run_task() {
  local dataset="$1"
  local task_key="$2"
  local task_arrow="${task_key/_to_/->}"
  if [ -z "$TASK_FILTER" ] || [ "$TASK_FILTER" = "all" ]; then
    return 0
  fi
  IFS=',' read -ra items <<< "$TASK_FILTER"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "$item" = "$task_key" ] || [ "$item" = "$task_arrow" ] || [ "$item" = "$dataset:$task_key" ] || [ "$item" = "$dataset:$task_arrow" ]; then
      return 0
    fi
  done
  return 1
}

configure_dataset() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local src_path="${4:-}"
  local tgt_path="${5:-}"
  case "$dataset" in
    REMOTE)
      SOURCE_EPOCHS="$REMOTE_SOURCE_EPOCHS"
      DA_EPOCHS="$REMOTE_DA_EPOCHS"
      DA_STEPS="$REMOTE_STEPS_PER_EPOCH"
      DA_LR="$REMOTE_DA_LR"
      MAX_SHIFT="$REMOTE_MAX_TEMPORAL_SHIFT"
      SHIFT_SAMPLE="$REMOTE_SHIFT_SAMPLE_SIZE"
      DATA_ARGS=(
        --data_root "$REMOTE_DATA_ROOT"
        --source "$src_path"
        --target "$tgt_path"
        --closed_set True
        --num_folds 1
        --val_ratio "$REMOTE_VAL_RATIO"
        --test_ratio "$REMOTE_TEST_RATIO"
        --batch_size "$REMOTE_BATCH_SIZE"
        --lr "$REMOTE_LR"
        --weight_decay "$REMOTE_WEIGHT_DECAY"
        --input_dim 10
        --num_pixels 64
        --seq_length 30
        --model pseltae
      )
      ;;
    HAR)
      SOURCE_EPOCHS="$HAR_SOURCE_EPOCHS"
      DA_EPOCHS="$HAR_DA_EPOCHS"
      DA_STEPS="$HAR_STEPS_PER_EPOCH"
      DA_LR="$HAR_DA_LR"
      MAX_SHIFT="$HAR_MAX_TEMPORAL_SHIFT"
      SHIFT_SAMPLE="$HAR_SHIFT_SAMPLE_SIZE"
      DATA_ARGS=(
        --dataset_type har
        --har_dataset_name HAR
        --data_root "$HAR_DATA_ROOT"
        --source "$src"
        --target "$tgt"
        --closed_set True
        --num_folds 1
        --val_ratio "$HAR_VAL_RATIO"
        --test_ratio 0.0
        --batch_size "$HAR_BATCH_SIZE"
        --lr "$HAR_LR"
        --weight_decay "$HAR_WEIGHT_DECAY"
        --input_dim 9
        --num_pixels 1
        --seq_length 128
        --model pseltae
      )
      ;;
    HHAR|HHAR_SA)
      SOURCE_EPOCHS="$HAR_SOURCE_EPOCHS"
      DA_EPOCHS="$HAR_DA_EPOCHS"
      DA_STEPS="$HAR_STEPS_PER_EPOCH"
      DA_LR="$HAR_DA_LR"
      MAX_SHIFT="$HAR_MAX_TEMPORAL_SHIFT"
      SHIFT_SAMPLE="$HAR_SHIFT_SAMPLE_SIZE"
      DATA_ARGS=(
        --dataset_type har
        --har_dataset_name HHAR_SA
        --data_root "$HHAR_DATA_ROOT"
        --source "$src"
        --target "$tgt"
        --closed_set True
        --num_folds 1
        --val_ratio "$HAR_VAL_RATIO"
        --test_ratio 0.0
        --batch_size "$HAR_BATCH_SIZE"
        --lr "$HAR_LR"
        --weight_decay "$HAR_WEIGHT_DECAY"
        --input_dim 3
        --num_pixels 1
        --seq_length 128
        --model pseltae
      )
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

configure_structure() {
  local mode="$1"
  local trend_weight="$2"
  local loss_version="v271_global"
  local gtw_radius_steps="0.0"

  case "$mode" in
    G_global)
      loss_version="v271_global"
      ;;
    G_gtw_r1)
      loss_version="v271_global_gtw"
      gtw_radius_steps="1.0"
      ;;
    G_gtw_r2)
      loss_version="v271_global_gtw"
      gtw_radius_steps="2.0"
      ;;
    G_gtw_r4)
      loss_version="v271_global_gtw"
      gtw_radius_steps="4.0"
      ;;
    *)
      echo "Unsupported mode: $mode" >&2
      exit 1
      ;;
  esac

  STRUCT_ARGS=(
    --source_feature_reshaper "$SOURCE_FEATURE_RESHAPER"
    --source_feature_reshaper_strength "$RESHAPER_STRENGTH"
    --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE"
    --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF"
    --source_feature_dual_path True
    --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF"
    --source_feature_dual_relation_trade_off "$DUAL_RELATION_TRADE_OFF"
    --source_phase_partition_mode uniform
    --source_segment_partition_mode uniform
    --source_phase_count 1
    --source_segment_count 1
    --source_phase_min_sample_points 1
    --source_structure_loss_version "$loss_version"
    --source_structure_intra_trade_off 1.0
    --source_structure_amplitude_trade_off 0.0
    --source_structure_interphase_trade_off 0.0
    --source_structure_shape_trade_off 0.0
    --source_structure_trend_trade_off "$trend_weight"
    --source_structure_season_trade_off 0.0
    --source_structure_segment_inter_trade_off 0.0
    --source_structure_boundary_window_trade_off 0.0
    --source_structure_v271_trend_kernel_size "$TREND_KERNEL_SIZE"
    --source_structure_v271_trend_smoothing_mode "$TREND_SMOOTHING_MODE"
    --source_structure_v271_trend_bandwidth "$TREND_BANDWIDTH"
    --source_structure_v271_trend_kernel "$TREND_KERNEL"
    --source_structure_v271_trend_dynamics_trade_off "$TREND_DYNAMICS_TRADE_OFF"
    --source_structure_v271_residual_variance_trade_off "$RESIDUAL_VARIANCE_TRADE_OFF"
    --source_structure_v271_residual_energy_trade_off "$RESIDUAL_ENERGY_TRADE_OFF"
    --source_structure_v271_residual_energy_margin "$RESIDUAL_ENERGY_MARGIN"
    --source_structure_v271_gtw_shift_radius_steps "$gtw_radius_steps"
    --source_structure_v271_gtw_shift_count "$GTW_SHIFT_COUNT"
    --source_structure_v271_gtw_temperature "$GTW_TEMPERATURE"
  )
}

run_pipeline() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local src_path="${4:-}"
  local tgt_path="${5:-}"
  local trend_weight="$6"
  local mode="$7"
  local gpu="$8"
  local task_key="${src}_to_${tgt}"
  local dataset_key="${dataset,,}"
  local mode_key="${mode#G_}"
  local exp_base="${dataset_key}_${task_key}_${mode_key}_t${trend_weight}_sc${GTW_SHIFT_COUNT}_temp${GTW_TEMPERATURE}"
  exp_base="${exp_base//./}"

  local log_dir="${LOG_ROOT}/${dataset_key}/${mode_key}"
  local out_dir="${OUT_ROOT}/${dataset_key}/${mode_key}"
  local run_dir="${RUN_ROOT}/${dataset_key}/${mode_key}"
  mkdir -p "$log_dir" "$out_dir" "$run_dir"

  configure_dataset "$dataset" "$src" "$tgt" "$src_path" "$tgt_path"
  configure_structure "$mode" "$trend_weight"

  local source_exp="${exp_base}_source_${STAMP}"
  local source_out="${out_dir}/${source_exp}"
  local source_log="${log_dir}/${source_exp}.log"
  local eval_log="${log_dir}/${source_exp}_source_on_target.log"
  local da_exp="${exp_base}_timematch_${STAMP}"
  local da_out="${out_dir}/${da_exp}"
  local da_log="${log_dir}/${da_exp}.log"

  echo "START|${dataset}|${task_key}|${mode}|gpu=${gpu}|source=${source_exp}"

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "${source_out}/fold_0/model.pt" ]; then
    append_summary "$dataset" "$task_key" "$mode" "source" "skipped" "" "$source_log" "$source_out"
  else
    if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
        "${DATA_ARGS[@]}" \
        --seed "$SEED" \
        --num_workers "$NUM_WORKERS" \
        --epochs "$SOURCE_EPOCHS" \
        --output_dir "$out_dir" \
        --tensorboard_log_dir "$run_dir" \
        --experiment_name "$source_exp" \
        "${STRUCT_ARGS[@]}" \
        sourcephasecompact \
        > "$source_log" 2>&1; then
      append_summary "$dataset" "$task_key" "$mode" "source" "ok" "$(parse_test_f1 "$source_log")" "$source_log" "$source_out"
    else
      append_summary "$dataset" "$task_key" "$mode" "source" "failed" "" "$source_log" "$source_out"
      return 0
    fi
  fi

  if [ "$EVAL_SOURCE_ON_TARGET" = "true" ]; then
    if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
        "${DATA_ARGS[@]}" \
        --seed "$SEED" \
        --num_workers "$NUM_WORKERS" \
        --output_dir "$out_dir" \
        --tensorboard_log_dir "$run_dir" \
        --experiment_name "$source_exp" \
        "${STRUCT_ARGS[@]}" \
        --eval \
        > "$eval_log" 2>&1; then
      append_summary "$dataset" "$task_key" "$mode" "source_on_target" "ok" "$(parse_test_f1 "$eval_log")" "$eval_log" "$source_out"
    else
      append_summary "$dataset" "$task_key" "$mode" "source_on_target" "failed" "" "$eval_log" "$source_out"
    fi
  fi

  if CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      "${DATA_ARGS[@]}" \
      --seed "$SEED" \
      --num_workers "$NUM_WORKERS" \
      --epochs "$SOURCE_EPOCHS" \
      --output_dir "$out_dir" \
      --tensorboard_log_dir "$run_dir" \
      --experiment_name "$da_exp" \
      "${STRUCT_ARGS[@]}" \
      timematch \
      --weights "$source_out" \
      --lr "$DA_LR" \
      --epochs "$DA_EPOCHS" \
      --steps_per_epoch "$DA_STEPS" \
      --estimate_shift True \
      --max_temporal_shift "$MAX_SHIFT" \
      --sample_size "$SHIFT_SAMPLE" \
      --shift_source True \
      --balance_source True \
      --timematch_source_structure_trade_off "$TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF" \
      > "$da_log" 2>&1; then
    append_summary "$dataset" "$task_key" "$mode" "da" "ok" "$(parse_test_f1 "$da_log")" "$da_log" "$da_out"
  else
    append_summary "$dataset" "$task_key" "$mode" "da" "failed" "" "$da_log" "$da_out"
  fi

  echo "DONE|${dataset}|${task_key}|${mode}|gpu=${gpu}"
}

schedule_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local src_path="${4:-}"
  local tgt_path="${5:-}"
  local trend_weight="$6"
  local task_key="${src}_to_${tgt}"

  if ! should_run_task "$dataset" "$task_key"; then
    return 0
  fi
  for mode in $MODES; do
    wait_for_slot
    local gpu="${GPUS[$((JOB_INDEX % ${#GPUS[@]}))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
    run_pipeline "$dataset" "$src" "$tgt" "$src_path" "$tgt_path" "$trend_weight" "$mode" "$gpu" &
    sleep 2
  done
}

JOB_INDEX=0
echo "RUN_TAG=$RUN_TAG"
echo "DATASETS=$DATASETS"
echo "MODES=$MODES"
echo "GPUS=${GPUS[*]}"
echo "Logs: $LOG_ROOT"
echo "Outputs: $OUT_ROOT"
echo "Summary: $SUMMARY_FILE"

for dataset in $DATASETS; do
  case "$dataset" in
    REMOTE)
      for spec in "${REMOTE_TASKS[@]}"; do
        IFS='|' read -r src tgt src_path tgt_path trend_weight <<< "$spec"
        schedule_task REMOTE "$src" "$tgt" "$src_path" "$tgt_path" "$trend_weight"
      done
      ;;
    HAR)
      for spec in "${HAR_TASKS[@]}"; do
        IFS='|' read -r src tgt <<< "$spec"
        schedule_task HAR "$src" "$tgt" "" "" "0.05"
      done
      ;;
    HHAR|HHAR_SA)
      for spec in "${HHAR_TASKS[@]}"; do
        IFS='|' read -r src tgt <<< "$spec"
        schedule_task HHAR_SA "$src" "$tgt" "" "" "0.05"
      done
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

wait
echo "v2.7.2 GTW-inspired full controller finished."
echo "Summary: $SUMMARY_FILE"
