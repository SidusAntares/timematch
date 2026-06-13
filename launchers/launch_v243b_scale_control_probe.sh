#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"

RUN_TAG="${RUN_TAG:-v243b_scale_control_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR2_to_FR1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
DRY_RUN="${DRY_RUN:-False}"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export DEVICE="${DEVICE:-cuda}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_FEATURE_TARGET="${SOURCE_STRUCTURE_FEATURE_TARGET:-raw}"
export SOURCE_STRUCTURE_DETACH_FEATURES="${SOURCE_STRUCTURE_DETACH_FEATURES:-False}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"
export TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.0}"

NORM_FLOOR_VALUE="${NORM_FLOOR_VALUE:-1.0}"
NORM_FLOOR_TRADE_OFF="${NORM_FLOOR_TRADE_OFF:-0.01}"

mkdir -p "$LOG_DIR"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

JOBS="$LOG_DIR/jobs.tsv"
MANIFEST="$LOG_DIR/config_manifest.tsv"
: > "$JOBS"
printf "config\tcompact_weight\tcompact_distance\tnorm_preserve_trade_off\tnorm_preserve_target\tnorm_preserve_value\tpurpose\n" > "$MANIFEST"

task_spec() {
  case "$1" in
    FR1_to_FR2) echo "france/30TXT/2017 france/31TCJ/2017 4" ;;
    FR1_to_DK1) echo "france/30TXT/2017 denmark/32VNH/2017 4" ;;
    FR1_to_AT1) echo "france/30TXT/2017 austria/33UVP/2017 4" ;;
    FR2_to_FR1) echo "france/31TCJ/2017 france/30TXT/2017 3" ;;
    FR2_to_DK1) echo "france/31TCJ/2017 denmark/32VNH/2017 3" ;;
    FR2_to_AT1) echo "france/31TCJ/2017 austria/33UVP/2017 3" ;;
    DK1_to_FR1) echo "denmark/32VNH/2017 france/30TXT/2017 2" ;;
    DK1_to_FR2) echo "denmark/32VNH/2017 france/31TCJ/2017 2" ;;
    DK1_to_AT1) echo "denmark/32VNH/2017 austria/33UVP/2017 2" ;;
    AT1_to_FR1) echo "austria/33UVP/2017 france/30TXT/2017 2" ;;
    AT1_to_FR2) echo "austria/33UVP/2017 france/31TCJ/2017 2" ;;
    AT1_to_DK1) echo "austria/33UVP/2017 denmark/32VNH/2017 2" ;;
    *)
      echo "ERROR unknown task: $1" >&2
      return 1
      ;;
  esac
}

add_manifest() {
  local config="$1"
  if grep -F -q "${config}"$'\t' "$MANIFEST"; then
    return
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$MANIFEST"
}

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local compact_weight="$6"
  local compact_distance="$7"
  local norm_trade_off="$8"
  local norm_target="$9"
  local norm_value="${10}"
  local est_weight="${11}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$compact_weight" \
    "$compact_distance" "$norm_trade_off" "$norm_target" "$norm_value" "$est_weight" >> "$JOBS"
}

add_task_seed_jobs() {
  local task="$1"
  local seed="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local est_weight="$5"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "plain" "0.0" "mse" "0.0" "none" "0.0" "$est_weight"
  add_manifest "plain" "0.0" "mse" "0.0" "none" "0.0" "No source compactness."

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_w0p75_source_only" "0.75" "mse" "0.0" "none" "0.0" "$est_weight"
  add_manifest "raw_global_w0p75_source_only" "0.75" "mse" "0.0" "none" "0.0" "Empirical best moderate raw MSE compactness reference."

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_w1_source_only" "1.0" "mse" "0.0" "none" "0.0" "$est_weight"
  add_manifest "raw_global_w1_source_only" "1.0" "mse" "0.0" "none" "0.0" "Strong raw MSE compactness reference with observed DA drop."

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_w1_normmse_source_only" "1.0" "normalized_mse" "0.0" "none" "0.0" "$est_weight"
  add_manifest "raw_global_w1_normmse_source_only" "1.0" "normalized_mse" "0.0" "none" "0.0" "Scale-free L2-normalized MSE compactness at strong weight."

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_w1_normfloor_source_only" "1.0" "mse" "$NORM_FLOOR_TRADE_OFF" "min_mean" "$NORM_FLOOR_VALUE" "$est_weight"
  add_manifest "raw_global_w1_normfloor_source_only" "1.0" "mse" "$NORM_FLOOR_TRADE_OFF" "min_mean" "$NORM_FLOOR_VALUE" "Strong MSE compactness with source-only mean-norm floor."

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_w0p75_normfloor_source_only" "0.75" "mse" "$NORM_FLOOR_TRADE_OFF" "min_mean" "$NORM_FLOOR_VALUE" "$est_weight"
  add_manifest "raw_global_w0p75_normfloor_source_only" "0.75" "mse" "$NORM_FLOOR_TRADE_OFF" "min_mean" "$NORM_FLOOR_VALUE" "Moderate MSE compactness with source-only mean-norm floor."
}

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
for seed in $SEEDS; do
  for task in "${TASK_NAMES[@]}"; do
    task="$(echo "$task" | xargs)"
    spec="$(task_spec "$task")" || exit 2
    read -r source_dataset target_dataset est_weight <<< "$spec"
    add_task_seed_jobs "$task" "$seed" "$source_dataset" "$target_dataset" "$est_weight"
  done
done

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
sort -t $'\t' -k11,11nr "$JOBS" > "$SORTED_JOBS"
job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$SORTED_JOBS"

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed config compact_weight compact_distance norm_trade_off norm_target norm_value est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config compact_weight compact_distance norm_trade_off norm_target norm_value est_weight; do
    [ -z "$task" ] && continue
    local tag log_file
    tag="v243b_scale_${RUN_TAG}_${task}_seed${seed}_${config}"
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"

    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|compact_weight=$compact_weight|distance=$compact_distance|norm_trade_off=$norm_trade_off|log=$log_file"
    (
      export SEED="$seed"
      export RESHAPER_TAG="$tag"
      export SOURCE_PHASE_PARTITION_MODE="uniform"
      export SOURCE_SEGMENT_PARTITION_MODE="uniform"
      export SOURCE_PHASE_COUNT="1"
      export SOURCE_SEGMENT_COUNT="1"
      export SOURCE_FEATURE_RESHAPER="none"
      export SOURCE_FEATURE_RESHAPER_STRENGTH="0.00"
      export SOURCE_FEATURE_RESHAPER_TRAINABLE="False"
      export SOURCE_FEATURE_RESHAPER_INIT_SEED="-1"
      export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="0.00"
      export SOURCE_FEATURE_DUAL_PATH="False"
      export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="1.00"
      export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="0.00"
      export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$compact_weight"
      export SOURCE_STRUCTURE_COMPACT_DISTANCE="$compact_distance"
      export SOURCE_STRUCTURE_NORM_PRESERVE_TRADE_OFF="$norm_trade_off"
      export SOURCE_STRUCTURE_NORM_PRESERVE_TARGET="$norm_target"
      export SOURCE_STRUCTURE_NORM_PRESERVE_VALUE="$norm_value"
      export TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF="0.0"
      export TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF="0.0"
      export TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="0.0"
      export TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="0.0"

      CUDA_VISIBLE_DEVICES="$gpu" \
        SOURCE="$source_dataset" \
        TARGETS_BLOCK="$target_dataset" \
        bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_source_block.sh"
    ) > "$log_file" 2>&1

    status="$?"
    if [ "$status" -eq 0 ]; then
      echo "DONE|gpu=$gpu|task=$task|seed=$seed|config=$config"
    else
      echo "FAIL|gpu=$gpu|task=$task|seed=$seed|config=$config|status=$status"
      worker_failed=1
    fi
  done < "$queue"
  return "$worker_failed"
}

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "NORM_FLOOR_VALUE=$NORM_FLOOR_VALUE"
echo "NORM_FLOOR_TRADE_OFF=$NORM_FLOOR_TRADE_OFF"
echo "SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS"
echo "TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "TIMEMATCH_STEPS_PER_EPOCH=$TIMEMATCH_STEPS_PER_EPOCH"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "JOBS=$(wc -l < "$JOBS")"
echo "MANIFEST=$MANIFEST"

case "$(echo "$DRY_RUN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "DRY_RUN=True"
    echo "Jobs: $JOBS"
    for gpu in "${GPU_IDS[@]}"; do
      echo "Queue gpu${gpu}: $LOG_DIR/queue_gpu${gpu}.tsv ($(wc -l < "$LOG_DIR/queue_gpu${gpu}.tsv") jobs)"
    done
    exit 0
    ;;
esac

pids=()
failed=0
for gpu in "${GPU_IDS[@]}"; do
  run_worker "$gpu" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

python "$ROOT_DIR/analysis/summarize_v243b_raw_strength_intervention.py" "$LOG_DIR" || failed=1
python "$ROOT_DIR/analysis/analyze_v243b_dose_epoch_trajectory.py" "$LOG_DIR" || failed=1
python "$ROOT_DIR/analysis/v243b_target_readiness_diagnostic.py" "$LOG_DIR" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --max_batches "${TARGET_READINESS_MAX_BATCHES:-64}" \
  --max_metric_samples "${TARGET_READINESS_MAX_METRIC_SAMPLES:-2048}" \
  --feature_kind final \
  --output_prefix target_readiness_final || failed=1
python "$ROOT_DIR/analysis/v243b_target_readiness_diagnostic.py" "$LOG_DIR" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --max_batches "${TARGET_READINESS_MAX_BATCHES:-64}" \
  --max_metric_samples "${TARGET_READINESS_MAX_METRIC_SAMPLES:-2048}" \
  --feature_kind raw_pooled \
  --output_prefix target_readiness_raw_pooled || failed=1

python "$ROOT_DIR/analysis/v243b_representation_content_probe.py" "$LOG_DIR" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --max_batches "${CONTENT_PROBE_MAX_BATCHES:-64}" \
  --max_metric_samples "${CONTENT_PROBE_MAX_METRIC_SAMPLES:-4096}" \
  --feature_kind raw_pooled \
  --output_prefix representation_content_raw_pooled || failed=1
python "$ROOT_DIR/analysis/v243b_representation_content_probe.py" "$LOG_DIR" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --max_batches "${CONTENT_PROBE_MAX_BATCHES:-64}" \
  --max_metric_samples "${CONTENT_PROBE_MAX_METRIC_SAMPLES:-4096}" \
  --feature_kind final \
  --output_prefix representation_content_final || failed=1

echo "Logs saved to: $LOG_DIR"
exit "$failed"
