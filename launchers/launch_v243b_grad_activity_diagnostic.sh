#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_grad_activity_diagnostic}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
DIAG_MODE="${DIAG_MODE:-topology}"

if [ "$DIAG_MODE" = "topology" ]; then
  TASKS="${TASKS:-FR1_to_AT1}"
  SEEDS="${SEEDS:-1}"
  COMPACT_WEIGHTS="${COMPACT_WEIGHTS:-1}"
else
  TASKS="${TASKS:-FR1_to_AT1,AT1_to_DK1}"
  SEEDS="${SEEDS:-1 2 3}"
  COMPACT_WEIGHTS="${COMPACT_WEIGHTS:-1 3}"
fi

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export DEVICE="${DEVICE:-cuda}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"
export SOURCE_STRUCTURE_GRAD_DIAGNOSTIC="${SOURCE_STRUCTURE_GRAD_DIAGNOSTIC:-True}"
export SOURCE_STRUCTURE_GRAD_DIAG_STEPS="${SOURCE_STRUCTURE_GRAD_DIAG_STEPS:-1,10,50,100,200,500}"

export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

# Keep DA-stage source structure off in this diagnostic. We are measuring the
# source-stage gradient topology, not re-testing continued DA shaping.
export TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.0}"
export TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.0}"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

MANIFEST="$LOG_DIR/config_manifest.tsv"
JOBS="$LOG_DIR/jobs.tsv"
: > "$JOBS"
printf "mode\t%s\n" "$DIAG_MODE" > "$MANIFEST"
printf "config\treshaper\tcompactness_weight\tpurpose\n" >> "$MANIFEST"
printf "plain\toff\t0\tNo reshaper and no compactness.\n" >> "$MANIFEST"
printf "reshaper_only\ton\t0\tReshaper path without compactness.\n" >> "$MANIFEST"
printf "global_compact_wX\toff\tX\tGlobal compactness directly on raw source features.\n" >> "$MANIFEST"
printf "reshaper_global_compact_wX\ton\tX\tGlobal compactness on reshaper output.\n" >> "$MANIFEST"

task_spec() {
  case "$1" in
    FR1_to_AT1)
      echo "france/30TXT/2017 austria/33UVP/2017 3"
      ;;
    AT1_to_DK1)
      echo "austria/33UVP/2017 denmark/32VNH/2017 2"
      ;;
    *)
      echo "ERROR unknown task: $1" >&2
      return 1
      ;;
  esac
}

weight_tag() {
  echo "$1" | sed 's/\./p/g'
}

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local intra="$6"
  local reshaper="$7"
  local reshaper_strength="$8"
  local dual_path="$9"
  local est_weight="${10}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$intra" \
    "$reshaper" "$reshaper_strength" "$dual_path" "$est_weight" >> "$JOBS"
}

add_task_seed_jobs() {
  local task="$1"
  local seed="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local est_weight="$5"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "plain" \
    "0.0" "none" "0.00" "False" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "reshaper_only" \
    "0.0" "residual_temporal_conv" "0.10" "True" "$est_weight"

  for weight in $COMPACT_WEIGHTS; do
    local tag
    tag="$(weight_tag "$weight")"
    add_job "$task" "$source_dataset" "$target_dataset" "$seed" "global_compact_w${tag}" \
      "$weight" "none" "0.00" "False" "$est_weight"
    add_job "$task" "$source_dataset" "$target_dataset" "$seed" "reshaper_global_compact_w${tag}" \
      "$weight" "residual_temporal_conv" "0.10" "True" "$est_weight"
  done
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

for idx in "${!GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${GPU_IDS[$idx]}.tsv"
done

SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
sort -t $'\t' -k10,10nr "$JOBS" > "$SORTED_JOBS"
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
  local task source_dataset target_dataset seed config intra reshaper reshaper_strength dual_path est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config intra reshaper reshaper_strength dual_path est_weight; do
    local tag="v243b_grad_${task}_seed${seed}_${config}"
    local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|intra=$intra|reshaper=$reshaper|log=$log_file"
    (
      export SEED="$seed"
      export RESHAPER_TAG="$tag"
      export SOURCE_PHASE_PARTITION_MODE="uniform"
      export SOURCE_SEGMENT_PARTITION_MODE="uniform"
      export SOURCE_PHASE_COUNT="1"
      export SOURCE_SEGMENT_COUNT="1"
      export SOURCE_FEATURE_RESHAPER="$reshaper"
      export SOURCE_FEATURE_RESHAPER_STRENGTH="$reshaper_strength"
      export SOURCE_FEATURE_DUAL_PATH="$dual_path"
      export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra"
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

echo "Logs saved to: $LOG_DIR"
python "$ROOT_DIR/analysis/summarize_v243b_grad_diagnostic.py" "$LOG_DIR" || true

exit "$failed"
