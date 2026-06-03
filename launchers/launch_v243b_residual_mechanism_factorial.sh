#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_residual_mechanism_factorial}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
SEEDS="${SEEDS:-1 2 3}"
COMPACT_INTRA="${COMPACT_INTRA:-1.0}"

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
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

# This diagnostic isolates source-stage residual mechanisms. The previous
# mechanism run already showed DA-stage continued structure was not the cause.
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
printf "config\treshaper\tcompactness\tpartition\tphase_count\tpurpose\n" > "$MANIFEST"

config_doc() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$6" >> "$MANIFEST"
}

config_doc "plain" "off" "off" "uniform" "1" "Pure no-structure control."
config_doc "reshaper_only" "on" "off" "uniform" "1" "Tests whether the residual temporal conv reshaper is useful without structure loss."
config_doc "global_compact" "off" "on" "uniform" "1" "Tests weak global compactness directly on raw source features."
config_doc "reshaper_global_compact" "on" "on" "uniform" "1" "Tests reshaper x global compactness interaction."
config_doc "local_compact" "off" "on" "uniform" "5" "Tests whether local pooling compactness helps without reshaper."
config_doc "reshaper_local_compact" "on" "on" "uniform" "5" "Tests local pooling compactness under reshaper."

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local partition="$6"
  local phase_count="$7"
  local intra="$8"
  local reshaper="$9"
  local reshaper_strength="${10}"
  local dual_path="${11}"
  local est_weight="${12}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$partition" "$phase_count" \
    "$intra" "$reshaper" "$reshaper_strength" "$dual_path" "$est_weight" >> "$JOBS"
}

add_task_jobs() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local est_weight="$5"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "plain" \
    "uniform" "1" "0.0" "none" "0.00" "False" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "reshaper_only" \
    "uniform" "1" "0.0" "residual_temporal_conv" "0.10" "True" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "global_compact" \
    "uniform" "1" "$COMPACT_INTRA" "none" "0.00" "False" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "reshaper_global_compact" \
    "uniform" "1" "$COMPACT_INTRA" "residual_temporal_conv" "0.10" "True" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "local_compact" \
    "uniform" "5" "$COMPACT_INTRA" "none" "0.00" "False" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "reshaper_local_compact" \
    "uniform" "5" "$COMPACT_INTRA" "residual_temporal_conv" "0.10" "True" "$est_weight"
}

for seed in $SEEDS; do
  # FR1 source is the slowest group; keep it in the manifest with a larger scheduling weight.
  add_task_jobs "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "$seed" "3"
  add_task_jobs "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "$seed" "2"
  add_task_jobs "DK1_to_AT1" "denmark/32VNH/2017" "austria/33UVP/2017" "$seed" "1"
done

for idx in "${!GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${GPU_IDS[$idx]}.tsv"
done

# Sort heavy jobs first and round-robin them so FR1-source runs are spread over all GPUs.
SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
sort -t $'\t' -k12,12nr "$JOBS" > "$SORTED_JOBS"
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
  local task source_dataset target_dataset seed config partition phase_count intra reshaper
  local reshaper_strength dual_path est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config partition phase_count intra reshaper reshaper_strength dual_path est_weight; do
    local tag="v243b_residual_${task}_seed${seed}_${config}"
    local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|partition=$partition|phase_count=$phase_count|reshaper=$reshaper|intra=$intra|log=$log_file"
    (
      export SEED="$seed"
      export RESHAPER_TAG="$tag"
      export SOURCE_PHASE_PARTITION_MODE="$partition"
      export SOURCE_SEGMENT_PARTITION_MODE="$partition"
      export SOURCE_PHASE_COUNT="$phase_count"
      export SOURCE_SEGMENT_COUNT="$phase_count"
      export SOURCE_FEATURE_RESHAPER="$reshaper"
      export SOURCE_FEATURE_RESHAPER_STRENGTH="$reshaper_strength"
      export SOURCE_FEATURE_DUAL_PATH="$dual_path"
      export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra"
      export SOURCE_STRUCTURE_TREND_TRADE_OFF="0.00"
      export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="0.00"
      export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="0.00"
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
python "$ROOT_DIR/analysis/summarize_v243b_residual_mechanism.py" "$LOG_DIR" || true

exit "$failed"
