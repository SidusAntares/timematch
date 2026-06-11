#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"

RUN_TAG="${RUN_TAG:-v243b_raw_strength_causal}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR2_to_FR1,FR1_to_AT1}"
SEEDS="${SEEDS:-1 2 3}"
RAW_WEIGHTS="${RAW_WEIGHTS:-0.25 0.5 1.0 2.0}"
DRY_RUN="${DRY_RUN:-False}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export DEVICE="${DEVICE:-cuda}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

# Isolate the causal variable: source-stage raw global compactness strength.
export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
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

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

JOBS="$LOG_DIR/jobs.tsv"
MANIFEST="$LOG_DIR/config_manifest.tsv"
: > "$JOBS"
printf "config\tcompact_weight\treshaper\tdual_path\tstructure_target\tsource_intra\tda_intra\tpurpose\n" > "$MANIFEST"

weight_tag() {
  local value="$1"
  value="${value%.0}"
  echo "$value" | sed 's/\./p/g'
}

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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$MANIFEST"
}

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local compact_weight="$6"
  local est_weight="$7"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$compact_weight" "$est_weight" >> "$JOBS"
}

add_task_seed_jobs() {
  local task="$1"
  local seed="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local est_weight="$5"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "plain" "0.0" "$est_weight"
  add_manifest "plain" "0.0" "none" "False" "raw" "0.0" "0.0" \
    "No source compactness, no reshaper, no DA-stage structure."

  local weight tag config
  for weight in $RAW_WEIGHTS; do
    tag="w$(weight_tag "$weight")"
    config="raw_global_${tag}_source_only"
    add_job "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$weight" "$est_weight"
    add_manifest "$config" "$weight" "none" "False" "raw" "$weight" "0.0" \
      "Source-stage raw global compactness with DA-stage structure off."
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

for gpu in "${GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${gpu}.tsv"
done

SORTED_JOBS="$LOG_DIR/jobs_sorted.tsv"
sort -t $'\t' -k7,7nr "$JOBS" > "$SORTED_JOBS"
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
  local task source_dataset target_dataset seed config compact_weight est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config compact_weight est_weight; do
    [ -z "$task" ] && continue
    local tag log_file
    tag="v243b_raw_strength_${RUN_TAG}_${task}_seed${seed}_${config}"
    log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"

    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|compact_weight=$compact_weight|log=$log_file"
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
      export SOURCE_STRUCTURE_FEATURE_TARGET="raw"
      export SOURCE_STRUCTURE_DETACH_FEATURES="False"
      export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$compact_weight"
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
echo "RAW_WEIGHTS=$RAW_WEIGHTS"
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

echo "Logs saved to: $LOG_DIR"
exit "$failed"
