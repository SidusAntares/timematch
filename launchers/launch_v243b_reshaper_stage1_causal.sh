#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_reshaper_stage1_causal}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-FR1_to_AT1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
FROZEN_INITS="${FROZEN_INITS:-101 102 103}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export DEVICE="${DEVICE:-cuda}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_INTRA_TRADE_OFF="${SOURCE_STRUCTURE_INTRA_TRADE_OFF:-0.0}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_TREND_TRADE_OFF="${SOURCE_STRUCTURE_TREND_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="${SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"
export SOURCE_STRUCTURE_GRAD_DIAGNOSTIC="${SOURCE_STRUCTURE_GRAD_DIAGNOSTIC:-False}"
export SOURCE_STRUCTURE_GRAD_DIAG_STEPS="${SOURCE_STRUCTURE_GRAD_DIAG_STEPS:-1,10,50,100,200,500}"

export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"

# Keep DA-stage source structure off. This experiment isolates reshaper /
# dual-path perturbation rather than continued DA structure shaping.
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
printf "config\tkind\tstrength\ttrainable\tinit_seed\treg_trade_off\tdual_relation\tpurpose\n" > "$MANIFEST"

config_doc() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$MANIFEST"
}

config_doc "plain" "none" "0" "False" "-1" "0" "0" "No reshaper and no dual path."
config_doc "strength0_dualcls_rel000" "residual_temporal_conv" "0.00" "False" "101" "0" "0.00" "Dual classifier path with identity perturbation and no relation loss."
config_doc "strength0_dualcls_rel003" "residual_temporal_conv" "0.00" "False" "101" "0" "0.03" "Dual classifier path with identity perturbation and default relation loss."
config_doc "trainable_s003_reg005" "residual_temporal_conv" "0.03" "True" "101" "0.05" "0.03" "Trainable near-identity reshaper with identity regularization."
config_doc "trainable_s003_reg000" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "Trainable near-identity reshaper without identity regularization."
config_doc "trainable_s010_reg005" "residual_temporal_conv" "0.10" "True" "101" "0.05" "0.03" "Historical strength trainable reshaper."
config_doc "frozen_s003_initX" "residual_temporal_conv" "0.03" "False" "X" "0.00" "0.03" "Frozen random near-identity perturbation, repeated across init seeds."
config_doc "frozen_s010_initX" "residual_temporal_conv" "0.10" "False" "X" "0.00" "0.03" "Frozen random historical-strength perturbation, repeated across init seeds."

task_spec() {
  case "$1" in
    FR1_to_AT1)
      echo "france/30TXT/2017 austria/33UVP/2017 3"
      ;;
    AT1_to_DK1)
      echo "austria/33UVP/2017 denmark/32VNH/2017 2"
      ;;
    FR2_to_DK1)
      echo "france/31TCJ/2017 denmark/32VNH/2017 3"
      ;;
    *)
      echo "ERROR unknown task: $1" >&2
      return 1
      ;;
  esac
}

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local reshaper="$6"
  local strength="$7"
  local trainable="$8"
  local init_seed="$9"
  local reg_trade_off="${10}"
  local dual_relation="${11}"
  local est_weight="${12}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$reshaper" \
    "$strength" "$trainable" "$init_seed" "$reg_trade_off" "$dual_relation" "$est_weight" >> "$JOBS"
}

add_task_seed_jobs() {
  local task="$1"
  local seed="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local est_weight="$5"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "plain" \
    "none" "0.00" "False" "-1" "0.00" "0.00" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "strength0_dualcls_rel000" \
    "residual_temporal_conv" "0.00" "False" "101" "0.00" "0.00" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "strength0_dualcls_rel003" \
    "residual_temporal_conv" "0.00" "False" "101" "0.00" "0.03" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_reg005" \
    "residual_temporal_conv" "0.03" "True" "101" "0.05" "0.03" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_reg000" \
    "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "$est_weight"
  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s010_reg005" \
    "residual_temporal_conv" "0.10" "True" "101" "0.05" "0.03" "$est_weight"

  for init_seed in $FROZEN_INITS; do
    add_job "$task" "$source_dataset" "$target_dataset" "$seed" "frozen_s003_init${init_seed}" \
      "residual_temporal_conv" "0.03" "False" "$init_seed" "0.00" "0.03" "$est_weight"
    add_job "$task" "$source_dataset" "$target_dataset" "$seed" "frozen_s010_init${init_seed}" \
      "residual_temporal_conv" "0.10" "False" "$init_seed" "0.00" "0.03" "$est_weight"
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
  local task source_dataset target_dataset seed config reshaper strength trainable init_seed reg_trade_off dual_relation est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config reshaper strength trainable init_seed reg_trade_off dual_relation est_weight; do
    local tag="v243b_reshaper_${task}_seed${seed}_${config}"
    local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|strength=$strength|trainable=$trainable|init=$init_seed|reg=$reg_trade_off|dualrel=$dual_relation|log=$log_file"
    (
      export SEED="$seed"
      export RESHAPER_TAG="$tag"
      export SOURCE_PHASE_PARTITION_MODE="uniform"
      export SOURCE_SEGMENT_PARTITION_MODE="uniform"
      export SOURCE_PHASE_COUNT="1"
      export SOURCE_SEGMENT_COUNT="1"
      export SOURCE_FEATURE_RESHAPER="$reshaper"
      export SOURCE_FEATURE_RESHAPER_STRENGTH="$strength"
      export SOURCE_FEATURE_RESHAPER_TRAINABLE="$trainable"
      export SOURCE_FEATURE_RESHAPER_INIT_SEED="$init_seed"
      export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$reg_trade_off"
      export SOURCE_FEATURE_DUAL_PATH="True"
      export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="1.00"
      export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$dual_relation"
      if [ "$reshaper" = "none" ]; then
        export SOURCE_FEATURE_DUAL_PATH="False"
      fi
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
python "$ROOT_DIR/analysis/summarize_v243b_reshaper_stage1.py" "$LOG_DIR" || true

exit "$failed"
