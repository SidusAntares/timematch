#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_counterfactual_chain}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-AT1_to_DK1,FR1_to_AT1,FR2_to_DK1,DK1_to_FR1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
COMPACT_WEIGHT="${COMPACT_WEIGHT:-1.0}"
INCLUDE_SCOPE_CONFIGS="${INCLUDE_SCOPE_CONFIGS:-False}"

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

export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"

# Keep DA-stage source structure off. This chain attributes source-stage
# compactness/reshaper effects before adding stage adaptation.
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
printf "config\tmechanism\tstructure_target\tdetach\tcompact_weight\tstrength\ttrainable\tinit_seed\treg_trade_off\tdual_relation\tpurpose\n" > "$MANIFEST"

truthy() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

weight_tag() {
  local value="$1"
  value="${value%.0}"
  echo "$value" | sed 's/\./p/g'
}

COMPACT_TAG="w$(weight_tag "$COMPACT_WEIGHT")"

config_doc() {
  local key="$1"
  if grep -F -q "${key}"$'\t' "$MANIFEST"; then
    return
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" >> "$MANIFEST"
}

task_spec() {
  case "$1" in
    FR1_to_FR2)
      echo "france/30TXT/2017 france/31TCJ/2017 3"
      ;;
    FR1_to_DK1)
      echo "france/30TXT/2017 denmark/32VNH/2017 3"
      ;;
    FR1_to_AT1)
      echo "france/30TXT/2017 austria/33UVP/2017 3"
      ;;
    FR2_to_FR1)
      echo "france/31TCJ/2017 france/30TXT/2017 3"
      ;;
    FR2_to_DK1)
      echo "france/31TCJ/2017 denmark/32VNH/2017 3"
      ;;
    FR2_to_AT1)
      echo "france/31TCJ/2017 austria/33UVP/2017 3"
      ;;
    DK1_to_FR1)
      echo "denmark/32VNH/2017 france/30TXT/2017 2"
      ;;
    DK1_to_FR2)
      echo "denmark/32VNH/2017 france/31TCJ/2017 2"
      ;;
    DK1_to_AT1)
      echo "denmark/32VNH/2017 austria/33UVP/2017 2"
      ;;
    AT1_to_FR1)
      echo "austria/33UVP/2017 france/30TXT/2017 2"
      ;;
    AT1_to_FR2)
      echo "austria/33UVP/2017 france/31TCJ/2017 2"
      ;;
    AT1_to_DK1)
      echo "austria/33UVP/2017 denmark/32VNH/2017 2"
      ;;
    * )
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
  local intra="${12}"
  local structure_target="${13}"
  local detach_features="${14}"
  local est_weight="${15}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$reshaper" \
    "$strength" "$trainable" "$init_seed" "$reg_trade_off" "$dual_relation" \
    "$intra" "$structure_target" "$detach_features" "$est_weight" >> "$JOBS"
}

add_config() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local mechanism="$6"
  local reshaper="$7"
  local strength="$8"
  local trainable="$9"
  local init_seed="${10}"
  local reg_trade_off="${11}"
  local dual_relation="${12}"
  local intra="${13}"
  local structure_target="${14}"
  local detach_features="${15}"
  local est_weight="${16}"
  local purpose="${17}"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "$config" \
    "$reshaper" "$strength" "$trainable" "$init_seed" "$reg_trade_off" "$dual_relation" \
    "$intra" "$structure_target" "$detach_features" "$est_weight"
  config_doc "$config" "$mechanism" "$structure_target" "$detach_features" "$intra" \
    "$strength" "$trainable" "$init_seed" "$reg_trade_off" "$dual_relation" "$purpose"
}

add_task_seed_jobs() {
  local task="$1"
  local seed="$2"
  local source_dataset="$3"
  local target_dataset="$4"
  local est_weight="$5"

  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "plain" \
    "none" "none" "0.00" "False" "-1" "0.00" "0.00" "0.0" "auto" "False" "$est_weight" \
    "No reshaper and no compactness."
  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_${COMPACT_TAG}_detached" \
    "raw_compact_value_control" "none" "0.00" "False" "-1" "0.00" "0.00" "$COMPACT_WEIGHT" "raw" "True" "$est_weight" \
    "Compactness loss value is computed on raw features, but detached from gradients."
  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "raw_global_${COMPACT_TAG}" \
    "raw_compact_gradient" "none" "0.00" "False" "-1" "0.00" "0.00" "$COMPACT_WEIGHT" "raw" "False" "$est_weight" \
    "Global compactness directly back-propagates through raw encoder features."

  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_reg000" \
    "reshaper_aug_only" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "0.0" "auto" "False" "$est_weight" \
    "Near-identity reshaper/dual-path augmentation without compactness."
  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_raw_global_${COMPACT_TAG}_detached" \
    "reshaper_aug_plus_compact_value_control" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "$COMPACT_WEIGHT" "raw" "True" "$est_weight" \
    "Near-identity reshaper/dual-path augmentation plus detached raw compactness value."
  add_config "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_raw_global_${COMPACT_TAG}" \
    "reshaper_aug_plus_raw_compact_gradient" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "$COMPACT_WEIGHT" "raw" "False" "$est_weight" \
    "Near-identity reshaper/dual-path augmentation plus raw compactness gradient."

  if truthy "$INCLUDE_SCOPE_CONFIGS"; then
    add_config "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_reshaped_global_${COMPACT_TAG}" \
      "reshaper_scope_reshaped_compact_gradient" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "$COMPACT_WEIGHT" "reshaped" "False" "$est_weight" \
      "Scope add-on: compactness back-propagates through reshaper output."
    add_config "$task" "$source_dataset" "$target_dataset" "$seed" "trainable_s003_both_global_${COMPACT_TAG}" \
      "reshaper_scope_raw_and_reshaped_compact_gradient" "residual_temporal_conv" "0.03" "True" "101" "0.00" "0.03" "$COMPACT_WEIGHT" "both" "False" "$est_weight" \
      "Scope add-on: raw compactness shapes encoder and reshaped compactness shapes reshaper."
  fi
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
sort -t $'\t' -k15,15nr "$JOBS" > "$SORTED_JOBS"
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
  local task source_dataset target_dataset seed config reshaper strength trainable init_seed reg_trade_off dual_relation intra structure_target detach_features est_weight

  while IFS=$'\t' read -r task source_dataset target_dataset seed config reshaper strength trainable init_seed reg_trade_off dual_relation intra structure_target detach_features est_weight; do
    local tag="v243b_cf_${task}_seed${seed}_${config}"
    local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|target=$structure_target|detach=$detach_features|intra=$intra|strength=$strength|trainable=$trainable|init=$init_seed|log=$log_file"
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
      export SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra"
      export SOURCE_STRUCTURE_FEATURE_TARGET="$structure_target"
      export SOURCE_STRUCTURE_DETACH_FEATURES="$detach_features"
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
python "$ROOT_DIR/analysis/summarize_v243b_counterfactual_chain.py" "$LOG_DIR" || true

exit "$failed"
