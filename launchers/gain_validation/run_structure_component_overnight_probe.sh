#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLOCK_SCRIPT="$ROOT_DIR/launchers/gain_validation/run_source_structure_block.sh"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-structure_component_overnight_probe_${BATCH_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-model.pt}"
export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-50}"
export SOURCE_SKIP_TRAIN="${SOURCE_SKIP_TRAIN:-0}"

# Representative tasks:
# - FR1->FR2: no-seg/global compact previously best.
# - FR2->FR1: strong v2.4.3b segmented gain.
# - DK1->FR1: sensitive task where lighter structure and dynamics need checking.
TASK_SPECS="${TASK_SPECS:-FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017,FR2_to_FR1|france/31TCJ/2017|france/30TXT/2017,DK1_to_FR1|denmark/32VNH/2017|france/30TXT/2017}"

# name:loss_version:phase_count:segment_count:intra:trend:segment_inter:boundary:prototype_dynamics:trajectory_pooling:dynamics_mode:dual_cls:dual_relation
VARIANT_SPECS="${VARIANT_SPECS:-seg_full:segment_boundary_window_residual:5:5:1.0:0.05:0.02:0.20:0.00:meanmax:cosine:1.00:0.03,seg_no_dual:segment_boundary_window_residual:5:5:1.0:0.05:0.02:0.20:0.00:meanmax:cosine:0.00:0.00,seg_intra_only:segment_boundary_window_residual:5:5:1.0:0.00:0.00:0.00:0.00:meanmax:cosine:1.00:0.03,seg_transition_only:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.00:0.00:meanmax:cosine:1.00:0.03,seg_boundary_weighted:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.20:0.00:meanmax:cosine:1.00:0.03,global_compact:segment_boundary_window_residual:1:1:5.0:0.00:0.00:0.00:0.00:meanmax:cosine:1.00:0.03,global_dynamics:trajectory_prototype_dynamics_v244b:1:1:5.0:0.00:0.00:0.00:0.01:meanmax:cosine:1.00:0.03}"

IFS=',' read -r -a GPU_ITEMS <<< "$GPU_IDS"
IFS=',' read -r -a TASK_ITEMS <<< "$TASK_SPECS"
IFS=',' read -r -a VARIANT_ITEMS <<< "$VARIANT_SPECS"

job_count=0
gpu_cursor=0

run_one() {
  local task_name="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local variant_spec="$4"

  IFS=':' read -r variant_name loss_version phase_count segment_count intra_weight trend_weight segment_inter_weight boundary_weight proto_dyn_weight trajectory_pooling dynamics_mode dual_cls dual_relation <<< "$variant_spec"

  local gpu_id="${GPU_ITEMS[$((gpu_cursor % ${#GPU_ITEMS[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))

  local source_tile
  local target_tile
  source_tile="$(echo "$source_dataset" | cut -d'/' -f2)"
  target_tile="$(echo "$target_dataset" | cut -d'/' -f2)"

  local job_tag="${variant_name}_${task_name}_${BATCH_STAMP}"
  local log_file="$LOG_DIR/${job_tag}.log"

  local partition_mode="doy_gap"
  if [ "$phase_count" = "1" ] && [ "$segment_count" = "1" ]; then
    partition_mode="uniform"
  fi

  echo "START|${job_tag}|gpu=${gpu_id}|loss=${loss_version}|partition=${partition_mode}|intra=${intra_weight}|trend=${trend_weight}|segment_inter=${segment_inter_weight}|boundary=${boundary_weight}|proto_dyn=${proto_dyn_weight}|dual_cls=${dual_cls}|dual_relation=${dual_relation}"
  (
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      SOURCE="$source_dataset" \
      TARGETS_BLOCK="$target_dataset" \
      VARIANT_NAME="$variant_name" \
      SOURCE_PHASE_PARTITION_MODE="$partition_mode" \
      SOURCE_SEGMENT_PARTITION_MODE="$partition_mode" \
      SOURCE_PHASE_COUNT="$phase_count" \
      SOURCE_SEGMENT_COUNT="$segment_count" \
      SOURCE_STRUCTURE_LOSS_VERSION="$loss_version" \
      SOURCE_STRUCTURE_INTRA_TRADE_OFF="$intra_weight" \
      SOURCE_STRUCTURE_TREND_TRADE_OFF="$trend_weight" \
      SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$segment_inter_weight" \
      SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$boundary_weight" \
      SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_TRADE_OFF="$proto_dyn_weight" \
      SOURCE_STRUCTURE_TRAJECTORY_POOLING="$trajectory_pooling" \
      SOURCE_STRUCTURE_PROTOTYPE_DYNAMICS_MODE="$dynamics_mode" \
      SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="$dual_cls" \
      SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$dual_relation" \
      SOURCE_MODEL_TAG="${job_tag}_${source_tile}_to_${target_tile}_source" \
      RESHAPER_TAG="${job_tag}_${source_tile}_to_${target_tile}" \
      TARGET_MODEL_TAG="${job_tag}_${source_tile}_to_${target_tile}" \
      bash "$BLOCK_SCRIPT"
  ) > "$log_file" 2>&1 &

  job_count=$((job_count + 1))
  if [ $((job_count % MAX_PARALLEL)) -eq 0 ]; then
    wait
  fi
}

echo "Logs saved under: $LOG_DIR"
echo "TASK_SPECS=$TASK_SPECS"
echo "VARIANT_SPECS=$VARIANT_SPECS"

for task_spec in "${TASK_ITEMS[@]}"; do
  IFS='|' read -r task_name source_dataset target_dataset <<< "$task_spec"
  for variant_spec in "${VARIANT_ITEMS[@]}"; do
    run_one "$task_name" "$source_dataset" "$target_dataset" "$variant_spec"
  done
done

wait

echo "Overnight structure-component probe finished."
echo "Logs saved under: $LOG_DIR"
