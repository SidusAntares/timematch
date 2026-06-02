#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
RUN_TAG="${RUN_TAG:-v243b_segment_mechanism_core}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
SEEDS="${SEEDS:-1 2 3 4 5}"

mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
export DEVICE="${DEVICE:-cuda}"

export SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
export SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
export SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
export SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"

export SOURCE_PHASE_GAP_THRESHOLD="${SOURCE_PHASE_GAP_THRESHOLD:-45}"
export SOURCE_PHASE_MIN_POINTS="${SOURCE_PHASE_MIN_POINTS:-3}"
export SOURCE_PHASE_MAX_POINTS="${SOURCE_PHASE_MAX_POINTS:-8}"
export SOURCE_PHASE_MAX_SPAN="${SOURCE_PHASE_MAX_SPAN:-120}"
export SOURCE_PHASE_MIN_SAMPLE_POINTS="${SOURCE_PHASE_MIN_SAMPLE_POINTS:-2}"

export SOURCE_STRUCTURE_LOSS_VERSION="${SOURCE_STRUCTURE_LOSS_VERSION:-segment_boundary_window_residual}"
export SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF="${SOURCE_STRUCTURE_AMPLITUDE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF="${SOURCE_STRUCTURE_INTERPHASE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SHAPE_TRADE_OFF="${SOURCE_STRUCTURE_SHAPE_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_SEASON_TRADE_OFF="${SOURCE_STRUCTURE_SEASON_TRADE_OFF:-0.00}"
export SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE="${SOURCE_STRUCTURE_BOUNDARY_WINDOW_SIZE:-2}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

read -r -a GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

MANIFEST="$LOG_DIR/config_manifest.tsv"
JOBS="$LOG_DIR/jobs.tsv"
: > "$JOBS"
printf "config\tpurpose\n" > "$MANIFEST"

config_doc() {
  printf "%s\t%s\n" "$1" "$2" >> "$MANIFEST"
}

config_doc "nostruct_global" "No structure loss with identity/global support; controls reshaper/dual-path without structure shaping."
config_doc "full_doy" "Complete v2.4.3b-style DOY-gap partition with intra, trend, segment transition, and boundary transition modulation."
config_doc "full_uniform" "Same loss weights as full_doy, but uniform partitions; isolates DOY-gap partition semantics."
config_doc "doy_intra_trend" "DOY-gap local carrier with intra + trend only; tests whether local segment pooling alone is enough."
config_doc "doy_no_trend_renorm" "Remove trend and renormalize intra/segment-inter mass; tests trend functional necessity under similar loss scale."
config_doc "doy_no_segment_inter_renorm" "Remove segment transition and renormalize intra/trend mass; tests transition functional necessity."
config_doc "doy_no_boundary_mod" "Keep segment transition but disable boundary-window modulation; tests whether boundary saliency matters."
config_doc "full_doy_daoff" "Full source-stage shaping, but TimeMatch-stage structure weights are zero; tests DA-stage continued shaping."
config_doc "no_reshaper" "No reshaper and no structure loss path; coarse control for whether the reshaper stack itself is necessary."

task_weights() {
  local task="$1"
  if [ "$task" = "FR1_to_AT1" ]; then
    echo "0.03 0.01 0.10"
  else
    echo "0.05 0.02 0.20"
  fi
}

renorm_without_trend() {
  local trend="$1"
  local si="$2"
  python - "$trend" "$si" <<'PY'
import sys
t=float(sys.argv[1]); si=float(sys.argv[2])
total=1.0+t+si
scale=total/(1.0+si)
print(f"{1.0*scale:.6f} 0.000000 {si*scale:.6f}")
PY
}

renorm_without_segment_inter() {
  local trend="$1"
  local si="$2"
  python - "$trend" "$si" <<'PY'
import sys
t=float(sys.argv[1]); si=float(sys.argv[2])
total=1.0+t+si
scale=total/(1.0+t)
print(f"{1.0*scale:.6f} {t*scale:.6f} 0.000000")
PY
}

add_job() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local config="$5"
  local partition="$6"
  local phase_count="$7"
  local intra="$8"
  local trend="$9"
  local segment_inter="${10}"
  local boundary="${11}"
  local reshaper="${12}"
  local reshaper_strength="${13}"
  local dual_path="${14}"
  local da_intra="${15}"
  local da_trend="${16}"
  local da_segment_inter="${17}"
  local da_boundary="${18}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$source_dataset" "$target_dataset" "$seed" "$config" "$partition" "$phase_count" \
    "$intra" "$trend" "$segment_inter" "$boundary" "$reshaper" "$reshaper_strength" "$dual_path" \
    "$da_intra" "$da_trend" "$da_segment_inter" "$da_boundary" >> "$JOBS"
}

add_task_jobs() {
  local task="$1"
  local source_dataset="$2"
  local target_dataset="$3"
  local seed="$4"
  local weights
  weights="$(task_weights "$task")"
  read -r base_trend base_si base_boundary <<< "$weights"

  read -r no_trend_intra no_trend no_trend_si <<< "$(renorm_without_trend "$base_trend" "$base_si")"
  read -r no_si_intra no_si_trend no_si <<< "$(renorm_without_segment_inter "$base_trend" "$base_si")"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "nostruct_global" "uniform" "1" \
    "0.0" "0.00" "0.00" "0.00" "residual_temporal_conv" "0.10" "True" \
    "0.0" "0.00" "0.00" "0.00"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "full_doy" "doy_gap" "5" \
    "1.0" "$base_trend" "$base_si" "$base_boundary" "residual_temporal_conv" "0.10" "True" \
    "1.0" "$base_trend" "$base_si" "$base_boundary"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "full_uniform" "uniform" "5" \
    "1.0" "$base_trend" "$base_si" "$base_boundary" "residual_temporal_conv" "0.10" "True" \
    "1.0" "$base_trend" "$base_si" "$base_boundary"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "doy_intra_trend" "doy_gap" "5" \
    "1.0" "$base_trend" "0.00" "0.00" "residual_temporal_conv" "0.10" "True" \
    "1.0" "$base_trend" "0.00" "0.00"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "doy_no_trend_renorm" "doy_gap" "5" \
    "$no_trend_intra" "$no_trend" "$no_trend_si" "$base_boundary" "residual_temporal_conv" "0.10" "True" \
    "$no_trend_intra" "$no_trend" "$no_trend_si" "$base_boundary"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "doy_no_segment_inter_renorm" "doy_gap" "5" \
    "$no_si_intra" "$no_si_trend" "$no_si" "0.00" "residual_temporal_conv" "0.10" "True" \
    "$no_si_intra" "$no_si_trend" "$no_si" "0.00"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "doy_no_boundary_mod" "doy_gap" "5" \
    "1.0" "$base_trend" "$base_si" "0.00" "residual_temporal_conv" "0.10" "True" \
    "1.0" "$base_trend" "$base_si" "0.00"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "full_doy_daoff" "doy_gap" "5" \
    "1.0" "$base_trend" "$base_si" "$base_boundary" "residual_temporal_conv" "0.10" "True" \
    "0.0" "0.00" "0.00" "0.00"

  add_job "$task" "$source_dataset" "$target_dataset" "$seed" "no_reshaper" "doy_gap" "5" \
    "0.0" "0.00" "0.00" "0.00" "none" "0.00" "False" \
    "0.0" "0.00" "0.00" "0.00"
}

for seed in $SEEDS; do
  add_task_jobs "AT1_to_DK1" "austria/33UVP/2017" "denmark/32VNH/2017" "$seed"
  add_task_jobs "FR1_to_AT1" "france/30TXT/2017" "austria/33UVP/2017" "$seed"
done

for idx in "${!GPU_IDS[@]}"; do
  : > "$LOG_DIR/queue_gpu${GPU_IDS[$idx]}.tsv"
done

job_index=0
while IFS= read -r line; do
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  printf "%s\n" "$line" >> "$LOG_DIR/queue_gpu${gpu}.tsv"
  job_index=$((job_index + 1))
done < "$JOBS"

run_worker() {
  local gpu="$1"
  local queue="$LOG_DIR/queue_gpu${gpu}.tsv"
  local worker_failed=0
  local task source_dataset target_dataset seed config partition phase_count intra trend segment_inter boundary
  local reshaper reshaper_strength dual_path da_intra da_trend da_segment_inter da_boundary

  while IFS=$'\t' read -r task source_dataset target_dataset seed config partition phase_count intra trend segment_inter boundary reshaper reshaper_strength dual_path da_intra da_trend da_segment_inter da_boundary; do
    local tag="v243b_mech_${task}_seed${seed}_${config}"
    local log_file="$LOG_DIR/gpu${gpu}_${task}_seed${seed}_${config}.log"
    echo "START|gpu=$gpu|task=$task|seed=$seed|config=$config|partition=$partition|log=$log_file"
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
      export SOURCE_STRUCTURE_TREND_TRADE_OFF="$trend"
      export SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$segment_inter"
      export SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$boundary"
      export TIMEMATCH_SOURCE_STRUCTURE_INTRA_TRADE_OFF="$da_intra"
      export TIMEMATCH_SOURCE_STRUCTURE_TREND_TRADE_OFF="$da_trend"
      export TIMEMATCH_SOURCE_STRUCTURE_SEGMENT_INTER_TRADE_OFF="$da_segment_inter"
      export TIMEMATCH_SOURCE_STRUCTURE_BOUNDARY_WINDOW_TRADE_OFF="$da_boundary"
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
python "$ROOT_DIR/analysis/summarize_v243b_mechanism.py" "$LOG_DIR" || true

exit "$failed"
