#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Controlled dose-response test:
# fixed source recipe, no reshaper, no DA-stage structure, full source compactness.
# This is a mechanism probe, not a selector or best-weight search.
export RUN_TAG="${RUN_TAG:-v243b_raw_compactness_dose_response_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_TAG}"
export TASKS="${TASKS:-FR2_to_FR1}"
export SEEDS="${SEEDS:-1 2 3 4 5}"
export RAW_WEIGHTS="${RAW_WEIGHTS:-0.25 0.5 0.75 1.0}"

export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export TIMEMATCH_STEPS_PER_EPOCH="${TIMEMATCH_STEPS_PER_EPOCH:-500}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GPUS="${GPUS:-0 1 2 3}"

echo "RUN_TAG=$RUN_TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "RAW_WEIGHTS=$RAW_WEIGHTS"
echo "SOURCE_PRETRAIN_EPOCHS=$SOURCE_PRETRAIN_EPOCHS"
echo "TIMEMATCH_EPOCHS=$TIMEMATCH_EPOCHS"
echo "TIMEMATCH_STEPS_PER_EPOCH=$TIMEMATCH_STEPS_PER_EPOCH"
echo "NOTE=Intensity-only raw compactness dose-response; timing fixed to full source training."

bash "$SCRIPT_DIR/launch_v243b_raw_strength_causal_intervention.sh"

python "$ROOT_DIR/analysis/analyze_v243b_dose_epoch_trajectory.py" "$LOG_DIR"

case "$(echo "${RUN_TARGET_READINESS_DIAG:-True}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    target_readiness_kind="${TARGET_READINESS_FEATURE_KIND:-final}"
    target_readiness_prefix="${TARGET_READINESS_OUTPUT_PREFIX:-target_readiness_${target_readiness_kind}}"
    python "$ROOT_DIR/analysis/v243b_target_readiness_diagnostic.py" "$LOG_DIR" \
      --data_root "$DATA_ROOT" \
      --device "$DEVICE" \
      --num_workers "$NUM_WORKERS" \
      --max_batches "${TARGET_READINESS_MAX_BATCHES:-64}" \
      --max_metric_samples "${TARGET_READINESS_MAX_METRIC_SAMPLES:-2048}" \
      --feature_kind "$target_readiness_kind" \
      --output_prefix "$target_readiness_prefix"
    ;;
esac

case "$(echo "${RUN_FROZEN_PROBE:-False}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    frozen_probe_kind="${FROZEN_PROBE_FEATURE_KIND:-final}"
    frozen_probe_prefix="${FROZEN_PROBE_OUTPUT_PREFIX:-frozen_probe_${frozen_probe_kind}}"
    python "$ROOT_DIR/analysis/v243b_frozen_encoder_transfer_probe.py" "$LOG_DIR" \
      --data_root "$DATA_ROOT" \
      --device "$DEVICE" \
      --num_workers "$NUM_WORKERS" \
      --max_batches "${FROZEN_PROBE_MAX_BATCHES:-64}" \
      --probe_epochs "${FROZEN_PROBE_EPOCHS:-300}" \
      --feature_kind "$frozen_probe_kind" \
      --output_prefix "$frozen_probe_prefix"
    ;;
esac
