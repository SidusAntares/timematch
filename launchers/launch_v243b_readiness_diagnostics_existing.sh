#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DIAG_LOG_DIR="${DIAG_LOG_DIR:-}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_BATCHES="${MAX_BATCHES:-64}"
MAX_METRIC_SAMPLES="${MAX_METRIC_SAMPLES:-2048}"
FEATURE_KIND="${FEATURE_KIND:-final}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-target_readiness_${FEATURE_KIND}}"
RUN_FROZEN_PROBE="${RUN_FROZEN_PROBE:-True}"
RUN_TARGET_READINESS="${RUN_TARGET_READINESS:-True}"
RUN_EPOCH_TRAJECTORY="${RUN_EPOCH_TRAJECTORY:-True}"

if [ -z "$DIAG_LOG_DIR" ]; then
  echo "ERROR: DIAG_LOG_DIR is required, e.g. DIAG_LOG_DIR=logs/v243b_raw_compactness_dose_response_20260612_164026" >&2
  exit 2
fi

if [ ! -d "$DIAG_LOG_DIR" ]; then
  echo "ERROR: DIAG_LOG_DIR does not exist: $DIAG_LOG_DIR" >&2
  exit 2
fi

if [ ! -f "$DIAG_LOG_DIR/jobs.tsv" ]; then
  echo "ERROR: jobs.tsv does not exist under DIAG_LOG_DIR: $DIAG_LOG_DIR/jobs.tsv" >&2
  exit 2
fi

failed=0

run_bool() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

echo "DIAG_LOG_DIR=$DIAG_LOG_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "DEVICE=$DEVICE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "MAX_BATCHES=$MAX_BATCHES"
echo "MAX_METRIC_SAMPLES=$MAX_METRIC_SAMPLES"
echo "FEATURE_KIND=$FEATURE_KIND"
echo "OUTPUT_PREFIX=$OUTPUT_PREFIX"

if run_bool "$RUN_EPOCH_TRAJECTORY"; then
  echo "STEP=epoch_trajectory"
  python "$ROOT_DIR/analysis/analyze_v243b_dose_epoch_trajectory.py" "$DIAG_LOG_DIR" || failed=1
fi

if run_bool "$RUN_TARGET_READINESS"; then
  echo "STEP=target_readiness"
  python "$ROOT_DIR/analysis/v243b_target_readiness_diagnostic.py" "$DIAG_LOG_DIR" \
    --data_root "$DATA_ROOT" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS" \
    --max_batches "$MAX_BATCHES" \
    --max_metric_samples "$MAX_METRIC_SAMPLES" \
    --feature_kind "$FEATURE_KIND" \
    --output_prefix "$OUTPUT_PREFIX" || failed=1
fi

if run_bool "$RUN_FROZEN_PROBE"; then
  echo "STEP=frozen_probe"
  python "$ROOT_DIR/analysis/v243b_frozen_encoder_transfer_probe.py" "$DIAG_LOG_DIR" \
    --data_root "$DATA_ROOT" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS" \
    --max_batches "$MAX_BATCHES" \
    --probe_epochs "${FROZEN_PROBE_EPOCHS:-300}" \
    --feature_kind "$FEATURE_KIND" \
    --output_prefix "frozen_probe_${FEATURE_KIND}" || failed=1
fi

echo "Diagnostics saved under: $DIAG_LOG_DIR"
exit "$failed"
