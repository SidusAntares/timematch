#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# v2.5 theory validation:
# run a fixed structure-view bank instead of stopping at the first gain.
#
# Default is one seed for a quick check. For a stronger run:
#   SEEDS="1 2 3" bash launchers/analysis/run_har_hhar_v25_multiview_validation_4gpu.sh

SEEDS="${SEEDS:-1}"
DATASETS="${DATASETS:-HAR HHAR_SA}"
GPUS="${GPUS:-0 1 2 3}"

VARIANTS="${VARIANTS:-compact_k5 compact_k3 compact_k8 intra_light_k5 noseg_global dynamics_cosine dynamics_mse}"
STOP_ON_GAIN="${STOP_ON_GAIN:-false}"

SOURCE_EPOCHS="${SOURCE_EPOCHS:-40}"
DA_EPOCHS="${DA_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"

STAMP_PREFIX="${STAMP_PREFIX:-v25_har_hhar_multiview_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT_DIR}/result/_summary/${STAMP_PREFIX}}"
mkdir -p "${SUMMARY_ROOT}"

ROOTS=()
for seed in ${SEEDS}; do
  STAMP="${STAMP_PREFIX}_seed${seed}"
  LOG_ROOT="${ROOT_DIR}/logs/${STAMP}"
  OUT_ROOT="${ROOT_DIR}/outputs/${STAMP}"
  RUN_ROOT="${ROOT_DIR}/runs/${STAMP}"

  echo "=== v2.5 HAR/HHAR multi-view validation: seed=${seed} ==="
  echo "Logs: ${LOG_ROOT}"
  echo "Variants: ${VARIANTS}"

  SEED="${seed}" \
  STAMP="${STAMP}" \
  LOG_ROOT="${LOG_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  DATASETS="${DATASETS}" \
  GPUS="${GPUS}" \
  VARIANTS="${VARIANTS}" \
  STOP_ON_GAIN="${STOP_ON_GAIN}" \
  SOURCE_EPOCHS="${SOURCE_EPOCHS}" \
  DA_EPOCHS="${DA_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  bash launchers/analysis/run_har_hhar_structure_overnight_controller.sh

  ROOTS+=("${LOG_ROOT}")
done

python analysis/summarize_har_hhar_multiview.py "${ROOTS[@]}" --output_dir "${SUMMARY_ROOT}"

echo "v2.5 HAR/HHAR multi-view validation finished."
echo "Summary: ${SUMMARY_ROOT}"
