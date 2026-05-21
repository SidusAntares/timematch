#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-quick}"
GPUS=(${GPUS:-0 1 2 3})
FEATURE_MODE="${FEATURE_MODE:-raw_input}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
CHECKPOINT_MATCH_TARGET="${CHECKPOINT_MATCH_TARGET:-false}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/user/DBL/timematch_data}"
HAR_DATA_ROOT="${HAR_DATA_ROOT:-/data/user/dataset/UCIHAR/HAR}"
HHAR_DATA_ROOT="${HHAR_DATA_ROOT:-/data/user/dataset/HHAR/HHAR_SA}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/result/_summary/v260_structure_reliability_audit_${STAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/v260_structure_reliability_audit_${STAMP}}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-2048}"
MAX_TARGET_SAMPLES="${MAX_TARGET_SAMPLES:-2048}"

CHECKPOINT_ARGS=()
if [ -n "${CHECKPOINT_ROOT}" ]; then
  CHECKPOINT_ARGS+=(--checkpoint_root "${CHECKPOINT_ROOT}")
fi
if [ "${CHECKPOINT_MATCH_TARGET}" = "true" ]; then
  CHECKPOINT_ARGS+=(--checkpoint_match_target)
fi

REMOTE_TASKS=(
  "remote FR1 FR2"
  "remote FR1 DK1"
  "remote FR1 AT1"
  "remote FR2 FR1"
  "remote FR2 DK1"
  "remote FR2 AT1"
  "remote DK1 FR1"
  "remote DK1 FR2"
  "remote DK1 AT1"
  "remote AT1 FR1"
  "remote AT1 FR2"
  "remote AT1 DK1"
)

HAR_TASKS=(
  "HAR 12 16"
  "HAR 2 11"
  "HAR 6 23"
  "HAR 7 13"
  "HAR 9 18"
)

HHAR_TASKS=(
  "HHAR_SA 0 6"
  "HHAR_SA 1 6"
  "HHAR_SA 2 7"
  "HHAR_SA 3 8"
  "HHAR_SA 4 5"
)

case "${MODE}" in
  quick)
    TASKS=(
      "remote FR1 FR2"
      "remote FR2 FR1"
      "remote DK1 FR1"
      "remote FR2 DK1"
      "HAR 6 23"
      "HHAR_SA 2 7"
    )
    ;;
  remote12)
    TASKS=("${REMOTE_TASKS[@]}")
    ;;
  har_hhar)
    TASKS=("${HAR_TASKS[@]}" "${HHAR_TASKS[@]}")
    ;;
  full)
    TASKS=("${REMOTE_TASKS[@]}" "${HAR_TASKS[@]}" "${HHAR_TASKS[@]}")
    ;;
  feature_failed_20260521)
    TASKS=(
      "HAR 2 11"
      "remote AT1 DK1"
      "remote AT1 FR1"
      "remote AT1 FR2"
    )
    ;;
  har_2_11)
    TASKS=(
      "HAR 2 11"
    )
    ;;
  *)
    echo "Unsupported MODE=${MODE}; expected quick, remote12, har_hhar, full, feature_failed_20260521, har_2_11" >&2
    exit 1
    ;;
esac

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "${running}" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 15
  done
}

job_idx=0
for task in "${TASKS[@]}"; do
  read -r dataset source target <<< "${task}"
  wait_for_slot
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  tag="${dataset}_${source}_to_${target}"
  task_out="${OUT_DIR}/${tag}"
  mkdir -p "${task_out}"
  echo "[v2.6.0] ${dataset} ${source}->${target} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python analysis/v260_structure_reliability_audit.py \
    --dataset_name "${dataset}" \
    --source "${source}" \
    --target "${target}" \
    --feature_mode "${FEATURE_MODE}" \
    "${CHECKPOINT_ARGS[@]}" \
    --remote_data_root "${REMOTE_DATA_ROOT}" \
    --har_data_root "${HAR_DATA_ROOT}" \
    --hhar_data_root "${HHAR_DATA_ROOT}" \
    --output_dir "${task_out}" \
    --output_stem "${tag}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_source_samples "${MAX_SOURCE_SAMPLES}" \
    --max_target_samples "${MAX_TARGET_SAMPLES}" \
    > "${LOG_DIR}/${tag}.log" 2>&1 &
  job_idx=$((job_idx + 1))
done

set +e
wait
STATUS=$?
set -e

python analysis/v260_structure_reliability_audit.py \
  --collect_dir "${OUT_DIR}" \
  --output_dir "${OUT_DIR}" \
  --output_stem "v260_structure_reliability_audit_${MODE}"

echo "v2.6.0 structure reliability audit finished."
echo "Logs: ${LOG_DIR}"
echo "Summary: ${OUT_DIR}/v260_structure_reliability_audit_${MODE}.md"
exit "${STATUS}"
