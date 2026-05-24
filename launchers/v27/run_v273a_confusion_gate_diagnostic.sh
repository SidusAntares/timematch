#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v273a_confusion_gate_diagnostic_${STAMP}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${ROOT_DIR}/outputs/v271_basis_view_probe_20260522_162224}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
SOURCE_MAX_BATCHES="${SOURCE_MAX_BATCHES:-64}"
TARGET_MAX_BATCHES="${TARGET_MAX_BATCHES:-64}"
SHIFT_SAMPLE_SIZE="${SHIFT_SAMPLE_SIZE:-40}"
ATOMIC_BINS="${ATOMIC_BINS:-12}"
MAX_MARGIN="${MAX_MARGIN:-0.20}"
MIN_TOP2_MASS="${MIN_TOP2_MASS:-0.35}"
BASELINE_PAIRS_PER_SAMPLE="${BASELINE_PAIRS_PER_SAMPLE:-4}"
MIN_SEGMENT_RATIO="${MIN_SEGMENT_RATIO:-1.0}"
MIN_SEGMENT_SCORE="${MIN_SEGMENT_SCORE:-0.0}"
GATE_SCORE_LOW="${GATE_SCORE_LOW:-0.02}"
GATE_SCORE_HIGH="${GATE_SCORE_HIGH:-0.10}"
GATE_RATIO_LOW="${GATE_RATIO_LOW:-1.00}"
GATE_RATIO_HIGH="${GATE_RATIO_HIGH:-1.20}"
GATE_ACCEPT_LOW="${GATE_ACCEPT_LOW:-0.01}"
GATE_ACCEPT_HIGH="${GATE_ACCEPT_HIGH:-0.08}"
GATE_ACCEPT_TOO_HIGH="${GATE_ACCEPT_TOO_HIGH:-0.16}"
GATE_OVERLAP_HIGH="${GATE_OVERLAP_HIGH:-0.50}"
GATE_POST_PRE_LOW="${GATE_POST_PRE_LOW:-0.15}"
GATE_POST_PRE_HIGH="${GATE_POST_PRE_HIGH:-0.70}"
TASK_FILTER="${TASK_FILTER:-all}"

TASKS=(
  "remote DK1 FR1 remote_DK1_to_FR1_soft_bank_v271_source_20260522_162227"
  "remote FR2 DK1 remote_FR2_to_DK1_dynamics_v271_source_20260522_162227"
  "HHAR_SA 1 6 hhar_sa_1_to_6_soft_bank_v271_source_20260522_162227"
)

should_run_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local filter="${TASK_FILTER}"
  if [ "${filter}" = "all" ] || [ -z "${filter}" ]; then
    return 0
  fi
  local task_key="${src}->${tgt}"
  local task_key_alt="${src}_to_${tgt}"
  local dataset_task_key="${dataset}:${src}->${tgt}"
  local dataset_task_key_alt="${dataset}:${src}_to_${tgt}"
  IFS=',' read -ra filter_items <<< "${filter}"
  for item in "${filter_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ "${item}" = "${task_key}" ] || [ "${item}" = "${task_key_alt}" ]; then
      return 0
    fi
    if [ "${item}" = "${dataset_task_key}" ] || [ "${item}" = "${dataset_task_key_alt}" ]; then
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if [ "${running}" -lt "${#GPUS[@]}" ]; then
      return 0
    fi
    sleep 10
  done
}

run_task() {
  local dataset="$1"
  local src="$2"
  local tgt="$3"
  local run_name="$4"
  local gpu="$5"
  local key="${dataset}_${src}_to_${tgt}"
  key="${key//\//_}"
  local run_dir="${SOURCE_RUN_ROOT}/${run_name}"
  local out_dir="${OUT_ROOT}/${key}"
  local log_file="${LOG_ROOT}/${key}.log"

  echo "[v2.7.3a] ${dataset} ${src}->${tgt} run_dir=${run_dir} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python analysis/v273_confusion_gate_diagnostic.py \
    --run_dir "${run_dir}" \
    --output_dir "${out_dir}" \
    --device "${DEVICE_PREFIX}:0" \
    --source_max_batches "${SOURCE_MAX_BATCHES}" \
    --target_max_batches "${TARGET_MAX_BATCHES}" \
    --shift_sample_size "${SHIFT_SAMPLE_SIZE}" \
    --atomic_bins "${ATOMIC_BINS}" \
    --max_margin "${MAX_MARGIN}" \
    --min_top2_mass "${MIN_TOP2_MASS}" \
    --baseline_pairs_per_sample "${BASELINE_PAIRS_PER_SAMPLE}" \
    --min_segment_score "${MIN_SEGMENT_SCORE}" \
    --min_segment_ratio "${MIN_SEGMENT_RATIO}" \
    --gate_score_low "${GATE_SCORE_LOW}" \
    --gate_score_high "${GATE_SCORE_HIGH}" \
    --gate_ratio_low "${GATE_RATIO_LOW}" \
    --gate_ratio_high "${GATE_RATIO_HIGH}" \
    --gate_accept_low "${GATE_ACCEPT_LOW}" \
    --gate_accept_high "${GATE_ACCEPT_HIGH}" \
    --gate_accept_too_high "${GATE_ACCEPT_TOO_HIGH}" \
    --gate_overlap_high "${GATE_OVERLAP_HIGH}" \
    --gate_post_pre_low "${GATE_POST_PRE_LOW}" \
    --gate_post_pre_high "${GATE_POST_PRE_HIGH}" \
    > "${log_file}" 2>&1

  echo -e "${dataset}\t${src}->${tgt}\t${run_dir}\t${out_dir}\t${log_file}" >> "${LOG_ROOT}/task_logs.tsv"
}

echo -e "dataset\ttask\trun_dir\toutput_dir\tlog_file" > "${LOG_ROOT}/task_logs.tsv"
echo "v2.7.3a confusion gate diagnostic"
echo "SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT}"
echo "Logs: ${LOG_ROOT}"
echo "Outputs: ${OUT_ROOT}"
echo "TASK_FILTER=${TASK_FILTER} GPUS=${GPUS[*]}"

job_idx=0
for task in "${TASKS[@]}"; do
  read -r dataset src tgt run_name <<< "${task}"
  if ! should_run_task "${dataset}" "${src}" "${tgt}"; then
    continue
  fi
  wait_for_slot
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  run_task "${dataset}" "${src}" "${tgt}" "${run_name}" "${gpu}" &
  job_idx=$((job_idx + 1))
done

wait
echo "v2.7.3a diagnostic finished."
echo "Task log index: ${LOG_ROOT}/task_logs.tsv"
