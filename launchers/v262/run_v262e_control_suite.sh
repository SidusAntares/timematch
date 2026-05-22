#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="${RUN_GROUP:-v262e_control_suite_${STAMP}}"

GROUP_LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_GROUP}}"
GROUP_OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_GROUP}}"
GROUP_RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_GROUP}}"
mkdir -p "${GROUP_LOG_ROOT}" "${GROUP_OUT_ROOT}" "${GROUP_RUN_ROOT}"

TASK_FILTER="${TASK_FILTER:-DK1_to_FR1,FR2_to_DK1}"
GPUS="${GPUS:-0 1 2 3}"
GPU_GROUP_A="${GPU_GROUP_A:-0 1}"
GPU_GROUP_B="${GPU_GROUP_B:-2 3}"
GPU_GROUP_C="${GPU_GROUP_C:-0 1 2 3}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
DA_EPOCHS="${DA_EPOCHS:-20}"

STATIC_MASK_WARMUP_EPOCHS="${STATIC_MASK_WARMUP_EPOCHS:-3}"
STATIC_MASK_MAX_BATCHES="${STATIC_MASK_MAX_BATCHES:-64}"
TEMPORAL_SUPPORT_SMOOTH_KERNEL="${TEMPORAL_SUPPORT_SMOOTH_KERNEL:-5}"
TEMPORAL_MASK_RELIABILITY_GATE="${TEMPORAL_MASK_RELIABILITY_GATE:-true}"
TEMPORAL_MASK_RELIABILITY_LOW="${TEMPORAL_MASK_RELIABILITY_LOW:-0.0005}"
TEMPORAL_MASK_RELIABILITY_HIGH="${TEMPORAL_MASK_RELIABILITY_HIGH:-0.04}"

REMOTE_MIN_SAMPLE_POINTS="${REMOTE_MIN_SAMPLE_POINTS:-1}"
ADAPT_MIN_FACTOR="${ADAPT_MIN_FACTOR:-1.00}"
ADAPT_MAX_FACTOR="${ADAPT_MAX_FACTOR:-1.00}"

SUITE_INDEX="${GROUP_LOG_ROOT}/suite_index.tsv"
echo -e "subrun\twindow\tpartition\tphase_count\tlog_root\tout_root\trun_root" > "${SUITE_INDEX}"

run_subexperiment() {
  local subrun="$1"
  local windows="$2"
  local partition_mode="$3"
  local phase_count="$4"
  local min_weight="$5"
  local gpus="$6"

  local sub_log_root="${GROUP_LOG_ROOT}/${subrun}"
  local sub_out_root="${GROUP_OUT_ROOT}/${subrun}"
  local sub_run_root="${GROUP_RUN_ROOT}/${subrun}"
  mkdir -p "${sub_log_root}" "${sub_out_root}" "${sub_run_root}"

  echo -e "${subrun}\t${windows}\t${partition_mode}\t${phase_count}\t${sub_log_root}\t${sub_out_root}\t${sub_run_root}" >> "${SUITE_INDEX}"

  echo "============================================================"
  echo "[v262e control suite] subrun=${subrun}"
  echo "  windows=${windows}"
  echo "  partition=${partition_mode}, phase_count=${phase_count}"
  echo "  task_filter=${TASK_FILTER}, gpus=${gpus}"
  echo "  logs=${sub_log_root}"
  echo "  outputs=${sub_out_root}"
  echo "============================================================"

  env \
    STAMP="${STAMP}" \
    RUN_TAG="${RUN_GROUP}_${subrun}" \
    LOG_ROOT="${sub_log_root}" \
    OUT_ROOT="${sub_out_root}" \
    RUN_ROOT="${sub_run_root}" \
    TASK_FILTER="${TASK_FILTER}" \
    WINDOWS="${windows}" \
    GPUS="${gpus}" \
    SOURCE_EPOCHS="${SOURCE_EPOCHS}" \
    DA_EPOCHS="${DA_EPOCHS}" \
    ADAPT_MIN_FACTOR="${ADAPT_MIN_FACTOR}" \
    ADAPT_MAX_FACTOR="${ADAPT_MAX_FACTOR}" \
    WINDOW_MIN_WEIGHT="${min_weight}" \
    STATIC_MASK_WARMUP_EPOCHS="${STATIC_MASK_WARMUP_EPOCHS}" \
    STATIC_MASK_MAX_BATCHES="${STATIC_MASK_MAX_BATCHES}" \
    TEMPORAL_SUPPORT_SMOOTH_KERNEL="${TEMPORAL_SUPPORT_SMOOTH_KERNEL}" \
    TEMPORAL_MASK_RELIABILITY_GATE="${TEMPORAL_MASK_RELIABILITY_GATE}" \
    TEMPORAL_MASK_RELIABILITY_LOW="${TEMPORAL_MASK_RELIABILITY_LOW}" \
    TEMPORAL_MASK_RELIABILITY_HIGH="${TEMPORAL_MASK_RELIABILITY_HIGH}" \
    REMOTE_PARTITION_MODE="${partition_mode}" \
    REMOTE_PHASE_COUNT="${phase_count}" \
    REMOTE_MIN_SAMPLE_POINTS="${REMOTE_MIN_SAMPLE_POINTS}" \
    bash "${SCRIPT_DIR}/run_v262e_soft_support_probe.sh" \
    > "${sub_log_root}/nohup.out" 2>&1
}

echo "v2.6.2e control suite"
echo "RUN_GROUP=${RUN_GROUP}"
echo "GROUP_LOG_ROOT=${GROUP_LOG_ROOT}"
echo "GROUP_OUT_ROOT=${GROUP_OUT_ROOT}"
echo "TASK_FILTER=${TASK_FILTER}"
echo "SOURCE_EPOCHS=${SOURCE_EPOCHS} DA_EPOCHS=${DA_EPOCHS}"
echo "ADAPT_MIN_FACTOR=${ADAPT_MIN_FACTOR} ADAPT_MAX_FACTOR=${ADAPT_MAX_FACTOR}"
echo "GPU_GROUP_A=${GPU_GROUP_A} GPU_GROUP_B=${GPU_GROUP_B} GPU_GROUP_C=${GPU_GROUP_C}"

run_subexperiment "global_full" "full" "uniform" "1" "1.00" "${GPU_GROUP_A}" &
pid_global_full=$!
run_subexperiment "global_support" "source_target_soft_support" "uniform" "1" "0.20" "${GPU_GROUP_B}" &
pid_global_support=$!
wait "${pid_global_full}" "${pid_global_support}"

run_subexperiment "segmented_support" "source_target_soft_support" "uniform" "5" "0.20" "${GPU_GROUP_C}"

echo "v2.6.2e control suite finished."
echo "Suite index: ${SUITE_INDEX}"
echo "Logs: ${GROUP_LOG_ROOT}"
echo "Outputs: ${GROUP_OUT_ROOT}"
