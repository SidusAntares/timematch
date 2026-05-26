#!/bin/bash

set -uo pipefail

MASTER_STAMP="${MASTER_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_TAG="${MASTER_TAG:-two_version_full_${MASTER_STAMP}}"
MASTER_LOG_ROOT="${MASTER_LOG_ROOT:-/data/user/${MASTER_TAG}}"
MASTER_SUMMARY="${MASTER_SUMMARY:-${MASTER_LOG_ROOT}/summary.tsv}"

EVENT_ROOT="${EVENT_ROOT:-/data/user/timematch_event_support}"
GTW_ROOT="${GTW_ROOT:-/data/user/ti}"

EVENT_SCRIPT="${EVENT_SCRIPT:-launchers/gain_validation/run_v271_event_support_full_controller.sh}"
GTW_SCRIPT="${GTW_SCRIPT:-launchers/gain_validation/run_v272_gtw_full_controller.sh}"

DATASETS="${DATASETS:-REMOTE HAR HHAR_SA}"
GPUS="${GPUS:-0 1 2 3}"
TASK_FILTER="${TASK_FILTER:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

REMOTE_SOURCE_EPOCHS="${REMOTE_SOURCE_EPOCHS:-50}"
REMOTE_DA_EPOCHS="${REMOTE_DA_EPOCHS:-20}"
REMOTE_STEPS_PER_EPOCH="${REMOTE_STEPS_PER_EPOCH:-500}"
HAR_SOURCE_EPOCHS="${HAR_SOURCE_EPOCHS:-40}"
HAR_DA_EPOCHS="${HAR_DA_EPOCHS:-40}"
HAR_STEPS_PER_EPOCH="${HAR_STEPS_PER_EPOCH:-0}"

EVENT_MODES="${EVENT_MODES:-G_global G_event_support G_random_event}"
EVENT_SUPPORT_COUNT="${EVENT_SUPPORT_COUNT:-2}"
EVENT_SUPPORT_SIGMA_RATIO="${EVENT_SUPPORT_SIGMA_RATIO:-0.20}"
EVENT_SUPPORT_TRADE_OFF="${EVENT_SUPPORT_TRADE_OFF:-1.0}"

GTW_MODES="${GTW_MODES:-G_global G_gtw_r1 G_gtw_r2 G_gtw_r4}"
GTW_SHIFT_COUNT="${GTW_SHIFT_COUNT:-5}"
GTW_TEMPERATURE="${GTW_TEMPERATURE:-0.05}"

TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF:-0.0}"

mkdir -p "${MASTER_LOG_ROOT}"
printf "version\tstatus\texit_code\tcwd\trun_tag\tlog_file\tstarted_at\tfinished_at\n" > "${MASTER_SUMMARY}"

run_version() {
  local version_name="$1"
  local root_dir="$2"
  local script_path="$3"
  local modes="$4"
  local run_tag="$5"
  local log_file="${MASTER_LOG_ROOT}/${version_name}_${MASTER_STAMP}.out"
  local started_at finished_at status exit_code

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "========== START ${version_name} =========="
  echo "cwd=${root_dir}"
  echo "script=${script_path}"
  echo "run_tag=${run_tag}"
  echo "log=${log_file}"

  if [ ! -d "${root_dir}" ]; then
    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "ERROR: root dir does not exist: ${root_dir}" | tee "${log_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${version_name}" "missing_root" "127" "${root_dir}" "${run_tag}" "${log_file}" "${started_at}" "${finished_at}" \
      >> "${MASTER_SUMMARY}"
    return 0
  fi
  if [ ! -f "${root_dir}/${script_path}" ]; then
    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "ERROR: script does not exist: ${root_dir}/${script_path}" | tee "${log_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${version_name}" "missing_script" "127" "${root_dir}" "${run_tag}" "${log_file}" "${started_at}" "${finished_at}" \
      >> "${MASTER_SUMMARY}"
    return 0
  fi

  (
    cd "${root_dir}" || exit 127
    mkdir -p logs outputs runs
    env \
      RUN_TAG="${run_tag}" \
      DATASETS="${DATASETS}" \
      MODES="${modes}" \
      GPUS="${GPUS}" \
      TASK_FILTER="${TASK_FILTER}" \
      SKIP_EXISTING="${SKIP_EXISTING}" \
      REMOTE_SOURCE_EPOCHS="${REMOTE_SOURCE_EPOCHS}" \
      REMOTE_DA_EPOCHS="${REMOTE_DA_EPOCHS}" \
      REMOTE_STEPS_PER_EPOCH="${REMOTE_STEPS_PER_EPOCH}" \
      HAR_SOURCE_EPOCHS="${HAR_SOURCE_EPOCHS}" \
      HAR_DA_EPOCHS="${HAR_DA_EPOCHS}" \
      HAR_STEPS_PER_EPOCH="${HAR_STEPS_PER_EPOCH}" \
      EVENT_SUPPORT_COUNT="${EVENT_SUPPORT_COUNT}" \
      EVENT_SUPPORT_SIGMA_RATIO="${EVENT_SUPPORT_SIGMA_RATIO}" \
      EVENT_SUPPORT_TRADE_OFF="${EVENT_SUPPORT_TRADE_OFF}" \
      GTW_SHIFT_COUNT="${GTW_SHIFT_COUNT}" \
      GTW_TEMPERATURE="${GTW_TEMPERATURE}" \
      TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF="${TIMEMATCH_SOURCE_STRUCTURE_TRADE_OFF}" \
      bash "${script_path}"
  ) > "${log_file}" 2>&1
  exit_code=$?

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
  if [ "${exit_code}" -eq 0 ]; then
    status="ok"
  else
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${version_name}" "${status}" "${exit_code}" "${root_dir}" "${run_tag}" "${log_file}" "${started_at}" "${finished_at}" \
    >> "${MASTER_SUMMARY}"
  echo "========== END ${version_name}: ${status} exit=${exit_code} =========="

  return 0
}

echo "MASTER_TAG=${MASTER_TAG}"
echo "MASTER_LOG_ROOT=${MASTER_LOG_ROOT}"
echo "MASTER_SUMMARY=${MASTER_SUMMARY}"
echo "DATASETS=${DATASETS}"
echo "GPUS=${GPUS}"
echo "TASK_FILTER=${TASK_FILTER}"

run_version \
  "event_support" \
  "${EVENT_ROOT}" \
  "${EVENT_SCRIPT}" \
  "${EVENT_MODES}" \
  "event_support_full_${MASTER_STAMP}"

run_version \
  "gtw" \
  "${GTW_ROOT}" \
  "${GTW_SCRIPT}" \
  "${GTW_MODES}" \
  "gtw_full_${MASTER_STAMP}"

echo "All requested versions have been attempted."
echo "Master summary: ${MASTER_SUMMARY}"
