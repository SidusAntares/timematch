#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/v23_full_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ti}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[INFO] Using python: $(which python)"
python -c "import sys; print('[INFO] Python executable:', sys.executable)"

SOURCE_FEATURE_RESHAPER="${SOURCE_FEATURE_RESHAPER:-componentized_residual_temporal_conv}"
SOURCE_FEATURE_RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"
SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS="${SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS:-False}"
SOURCE_DOMAIN_PHASE_BLEND_ALPHA="${SOURCE_DOMAIN_PHASE_BLEND_ALPHA:-0.00}"
RESHAPER_TAG="${RESHAPER_TAG:-v23_componentized_s010_rel003}"

TASKS_GPU0=$'france/30TXT/2017 france/31TCJ/2017\nfrance/30TXT/2017 denmark/32VNH/2017\nfrance/30TXT/2017 austria/33UVP/2017'
TASKS_GPU1=$'france/31TCJ/2017 france/30TXT/2017\nfrance/31TCJ/2017 denmark/32VNH/2017\nfrance/31TCJ/2017 austria/33UVP/2017'
TASKS_GPU2=$'denmark/32VNH/2017 france/30TXT/2017\ndenmark/32VNH/2017 france/31TCJ/2017\ndenmark/32VNH/2017 austria/33UVP/2017'
TASKS_GPU3=$'austria/33UVP/2017 france/30TXT/2017\naustria/33UVP/2017 france/31TCJ/2017\naustria/33UVP/2017 denmark/32VNH/2017'

run_task_group() {
  local gpu_id="$1"
  local tasks_block="$2"
  local log_file="$3"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
    SOURCE_FEATURE_RESHAPER="$SOURCE_FEATURE_RESHAPER" \
    SOURCE_FEATURE_RESHAPER_STRENGTH="$SOURCE_FEATURE_RESHAPER_STRENGTH" \
    SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$SOURCE_FEATURE_RESHAPER_KERNEL_SIZE" \
    SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF" \
    SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="$SOURCE_FEATURE_DUAL_CLS_TRADE_OFF" \
    SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF" \
    SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS="$SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS" \
    SOURCE_DOMAIN_PHASE_BLEND_ALPHA="$SOURCE_DOMAIN_PHASE_BLEND_ALPHA" \
    RESHAPER_TAG="$RESHAPER_TAG" \
    TASKS_BLOCK="$tasks_block" \
    bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_tasklist.sh" \
    > "$log_file" 2>&1
}

run_task_group 0 "$TASKS_GPU0" "$LOG_DIR/gpu0_group_30TXT.log" &
run_task_group 1 "$TASKS_GPU1" "$LOG_DIR/gpu1_group_31TCJ.log" &
run_task_group 2 "$TASKS_GPU2" "$LOG_DIR/gpu2_group_32VNH.log" &
run_task_group 3 "$TASKS_GPU3" "$LOG_DIR/gpu3_group_33UVP.log" &

wait

echo "[SUCCESS] v2.3 componentized full run finished. Logs in: $LOG_DIR"
