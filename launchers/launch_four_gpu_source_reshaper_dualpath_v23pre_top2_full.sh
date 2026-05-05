#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/v23pre_full_top2_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# If the caller already activated the right environment, keep it.
# Otherwise try to activate the requested conda env in a conservative way.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ti}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[INFO] Using python: $(which python)"
python -c "import sys; print('[INFO] Python executable:', sys.executable)"

run_partition() {
  local gpu_id="$1"
  local partition="$2"
  local blend_alpha="$3"
  local tag="$4"
  local log_file="$5"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
    SOURCE_FEATURE_RESHAPER=adaptive_residual_temporal_conv \
    SOURCE_FEATURE_RESHAPER_STRENGTH=0.10 \
    SOURCE_FEATURE_RESHAPER_KERNEL_SIZE=3 \
    SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF=0.05 \
    SOURCE_FEATURE_DUAL_CLS_TRADE_OFF=1.00 \
    SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF=0.03 \
    SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS=True \
    SOURCE_DOMAIN_PHASE_BLEND_ALPHA="$blend_alpha" \
    RESHAPER_TAG="$tag" \
    PARTITION="$partition" \
    bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_partition.sh" \
    > "$log_file" 2>&1
}

run_partition 0 A 0.20 v23pre2_adaptive_s010_rel003_a020 "$LOG_DIR/gpu0_v23pre2_partition_A.log" &
run_partition 1 B 0.20 v23pre2_adaptive_s010_rel003_a020 "$LOG_DIR/gpu1_v23pre2_partition_B.log" &
run_partition 2 A 0.35 v23pre3_adaptive_s010_rel003_a035 "$LOG_DIR/gpu2_v23pre3_partition_A.log" &
run_partition 3 B 0.35 v23pre3_adaptive_s010_rel003_a035 "$LOG_DIR/gpu3_v23pre3_partition_B.log" &

wait

echo "[SUCCESS] v23pre top2 full run finished. Logs in: $LOG_DIR"
