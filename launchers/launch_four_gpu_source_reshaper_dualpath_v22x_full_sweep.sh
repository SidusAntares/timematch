#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/source_reshaper_dualpath_v22x_full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# Default v2.2.x hyperparameter groups.
# Format per line:
#   reshaper_kind|strength|kernel_size|reg_trade_off|dual_cls_trade_off|dual_relation_trade_off|tag
DEFAULT_CONFIGS=$(cat <<'EOF'
residual_temporal_conv|0.08|3|0.05|1.00|0.03|v221_s008_rel003
residual_temporal_conv|0.10|3|0.05|1.00|0.03|v222_s010_rel003
residual_temporal_conv|0.10|3|0.05|1.00|0.05|v223_s010_rel005
residual_temporal_conv|0.12|3|0.05|1.00|0.05|v224_s012_rel005
EOF
)

CONFIG_BLOCK="${CONFIG_BLOCK:-$DEFAULT_CONFIGS}"

mapfile -t CONFIGS < <(printf '%s\n' "$CONFIG_BLOCK" | sed '/^\s*$/d')
GPU_COUNT=4

run_config() {
  local gpu_id="$1"
  local cfg_line="$2"

  IFS='|' read -r reshaper_kind strength kernel_size reg_trade dual_cls_trade dual_rel_trade tag <<< "$cfg_line"
  local log_file="$LOG_DIR/gpu${gpu_id}_${tag}.log"

  echo "[GPU ${gpu_id}] start ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    SOURCE_FEATURE_RESHAPER="$reshaper_kind" \
    SOURCE_FEATURE_RESHAPER_STRENGTH="$strength" \
    SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$kernel_size" \
    SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$reg_trade" \
    SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="$dual_cls_trade" \
    SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$dual_rel_trade" \
    RESHAPER_TAG="$tag" \
    bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_alltasks.sh" \
    > "$log_file" 2>&1
  echo "[GPU ${gpu_id}] done ${tag}"
}

for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
  (
    idx=$gpu
    while [ "$idx" -lt "${#CONFIGS[@]}" ]; do
      run_config "$gpu" "${CONFIGS[$idx]}"
      idx=$((idx + GPU_COUNT))
    done
  ) &
done

wait
