#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDEA_DIR="$SCRIPT_DIR/ideas"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/source_reshaper_dualpath_v23pre_minival_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# Default v2.3 pre-version configs.
# Format per line:
#   reshaper_kind|strength|kernel_size|reg_trade_off|dual_cls_trade_off|dual_relation_trade_off|domain_phase_weights|domain_phase_blend_alpha|tag
DEFAULT_CONFIGS=$(cat <<'EOF'
adaptive_residual_temporal_conv|0.10|3|0.05|1.00|0.03|False|0.00|v23pre1_adaptive_s010_rel003_a000
adaptive_residual_temporal_conv|0.10|3|0.05|1.00|0.03|True|0.20|v23pre2_adaptive_s010_rel003_a020
adaptive_residual_temporal_conv|0.10|3|0.05|1.00|0.03|True|0.35|v23pre3_adaptive_s010_rel003_a035
adaptive_residual_temporal_conv|0.10|3|0.05|1.00|0.02|True|0.20|v23pre4_adaptive_s010_rel002_a020
EOF
)

CONFIG_BLOCK="${CONFIG_BLOCK:-$DEFAULT_CONFIGS}"

mapfile -t CONFIGS < <(printf '%s\n' "$CONFIG_BLOCK" | sed '/^\s*$/d')
GPU_COUNT=4

run_config() {
  local gpu_id="$1"
  local cfg_line="$2"

  IFS='|' read -r reshaper_kind strength kernel_size reg_trade dual_cls_trade dual_rel_trade domain_phase_weights domain_phase_blend_alpha tag <<< "$cfg_line"
  local log_file="$LOG_DIR/gpu${gpu_id}_${tag}.log"

  echo "[GPU ${gpu_id}] start ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    SOURCE_FEATURE_RESHAPER="$reshaper_kind" \
    SOURCE_FEATURE_RESHAPER_STRENGTH="$strength" \
    SOURCE_FEATURE_RESHAPER_KERNEL_SIZE="$kernel_size" \
    SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF="$reg_trade" \
    SOURCE_FEATURE_DUAL_CLS_TRADE_OFF="$dual_cls_trade" \
    SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF="$dual_rel_trade" \
    SOURCE_DOMAIN_ADAPTIVE_PHASE_WEIGHTS="$domain_phase_weights" \
    SOURCE_DOMAIN_PHASE_BLEND_ALPHA="$domain_phase_blend_alpha" \
    RESHAPER_TAG="$tag" \
    bash "$IDEA_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_minival_alltasks.sh" \
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
