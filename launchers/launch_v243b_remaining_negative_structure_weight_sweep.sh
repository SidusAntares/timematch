#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG="${RUN_TAG:-v243b_remaining_negative_structure_weight_sweep_${BATCH_STAMP}}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/${RUN_TAG}}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"

# Remaining negative tasks after combining existing v2.4.3b and the first sweep.
export TASK_SPECS="${TASK_SPECS:-FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017,DK1_to_FR1|denmark/32VNH/2017|france/30TXT/2017,DK1_to_AT1|denmark/32VNH/2017|austria/33UVP/2017}"

# Direction: reduce structural pressure first. These tasks are currently harmed by
# the v2.4.3b default trend/segment/boundary setting.
export VARIANT_SPECS="${VARIANT_SPECS:-base:0.05:0.02:0.20,very_light:0.01:0.005:0.05,no_trend_weak_boundary:0.00:0.01:0.05,no_inter_boundary_only:0.03:0.00:0.10,no_boundary_transition:0.03:0.01:0.00,trend_only_light:0.02:0.00:0.00,boundary_mid_inter_off:0.00:0.00:0.15,trend_mid_boundary_off:0.05:0.005:0.00}"

bash "$ROOT_DIR/launchers/launch_v243b_negative_task_structure_weight_sweep.sh"
