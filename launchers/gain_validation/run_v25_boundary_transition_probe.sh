#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export BATCH_STAMP
export RUN_TAG="${RUN_TAG:-v25_boundary_transition_probe_${BATCH_STAMP}}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export SOURCE_PRETRAIN_EPOCHS="${SOURCE_PRETRAIN_EPOCHS:-50}"
export TIMEMATCH_EPOCHS="${TIMEMATCH_EPOCHS:-20}"
export SOURCE_WEIGHTS_CHECKPOINTS="${SOURCE_WEIGHTS_CHECKPOINTS:-model.pt}"
export SOURCE_CHECKPOINT_EPOCHS="${SOURCE_CHECKPOINT_EPOCHS:-50}"

# Keep the same four theory tasks as the v2.5 follow-up probe so results can be
# compared directly against global / phase compactness views.
export TASK_SPECS="${TASK_SPECS:-FR2_to_DK1|france/31TCJ/2017|denmark/32VNH/2017,DK1_to_FR2|denmark/32VNH/2017|france/31TCJ/2017,AT1_to_DK1|austria/33UVP/2017|denmark/32VNH/2017,FR1_to_AT1|france/30TXT/2017|austria/33UVP/2017}"

# v2.5 boundary/transition probe.
#
# Current boundary_window is not an independent loss. It reweights the adjacent
# segment_inter transition term around phase boundaries. Therefore the clean
# validation is a factorial comparison:
#   1. intra only:              segment_inter=0,    boundary=0
#   2. transition only:         segment_inter=0.02, boundary=0
#   3. boundary-weighted trans: segment_inter=0.02, boundary=0.20
#   4. stronger boundary check: segment_inter=0.02, boundary=0.50
#
# Trend is kept at 0 in the main variants so local-transition effects are not
# confounded with the older smooth-trend regularizer.
export VARIANT_SPECS="${VARIANT_SPECS:-uniform_k5_intra_ref:uniform:segment_boundary_window_residual:5:5:1.0:0.00:0.00:0.00:0.00:meanmax:cosine:1.00:0.03,uniform_k5_transition:uniform:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.00:0.00:meanmax:cosine:1.00:0.03,uniform_k5_boundary02:uniform:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.20:0.00:meanmax:cosine:1.00:0.03,uniform_k5_boundary05:uniform:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.50:0.00:meanmax:cosine:1.00:0.03,doy_k5_transition:doy_gap:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.00:0.00:meanmax:cosine:1.00:0.03,doy_k5_boundary02:doy_gap:segment_boundary_window_residual:5:5:1.0:0.00:0.02:0.20:0.00:meanmax:cosine:1.00:0.03}"

bash "$SCRIPT_DIR/run_v25_theory_followup_probe.sh"
