#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:-denmark/32VNH/2017}"
TARGET="${TARGET:-france/30TXT/2017}"
SOURCE_TILE="$(echo "$SOURCE" | cut -d'/' -f2)"
TARGET_TILE="$(echo "$TARGET" | cut -d'/' -f2)"
SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift}"
TIMEMATCH_MODEL="${TIMEMATCH_MODEL:-timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_tgtphasecompact_p5}"
TARGET_STRUCT_TRADE_OFF="${TARGET_STRUCT_TRADE_OFF:-0.05}"
TARGET_STRUCT_WARMUP_EPOCHS="${TARGET_STRUCT_WARMUP_EPOCHS:-2}"

cd "$ROOT_DIR"

# Source-only baseline
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$SOURCE"

# Baseline source model direct transfer result
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  --eval

# TimeMatch + target-domain phase compactness on high-confidence pseudo labels
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$TIMEMATCH_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  timematchtgtphasecompact \
  --weights "outputs/$SOURCE_MODEL" \
  --target_struct_trade_off "$TARGET_STRUCT_TRADE_OFF" \
  --target_struct_warmup_epochs "$TARGET_STRUCT_WARMUP_EPOCHS"
