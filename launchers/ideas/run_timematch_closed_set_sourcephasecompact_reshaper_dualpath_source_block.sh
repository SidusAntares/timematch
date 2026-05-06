#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:?SOURCE is required}"
TARGETS_BLOCK="${TARGETS_BLOCK:?TARGETS_BLOCK is required}"
SOURCE_TILE="$(echo "$SOURCE" | cut -d'/' -f2)"

RESHAPER_KIND="${SOURCE_FEATURE_RESHAPER:-residual_temporal_conv}"
RESHAPER_STRENGTH="${SOURCE_FEATURE_RESHAPER_STRENGTH:-0.10}"
RESHAPER_KERNEL_SIZE="${SOURCE_FEATURE_RESHAPER_KERNEL_SIZE:-3}"
RESHAPER_REG_TRADE_OFF="${SOURCE_FEATURE_RESHAPER_REG_TRADE_OFF:-0.05}"
DUAL_CLS_TRADE_OFF="${SOURCE_FEATURE_DUAL_CLS_TRADE_OFF:-1.00}"
DUAL_REL_TRADE_OFF="${SOURCE_FEATURE_DUAL_RELATION_TRADE_OFF:-0.03}"
RESHAPER_TAG="${RESHAPER_TAG:-v223_current_s010_rel003}"

SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift_sourcephasecompact_p5_${RESHAPER_TAG}}"

cd "$ROOT_DIR"

python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  --source_feature_reshaper "$RESHAPER_KIND" \
  --source_feature_reshaper_strength "$RESHAPER_STRENGTH" \
  --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE" \
  --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF" \
  --source_feature_dual_path True \
  --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF" \
  --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF" \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$SOURCE" \
  sourcephasecompact

while IFS= read -r TARGET; do
  if [ -z "$TARGET" ]; then
    continue
  fi

  TARGET_TILE="$(echo "$TARGET" | cut -d'/' -f2)"
  TIMEMATCH_MODEL="timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_sourcephasecompact_p5_${RESHAPER_TAG}"

  python train.py \
    --data_root "$DATA_ROOT" \
    --closed_set True \
    --with_shift_aug False \
    --source_feature_reshaper "$RESHAPER_KIND" \
    --source_feature_reshaper_strength "$RESHAPER_STRENGTH" \
    --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE" \
    --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF" \
    --source_feature_dual_path True \
    --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF" \
    --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF" \
    -e "$SOURCE_MODEL" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --eval

  python train.py \
    --data_root "$DATA_ROOT" \
    --closed_set True \
    --with_shift_aug False \
    --source_feature_reshaper "$RESHAPER_KIND" \
    --source_feature_reshaper_strength "$RESHAPER_STRENGTH" \
    --source_feature_reshaper_kernel_size "$RESHAPER_KERNEL_SIZE" \
    --source_feature_reshaper_reg_trade_off "$RESHAPER_REG_TRADE_OFF" \
    --source_feature_dual_path True \
    --source_feature_dual_cls_trade_off "$DUAL_CLS_TRADE_OFF" \
    --source_feature_dual_relation_trade_off "$DUAL_REL_TRADE_OFF" \
    -e "$TIMEMATCH_MODEL" \
    --source "$SOURCE" \
    --target "$TARGET" \
    timematch \
    --weights "outputs/$SOURCE_MODEL"
done <<< "$TARGETS_BLOCK"
