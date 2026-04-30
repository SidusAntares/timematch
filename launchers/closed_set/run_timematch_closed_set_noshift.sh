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
TIMEMATCH_MODEL="${TIMEMATCH_MODEL:-timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift}"

cd "$ROOT_DIR"

# Source-only
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$SOURCE"

# Baseline result
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  --eval

# TimeMatch
python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  -e "$TIMEMATCH_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  timematch \
  --weights "outputs/$SOURCE_MODEL"
