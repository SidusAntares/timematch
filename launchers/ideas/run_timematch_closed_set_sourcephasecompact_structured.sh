#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
SOURCE="${SOURCE:-denmark/32VNH/2017}"
TARGET="${TARGET:-france/30TXT/2017}"
SOURCE_TILE="$(echo "$SOURCE" | cut -d'/' -f2)"
TARGET_TILE="$(echo "$TARGET" | cut -d'/' -f2)"
STRUCTURE_TRANSFORM="${SOURCE_STRUCTURE_TRANSFORM:-none}"
STRUCTURE_STRENGTH="${SOURCE_STRUCTURE_STRENGTH:-0.0}"
STRUCTURE_PHASE_COUNT="${SOURCE_STRUCTURE_PHASE_COUNT:-5}"
STRUCTURE_TAG="${STRUCTURE_TAG:-${STRUCTURE_TRANSFORM}_s${STRUCTURE_STRENGTH}_p${STRUCTURE_PHASE_COUNT}}"

SOURCE_MODEL="${SOURCE_MODEL:-pseltae_${SOURCE_TILE}_closedset_noshift_sourcephasecompact_p5_${STRUCTURE_TAG}}"
TIMEMATCH_MODEL="${TIMEMATCH_MODEL:-timematch_${SOURCE_TILE}_to_${TARGET_TILE}_closedset_noshift_sourcephasecompact_p5_${STRUCTURE_TAG}}"

cd "$ROOT_DIR"

python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  --source_structure_transform "$STRUCTURE_TRANSFORM" \
  --source_structure_strength "$STRUCTURE_STRENGTH" \
  --source_structure_phase_count "$STRUCTURE_PHASE_COUNT" \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$SOURCE" \
  sourcephasecompact

python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  --source_structure_transform "$STRUCTURE_TRANSFORM" \
  --source_structure_strength "$STRUCTURE_STRENGTH" \
  --source_structure_phase_count "$STRUCTURE_PHASE_COUNT" \
  -e "$SOURCE_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  --eval

python train.py \
  --data_root "$DATA_ROOT" \
  --closed_set True \
  --with_shift_aug False \
  --source_structure_transform "$STRUCTURE_TRANSFORM" \
  --source_structure_strength "$STRUCTURE_STRENGTH" \
  --source_structure_phase_count "$STRUCTURE_PHASE_COUNT" \
  -e "$TIMEMATCH_MODEL" \
  --source "$SOURCE" \
  --target "$TARGET" \
  timematch \
  --weights "outputs/$SOURCE_MODEL"
