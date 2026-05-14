#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${RUN_TAG:-v255a_source_descriptor_audit}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/result/_summary}"

mkdir -p "$OUT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/${RUN_TAG}.json}"
OUTPUT_MD="${OUTPUT_MD:-$OUT_DIR/${RUN_TAG}.md}"
MAX_SAMPLES_PER_CLASS="${MAX_SAMPLES_PER_CLASS:-128}"

cd "$ROOT_DIR"

python analysis/source_structure_descriptor_audit.py \
  --data_root "$DATA_ROOT" \
  --sources \
    france/30TXT/2017 \
    france/31TCJ/2017 \
    denmark/32VNH/2017 \
    austria/33UVP/2017 \
  --closed_set True \
  --max_samples_per_class "$MAX_SAMPLES_PER_CLASS" \
  --source_segment_partition_mode doy_gap \
  --source_segment_count 5 \
  --source_phase_gap_threshold 45 \
  --source_phase_min_points 3 \
  --source_phase_max_points 8 \
  --source_phase_max_span 120 \
  --output_json "$OUTPUT_JSON" \
  --output_md "$OUTPUT_MD"

echo "v2.5.5a source descriptor audit written to:"
echo "  $OUTPUT_JSON"
echo "  $OUTPUT_MD"
