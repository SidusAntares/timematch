#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/result/v25_temporal_shift_diagnostic_${STAMP}}"
mkdir -p "${OUT_DIR}"

python analysis/temporal_shift_diagnostic.py \
  --data_root "${DATA_ROOT}" \
  --tasks \
    "FR1_to_FR2|france/30TXT/2017|france/31TCJ/2017" \
    "FR2_to_FR1|france/31TCJ/2017|france/30TXT/2017" \
    "DK1_to_FR1|denmark/32VNH/2017|france/30TXT/2017" \
  --partitions uniform:1 uniform:2 uniform:5 uniform:10 doy_gap:5 \
  --max_samples_per_class "${MAX_SAMPLES_PER_CLASS:-128}" \
  --boundary_window "${BOUNDARY_WINDOW:-1}" \
  --output_csv "${OUT_DIR}/temporal_shift_diagnostic.csv" \
  --output_json "${OUT_DIR}/temporal_shift_diagnostic.json" \
  --output_md "${OUT_DIR}/temporal_shift_diagnostic.md"
