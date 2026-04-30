#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$ROOT_DIR/analysis/recompute_transfer_metrics.py}"
DATA_ROOT="${DATA_ROOT:-/data/user/DBL/timematch_data}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-result/baseline_analysis/closed_set_transfer_metrics_recomputed.csv}"

cd "$ROOT_DIR"

if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
  echo "[ERROR] Missing analysis script: $ANALYSIS_SCRIPT"
  echo "[INFO] Restore the metric recomputation script, then rerun this launcher."
  exit 1
fi

wait_for_closed_set_outputs() {
  python - <<'PY'
import sys
import time
from pathlib import Path

tag_to_tile = {
    "FR1": "30TXT",
    "FR2": "31TCJ",
    "DK1": "32VNH",
    "AT1": "33UVP",
}

tasks = [
    ("FR1", "FR2"),
    ("FR1", "DK1"),
    ("FR1", "AT1"),
    ("FR2", "FR1"),
    ("FR2", "DK1"),
    ("FR2", "AT1"),
    ("DK1", "FR1"),
    ("DK1", "FR2"),
    ("DK1", "AT1"),
    ("AT1", "FR1"),
    ("AT1", "FR2"),
    ("AT1", "DK1"),
]

root = Path("outputs")

def task_ready(source_tag, target_tag):
    exp = root / f"timematch_{tag_to_tile[source_tag]}_to_{tag_to_tile[target_tag]}_closedset_noshift"
    if not exp.exists():
        return False
    has_ckpt = any((fold_dir / "model.pt").exists() for fold_dir in exp.glob("fold_*"))
    target_dataset = {
        "FR1": "france_30TXT_2017",
        "FR2": "france_31TCJ_2017",
        "DK1": "denmark_32VNH_2017",
        "AT1": "austria_33UVP_2017",
    }[target_tag]
    has_metrics = (exp / f"overall_{target_dataset}.json").exists() or any(
        (fold_dir / f"test_metrics_{target_dataset}.json").exists() for fold_dir in exp.glob("fold_*")
    )
    return has_ckpt and has_metrics

while True:
    missing = [f"{s}->{t}" for s, t in tasks if not task_ready(s, t)]
    if not missing:
        print("[INFO] All closed-set baseline outputs detected. Starting metric recomputation.")
        break
    print(f"[INFO] Waiting for closed-set outputs ({len(missing)} missing): {', '.join(missing)}", flush=True)
    time.sleep(120)
PY
}

wait_for_closed_set_outputs

python "$ANALYSIS_SCRIPT" \
  --data_root "$DATA_ROOT" \
  --outputs_root "$OUTPUTS_ROOT" \
  --output_csv "$OUTPUT_CSV" \
  --closed_set True
