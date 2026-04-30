# Analysis Entry Points

This folder is reserved for rebuilt analysis scripts that recompute transferability metrics
from the current clean baseline runs.

The launcher scripts expect the main entry point to be:

- `analysis/recompute_transfer_metrics.py`

It should support at least:

- `--data_root`
- `--outputs_root`
- `--output_csv`
- `--closed_set`

Current launcher mapping:

- open-set metrics:
  - `launchers/analysis/run_recompute_metrics_open.sh`
- closed-set metrics:
  - `launchers/analysis/run_recompute_metrics_closed.sh`

Expected outputs:

- `result/baseline_analysis/open_set_transfer_metrics_recomputed.csv`
- `result/baseline_analysis/closed_set_transfer_metrics_recomputed.csv`

Until `recompute_transfer_metrics.py` is restored, GPU2/GPU3 launcher jobs will fail fast
with a clear missing-script message rather than running with an inconsistent analysis entry.
