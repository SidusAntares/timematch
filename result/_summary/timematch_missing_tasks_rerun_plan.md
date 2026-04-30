# Missing TimeMatch Tasks Rerun Plan

These five adaptation tasks did not produce a usable best validation macro-F1 in the copied log files:

1. `DK1 -> FR1`
2. `FR1 -> DK1`
3. `FR2 -> DK1`
4. `FR2 -> FR1`
5. `AT1 -> FR2`

## Why They Are Missing

- `DK1 -> FR1`
  - Interrupted by an existing-output overwrite prompt before training started.
- `FR1 -> DK1`
  - Crashed very early in adaptation.
- `FR2 -> DK1`
  - Crashed at fold 0 with `CUDA device-side assert`.
- `FR2 -> FR1`
  - Crashed at fold 0 with `CUDA device-side assert`.
- `AT1 -> FR2`
  - Crashed at fold 0 with `CUDA device-side assert`.

## Recommended Rerun Scripts

Follow the original command style as closely as possible with one worker per source:

- [run_missing_32VNH_to_30TXT.sh](/C:/Code/dev/PythonProject/timematch/code/scripts/run_missing_32VNH_to_30TXT.sh)
- [run_missing_30TXT_to_32VNH.sh](/C:/Code/dev/PythonProject/timematch/code/scripts/run_missing_30TXT_to_32VNH.sh)
- [run_missing_31TCJ_pair.sh](/C:/Code/dev/PythonProject/timematch/code/scripts/run_missing_31TCJ_pair.sh)
- [run_missing_33UVP_to_31TCJ.sh](/C:/Code/dev/PythonProject/timematch/code/scripts/run_missing_33UVP_to_31TCJ.sh)

And the four-GPU launcher:

- [launch_missing_timematch_4gpu.sh](/C:/Code/dev/PythonProject/timematch/code/scripts/launch_missing_timematch_4gpu.sh)

## Suggested Server Command

Run inside the `code/` directory:

```bash
bash scripts/launch_missing_timematch_4gpu.sh
```

If you want to override the data path:

```bash
DATA_ROOT=/data/user/DBL/timematch_data bash scripts/launch_missing_timematch_4gpu.sh
```

## GPU Allocation

- GPU0: `DK1 -> FR1`
- GPU1: `FR1 -> DK1`
- GPU2: `FR2 -> DK1` and `FR2 -> FR1`
- GPU3: `AT1 -> FR2`

## Expected Output Directories

- `outputs/timematch_32VNH_to_30TXT`
- `outputs/timematch_30TXT_to_32VNH`
- `outputs/timematch_31TCJ_to_32VNH`
- `outputs/timematch_31TCJ_to_30TXT`
- `outputs/timematch_33UVP_to_31TCJ`
