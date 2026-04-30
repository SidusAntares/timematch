# ShiftAug Reset Notes (2026-04-28)

## What changed

- We confirmed that using a `ShiftAug`-trained source model can materially raise TimeMatch downstream macro-F1.
- That means older baseline comparisons are not clean enough to keep using as the main reference line.
- From now on, every experiment should record two separate switches explicitly:
  - `source_shift_aug`
  - `da_shift_aug`

## Canonical execution choice

- Keep the root project as the main runnable workspace.
- Treat `code/` as a frozen reference copy that helped us trace the older behavior.
- Reason:
  - root now contains the restored core TimeMatch behavior we wanted,
  - root has better experiment-control utilities (`overwrite_existing`, summary helpers, analysis scripts),
  - idea-validation code only exists in root.

## New baseline policy

- Main baseline reset:
  - source-only pretraining: `with_shift_aug=False`
  - TimeMatch adaptation: `with_shift_aug=False`
- Every experiment name and log name should encode the ShiftAug state.

## One-task validation policy

- Before re-running every structural idea everywhere, validate on one informative task:
  - primary: `DK1 -> FR1`
  - mirror: `FR1 -> DK1`
- Only ideas that still look promising after the reset should graduate to full multi-task reruns.

## Safe cleanup candidates

These look temporary or superseded and should be archived or removed after server sync is stable:

- `recheck_original_setting.sh`
- `restore_after_other_impl_test.sh`
- `test_other_impl_dk1_to_fr1.sh`
- `train.log`
- ad-hoc one-off logs in repo root

## Keep

- `code/` for reference until the reset results are settled
- `result/baseline_analysis/*`
- `result/_summary/structure_design_principles.md`
- scripts under `scripts/` that are still part of the reset workflow

## New scripts added for the reset

- `scripts/run_timematch_task.sh`
- `scripts/run_baselines_partition.sh`
- `scripts/run_idea_validation_queue.sh`
- `scripts/launch_four_gpu_plan.sh`
