# v3.2.1 Residual Local Shift

This directory contains the stage-wise residual local-shift implementation used
in v3.2.1.

The method follows this pipeline:

```text
source class-stage reference
-> target feature-change partition
-> soft stage correspondence
-> stage-wise residual shift
-> adjusted target positions
-> TimeMatch pseudo-label adaptation
```

After the Stage 3c equivalence control was repaired, `global_only` became
equivalent to the original TimeMatch control path. Under this valid control,
all Stage 3d residual variants underperformed on both audit tasks.

The implementation is therefore retained as a negative-boundary experiment
and reusable diagnostic code. It is not an active method and should not be
extended for v3.2.2.

Reusable components:

- `source_reference.py`
- `target_partition.py`
- `soft_alignment.py`
- `local_position.py`

Final results are documented in:

- `analysis/v321_stage3d_residual_safety_report.md`
- `analysis/v321_local_shift_design.md`
- `analysis/v321_ablation_plan.md`
