# v3.2.1 Launchers

`timematch_local_shift` is experimental and archived after Stage 3d.  It is
kept for diagnostics and reproducibility, not advertised as an active method.

## Stage 2 Smoke

`launch_v321_local_shift_smoke.sh` runs one minimal `timematch_local_shift`
job. It does not build the source reference; pass an existing source checkpoint
and `source_stage_reference.pt` explicitly:

```bash
cd /data/user/timematch

nohup env \
  RUN_TAG=v321_local_shift_smoke_$(date +%Y%m%d_%H%M%S) \
  SOURCE="france/30TXT/2017" \
  TARGET="france/31TCJ/2017" \
  SOURCE_WEIGHTS="/data/user/timematch/outputs/YOUR_SOURCE_EXPERIMENT" \
  SOURCE_STAGE_REFERENCE_PATH="/data/user/timematch/outputs/v321_smoke/source_stage_reference.pt" \
  MODE=residual \
  GPUS="0" \
  EPOCHS=1 \
  STEPS_PER_EPOCH=5 \
  SAMPLE_SIZE=8 \
  bash launchers/v321/launch_v321_local_shift_smoke.sh \
  > logs/v321_local_shift_smoke_nohup.out 2>&1 &
```

For the scalar-shift control:

```bash
MODE=global_only bash launchers/v321/launch_v321_local_shift_smoke.sh
```

This launcher is for runtime smoke only. Do not use its F1 as a method result.


## Stage 3 Probe

`launch_v321_stage3_probe.sh` runs the four-task cleaned-code comparison:

- `base`: cleaned TimeMatch with plain source checkpoint.
- `smooth_base`: cleaned TimeMatch with `smooth_k3` source checkpoint, skipped when the checkpoint is missing.
- `global_only`: `timematch_local_shift` with scalar-shift control.
- `residual`: `timematch_local_shift` with residual local shift.

It builds `source_stage_reference.pt` for local-shift jobs when needed and writes
one compact summary TSV under the run log directory.

```bash
cd /data/user/timematch

nohup env \
  MASTER_TAG=v321_stage3_probe_$(date +%Y%m%d_%H%M%S) \
  GPUS="0 1 2 3" \
  TASKS="FR1_to_FR2,FR2_to_AT1,FR2_to_FR1,AT1_to_DK1" \
  CONFIGS="base,smooth_base,global_only,residual" \
  SEED=1 \
  EPOCHS=20 \
  STEPS_PER_EPOCH=500 \
  SAMPLE_SIZE=100 \
  NUM_WORKERS=8 \
  LOCAL_SHIFT_SOURCE_KIND=smooth_if_available \
  bash launchers/v321/launch_v321_stage3_probe.sh \
  > logs/v321_stage3_probe_nohup.out 2>&1 &
```

Optional manifest format:

```text
task    plain_source    smooth_source
FR1_to_FR2    /data/user/timematch/outputs/...plain_source    /data/user/timematch/outputs/...v276_smooth_k3_w1_source
```

Pass it with:

```bash
SOURCE_CHECKPOINT_MANIFEST=/data/user/timematch/path/to/stage3_source_manifest.tsv
```


## Stage 3b Equivalence Audit

`launch_v321_stage3b_equivalence_audit.sh` runs only two tasks and checks where
`timematch_local_shift` first diverges from the cleaned TimeMatch path:

```bash
cd /data/user/timematch

nohup env \
  MASTER_TAG=v321_stage3b_equivalence_audit_$(date +%Y%m%d_%H%M%S) \
  GPUS="0 1 2 3" \
  TASKS="AT1_to_DK1,FR2_to_FR1" \
  CONFIGS="smooth_base,base_equiv,global_forward,global_only,residual" \
  SEED=1 \
  EPOCHS=20 \
  STEPS_PER_EPOCH=500 \
  SAMPLE_SIZE=100 \
  NUM_WORKERS=8 \
  bash launchers/v321/launch_v321_stage3b_equivalence_audit.sh \
  > logs/v321_stage3b_equivalence_audit_nohup.out 2>&1 &
```

Outputs:

```text
logs/<MASTER_TAG>/summary.tsv
analysis/v321_stage3b_equivalence_audit.md
```


## Stage 3c Equivalence Fix

`launch_v321_stage3c_equivalence_fix.sh` reruns the two-task equivalence audit
after the local-shift loop is made closer to base TimeMatch. It does not run
full12.

```bash
cd /data/user/timematch

nohup env \
  MASTER_TAG=v321_stage3c_equivalence_fix_$(date +%Y%m%d_%H%M%S) \
  GPUS="0 1 2 3" \
  TASKS="AT1_to_DK1,FR2_to_FR1" \
  CONFIGS="smooth_base,base_equiv,global_forward,global_only,residual" \
  SEED=1 \
  EPOCHS=20 \
  STEPS_PER_EPOCH=500 \
  SAMPLE_SIZE=100 \
  NUM_WORKERS=8 \
  EQUIV_DEBUG_STEPS=3 \
  bash launchers/v321/launch_v321_stage3c_equivalence_fix.sh \
  > logs/v321_stage3c_equivalence_fix_nohup.out 2>&1 &
```

Outputs:

```text
logs/<MASTER_TAG>/summary.tsv
analysis/v321_stage3c_equivalence_fix_report.md
```


## Stage 3d Residual Safety

`launch_v321_stage3d_residual_safety_4gpu.sh` runs the two-task residual
safety audit with independent single-GPU jobs. It builds each source-stage
reference once, then reuses it across the residual variants.

Stage 3d is the stopping experiment for residual local shift.  The residual
variants underperformed the corrected `global_only` control, so do not run
full12 from this path.

```bash
cd /data/user/timematch

nohup env \
  MASTER_TAG=v321_stage3d_residual_safety_$(date +%Y%m%d_%H%M%S) \
  GPUS="0 1 2 3" \
  TASKS="AT1_to_DK1,FR2_to_FR1" \
  CONFIGS="global_only,residual_raw,residual_zero_mean,residual_scaled_zero_mean_alpha05,residual_gated_scaled_zero_mean_alpha05_top065" \
  SEED=1 \
  EPOCHS=20 \
  STEPS_PER_EPOCH=500 \
  SAMPLE_SIZE=100 \
  NUM_WORKERS=8 \
  bash launchers/v321/launch_v321_stage3d_residual_safety_4gpu.sh \
  > logs/v321_stage3d_residual_safety_nohup.out 2>&1 &
```

Outputs:

```text
logs/<MASTER_TAG>/job_status.tsv
logs/<MASTER_TAG>/summary.tsv
logs/<MASTER_TAG>/<task>/<config>/train.log
logs/<MASTER_TAG>/<task>/<config>/local_shift.tsv
analysis/v321_stage3d_residual_safety_report.md
```
