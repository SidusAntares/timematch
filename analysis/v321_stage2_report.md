# v3.2.1 Stage 2 Report

## 1. Changed Files

```text
models/stclassifier.py
methods/local_shift/train_local_shift.py
methods/local_shift/soft_alignment.py
train.py
analysis/test_forward_from_temporal_features.py
analysis/test_v321_train_local_shift_smoke.py
analysis/v321_ablation_plan.md
launchers/v321/launch_v321_local_shift_smoke.sh
launchers/v321/README.md
```

## 2. Model Interface

All sequence classifiers now expose:

```python
forward_from_temporal_features(temporal_features, positions, return_feats=False)
```

`temporal_features` is the temporal sequence output of the spatial encoder,
shaped `[B,T,D]`. The method applies the existing temporal encoder and decoder
with the supplied positions. Existing `forward(...)`, `return_feats=True`, and
`return_temporal_features=True` behavior is preserved.

## 3. Local-Shift Training Flow

New method:

```text
timematch_local_shift
```

Training path:

```text
load source checkpoint
load source_stage_reference.pt
teacher pseudo labels with global TimeMatch shift
extract target H(t) from student spatial encoder
partition target H(t) by feature-change boundaries
soft-align target stages to pseudo-label-conditioned source class stages
generate local target positions
forward target H(t) through temporal encoder with local positions
optimize source CE + trade_off * target pseudo CE
update EMA teacher
write compact epoch TSV
```

No stage contrast loss is added.

## 4. CLI Parameters

New subcommand parameters include:

```text
--source_stage_reference_path
--local_shift_kmax
--local_shift_min_stage_len
--local_shift_topm
--local_shift_change_threshold
--local_shift_change_quantile
--local_shift_nms_radius
--local_shift_time_weight
--local_shift_duration_weight
--local_shift_feature_weight
--local_shift_clip
--local_shift_detach_correspondence
--local_shift_log_path
--local_shift_mode {global_only,residual}
```

`global_only` still runs partition/alignment logging but uses scalar shifted
positions. `residual` enables stage-wise residual local positions.

## 5. Source Reference Checks

The local-shift path requires:

```text
class_stage_feats
class_stage_centers
class_stage_durations
class_stage_mask
class_counts
```

It checks class count, feature dimension, Kmax compatibility, and that at least
one valid source stage exists. Invalid references raise `ValueError`; there is
no silent fallback.

## 6. Logging

`--local_shift_log_path` writes one compact TSV row per epoch with:

```text
global_shift
local_shift_mode
local_shift_mean/std/abs_mean/clip_fraction
stage_count_mean/std/max/p90
alignment_entropy/top1_mass/valid_ratio/fallback_ratio
position_min/max/clamp_ratio
target_pseudo_confidence/ratio
source_loss
target_loss
total_loss
```

No per-sample or per-stage logs are emitted.

## 7. Verification Status

Required local verification:

```text
py_compile for changed modules
```

Runtime tensor tests require a Python environment with torch installed:

```bash
python analysis/test_temporal_feature_exposure.py
python analysis/test_v321_local_shift_shapes.py
python analysis/test_forward_from_temporal_features.py
python analysis/test_v321_train_local_shift_smoke.py
```

## 8. Scope Limits

Stage 2 only proves the training path can run. It does not prove performance.

Not included:

```text
stage contrast
target true labels
fixed uniform stages
same-index hard alignment
DTW/GCTW loss
memory bank
trainable partition
full12 experiments
```

Formal comparison must use the same cleaned code for all settings:

```text
cleaned base TimeMatch
cleaned smooth source + base TimeMatch
source reference + global_only
source reference + residual local_shift
```
