# v3.2.1 Stage 1 Report

## Scope

This stage implements only reusable building blocks.  It does not register a new active CLI, does not connect full DA training, and does not restore v3.1 stage contrast.

## Implemented

### Temporal Feature Exposure

`models/stclassifier.py` now supports:

```python
model(..., return_temporal_features=True)
model(..., return_feats=True, return_temporal_features=True)
```

Behavior:

```text
default forward:
  logits

return_feats=True:
  logits, pooled_feature

return_temporal_features=True:
  logits, temporal_features

return_feats=True + return_temporal_features=True:
  logits, pooled_feature, temporal_features
```

`temporal_features` is the spatial/PSE temporal sequence feature before temporal pooling:

```text
[B, T, D]
```

It is not detached.

### Source Stage Reference

Implemented in:

```text
methods/local_shift/source_reference.py
```

Output keys:

```text
class_stage_feats
class_stage_centers
class_stage_durations
class_stage_mask
class_counts
stage_sample_counts
config
summary
```

The builder uses source data and source labels only.

### Target Partition

Implemented in:

```text
methods/local_shift/target_partition.py
```

Batch output keys:

```text
stage_feats
stage_mask
stage_centers
stage_durations
stage_to_time
stage_count
intervals
logs
```

The partition is feature-change based, budgeted by `Kmax`, and uses `min_stage_len` and NMS.  It is not fixed uniform segmentation.

### Soft Alignment

Implemented in:

```text
methods/local_shift/soft_alignment.py
```

The aligner uses pseudo labels to select only the corresponding source class.  It does not scan all classes and does not use target true labels.

Output keys:

```text
weights
source_indices
expected_source_centers
valid_mask
logs
```

The score includes feature cosine similarity, time gap, and duration gap.

### Local Position

Implemented in:

```text
methods/local_shift/local_position.py
```

Residual shift:

```text
residual_k = expected_source_center_k - (target_center_k + global_shift)
```

Local positions:

```text
positions_local = positions + global_shift + residual_shift_per_time
```

## Tool

Implemented:

```text
tools/build_source_stage_reference.py
```

Example:

```bash
python tools/build_source_stage_reference.py \
  --weights /path/to/model.pt \
  --source france/31TCJ/2017 \
  --num_classes 9 \
  --kmax 8 \
  --min_stage_len 3 \
  --change_quantile 0.75 \
  --output outputs/source_stage_reference.pt
```

## Tests

Added:

```text
analysis/test_temporal_feature_exposure.py
analysis/test_v321_local_shift_shapes.py
```

Local verification:

```text
py_compile: passed
runtime shape tests: not executed locally because this Python environment has no torch
```

Server verification commands:

```bash
python analysis/test_temporal_feature_exposure.py
python analysis/test_v321_local_shift_shapes.py
```
