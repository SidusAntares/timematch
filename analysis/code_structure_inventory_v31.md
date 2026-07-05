# Code Structure Inventory for v3.1 Stage Contrast

## 1. Entry Points

### Main CLI

- File: `train.py`
- CLI: `python train.py ... {method}`
- Method dispatch:
  - `timematch` -> `timematch.train_timematch(...)`
  - `sourcephasecompact` -> `ideas.train_source_phase_compactness.train_supervised_source_phase_compactness(...)`
  - no method / other supervised path -> `train_supervised(...)`

### Source Pretrain

Current structure-pretrain entry:

```bash
python train.py ... sourcephasecompact
```

Main function:

```python
ideas/train_source_phase_compactness.py
train_supervised_source_phase_compactness(...)
```

Important parameters:

```text
--source_structure_loss_version
--source_structure_feature_target
--source_structure_detach_features
--source_structure_intra_trade_off
--source_structure_lambda_schedule
--source_structure_lambda_base
--source_structure_lambda_final
--source_structure_time_smooth_kernel_size
--source_structure_time_permutation_seed
--source_segment_partition_mode
--source_segment_count
--epochs
--with_shift_aug
--closed_set
```

Source pretrain saves:

```text
outputs/<experiment>/fold_0/model.pt
outputs/<experiment>/train_config.json
outputs/<experiment>/fold_0/source_structure_lambda_curve.tsv
```

### TimeMatch DA

Current DA entry:

```bash
python train.py ... timematch --weights outputs/<source_experiment>
```

Main function:

```python
timematch.py
train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits)
```

Important parameters:

```text
--weights
--pseudo_threshold
--ema_decay
--trade_off
--estimate_shift
--shift_source
--sample_size
--max_temporal_shift
--shift_estimator
--domain_specific_bn
--timematch_shift_policy
--timematch_topk_shifts
--timematch_diagnostic_log_path
--epochs
--steps_per_epoch
```

TimeMatch saves:

```text
outputs/<timematch_experiment>/fold_0/model.pt
outputs/<timematch_experiment>/train_config.json
outputs/<timematch_experiment>/fold_0/test_metrics_<target>.json
outputs/<timematch_experiment>/fold_0/class_report_<target>.txt
outputs/<timematch_experiment>/fold_0/conf_mat_<target>.pkl
```

Optional TimeMatch diagnostic TSV:

```text
--timematch_diagnostic_log_path <path>
```

## 2. Model / Encoder

### Classifier Wrapper

File:

```text
models/stclassifier.py
```

Main model for current runs:

```python
PseLTae
```

Forward:

```python
logits = model.forward(pixels, mask, positions, extra)
logits, temporal_feats = model.forward(..., return_feats=True)
```

Important point:

```text
return_feats=True returns temporal encoder output after LTAE pooling: [B, D].
It does not return a per-time-step tensor [B, T, D].
```

### Spatial Encoder

File:

```text
models/pse.py
```

Class:

```python
PixelSetEncoder
```

Input:

```text
pixels: [B, T, C, S]
mask:   [B, T, S]
extra:  [B, T, E] or None
```

Output:

```text
spatial_feats: [B, T, D]
```

Pooling location:

```python
PixelSetEncoder.forward(...)
out = self.mlp1(out).transpose(1, 2)
out = torch.cat([pooling_methods[n](out, mask) ...], dim=1)
out = self.mlp2(out)
out = out.view(batch, temp, -1)
```

This pooling is pixel-set pooling inside each time step, not temporal pooling.

### Temporal Encoder

File:

```text
models/ltae.py
```

Class:

```python
LTAE
```

Input:

```text
spatial_feats: [B, T, D]
positions:     [B, T]
```

Internal sequence feature:

```python
x = self.inconv(x)
enc_output = x + self.positional_enc(positions + self.max_temporal_shift)
```

Temporal pooling location:

```python
enc_output, attn = self.attention_heads(enc_output)
enc_output = self.dropout(self.mlp(enc_output))
```

`MultiHeadAttention.forward(...)` collapses `[B,T,D]` into `[B,D]` using learned temporal attention.

### Can We Get H ∈ [B,T,D]?

Yes, but not from current `model.forward(..., return_feats=True)`.

Currently available without model API change:

```python
spatial_feats = model.spatial_encoder(pixels, mask, extra)  # [B,T,D]
```

If v3.1 needs temporal features after LTAE projection and positional encoding, add a small explicit hook:

```python
return_temporal_features=True
```

Candidate returned tensors:

```text
spatial_feats: PSE output, before LTAE, [B,T,D]
ltae_input:    after LTAE inconv + positional encoding, [B,T,D_model]
pooled_feats:  after LTAE attention, [B,D]
```

## 3. Existing Structure Losses

### Call Site

File:

```text
ideas/train_source_phase_compactness.py
```

Main source loop:

```python
spatial_feats_raw = model.spatial_encoder(pixels, mask, extra)
temporal_feats_raw = model.temporal_encoder(spatial_feats_raw, positions)
outputs_raw = model.decoder(temporal_feats_raw)
compact_raw_loss, raw_logs = _compute_source_structure_loss_on_features(...)
loss = cls_loss_raw + compact_loss
```

### Dispatcher

Function:

```python
_compute_source_structure_loss_on_features(...)
```

If `source_structure_loss_version` is a raw compactness version, it calls:

```python
ideas/source_raw_compactness.py
compute_source_raw_global_compactness_loss(...)
```

Otherwise it calls legacy segment structure loss:

```python
ideas/source_phase_compactness.py
compute_source_structure_loss(...)
```

### Raw Global Compactness

File:

```text
ideas/source_raw_compactness.py
```

Version names:

```text
v275_raw_global_compactness
raw_global_compactness
source_raw_global_compactness
```

Implementation:

```python
pooled_feats = spatial_feats.mean(dim=1)
compact_loss = class-wise compactness to batch class mean
```

Key parameters:

```text
--source_structure_intra_trade_off
--source_structure_compact_distance
--source_structure_detach_features
```

### Smooth k3 Timepoint Compactness

Version names:

```text
v276_raw_smoothed_timepoint_compactness
raw_smoothed_timepoint_compactness
source_raw_smoothed_timepoint_compactness
```

Implementation:

```python
smoothed_feats = _smooth_time_axis(spatial_feats, kernel_size)
compact_loss = _compute_timepoint_compactness(smoothed_feats, labels)
```

Default kernel:

```text
--source_structure_time_smooth_kernel_size 3
```

### Time-permuted Smooth k3

Version names:

```text
v303_time_permuted_smoothed_timepoint_compactness
time_permuted_smoothed_timepoint_compactness
source_time_permuted_smoothed_timepoint_compactness
```

Implementation:

```python
permuted = spatial_feats[:, perm, :]
smoothed = _smooth_time_axis(permuted, kernel_size)
smoothed[:, inv_perm, :]
compact_loss = _compute_timepoint_compactness(...)
```

Important:

```text
The permutation is inside the structure loss only.
It does not permute model input, target input, TimeMatch shift estimation, or DA batches.
```

Key parameter:

```text
--source_structure_time_permutation_seed
```

## 4. TimeMatch Components

### Teacher / Student

File:

```text
timematch.py
```

Setup:

```python
student.load_state_dict(pretrained_weights)
teacher = deepcopy(student)
```

Training mode:

```python
student.train()
teacher.eval()
```

EMA update:

```python
update_ema_variables(student, teacher, config.ema_decay)
```

### Pseudo-label Generation

Batch-level pseudo labels are generated in `train_timematch(...)`:

```python
teacher_preds = F.softmax(
    teacher.forward(pixels_t_weak, mask_t_weak, position_t_weak + target_to_source_shift, extra_t_weak),
    dim=1,
)
pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
pseudo_mask = pseudo_conf > config.pseudo_threshold
```

Available in the batch loop:

```text
teacher_preds:   [B,C]
pseudo_conf:     [B]
pseudo_targets:  [B]
pseudo_mask:     [B]
```

They are not stored in `sample_target_*`; they are local tensors inside the DA loop.

### TimeMatch Global Shift

Shift estimation:

```python
estimate_temporal_shift_details(...)
collect_shift_softmaxes(...)
score_shift_softmaxes(...)
```

Global variables inside DA loop:

```text
target_to_source_shift
source_to_target_shift
target_to_source_topk_shifts
last_shift_details
```

Usage:

```python
teacher.forward(..., position_t_weak + target_to_source_shift, ...)
student.forward(..., position_s + source_to_target_shift, ...)
```

Logging:

```text
print("TIMEMATCH_SHIFT_TRAJECTORY|...")
_append_diag_tsv(config.timematch_diagnostic_log_path, ...)
writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)
```

Risk:

```text
The estimated shift is a local variable in train_timematch.
It is logged, but not saved as an explicit model/checkpoint field.
```

### DA Loss

Source classification:

```python
loss_source = criterion(logits_source, source_labels)
```

Target pseudo-label loss:

```python
loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
```

Total:

```python
loss = loss_source + config.trade_off * loss_target
```

Current TimeMatch DA has no source structure loss call in the DA loop.

## 5. Batch Data Format

### Dataset Sample

File:

```text
dataset.py
PixelSetData.__getitem__
```

Raw keys:

```text
index
parcel_index
pixels
valid_pixels
positions
extra
label
```

After `ToTensor`:

```text
pixels:       torch.float32, [T,C,S]
valid_pixels: torch.float32, [T,S]
positions:    torch.long,    [T]
extra:        torch.float32, [E] before collate, effectively [B,E] after batching
label:        torch.long,    scalar before collate, [B] after batching
```

After DataLoader collate:

```text
sample["pixels"]:       [B,T,C,S]
sample["valid_pixels"]: [B,T,S]
sample["positions"]:    [B,T]
sample["extra"]:        [B,E]
sample["label"]:        [B]
sample["index"]:        [B]
sample["parcel_index"]: [B]
```

### Source Batch in TimeMatch

In `train_timematch(...)`:

```python
sample_source = next(source_iter)
pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
source_labels = sample_source["label"].cuda(...)
```

Source batch fields:

```text
sample_source["pixels"]
sample_source["valid_pixels"]
sample_source["positions"]
sample_source["extra"]
sample_source["label"]
```

### Target Batch in TimeMatch

Target loader returns a tuple:

```python
sample_target_weak, sample_target_strong = next(target_iter)
```

Weak target:

```text
Used by teacher for pseudo-label generation.
```

Strong target:

```text
Used by student for target pseudo-label training.
```

Target true label exists in batch:

```text
sample_target_weak["label"]
sample_target_strong["label"]
```

But it is used only for diagnostics / offline pseudo-label F1, not for training.

### CUDA Conversion

File:

```text
utils/train_utils.py
```

Function:

```python
to_cuda(sample, device)
```

Returns:

```python
pixels, valid_pixels, positions, extra
```

Labels are moved manually where needed.

## 6. Candidate Hook Points

### return_temporal_features

Best minimal hook:

```text
models/stclassifier.py
PseLTae.forward(...)
```

Recommended behavior:

```python
if return_temporal_features:
    return {
        "logits": logits,
        "spatial_feats": spatial_feats,   # [B,T,D]
        "pooled_feats": temporal_feats,   # [B,D]
    }
```

If stage contrast needs positional temporal features, add a helper in `models/ltae.py`:

```python
encode_sequence(x, positions) -> [B,T,D_model]
```

### TemporalStageExtractor

Recommended new file:

```text
ideas/stage_contrast.py
```

Inputs:

```text
H:         [B,T,D]
positions: [B,T]
stage spec: fixed K or learned/offline anchors
```

Outputs:

```text
stage_feats: [B,K,D]
stage_mask:  [B,K]
```

### StageContrastiveLoss

Recommended new file:

```text
ideas/stage_contrast.py
```

Inputs:

```text
source_stage_feats: [B_s,K,D]
source_labels:      [B_s]
target_stage_feats: [B_t,K,D]
target_pseudo:      [B_t]
target_conf:        [B_t]
target_mask:        [B_t]
```

### stage correspondence A(k,j)

Candidate hook inside TimeMatch DA loop:

```text
timematch.py
inside train_timematch, after pseudo_targets / pseudo_mask and after source/target forward feature extraction.
```

Reason:

```text
At this point source labels, target pseudo labels, pseudo confidence, source positions, target positions, teacher logits, and student features are all available in the same batch step.
```

### Logging

Current logging options:

```text
print("TAG|key=value|...")
writer.add_scalar(...)
_append_diag_tsv(...)
```

Recommended v3.1 logs:

```text
STAGE_CONTRAST|
stage_loss
stage_pair_count
stage_source_count
stage_target_count
pseudo_coverage
pseudo_conf_mean
correspondence_entropy
correspondence_diag_mass
```

If TSV is needed, follow TimeMatch:

```python
_append_diag_tsv(path, row, fields)
```

## 7. Risks

1. `model.forward(..., return_feats=True)` does not expose `[B,T,D]`.
   - It returns pooled temporal encoder features `[B,D]`.
   - Use `model.spatial_encoder(...)` or add an explicit temporal feature return.

2. LTAE currently collapses time inside `MultiHeadAttention`.
   - If stage contrast needs post-positional sequence features, add a pre-attention return hook.

3. Target pseudo labels are local tensors, not batch fields.
   - Hook must be inside `train_timematch(...)` after `pseudo_conf`, `pseudo_targets`, `pseudo_mask` are computed.

4. Global shift is local state.
   - `target_to_source_shift` and `source_to_target_shift` are not checkpoint fields.
   - Stage correspondence code should either use local shift variables directly or explicitly log/save them.

5. Source and target time axes may differ.
   - Remote tasks use true date positions.
   - DataLoader groups by shape for eval, but DA train loader can sample source/target separately.
   - Any stage extractor should accept `positions`, not assume index-only alignment.

6. Target true labels exist in the batch but must not be used for training.
   - They are diagnostic-only.

7. `domain_specific_bn=True` splits source and target forward calls.
   - If a stage loss needs source and target features simultaneously, handle both branches.

8. `pseudo_mask` can be empty or very small.
   - Existing target loss skips target logits if insufficient pseudo-labels in domain-specific BN branch.
   - Stage loss must gracefully return zero when no reliable target samples exist.

9. Existing structure loss is source-pretrain only.
   - Adding DA-stage stage contrast is a new DA hook; keep it behind explicit flags.

10. Existing logs can become very large.
    - Avoid per-sample or per-pair prints.
    - Prefer per-epoch summaries and compact TSV rows.

## 8. Minimal Implementation Plan

1. Expose temporal features.
   - Add `return_temporal_features` to `PseLTae.forward`.
   - Return `spatial_feats [B,T,D]` and `pooled_feats [B,D]`.
   - Optionally add an LTAE helper for pre-attention positional features.

2. Add fixed stage extractor.
   - New file: `ideas/stage_contrast.py`.
   - Start with deterministic fixed K stages over `positions`.
   - Return `[B,K,D]` stage features and `[B,K]` mask.

3. Add hard k-k stage contrast.
   - Use source labels and target pseudo labels.
   - Only use target samples passing `pseudo_mask`.
   - Start with same stage index `k -> k`, no learned correspondence.

4. Add config flags.
   - `--stage_contrast_trade_off`
   - `--stage_contrast_stage_count`
   - `--stage_contrast_feature_kind`
   - `--stage_contrast_pseudo_threshold`
   - `--stage_contrast_log_path`

5. Add logs.
   - Print one compact `STAGE_CONTRAST|...` line per epoch.
   - Optional TSV with one row per epoch.
   - Do not print per-sample alignment details during training.
