# v2.8 Old vs Cleaned Code Audit

## 1. Current Workspace

| item | value |
|---|---|
| project path | `<repository-root>` |
| branch | `main` |
| current commit | `7f36300768b31ca56724c44050eee2bf88207970` |
| cleaned full12 log | `logs/v28_cleaned_full12_20260710_120037` |
| old v275/v276 raw logs in local workspace | not found |

Current worktree is dirty before this audit. This audit does not revert or modify training code.

---

## 2. Likely Old Code Versions

| experiment | likely commit | evidence |
|---|---|---|
| old v275 baseline / raw full12 | `f04e1e06805270d4e98db688ae869fbdeb6493b6` | commit adds `launch_v275_closedset_baseline_v275_12tasks_3seeds.sh` and `launch_v276_closedset_strength_response_12tasks_3seeds.sh` |
| old v276 smooth / center probe | `89d9df4e52744cb955168b0d203a2ddd61c3199e` | commit adds `launch_v276_center_lambda_probe.sh` and `launch_v276_fr2_center_probe.sh`; modifies `ideas/source_raw_compactness.py` |
| cleaned-code reproduction | `7f36300768b31ca56724c44050eee2bf88207970` | current checked commit |

Important: `f04e1e0 -> 89d9df4` does not modify `timematch.py`. So old v275 baseline and old v276 smooth likely used the same TimeMatch DA implementation. The smooth commit mainly changes source-side structure loss and launchers.

---

## 3. Cleaned Full12 Results Found

From `logs/v28_cleaned_full12_20260710_120037/summary_by_config.tsv`:

| config | source-on-target | DA F1 | DA gain | jobs |
|---|---:|---:|---:|---:|
| base | 0.466050 | 0.648822 | 0.182772 | 36 |
| raw_global | 0.482808 | 0.651197 | 0.168389 | 36 |
| smooth_k3 | 0.480683 | 0.629147 | 0.148464 | 36 |
| elastic_r2 | 0.483289 | 0.644606 | 0.161317 | 36 |

Old reported values from the user:

| config | source-on-target | DA F1 | DA gain |
|---|---:|---:|---:|
| old base | 0.4633 | 0.6278 | 0.1644 |
| old raw_global | 0.4801 | 0.6499 | 0.1698 |
| old smooth_k3 | 0.4735 | 0.6525 | 0.1790 |

Observed difference:

| config | source-on-target delta | DA gain delta | DA F1 delta |
|---|---:|---:|---:|
| base | +0.0028 | +0.0184 | +0.0210 |
| raw_global | +0.0027 | -0.0014 | +0.0013 |
| smooth_k3 | +0.0072 | -0.0305 | -0.0234 |

This supports: source-stage behavior is broadly similar, while the base increase and smooth decrease mainly appear in DA dynamics. It does not by itself identify the cause.

---

## 4. Launcher / Config Comparison

### Old v275 / v276 launcher path

Old v275 wrapper:

```text
launchers/archive/older/launch_v275_closedset_baseline_v275_12tasks_3seeds.sh
```

Key settings:

```text
TASKS = 12 remote tasks
SEEDS = 1 2 3
CONFIGS = plain,v275_raw_w1
CLOSED_SET = True
SOURCE_EPOCHS = 100
DA_EPOCHS = 20
V275_WEIGHT = 1.0
```

Old v276 smooth path:

```text
launchers/archive/older/launch_v276_center_lambda_probe.sh
launchers/archive/older/launch_v276_fr2_center_probe.sh
launchers/archive/older/launch_v275_clean_baseline_4task_probe.sh
```

Key settings:

```text
SOURCE_EPOCHS = 100
DA_EPOCHS = 20
STEPS_PER_EPOCH = 500
CLOSED_SET = True
CONFIGS include v276_smoothed_timepoint_w1
```

Old lower-level launcher default `SOURCE_EPOCHS=50`, but v275/v276 wrappers override it to `100`. So old full/probe settings should not be treated as 50 epoch from the lower-level default alone.

### Cleaned full12 path

```text
launchers/v28/launch_v28_cleaned_full12_repro_4gpu.sh
launchers/v28/launch_v28_cleaned_full12_source_train_4gpu.sh
launchers/v28/launch_v28_cleaned_full12_da_4gpu.sh
```

Key settings:

```text
SOURCE_EPOCHS = 100
DA_EPOCHS = 20
STEPS_PER_EPOCH = 500
CLOSED_SET = True
with_shift_aug = False
```

Cleaned source inventory reuses checkpoints by `(source_domain, config, seed)`. Old launcher names source checkpoints by `(task, config, seed)`. This is a real run-organization difference, but it is not proven to be causal unless checkpoint hashes or deterministic source smoke tests show different weights.

---

## 5. Cleaned Source Training Check

Examples inspected from cleaned logs:

```text
logs/v28_cleaned_full12_20260710_120037/source_train/DK1/smooth_k3/seed1/train.log
logs/v28_cleaned_repro_20260709_194845/source_train/FR2/smooth_k3/seed1/train.log
```

Confirmed smooth source settings:

```text
method = sourcephasecompact
epochs = 100
closed_set = True
with_shift_aug = False
source_structure_loss_version = v276_raw_smoothed_timepoint_compactness
source_structure_feature_target = raw
source_structure_detach_features = False
source_structure_intra_trade_off = 1.0
source_structure_time_smooth_kernel_size = 3
source_structure_compact_distance = mse
legacy component weights = 0
```

Confirmed base source settings:

```text
plain supervised source training
epochs = 100
closed_set = True
with_shift_aug = False
no sourcephasecompact method
```

---

## 6. Cleaned DA Training Check

Example inspected:

```text
logs/v28_cleaned_full12_20260710_120037/da/FR2_FR1/base/seed1/train.log
logs/v28_cleaned_full12_20260710_120037/da/FR2_FR1/smooth_k3/seed1/train.log
```

Confirmed DA settings:

```text
method = timematch
epochs = 20
steps_per_epoch = 500
batch_size = 128
lr = 0.0001
weight_decay = 0.0001
with_shift_aug = False
closed_set = True
pseudo_threshold = 0.9
ema_decay = 0.9999
trade_off = 2.0
estimate_shift = True
balance_source = True
use_focal_loss = True
shift_source = True
sample_size = 100
max_temporal_shift = 60
domain_specific_bn = True
shift_estimator = AM
timematch_shift_policy = original_timematch
output_student = True
```

So cleaned DA was not accidentally run with local-shift or DA-stage source structure loss.

---

## 7. DA Implementation Differences

| component | old behavior | cleaned behavior | changed | could affect base | could affect smooth | evidence |
|---|---|---|---|---|---|---|
| TimeMatch location | `timematch.py` contains implementation | `timematch.py` wrapper to `methods/timematch_base/train_loop.py` | yes, refactor | possible | possible | code moved and expanded |
| student/teacher init | load source `model.pt`, `teacher=deepcopy(student)` | same | no obvious | no | no | code inspection |
| optimizer | Adam on student params | same | no | no | no | code inspection |
| scheduler | CosineAnnealingLR, step each iteration | same | no | no | no | code inspection |
| pseudo threshold | 0.9 | 0.9 | no | no | no | parser/logs |
| EMA decay/timing | update after optimizer step | same | no | no | no | code inspection |
| teacher mode | `teacher.eval()` before pseudo labels | same | no | no | no | code inspection |
| output model | student by default | same | no | no | no | parser/logs |
| target strong/weak aug | weak: pixels only; strong: random timesteps | same | no obvious | no | no | code inspection |
| shift estimator | IS initial if AM; then AM each epoch | same policy | mostly same | possible | possible | code inspection |
| shift scoring epsilon | `1e-5` in IS/AM/entropy logs | `1e-12` in `score_shift_softmaxes` | yes | possible | possible | code inspection |
| shift diagnostics | none | extra diagnostic fields/topk | yes | unlikely unless code path changes | unlikely unless code path changes | code inspection |
| no-shift/fixed/oracle policies | absent | added, default `original_timematch` | added but default off | unlikely | unlikely | logs show original policy |
| dataloader timeout | absent | added timeout argument | added | unlikely for metrics | unlikely | code inspection |
| tqdm/logging | tqdm | compact logs | changed | no intended | no intended | code inspection |
| validation/test metric code | same metric formulas; tqdm removed | same formulas, `+/-` output format changed | no metric formula change found | unlikely | unlikely | diff of `evaluation.py`, `utils/metrics.py` |

The clearest DA code-level candidate difference is the shift scoring epsilon change from `1e-5` to `1e-12`. Whether it changes chosen shifts must be tested on the same checkpoint/logits.

---

## 8. Smooth Loss Comparison

Old smooth implementation:

```text
ideas/source_raw_compactness.py @ 89d9df4
version = v276_raw_smoothed_timepoint_compactness
```

Old formula path:

```text
spatial_feats: [B, T, D]
smooth over T with replicate padding, uniform kernel size 3
per class:
  class_center = mean(class_feats, dim=0, keepdim=True)  # [1,T,D]
  loss_c = mean_{i,t} sum_d (H_i(t,d) - center_c(t,d))^2
loss = average over valid classes
total = lambda * loss
```

Cleaned implementation:

```text
methods/source_structure/losses.py
version = v276_raw_smoothed_timepoint_compactness
```

Cleaned formula path for `smooth_k3`:

```text
same spatial_feats [B,T,D]
same replicate padding
same uniform smoothing
time_smooth_kernel_size = 3
same per-timepoint class center
same MSE and valid-class averaging
same detach=False from launcher
```

Code-level result:

```text
For v276_raw_smoothed_timepoint_compactness with kernel=3,
the old and cleaned formulas appear equivalent by inspection.
```

Numerical unit comparison status:

```text
not executed locally: current local Python environment has no torch installed.
```

Server-side minimal check:

```bash
cd /data/user/timematch
python - <<'PY'
import subprocess, types, torch
old_src = subprocess.check_output(
    ["git", "show", "89d9df4e52744cb955168b0d203a2ddd61c3199e:ideas/source_raw_compactness.py"],
    text=True,
)
old_mod = types.ModuleType("old_source_raw_compactness")
exec(old_src, old_mod.__dict__)
from methods.source_structure.losses import compute_source_raw_global_compactness_loss as new_loss

torch.manual_seed(123)
H_old = torch.randn(10, 7, 5, requires_grad=True)
labels = torch.tensor([0,0,0,1,1,1,2,2,2,2], dtype=torch.long)
H_new = H_old.detach().clone().requires_grad_(True)
kwargs = dict(
    version="v276_raw_smoothed_timepoint_compactness",
    intra_trade_off=1.0,
    compact_distance="mse",
    norm_preserve_trade_off=0.0,
)
old_val, old_logs = old_mod.compute_source_raw_global_compactness_loss(H_old, labels, **kwargs)
new_val, new_logs = new_loss(H_new, labels, time_smooth_kernel_size=3, **kwargs)
old_val.backward()
new_val.backward()
print("old_loss", float(old_val.detach()))
print("new_loss", float(new_val.detach()))
print("loss_abs_diff", abs(float(old_val.detach()) - float(new_val.detach())))
print("old_grad_norm", float(H_old.grad.norm()))
print("new_grad_norm", float(H_new.grad.norm()))
print("grad_max_abs_diff", float((H_old.grad - H_new.grad).abs().max()))
PY
```

---

## 9. Source Checkpoint Equivalence

Cleaned inventory records checkpoint paths but local workspace does not contain server checkpoint files:

```text
/data/user/timematch/outputs/...
```

Local `Test-Path` for representative server checkpoint returns false, so file/state_dict SHA256 cannot be computed in this local audit.

Cleaned inventory confirms organization:

```text
AT1/FR1/FR2 source checkpoints mostly reused from probe
DK1 source checkpoints trained in full12
```

This confirms reuse, but does not prove reuse causes the result difference.

Required server hash audit:

```bash
cd /data/user/timematch
python - <<'PY'
import csv, hashlib, pathlib, torch
inv = pathlib.Path("logs/v28_cleaned_full12_20260710_120037/source_checkpoint_inventory.tsv")
rows = list(csv.DictReader(inv.open(), delimiter="\t"))
for row in rows[:]:
    p = pathlib.Path(row["checkpoint_path"])
    if not p.exists():
        print("MISSING", row["source_domain"], row["seed"], row["source_config"], p)
        continue
    data = p.read_bytes()
    file_sha = hashlib.sha256(data).hexdigest()
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    print(row["source_domain"], row["seed"], row["source_config"], file_sha, h.hexdigest(), p)
PY
```

Task-specific old-source deterministic smoke is not run in this local audit because the server data/checkpoint environment is not locally available.

---

## 10. Old Logs / Old Aggregation

Local search did not find:

```text
v275_closedset_baseline_v275_12tasks_3seeds_20260616_155121
v276_smoothed_lambda12_half_20260619_215420
```

Therefore:

```text
old base = 0.6278
old smooth = 0.6525
```

cannot currently be independently recomputed from raw old logs in this workspace.

Cleaned aggregation is verified from `tools/summarize_v28_cleaned_full12.py`:

```text
source-on-target: last test F1 from source eval log
DA test: last test F1 from DA log
DA val: max validation F1 in DA log
summary_by_config: equal mean over successful task × seed rows
```

Need old raw logs to confirm whether old aggregation used the same final-test/student/equal-row mean convention.

---

## 11. Confirmed / Not Confirmed

Confirmed:

```text
1. cleaned source and DA parameters match intended v2.8 reproduction settings;
2. old v275 baseline launcher likely came from f04e1e0;
3. old v276 smooth launcher likely came from 89d9df4;
4. f04e1e0 -> 89d9df4 did not change timematch.py;
5. cleaned TimeMatch is a refactor with added diagnostics and shift policies;
6. the most concrete DA formula difference found so far is shift scoring epsilon 1e-5 -> 1e-12;
7. cleaned smooth_k3 source loss appears formula-equivalent to old v276 smooth_k3 for kernel=3;
8. local workspace does not contain old raw logs/checkpoints needed to recompute old means.
```

Not confirmed:

```text
1. source checkpoint reuse is causal;
2. old task-specific source training produced different weights;
3. old 0.6278 and 0.6525 are the same aggregation口径 as cleaned summary;
4. the shift epsilon change actually changes selected shifts;
5. old vs cleaned DA are numerically equivalent on the same checkpoint.
```

---

## 12. Minimal Next Experiment

Do not run full12. First recover or reconstruct the smallest causal comparison.

Recommended order:

1. Server-side hash audit of cleaned checkpoints.
2. Server-side synthetic old-vs-cleaned smooth loss numerical comparison.
3. Recover old logs/checkpoints if still on server.
4. If old checkpoints exist, run:

```text
tasks:
  AT1->FR2
  FR2->FR1

methods:
  base
  smooth_k3

seed:
  1

cells:
  same source checkpoint + old DA
  same source checkpoint + cleaned DA
  old source checkpoint + same DA
  cleaned source checkpoint + same DA
```

Interpretation:

```text
same source checkpoint, old DA != cleaned DA:
  cause is DA implementation/config/evaluation.

same DA, old source != cleaned source:
  cause is source checkpoint/source training.

only smooth changes:
  smooth-specific source representation or smooth loss compatibility.

base and smooth move together:
  general DA/evaluation behavior.
```

Do not expand beyond this until old logs/checkpoints or same-checkpoint DA comparisons identify where equivalence first breaks.
