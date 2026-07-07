# v3.2.1 Cleanup Report

## Repository State

```text
branch: main
head: 63fde5a
origin: https://github.com/SidusAntares/timematch.git
upstream: https://github.com/jnyborg/timematch.git
```

Recent commits:

```text
63fde5a Clean generated artifacts from tracked files
38ae949 Ignore generated experiment artifacts
032bc18 v3.1.0优化
b46d17d v3.1.0优化
75de37a v3.1.0
```

## Cleanup Scope

The active repository has been reorganized around three layers:

```text
methods/timematch_base/
methods/source_structure/
methods/local_shift/
```

Legacy experimental code is archived under:

```text
methods/legacy/
analysis/archive/
launchers/archive/
```

## Active Entries

```text
timematch.py
methods/timematch_base/train_loop.py
methods/source_structure/losses.py
methods/source_structure/train_source_structure.py
methods/local_shift/
configs/
tools/
```

`timematch.py` is now a compatibility wrapper:

```python
from methods.timematch_base.train_loop import train_timematch
```

## Archived Entries

```text
ideas/adaptive_stage_contrast.py
  -> methods/legacy/adaptive_stage_contrast_v31.py

ideas/source_phase_compactness.py
  -> methods/legacy/old_source_phase_compactness.py

ideas/source_raw_compactness.py
  -> methods/source_structure/losses.py

ideas/train_source_phase_compactness.py
  -> methods/source_structure/train_source_structure.py

analysis/v31*
  -> analysis/archive/v31/

analysis/v30*
  -> analysis/archive/v30x/

analysis/v29*
  -> analysis/archive/v29x/

analysis/v28*
  -> analysis/archive/v28x/

older v2.x one-off analysis scripts
  -> analysis/archive/older/

v31/v30x/v29x/v28x launchers
  -> launchers/archive/
```

## v3.1 Status

v3.1 is archived.  Its stage correspondence matrix was used as a contrastive loss and is not part of the active method path.

The active v3.2.1 plan keeps stage correspondence only as a local shift estimator:

```text
source-target stage correspondence
-> stage-wise residual shift
-> adjusted positions
-> TimeMatch pseudo-label objective
```

## v3.2.1 Skeleton

Reserved implementation hooks:

```text
methods/local_shift/source_reference.py
methods/local_shift/target_partition.py
methods/local_shift/soft_alignment.py
methods/local_shift/local_position.py
methods/local_shift/train_local_shift.py
```

Supporting tools:

```text
tools/build_source_stage_reference.py
tools/summarize_local_shift_logs.py
```

Shape test:

```text
analysis/test_v321_local_shift_shapes.py
```

## Artifact Tracking

`.gitignore` now ignores:

```text
logs/
outputs/
runs/
result/
temp_docs/
*.tsv
*.csv
*.jsonl
*.pkl
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.tar.gz
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.idea/
```

Tracked generated experiment summaries under `result/` and `temp_docs/` were removed from the Git index with `git rm --cached`; local files were not deleted.

## Remaining TODO

```text
1. Expose temporal features before pooling in the encoder.
2. Implement source checkpoint/data loading in tools/build_source_stage_reference.py.
3. Wire a new timematch_local_shift entry point only after shape and runtime checks.
4. Move helper bodies from methods/timematch_base/train_loop.py into split modules when safe.
5. Keep v3.1 stage contrast out of the active parser and active training loop.
```

## Verification

```text
py_compile: passed with PYTHONPYCACHEPREFIX redirected to a temp directory
shape test: not executed locally because the active local Python has no torch module
```
