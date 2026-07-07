# v3.1 Archive

v3.1 implemented feature-change segmentation with shift-aware class-stage prototype contrast.

Its alignment matrix `A` was used directly by a stage contrast loss.  Experiments showed weak or unstable benefit, high runtime cost, and too much coupling with the active TimeMatch loop.  It is therefore archived as a negative/weak-result branch, not an active method.

The v3.2.1 direction keeps the useful idea of class-stage correspondence, but changes its role:

```text
v3.1:
  A -> stage contrast loss

v3.2.1:
  A -> stage-wise local shift estimate -> adjusted target positions -> TimeMatch pseudo-label loss
```

Archived code:

```text
methods/legacy/adaptive_stage_contrast_v31.py
analysis/archive/v31/
launchers/archive/v31/
```

No active `timematch.py` or active `train.py timematch` parser should import or expose v3.1 stage contrast.
