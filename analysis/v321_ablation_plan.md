# v3.2.1 Ablation Plan

Run only after local-shift training is implemented.

## Core Comparisons

| setting | purpose |
|---|---|
| base TimeMatch | clean reference |
| smooth source structure + base TimeMatch | source anchor/reference effect only |
| source reference + global shift | verify reference extraction without local shift |
| source reference + stage-wise local shift | first active v3.2.1 |
| local shift with detached correspondence | check whether correspondence only changes positions |

## Diagnostics

Record compact TSV only:

```text
task
seed
source_config
da_config
global_shift
local_shift_mean
local_shift_std
local_shift_clip_fraction
alignment_entropy
alignment_top1_mass
target_pseudo_confidence
target_pseudo_ratio
source_on_target_f1
da_f1
```

## Stop Conditions

Stop and debug before expanding if:

```text
local_shift_clip_fraction is high;
alignment entropy collapses at epoch 1;
runtime is more than 1.5x base TimeMatch;
DA F1 drops on most probe tasks while pseudo confidence rises.
```
