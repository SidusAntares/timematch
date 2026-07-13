# v3.2.1 Ablation Plan

Run only after local-shift training is implemented.

## Core Comparisons

| setting | purpose |
|---|---|
| cleaned base TimeMatch | same-code clean reference |
| cleaned smooth source + base TimeMatch | source anchor/reference effect only |
| source reference + global_only | loads reference and logs partition/alignment but uses scalar shift positions |
| source reference + residual local_shift | first active v3.2.1 local positions |

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

## Stage 3d Stop Condition

Stage 3d met the stop condition for residual local shift.

Results:

```text
AT1->DK1:
  global_only test = 0.7874
  best residual test = 0.7782
  best residual - global_only = -0.0092

FR2->FR1:
  global_only test = 0.7200
  best residual test = 0.7120
  best residual - global_only = -0.0080
```

All tested residual variants underperformed the corrected `global_only`
control, and residual runtime remained about 2.7-3.0x slower.

Decision:

```text
Do not proceed to full12 for v3.2.1 residual local shift.
Do not tune more residual local-shift hyperparameters.
Archive v3.2.1 residual local shift as a negative boundary experiment.
```

## Stage 2 Scope

Stage 2 only proves the runtime training loop exists:

```text
source reference loads;
target H(t) is extracted;
target stages are partitioned;
soft alignment produces source-stage centers;
local positions are generated;
target forward_from_temporal_features runs;
source CE + pseudo target CE backward succeeds;
EMA updates;
compact TSV logs are written.
```

It does not claim performance equivalence with older baselines. Formal
comparisons must rerun all settings under the same cleaned code.
