# v3.2.1 Local Shift Design

## Goal

Implement a cleaner successor to v3.1:

```text
structure-guided stage-wise local shift
```

The method should not add another stage contrast loss in the first version.  It should use source-side structure to build stable class-stage references, then use soft stage correspondence to estimate local target time shifts that enter the existing TimeMatch position pathway.

## Data Flow

```text
source structure checkpoint
-> source class-stage reference
-> target feature-change partition
-> soft source-target stage alignment
-> stage-wise residual local shift
-> adjusted target positions
-> base TimeMatch pseudo-label training
```

## Active Hook Locations

```text
methods/local_shift/source_reference.py
methods/local_shift/target_partition.py
methods/local_shift/soft_alignment.py
methods/local_shift/local_position.py
methods/local_shift/train_local_shift.py
```

## Design Constraints

```text
Do not use target true labels.
Do not use fixed uniform stages as the method definition.
Do not introduce a stage contrast loss in the first v3.2.1 implementation.
Do not add more v3.1 conditionals into timematch.py.
Keep base TimeMatch import-compatible through timematch.py.
```

## Current Status

This commit only reserves interfaces and shape-checked utilities.  It does not activate a new training method.

## Stage 3d Outcome

Stage 3c fixed the equivalence problem and made `global_only` a valid control:

```text
base_equiv ~= smooth_base
global_forward == base_equiv
global_only == global_forward
```

Stage 3d then tested safer residual local-shift variants:

```text
raw
zero_mean
scaled_zero_mean
gated_scaled_zero_mean
```

All residual variants failed to beat `global_only`:

```text
AT1->DK1:
  global_only test = 0.7874
  best residual test = 0.7782

FR2->FR1:
  global_only test = 0.7200
  best residual test = 0.7120
```

Direct position residuals from stage correspondence are too disruptive for the
current TimeMatch training path.  This method should be treated as an archived
negative boundary, not as the active v3 main method.

The source reference, target partition, soft alignment, and local-position
utilities remain useful diagnostic components.

Status: archived as negative boundary; not an active method.
