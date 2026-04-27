# Phase-Aware PSE Structure Alignment v4

## Why this v4 exists

The current project has reached a stable negative result on several prototype-only variants:

- `RGPA-v1`: `0.5999`
- `ProtoCon-pure-v1`: `0.5996`
- `MemProto-v1`: `0.5989`

while the closed-set baseline on `FR1 -> DK1` remains:

- `0.6391047`

So the next method should not continue to treat structure as only:

- one pooled sample representation
- one static class prototype
- or one global source-target shift

The `v3` temporal analysis now provides a better direction.

---

## Key evidence from v3

Strongest correlations with `target_f1`:

- `pse_trend_curve_distance`: `-0.6997`
- `pse_early_curve_distance`: `-0.6855`
- `pse_mid_curve_distance`: `-0.6845`
- `pse_temporal_curve_distance`: `-0.6573`
- `prototype_distance`: `-0.6518`
- `pse_frequency_magnitude_distance`: `-0.6393`

This means:

1. encoded temporal structure is more informative than raw curve geometry
2. early and mid season encoded structure matter more than late season
3. low-frequency semantic trend is the strongest current temporal signal

So the next method should be built around:

> **phase-aware encoded temporal structure**

rather than raw-curve alignment or more static prototype tweaking.

---

## Core idea

Instead of aligning one prototype per class, build and align:

- **phase-specific encoded features**
- **class-phase prototypes**
- and optionally **phase-level trend summaries**

The fundamental unit becomes:

- `sample phase -> class phase`

rather than:

- `sample -> class`

---

## v4 method definition

### Name

**Phase-Aware PSE Structure Alignment**

Short name:

- `PPSA-v4`

### Structure carrier

Use per-time-step features from the `PSE` stage or immediately after the temporal encoder input stage, before final whole-sequence pooling collapses temporal structure.

### Phase partition

Start with a simple 3-phase partition:

- early
- mid
- late

The first version should use fixed equal-length windows in encoded time index, not learned phase discovery.

Reason:

- it is easy to implement
- it matches the `v3` evidence that early/mid are already informative
- it gives a clean first test before adding clustering or learned phase boundaries

### Per-sample phase features

For each sample:

1. split the encoded temporal sequence into 3 windows
2. pool each window separately
3. obtain:
   - `z_early`
   - `z_mid`
   - `z_late`

Optional trend summary:

- `z_trend = z_late - z_early`

This keeps the method close to the strongest `v3` signal without requiring explicit ODE-style modeling.

---

## Target structure construction

### Source side

Use ground-truth labels to build:

- source class-phase prototypes

For each class `c` and phase `p`:

- `P_s[c, p]`

### Target side

Use high-confidence pseudo labels to build:

- target class-phase prototypes

For each class `c` and phase `p`:

- `P_t[c, p]`

Use EMA / momentum update if needed, but this should be optional in the first version.

### Shared semantic unit

The alignment object is not a single class center anymore.

It is:

- `class c in phase p`

This is the main conceptual change from the earlier prototype lines.

---

## Loss design

### 1. Base TimeMatch losses

Keep the original:

- source supervised loss
- target pseudo-label loss

### 2. Phase prototype alignment loss

For each shared class-phase prototype:

- align `P_s[c, p]` and `P_t[c, p]`

Simple first form:

- L2 or cosine distance

This gives:

- `L_phase_proto`

### 3. Phase sample-to-prototype loss

For each target or source sample phase feature:

- pull it toward its own class-phase prototype
- push it away from other class-phase prototypes

This gives:

- `L_phase_contrast`

The contrastive objective should use:

- positives: same class, same phase
- negatives: different class and optionally same class but different phase

That second option is important, because it teaches:

- same crop at different seasonal phases should not collapse completely

### 4. Trend structure loss

Use the per-class trend summary:

- `T_s[c] = P_s[c, late] - P_s[c, early]`
- `T_t[c] = P_t[c, late] - P_t[c, early]`

Then align:

- `T_s[c]`
- `T_t[c]`

This gives:

- `L_trend`

This term is motivated directly by the strongest `v3` metric:

- `pse_trend_curve_distance`

### First total loss

A clean first version is:

```text
L = L_source + lambda_t * L_target + lambda_p * L_phase_proto
  + lambda_c * L_phase_contrast + lambda_tr * L_trend
```

---

## What this v4 is trying to fix

Earlier methods assumed that the main semantic structure problem was:

- noisy target prototype
- or weak discriminative pressure around a static prototype

But `v3` suggests the bigger issue is:

- the **temporal semantic path** of a class across the season

So this v4 tries to fix:

1. structure collapse caused by whole-sequence pooling
2. loss of early/mid seasonal semantics
3. treating one class as one static point instead of one evolving trajectory

---

## Why this is better than the previous lines

### Compared with RGPA-v1

- RGPA-v1 tried to remove unstable temporal positions
- `PPSA-v4` instead keeps temporal structure and organizes it into phase-aware semantic units

### Compared with ProtoCon-pure-v1

- ProtoCon used one class prototype
- `PPSA-v4` uses one prototype per class-phase pair

### Compared with MemProto-v1

- MemProto stabilized the prototype memory
- `PPSA-v4` changes the structure definition itself

So this is not just a better bank or better contrastive loss.

It is a different structural hypothesis.

---

## Recommended first implementation scope

To keep risk controlled, the first version should be minimal:

1. fixed 3-way phase split
2. per-phase pooled PSE features
3. class-phase prototype alignment
4. optional trend alignment
5. no clustering-based phase discovery yet
6. no ODE-based trend model yet
7. no heavy joint completion module yet

This is important because the point of `v4` is to test:

> whether the **right structural unit** is class-phase encoded semantics

not to test every advanced mechanism at once.

---

## Possible experiment names

Suggested names:

- `timematch_FR1_to_DK1_ppsa_v4`
- `timematch_FR1_to_DK1_phasepse_v4`
- `timematch_FR1_to_DK1_phase_trend_v4`

Preferred:

- `timematch_FR1_to_DK1_phasepse_v4`

---

## What success would mean

The first success criterion is not beating every baseline immediately.

It is:

1. the method trains stably
2. the new phase/trend losses are active and interpretable
3. validation improves over the previous `~0.599` prototype family

The second success criterion is:

- whether it closes a meaningful part of the gap toward `0.6391`

---

## If this v4 works

Then the next upgrades become natural:

1. replace fixed 3 phases with clustering-based or confidence-based phase discovery
2. replace simple trend difference with stronger decomposition
3. add phase-specific memory banks
4. add phase-aware target sample selection

---

## Final takeaway

> The next method should stop asking only  
> "which class center should this sample approach?"  
> and start asking  
> "which class-phase semantic state should this sample phase approach?"
