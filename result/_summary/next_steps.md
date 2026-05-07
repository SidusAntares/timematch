# Project Next Steps

## Status Note (2026-05-07)

This file is now aligned with the current source-structure mainline.

Earlier versions of this note mixed together:

- old open-set / early closed-set baseline discussions
- source-target structure gap motivation
- prototype / clustering / target-side alignment ideas
- pre-implementation guesses about what the next mainline should be

Those ideas were not all wrong, but many of them have now either:

- been tested and found not to be the right mainline
- been partially absorbed into the current source-only structure design
- or been postponed to a later, more source-sensitive weighting stage

The current project status should therefore be read from the perspective of:

> **source-only temporal structure shaping for time-series UDA**

That is:

- structure loss is defined only on the source side
- target is not used to define the structure loss itself
- TimeMatch still handles adaptation
- the structure module is used to shape a better source initialization

---

## 1. Current Best Mainline

### Official best-performing mainline

Current best full 12-task result remains:

- **`v2.2.3 = 0.6761`**

using the `v222` hyperparameter set:

- `source_feature_reshaper = residual_temporal_conv`
- `source_feature_reshaper_strength = 0.10`
- `source_feature_dual_relation_trade_off = 0.03`
- `source_feature_dual_cls_trade_off = 1.00`
- `source_feature_reshaper_reg_trade_off = 0.05`
- `kernel_size = 3`

This is still the official performance reference line.

---

## 2. What Has Already Been Considered And Slowly Implemented

Several ideas that were previously only hypotheses have now already been tested in concrete versions.

### 2.1 Source-only structure shaping
This is no longer just a motivation; it has become the actual mainline.

What has already been established:

- source-side structure regularization is useful
- source-only structure shaping + ordinary TimeMatch is a valid training pipeline
- target should not be directly used to define the structure loss

### 2.2 Phase unit redesign
The earlier concern that the original phase split was too crude has already been acted on.

This was implemented in:

- **`v2.3.1`**

What changed:

- no longer uniform split by sorted index
- changed to source-domain shared `DOY-gap-aware` phase partition
- phase count became adaptive
- phase units became more faithful to real temporal support

Conclusion:

- this idea was reasonable and important
- but phase correction alone was **not sufficient** to improve final UDA performance

### 2.3 Multi-component structure loss
The idea that structure should not be only one compactness term has also already been tested.

This was implemented in:

- **`v2.3.2`**

What changed:

- tried to organize structure as multiple components:
  - `intra`
  - `amplitude`
  - `inter-phase`

Conclusion:

- this idea was also reasonable in motivation
- but directly turning correlated indicators into penalties was **not effective**
- performance became worse

### 2.4 More conservative temporal-structure regularization
After `v2.3.2` failed, a more conservative profile-based rewrite was tested.

This was:

- **`v2.3.3`**

Conclusion:

- the direction was healthier than `v2.3.2`
- but still too conservative
- it partially protected the original source structure instead of sufficiently reshaping it

### 2.5 Trend-residual restructuring
The next major correction was to move away from direct `amplitude/inter-phase` penalties and keep only:

- `residual / compactness` as the main term
- `trend` as a weak regularizer

This became:

- **`v2.3.4`**

Current judgment:

- this is the strongest post-`v2.2.3` direction so far
- full 12-task result reached:
  - **`v2.3.4 = 0.6710`**
- this is clearly better than `v2.3.1` and `v2.3.2`
- and only slightly below `v2.2.3`

So this idea has also already moved from “reasonable thought” to “partially validated implementation”.

### 2.6 Seasonal pattern regularization
The next step after `v2.3.4` is no longer to shrink seasonal structure, but to regulate its **pattern**.

This has now been implemented as:

- **`v2.3.5`**

Its current design adds a weak source-only seasonal pattern term on top of:

- `residual`
- `trend`

Status:

- implementation completed
- full 12-task run should be treated as the next formal verification

---

## 3. What We Have Learned So Far

The following judgments are now much clearer than before.

### 3.1 Structure loss should not be built directly from correlation tables
Many indicators found in analysis are useful as:

- descriptors
- explanatory signals
- candidate weighting signals

but not automatically as:

- direct optimization penalties

In particular:

- correlated structure indicators are not automatically good loss terms
- “related to better transfer” does not imply “should be pushed smaller”

### 3.2 Pattern regularization is better than magnitude penalty
This has become a core design principle.

Bad direction:

- penalize `amplitude` magnitude directly
- penalize `inter-phase jump` magnitude directly

Better direction:

- constrain temporal pattern regularity
- reduce structural disorder
- reduce redundancy
- preserve discriminative temporal organization

In short:

> **good temporal structure regularization should constrain pattern, not simply shrink magnitude**

### 3.3 Trend / seasonal / residual is a better semantic basis than intra / amplitude / inter
After reading decomposition papers and comparing our experiments, the cleaner decomposition view is now:

- `residual / noise`
- `trend`
- `seasonal / phenology-like variation`

This is more semantically stable than:

- `intra`
- `amplitude`
- `inter-phase`

because the latter are closer to measurement views, while the former are closer to structural components.

### 3.4 Source-only remains the correct constraint boundary
Even when borrowing ideas from:

- Barlow Twins
- multi-view causal regularization
- self-supervised structural losses

the rule remains:

- only borrow the **form of pattern regularization**
- do **not** turn the method into source-target structure alignment

This boundary should remain explicit.

---

## 4. Current Version Line

### `v2.2.3`
- best-performing official reference
- single compactness-style source structure loss

### `v2.3.1`
- phase mechanism fixed
- `DOY-gap-aware`
- adaptive phase count
- result worse than `v2.2.3`

### `v2.3.2`
- direct multi-component penalties
- failed

### `v2.3.3`
- profile consistency quickcheck
- healthier than `v2.3.2`
- still too conservative

### `v2.3.4`
- `residual + trend`
- best post-`v2.2.3` source-structure line so far

### `v2.3.5`
- `residual + trend + source-only seasonal-pattern`
- current active next verification line

---

## 5. What Still Needs To Be Solved

Although many earlier reasonable ideas have already been tested, several issues remain unresolved.

### 5.1 Phase is improved, but may still not be the final temporal unit
`DOY-gap-aware` phase partition is already much better than uniform split, but it is still a hard segmentation.

Open question:

- should the final temporal unit remain `phase`
- or should it move toward a more general local temporal unit

Possible direction:

- **coherent temporal segment**

This would be more general than a remote-sensing-specific phase notion.

### 5.2 Sliding-window local structure is still worth exploring
The current phase design is still segment-based.

A likely next structural improvement is:

- local sliding-window structure analysis

Why it still matters:

- it captures local continuity better
- it is less dependent on hard phase boundaries
- it is more general across domains such as:
  - remote sensing
  - medicine
  - weather
  - traffic

This has not yet been fully implemented as a mainline structure unit.

### 5.3 Intra-segment and inter-segment losses should likely be different
This is now an important design insight.

Inside one coherent temporal segment, it is reasonable to constrain:

- compactness
- local smoothness
- residual suppression

Across two different segments, it is **not** reasonable to use the same contraction logic.

Instead, inter-segment losses should focus on:

- transition regularity
- structural organization
- boundary consistency

not simple shrinkage.

This distinction has been conceptually identified, but has not yet been fully formalized as the next mainline loss family.

### 5.4 Seasonal structure still needs a correct role
Current judgment:

- seasonal structure should **not** be directly minimized
- seasonal magnitude itself is not the target
- seasonal pattern is the meaningful object

Open question:

- should seasonal pattern be used as:
  - a weak regularizer
  - a phase/component weighting signal
  - or a later source-sensitive gating signal

`v2.3.5` is the first concrete attempt along this line.

---

## 6. Generic Time-Series UDA Goal

The long-term goal is no longer just “remote-sensing phenology adaptation”.

The desired scope is:

> **generic time-series unsupervised domain adaptation**

This means the method should ideally be reusable on:

- remote sensing
- medical time-series classification
- weather-related time-series classification
- traffic time-series classification

This has two consequences.

### 6.1 What is already relatively generic
The following structure ideas are already close to being general:

- source-only structure shaping
- residual/noise suppression
- trend regularization
- source-internal seasonal pattern regularization

### 6.2 What is still domain-flavored
The following part is still more remote-sensing-flavored:

- the current `DOY-gap-aware` partition thresholds
- absolute gap/span settings such as `45` / `120`

Therefore, a later generalization step should reframe phase construction as:

- time-gap-aware segmentation
- scale-normalized temporal segmentation
- or local coherent temporal windows / segments

rather than remote-sensing-specific `DOY` heuristics.

---

## 7. Immediate Next Formal Check

The next formal step is:

- finish validating **`v2.3.5`** on the full 12 tasks

Main question:

> can a weak source-only seasonal-pattern term improve over `v2.3.4` without reintroducing the failure mode of `v2.3.2`

If `v2.3.5` works, then the next stage can move toward:

- more explicit source-sensitive component weighting
- but still source-only

If `v2.3.5` fails, then the next redesign should focus on:

- coherent temporal segment formulation
- intra-segment vs inter-segment loss separation
- window-based local structural constraints

---

## 8. Current Short Summary

Many previously recorded “reasonable ideas” have **already** been gradually considered and partially implemented.

In particular, we have already gone through:

- source-only structure shaping
- adaptive phase redesign
- multi-component direct penalties
- conservative profile regularization
- trend-residual restructuring
- source-only seasonal pattern regularization

So the project is no longer at the stage of “what if we try structure”.

The current stage is:

> **how to organize source-only temporal structure loss so that it is semantically correct, optimization-friendly, and eventually generalizable beyond remote sensing**

That is now the real mainline.
