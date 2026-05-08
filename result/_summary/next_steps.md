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
- full 12-task verification completed
- result shows this line is informative, but not the next mainline

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
- full run completed
- useful as a negative-but-informative structural attempt

### `v2.4.0`
- `phase` was lifted into a more general `segment` abstraction
- kept the `v2.3.4` loss unchanged
- first framework-level refactor toward segment-aware structure loss

### `v2.4.1`
- first explicit weak `inter-segment` transition regularization
- improved over `v2.4.0`
- confirmed that segment-aware intra/inter split is a valid mainline direction

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

## 7. Current Transition Point

The project is no longer waiting for `v2.3.5` verification.

What is now established:

- `v2.3.5` full run completed and did **not** outperform `v2.3.4`
- `v2.4.0` confirmed that `phase -> segment` abstraction can be introduced without breaking the mainline
- `v2.4.1` confirmed that a weak `inter-segment` transition term can improve over `v2.4.0`

So the immediate question is no longer:

- whether seasonal pattern can simply be added on top of `v2.3.4`

The immediate question is now:

> **how to improve the temporal unit itself and how to make segment boundaries more semantically meaningful**

This leads directly to:

- `v2.4.2`: semantic segment refinement
- `v2.4.3`: boundary-centered sliding-window inter-segment modeling
- `v2.5`: source-sensitive weighting / gating

---

## 8. Confirmed Next Route

The later route is now much clearer and can be explicitly separated into three lines.

### 8.1 `v2.4.2`: semantic segment refinement line

After `v2.4.0` and `v2.4.1`, the next structural line should first improve the quality of the segment unit itself.

Current judgment:

- the current `segment` is still largely geometry-driven
- it uses time support structure better than old phase splitting
- but it still does not truly group **semantically similar local regimes**

So `v2.4.2` should focus on:

- semantic segment refinement
- source-driven boundary detection
- possibly coarse/fine segment hierarchy later
- while keeping the main loss framework relatively stable

Recommended progression:

- `v2.4.2a`
  - semantic enhancement of current `DOY-gap-aware` segmentation
  - keep loss nearly unchanged
- `v2.4.2b`
  - test additional local semantic signals:
    - slope
    - curvature
    - local variance
    - local autocorrelation / periodic cues
- `v2.4.2c`
  - if useful, consider coarse/fine two-level segment views

The main conceptual shift is:

> **before making inter-segment windows more complex, first make the segment unit itself more semantically correct.**

Relevant paper inspiration for this line:

- **AMD / Adaptive multi-scale decomposition**
  - explicit multi-scale decomposition
  - coarse-to-fine structural organization
- **MSGNet**
  - scale discovery from periodic / frequency cues
  - scale importance from amplitude / energy

These are useful not because we do forecasting, but because they support:

- multi-scale segment design
- semantically informed temporal partitioning

### 8.2 `v2.4.3`: boundary-centered sliding-window line

Only after segment quality is improved should we introduce sliding windows for **inter-segment** modeling.

Current judgment:

- sliding windows should **not** replace the segment unit everywhere
- segment should remain the main unit for intra-segment structure
- windows should be used specifically around **segment boundaries**

So `v2.4.3` should focus on:

- boundary-centered sliding windows
- local transition modeling
- optional keypoint / turning-point emphasis
- later multi-scale windows

Recommended progression:

- `v2.4.3a`
  - fixed boundary-centered window
  - simple local encoder
- `v2.4.3b`
  - keypoint-aware boundary window
  - emphasize local extrema / turning points / strong transition cues
- `v2.4.3c`
  - multi-scale boundary windows
  - different window lengths around the same boundary

Relevant paper inspiration for this line:

- **TRNN**
  - sliding window is useful
  - but local structure should emphasize turning points, not all points equally
- **multi-head CNN / XAI window paper**
  - fixed look-back windows are a strong practical baseline
  - local conv encoding is a stable first choice
- **Pathformer**
  - multi-scale patch/window idea
  - supports later moving from one fixed window to multiple candidate windows
- **TT-ConvLSTM**
  - supports the idea that local detail and global structure can be handled by different pathways

This line should therefore be understood as:

> **window-based local structure for inter-segment boundaries, not full replacement of semantic segments.**

### 8.3 `v2.5`: source-sensitive weighting / gating

This should be treated as a new stage after the segment framework is stable.

Its goal is:

> **given different source-domain structures, adaptively determine which structure-loss parts should be stronger and which should be weaker.**

This is the natural place for:

- `moe`
- gating
- source-conditioned component weighting

What this stage should answer:

- which source domains need stronger residual suppression
- which source domains need stronger trend regularization
- which source domains should weaken inter-segment constraints
- whether different source domains should use different structural priorities

This should belong to:

- **`v2.5`**

and not be mixed into `v2.4`, because `v2.4` is still solving:

- the temporal unit itself
- the intra/inter structural organization itself

Relevant paper inspiration for this line:

- **Pathformer**
  - adaptive pathway selection
  - multi-scale router / top-k scale choice
- **MSGNet**
  - scale weighting by amplitude / importance

These are the most natural conceptual sources for later:

- `moe`
- gating
- source-conditioned structural weighting

### 8.4 Reshaper backbone line

The current reshaper remains:

- convolutional

This is acceptable for now because the main bottleneck is still:

- temporal unit design
- loss organization

not yet the reshaper backbone itself.

However, a later comparison line should still be recorded:

- convolution reshaper
- transformer reshaper
- Mamba reshaper

Current judgment:

- transformer and Mamba are both worth considering later
- but they should not block the current `v2.4` line
- there is not yet a sufficiently general reason to assume Mamba is automatically better than transformer here

Therefore:

> backbone replacement should be treated as a later comparison / extension line,  
> not as the current mainline bottleneck.

---

## 9. Current Short Summary

Many previously recorded “reasonable ideas” have **already** been gradually considered and partially implemented.

In particular, we have already gone through:

- source-only structure shaping
- adaptive phase redesign
- multi-component direct penalties
- conservative profile regularization
- trend-residual restructuring
- source-only seasonal pattern regularization
- segment abstraction
- first explicit inter-segment transition regularization

So the project is no longer at the stage of “what if we try structure”.

The current stage is:

> **how to organize source-only temporal structure loss so that it is semantically correct, optimization-friendly, and eventually generalizable beyond remote sensing**

That is now the real mainline.
