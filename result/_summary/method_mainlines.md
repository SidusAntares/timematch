# Candidate Method Mainlines for Structure-Based Time-Series UDA

## Status Note (2026-05-01)

This file is a **method-candidate history document**, not the current single-source-of-truth for implementation status.

Several parts below describe:

- earlier candidate rankings
- earlier prototype-centered priorities
- and the first adaptation-stage `srcphasecompact_p5` probe

Those sections are still useful historically, but the current status is now:

1. the project motivation has been corrected toward **source-domain self-structure**
2. the strongest source-side analysis signals are:
   - `phase compactness`
   - `phase margin`
3. the first adaptation-stage probe `srcphasecompact_p5` showed:
   - the idea is meaningful
   - but a global hard source compactness regularizer is too blunt
4. the cleaner current implementation direction is:
   - improve source-domain self-structure first
   - then initialize ordinary TimeMatch from that improved source model

So when this file conflicts with newer records, prefer:

- [structure_motivation_and_metrics.md](C:\Code\dev\PythonProject\timematch\result\_summary\structure_motivation_and_metrics.md)

for the latest analysis-based judgment.

## Why this document exists
- The current project has already shown that **prototype-level category structure** explains transfer performance better than global discrepancy metrics such as MMD/CORAL.
- Therefore, the next method should not be chosen by intuition alone, but by asking:
  - what structure should be aligned,
  - how target structure should be constructed,
  - and which design is most likely to produce a meaningful improvement over the current closed-set TimeMatch baseline.

- This document summarizes the **main candidate directions**, the **papers they borrow from**, and their **advantages / weaknesses / emphasis**.

## Current evidence that guides method choice
- `Prototype Distance` currently has the strongest correlation with target F1.
- `Relation Structure Distance` has some explanatory power, but is weaker than `Prototype Distance`.
- This suggests:
  1. **prototype quality and prototype stability are first-order issues**
  2. **relation / graph structure is a second-order issue**
  3. methods that directly improve target prototype construction are more promising than simply designing a stronger relation MSE

---

## Mainline 1: Prototype Contrastive Adaptation

### Core idea
- Use semantic prototypes as class anchors.
- Train source samples with supervised prototype contrastive learning.
- Train high-confidence target samples with pseudo-labeled prototype contrastive learning.
- Replace or weaken the current point/edge MSE-style alignment with a more discriminative prototype-centered contrastive loss.

### Main papers borrowed from
1. **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
2. **Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation**
3. **A Contrastive Representation Domain Adaptation Method for Industrial Time-Series Cross-Domain Prediction**

### What is borrowed from each paper
- **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
  - prototype as a semantic structure carrier
  - target-to-prototype soft alignment is more meaningful than pure global discrepancy matching
- **Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation**
  - sample-to-prototype / pixel-to-prototype contrastive alignment
  - positive = own class prototype, negative = all other class prototypes
- **A Contrastive Representation Domain Adaptation Method for Industrial Time-Series Cross-Domain Prediction**
  - local structure can be preserved and transferred with contrastive learning
  - structure alignment need not be expressed as relation matrix MSE

### Best fit for this project
- Strongly matches the current analysis result that `Prototype Distance` is the strongest signal.
- Avoids heavy graph modeling and is easier to implement on top of the current pipeline.
- More task-aligned than directly forcing relation matrices to match.

### Main strengths
- Clear semantic objective: same-class closer, different-class farther.
- Better discriminative structure than point/edge MSE.
- Computationally lighter than graph/GNN-based structure methods.
- Easy to explain and relatively easy to prototype.

### Main weaknesses
- Still depends on target structure quality.
- If pseudo labels are noisy, target-side contrastive terms may drift.
- May still need extra machinery for robust target prototype construction.

### Main emphasis
- **Semantic class structure**
- **Discriminative prototype alignment**
- **Replacing weak geometric MSE with stronger contrastive constraints**

### When to prioritize this line
- If the goal is to get a strong, implementable next method quickly.
- If the project wants a cleaner replacement for current PRA point/edge loss.

---

## Mainline 2: Refinement-Guided Prototype Adaptation

### Core idea
- Not all samples or time segments should contribute equally to prototype construction.
- First identify unstable / transition / uncertain temporal regions.
- Use those regions for harder representation learning or local consistency strengthening.
- Exclude or downweight those unstable regions when building prototypes.
- Only use stable segments to form reliable semantic prototypes for alignment.

### Main papers borrowed from
1. **Liu, BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation**
2. **Xu, Cross-Domain Detection via Graph-Induced Prototype Alignment**
3. **Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery with Invariant Domain-Level Prototype Memory**

### What is borrowed from each paper
- **BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation**
  - key / difficult regions should be treated differently from stable regions
  - critical regions may be emphasized during training but excluded from prototype construction
- **Cross-Domain Detection via Graph-Induced Prototype Alignment**
  - refinement before alignment
  - improve the input representation used to build prototypes
- **Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery with Invariant Domain-Level Prototype Memory**
  - stable prototypes require smoothing, accumulation, and noise control

### Best fit for this project
- Highly suitable if the method is intended to emphasize **structure** rather than merely “using prototypes”.
- Particularly promising for time series because temporal transitions, phase shifts, and unstable periods are natural analogues to image boundaries.
- Gives a clearer “new idea” than simply replacing one loss with another.

### Main strengths
- Strongest “own method” flavor among current candidates.
- Directly attacks the real bottleneck suggested by current analysis: prototype quality.
- Natural extension from spatial boundary thinking to temporal boundary / transition thinking.

### Main weaknesses
- Requires defining temporal boundaries / unstable segments in a principled way.
- Slightly more complex to implement and justify than straightforward contrastive prototype adaptation.
- Risk of overcomplicating the design if unstable-region detection is weak.

### Main emphasis
- **Prototype quality before alignment**
- **Different treatment for unstable and stable temporal regions**
- **Structure-aware data selection**

### When to prioritize this line
- If the project wants a more original structure-driven method.
- If the goal is not only to improve performance, but also to make a clearer methodological contribution.

---

## Mainline 3: Memory-Based Invariant Prototype Adaptation

### Core idea
- Use a momentum-updated prototype memory instead of relying on batch-only prototypes.
- Maintain stable semantic centers across batches.
- Update the memory using source labels and high-confidence target information.
- Align source/target features to a shared or invariant prototype memory.

### Main papers borrowed from
1. **Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery with Invariant Domain-Level Prototype Memory**
2. **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
3. **Xu, Cross-Domain Detection via Graph-Induced Prototype Alignment**

### What is borrowed from each paper
- **Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery with Invariant Domain-Level Prototype Memory**
  - momentum memory update
  - high-confidence target filtering
  - domain-invariant semantic memory center
- **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
  - prototypes as alignment anchors
  - target structure may be softly aligned rather than hard-assigned
- **Cross-Domain Detection via Graph-Induced Prototype Alignment**
  - prototype quality matters more than raw alignment mechanics

### Best fit for this project
- Very suitable for remote sensing and time-series data where classes can be imbalanced and batch statistics are unstable.
- Engineering-wise, it is close to the current bank-based prototype thinking, so it is lower-risk to implement.

### Main strengths
- Strong mechanism for suppressing target noise.
- Handles rare classes and batch imbalance better.
- Natural fit for closed-set remote sensing scenarios with unstable class presence across batches.

### Main weaknesses
- On its own, this may feel more like a stability upgrade than a new structural idea.
- If not combined with a stronger alignment objective, it may be seen as incremental.
- Needs careful design of shared vs separate memory, confidence filtering, and update timing.

### Main emphasis
- **Prototype stability**
- **Cross-batch semantic accumulation**
- **Noise suppression in target structure construction**

### When to prioritize this line
- If the goal is to build the most robust version of the current prototype track.
- If implementation risk needs to stay low.

---

## Mainline 4: Soft Prototype Transport / Assignment

### Core idea
- Do not necessarily build an explicit noisy target prototype.
- Use source semantic prototypes as anchors.
- Let target features align to source prototypes through soft transport or soft assignment.
- Avoid hard pseudo-label dependence at the earliest stage.

### Main papers borrowed from
1. **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
2. **A Contrastive Representation Domain Adaptation Method for Industrial Time-Series Cross-Domain Prediction**
3. **AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data**

### What is borrowed from each paper
- **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
  - target-to-prototype and prototype-to-target bidirectional soft alignment
  - soft conditional transport without explicit target prototype construction
- **A Contrastive Representation Domain Adaptation Method for Industrial Time-Series Cross-Domain Prediction**
  - local structure can guide alignment before hard semantic commitments are made
- **AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data**
  - strong DA results often come from better class-conditional alignment rather than overcomplicated modality-specific design

### Best fit for this project
- Attractive if pseudo labels are judged too noisy and clustering is still not trusted.
- A possible middle route between prototype methods and purely local-structure methods.

### Main strengths
- Avoids early hard target assignments.
- Provides a cleaner theoretical story than simple point MSE.
- Could reduce target prototype noise at the cost of more abstract alignment machinery.

### Main weaknesses
- More abstract and potentially harder to implement/debug than prototype contrastive.
- Less direct connection to the current analysis table than explicit prototype distance.
- May require more care in explaining why soft transport works for time-series semantics.

### Main emphasis
- **Soft semantic alignment**
- **Avoiding early hard target structure decisions**
- **Class-conditional transport**

### When to prioritize this line
- If the project wants a more elegant theory-driven route.
- If explicit target prototype construction keeps failing.

---

## Which line is currently the most promising?

### Recommended ranking
1. **Refinement-Guided Prototype Adaptation**
2. **Prototype Contrastive Adaptation**
3. **Memory-Based Invariant Prototype Adaptation**
4. **Soft Prototype Transport / Assignment**

### Why this ranking
- **Refinement-Guided Prototype Adaptation** is currently the most promising **research contribution** line:
  - it is closest to the current finding that prototype quality is the real bottleneck
  - it has stronger structural flavor than “just use prototype”
  - it gives a clearer time-series-specific story through temporal boundaries / transition regions

- **Prototype Contrastive Adaptation** is currently the most promising **practical next method**:
  - easiest to implement
  - strongest replacement for current point/edge MSE
  - tightly aligned with the evidence that prototype-level structure matters

- **Memory-Based Invariant Prototype Adaptation** is the most promising **stability upgrade**:
  - lower implementation risk
  - useful even if it becomes part of another mainline instead of standing alone

- **Soft Prototype Transport / Assignment** is the most promising **theory-driven backup direction**:
  - elegant
  - useful if explicit target prototype construction remains too noisy
  - but less directly actionable as the very next implementation

---

## Recommended reading priority for direct method design

### Must read carefully
1. **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**
2. **Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation**
3. **Xu, Cross-Domain Detection via Graph-Induced Prototype Alignment**
4. **Liu, BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation**
5. **Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery with Invariant Domain-Level Prototype Memory**

### Important supporting reading
6. **A Contrastive Representation Domain Adaptation Method for Industrial Time-Series Cross-Domain Prediction**
7. **AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data**

---

## Practical recommendation for the next implementation step

### If the goal is fastest useful progress
- Start with **Prototype Contrastive Adaptation**
- Because it gives the cleanest replacement for current point/edge alignment

### If the goal is strongest structural contribution
- Start with **Refinement-Guided Prototype Adaptation**
- Because it is the most likely to become a genuinely project-specific method rather than a prototype reimplementation

### If the goal is safest engineering path
- Start with **Memory-Based Invariant Prototype Adaptation**
- Because it is closest to the current bank-based idea and easiest to integrate incrementally

---

## Current final takeaway

> The project should not continue treating “structure” as a generic relation matrix problem.  
> The most promising next methods are the ones that treat **prototype construction quality** as the first problem, and only then perform prototype-level semantic alignment, relation modeling, or transport.
---

## Update After v3 Temporal Analysis

The `v3` analysis changes the next-method ranking in an important way.

Strongest newly observed correlations with `target_f1` are now:

- `pse_trend_curve_distance`: `-0.6997`
- `pse_early_curve_distance`: `-0.6855`
- `pse_mid_curve_distance`: `-0.6845`
- `pse_temporal_curve_distance`: `-0.6573`
- `prototype_distance`: `-0.6518`

This means:

1. encoded temporal structure is more informative than raw curve geometry
2. early / mid season encoded structure matters more than late-season structure
3. the next structural unit should likely become:
   - `class x phase`
   rather than only:
   - one static class prototype

So the most promising immediate mainline is now:

- **Phase-Aware PSE Structure Alignment**

This does not overturn the earlier prototype conclusion.

It refines it:

> prototypes still matter,  
> but the most useful prototype may be a **phase-specific encoded semantic state**  
> rather than only a single pooled class center

### Updated ranking for the next implementation

1. **Phase-Aware PSE Structure Alignment**
2. **Refinement-Guided Prototype Adaptation**
3. **Prototype Contrastive Adaptation**
4. **Memory-Based Invariant Prototype Adaptation**
5. **Soft Prototype Transport / Assignment**

### Why the ranking changed

- `Phase-Aware PSE Structure Alignment` is now first because:
  - it is directly supported by the newest analysis
  - it explains why prototype-only variants plateaued
  - it gives a more precise structural carrier than static prototype alignment

- `Refinement-Guided Prototype Adaptation` remains valuable, but should now be reframed around:
  - phase boundaries
  - encoded temporal segments
  - or phase-confidence filtering

- the earlier prototype lines still matter:
  - but more as components inside a phase-aware design
  - than as the full next method by themselves

---

## Retrospective Implementation Note: `srcphasecompact_p5`

This section records the **first implemented source-domain self-structure regularizer** that was actually run as:

- `timematch_*_closedset_noshift_srcphasecompact_p5`

This version should be treated as an **exploratory implementation record**, not the final recommended formulation.

### What this version was trying to test

The phase-metric analysis had already shown that:

- `source phase compactness` is the strongest source self-structure signal
- informative phases are not equally useful
- a source domain with tighter within-class phase structure is more likely to transfer well

So the first concrete question became:

> if we directly regularize source-domain phase-wise compactness during adaptation, can we improve downstream closed-set TimeMatch?

### Where the loss was attached in the framework

This first version was attached **inside the TimeMatch training stage**, not in source-only pretraining.

The logic was:

1. keep the original TimeMatch training objective
2. extract source-batch time-step features from the shared `PSE` / spatial encoder
3. compute a source-only phase compactness loss on those features
4. add that loss to the total training loss as a small auxiliary regularizer

So the total optimization target became:

```text
L_total = L_TimeMatch + L_src_phase_compactness
```

with:

```text
L_src_phase_compactness = lambda_compact * sum_k w_k * L_compact^(k)
```

This means:

- the additional supervision came only from **source labeled samples**
- but it was applied during the **domain adaptation stage**
- and gradients updated the same shared encoder used by TimeMatch

### Which features were used

The regularizer was built on the **PSE time-step feature curve**, not on:

- the pooled `LTAE` sequence embedding
- and not on the final decoder logits

For a source sample `i`, let:

```text
H_i = [h_i^(1), h_i^(2), ..., h_i^(T)]
```

where `h_i^(t)` is the `PSE` output at time step `t`.

This choice was made because the source self-structure analysis itself was phase-based and time-step-based, so the most faithful implementation should regularize the same structural carrier.

### How the phase representation was built

This first version used a **hard-coded uniform 5-phase split**.

If the sequence has `T` valid encoded time steps, split the time axis into:

- `p1`
- `p2`
- `p3`
- `p4`
- `p5`

with contiguous equal-length windows up to rounding.

For each sample `i` and phase `k`, compute the phase feature by averaging time-step PSE features inside that phase:

```text
z_i^(k) = mean_{t in T_k} h_i^(t)
```

where `T_k` is the set of time indices belonging to phase `k`.

### How the compactness loss was computed

For each phase `k` and class `c`, compute the phase-wise class center:

```text
mu_c^(k) = (1 / N_c) * sum_{i: y_i = c} z_i^(k)
```

Then define the phase-wise within-class compactness penalty:

```text
L_compact^(k) = (1 / N) * sum_i || z_i^(k) - mu_{y_i}^(k) ||_2^2
```

This loss becomes small when:

- same-class source samples form tighter clusters
- especially inside a specific temporal phase

### How the phase weights were chosen

This version did **not** learn phase weights.

Instead, phase weights were hard-coded from the earlier `uniform 5-phase` correlation analysis, using the absolute strength of:

- `source_phase_compactness_p1`
- `source_phase_compactness_p2`
- `source_phase_compactness_p3`
- `source_phase_compactness_p4`
- `source_phase_compactness_p5`

The normalized weights used were:

```text
p1 = 0.2472
p2 = 0.2318
p3 = 0.2127
p4 = 0.1578
p5 = 0.1466
```

So this first implementation emphasized:

- `p1`
- `p2`
- `p3`

more than:

- `p4`
- `p5`

### Hyperparameters used in this version

This exploratory version used:

- partition type: `uniform`
- phase count: `5`
- extra loss: `source phase compactness` only
- `lambda_compact = 0.05`
- training condition: `closed-set`
- augmentation condition: `no shift aug`

No extra margin loss was added in this version.

### Why this design was chosen

This design was chosen because it was the **smallest direct test** of the source self-structure hypothesis:

- use the strongest source self-structure indicator already found by analysis
- regularize the same temporal feature carrier used in the analysis
- avoid introducing clustering, dynamic boundaries, margin mining, or target-side extra machinery

So the purpose of `srcphasecompact_p5` was not to be the final method, but to answer:

> does source-domain phase compactness act as a useful inductive bias at all?

### What limitation this version had

This version mixed two ideas together:

- source-domain self-structure motivation
- and adaptation-stage shared-encoder optimization

That means it could improve some transfer tasks, but it could also pull the target representation in a source-specific direction.

This is why later discussion shifted toward a cleaner formulation:

- first improve source-domain self-structure in source-only training
- then use the improved source model to initialize ordinary TimeMatch

So `srcphasecompact_p5` should be read as:

- a valid first probe
- but not yet the best-isolated implementation of the method motivation
