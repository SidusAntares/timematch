# Candidate Method Mainlines for Structure-Based Time-Series UDA

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
