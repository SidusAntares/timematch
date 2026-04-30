# Structure Design Principles for This Project

## Why this note exists

This project has already gone through several prototype-centered training variants:

- `RGPA-v1`
- `ProtoCon-pure-v1`
- `MemProto-v1`
- `PhasePSE-v4`

None of them surpassed the closed-set `FR1 -> DK1` baseline.

So the next step should not be:

- adding more loosely motivated structure modules
- or continuing to tune prototype losses without a stronger design rationale

Instead, the next step should be guided by a compact set of **structure design principles** grounded in:

1. the core papers most relevant to this project
2. the existing local experimental evidence in this repository

---

## Current project evidence that any principle must respect

### Stable negative result on prototype-only variants

- closed-set baseline on `FR1 -> DK1`: `0.6391`
- `RGPA-v1`: `0.5999`
- `ProtoCon-pure-v1`: `0.5996`
- `MemProto-v1`: `0.5989`
- `PhasePSE-v4`: `0.5989`

This means:

- prototype structure is important as an **analysis signal**
- but the current prototype-side optimization routes are not sufficient by themselves

### Best explanatory signals so far

From the current analysis track:

- `prototype_distance` is a strong explanatory metric
- `pse_trend_curve_distance` is even slightly stronger
- `pse_early_curve_distance` and `pse_mid_curve_distance` are also very strong

This means:

- semantic class structure matters
- encoded temporal structure matters
- early/mid seasonal structure matters more than late-season structure

### Working interpretation

The project should now assume:

> the missing transferable signal is not only “where the class center is”,  
> but also “how class identity evolves over encoded time”

---

## Paper Core Map

This section compresses the key takeaways from the most relevant papers into one place.

| Paper | Core thing worth taking | What not to copy literally |
|---|---|---|
| `A Prototype-Oriented Framework for UDA` | Use prototypes as semantic anchors instead of raw source-target sample matching; soft conditional transport is meaningful | Do not assume the original transport formulation is automatically the best fit for TimeMatch |
| `Prototypical Contrast Adaptation` | Inter-class structure matters; same-class prototype attraction and other-class prototype repulsion can improve discriminability | Do not assume prototype contrast alone solves target noise |
| `Cross-Domain Detection via Graph-Induced Prototype Alignment` | Prototype alignment works better after refinement; instance quality before prototype construction matters | Do not import graph modules just for architectural novelty |
| `BAPA-Net` | Difficult / unstable regions and stable regions should be treated differently; boundaries are useful for learning but risky for prototype construction | Do not directly reuse image boundary logic; translate it into temporal transition logic |
| `Invariant Domain-Level Prototype Memory` | Prototype memory, momentum update, and confidence-based target updates improve anchor stability | Do not assume memory alone creates a new structural idea |
| `A Contrastive Representation DA Method for Industrial Time-Series Prediction` | Contrastive learning can preserve prediction-relevant local structure and reduce semantically bad alignment in time series | Do not copy the exact industrial regression framing into crop classification |
| `AdaTime` | Fair TS-UDA evaluation matters; many gains come from sound class-conditional alignment and realistic model selection, not just complicated methods | Do not over-interpret single-paper wins without standardized comparison |
| `ACSSM` | Irregular time handling, interpolation, and latent continuous dynamics can provide better temporal infrastructure when observations are sparse or misaligned | Do not turn this project into a full generative irregular-time modeling project unless structure evidence clearly demands it |
| `Self-attention + Frequency Augmentation` (previous local reading) | Relative class structure matters more than simple linear shift; source variability and class-structure geometry can explain transferability | Do not rely on one narrow MSE metric as the only complexity measure |
| `Dynamic Snake Convolution` | Structure priors should be injected as constrained operators, not fully free deformations; continuity is a first-class objective | Do not transplant spatial tubular modules into temporal DA directly |

---

## Structure Design Principles Table

These are the principles that are most useful for **this** project, not generic DA advice.

| Principle | Meaning in our setting | Strongest paper support | Support from current project evidence | Immediate implication |
|---|---|---|---|---|
| `1. Use class-conditional structure, not only global alignment` | The alignment unit should be semantic class structure rather than just global MMD/CORAL-style matching | `Prototype-Oriented`, `ProCA`, `AdaTime` | `prototype_distance` is much stronger than `mmd/coral` | Future methods should keep semantic pseudo-label structure in the loop |
| `2. Preserve inter-class separability, not only intra-class closeness` | Good adaptation should not collapse multiple target classes into a single aligned blob | `ProCA`, `Industrial Contrastive`, `Prototype-Oriented` | Prototype metrics explain F1 better than global discrepancy metrics | Losses should explicitly push different classes apart, not only pull same-class features together |
| `3. Prototype quality comes before prototype alignment` | Bad or noisy target prototypes will poison any downstream alignment objective | `GPA`, `BAPA-Net`, `Prototype Memory` | Several prototype-loss variants plateaued near `0.599` | Improve the inputs to prototype construction before adding stronger losses |
| `4. Unstable regions should not be treated the same as stable regions` | Temporal transitions, uncertain windows, or weak pseudo-label phases should not contribute equally | `BAPA-Net`, `GPA` | `RGPA-v1` showed the idea is relevant even though the first rule was weak | Replace naive all-time equal weighting with confidence- or phase-aware construction |
| `5. Structure should be temporal and phase-aware, not purely static` | One class should not be represented only by a single pooled point if its seasonal evolution carries the transferable signal | `Industrial Contrastive`, `ACSSM`, prior `self-attention + frequency` reading | `pse_trend`, `pse_early`, `pse_mid` are among the strongest metrics | Focus on `class x phase` or encoded trajectory structure |
| `6. Encoded temporal structure is more useful than raw curve geometry` | Raw spectral curve distance is weaker than model-encoded temporal feature distance | Prior local analysis + `AdaTime` style evaluation caution | `pse_*` metrics beat raw-curve metrics in `v3` | Build temporal structure methods on `PSE` / encoded features, not raw curves alone |
| `7. Constrain structure modeling with domain knowledge` | The model should not be completely free to align arbitrary time positions or shapes | `Dynamic Snake`, `ACSSM` | Simple global shift and simple static prototype both appear too crude | Use ordered phases, trend continuity, or monotonic phase relations as constraints |
| `8. Separate shift correction from structure alignment` | A baseline’s shift mechanism can hide or absorb structural differences | Prior `self-attention + frequency` reading, `AdaTime` | We now explicitly test `shift_aligned_curve_distance` and related metrics | Structural conclusions should be checked both before and after shift-aware comparison |
| `9. Target structure should be accumulated, filtered, or smoothed` | Target-side semantics are too noisy if built from single batches only | `Prototype Memory`, `Prototype-Oriented` | Batchwise prototype variants are unstable and underperform | EMA memory, confidence filtering, or delayed updates should be default options |
| `10. Infrastructure matters: incomplete or irregular timing is not just preprocessing` | Interpolation / completion can define the coordinate system in which structure becomes comparable | `ACSSM`, prior phenology papers, `Dynamic Snake` in a broader sense | Current curve analysis already depends on interpolation to be computable | Treat completion as an analysis or representation scaffold, even if not yet a full training module |

---

## Which principles matter the most right now?

Not all ten principles should be acted on at once.

The most actionable ones for the current project are:

1. `Prototype quality before prototype alignment`
2. `Structure should be temporal and phase-aware`
3. `Encoded temporal structure is more useful than raw curve geometry`
4. `Target structure should be accumulated, filtered, or smoothed`
5. `Constrain structure modeling with domain knowledge`

These five together define the strongest current direction:

> **phase-aware encoded temporal structure with filtered or stabilized target construction**

---

## What this means for method choice

### Directions that now look weaker

- more point/edge geometric MSE tuning
- more static prototype-loss variants without changing the structure carrier
- more raw-curve whole-season comparison as the main structural object

### Directions that now look stronger

- `class x phase` prototypes
- encoded trend alignment
- early/mid-window semantic alignment
- confidence-aware temporal prototype construction
- phase-constrained or order-constrained target alignment

---

## Recommended next-mainline definition

If the project wants a structure method that is both:

- meaningfully different from current prototype-only attempts
- and still grounded in the current evidence

then the best current description is:

### `Phase-Aware Encoded Structure Adaptation`

This mainline should:

- preserve semantic class supervision via pseudo labels
- operate on encoded temporal features rather than raw curves
- build phase-specific target structure
- avoid treating all phases as equally reliable
- optionally use memory or smoothing for target phase anchors

In other words:

> not `sample -> prototype` only,  
> but `sample phase -> class phase structure`

---

## Final conclusion

The papers do **not** support continuing to add generic prototype losses indefinitely.

Taken together, they support a more specific conclusion:

> The right structural route for this project is likely not a stronger static prototype penalty,  
> but a **constrained, phase-aware, encoded temporal structure model** with better target structure construction.

That is the clearest common design lesson across:

- prototype papers
- contrastive papers
- refinement papers
- memory papers
- time-series DA benchmarking
- irregular-time modeling
- and the prior local evidence already collected in this repository.
