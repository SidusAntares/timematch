# Source Structure Foundation

## Current problem definition

The current research question is no longer "how do we force source and target to become closer."
The focus is:

1. what kind of **source-domain structure** helps domain adaptation;
2. whether the **target domain** has its own structural preference for adaptation;
3. whether those preferences are domain-sensitive or class-sensitive;
4. where constraints should be applied: raw data, PSE output, or an intermediate source-only reshaping module.

## Why target-only / gap-only is no longer the main line

- Gap-style indicators can still help explain transfer behavior, but they are not the main modeling direction.
- The current goal is to improve domain adaptation by changing source structure itself, not by explicitly forcing source-target closeness.

## Main concern with current source-phase regularization

The current source structure regularization is applied on top of PSE features.
That means:

- the regularizer shapes the shared encoder;
- the target representation is then indirectly affected by source-side constraints;
- this makes it hard to isolate whether the improvement comes from source structure change itself or from encoder-level side effects.

## Two implementation routes that emerged

### Route A: source-only feature reshaping after PSE

This includes both:

- a source-specific mapping layer between PSE and LTAE;
- a source-only learned preprocessor that reshapes source features before temporal modeling.

Shared requirement:

- the reshaped feature still needs to stay compatible with the downstream temporal encoder and decoder;
- otherwise the model may learn to recognize only reshaped source features, while target still arrives as raw PSE output.

This route is safest when the reshaping is near-identity and conservative.

### Route B: source data-level structural transformation

Instead of modifying the shared encoder path, transform only the source data before PSE.
This matches the original scientific question more directly:

- if a certain kind of source structure helps domain adaptation,
- can we construct that structure in the source data itself?

This route has the benefit of not directly polluting the target feature path.

## What the current analysis already suggests

- Indicators are both **domain-sensitive** and **class-sensitive**.
- Therefore, a single global structural direction is unlikely to work well for all source domains.
- A more realistic next step is to screen source-side metrics by family and stability, then route different domains toward different metric combinations.

## Implication for the next stage

The immediate next-stage work is:

1. use **source-side metric family screening** to identify stable candidate signals;
2. test **rule-based source data transforms** first, because they are controlled and interpretable;
3. only after that, consider a source-only reshaping module or dynamic phase-count ideas.
