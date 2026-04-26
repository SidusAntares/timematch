# Refinement-Guided Prototype Adaptation Experiments

## Purpose

This note tracks experiments specifically under the `Refinement-Guided Prototype Adaptation` mainline.

The goal of this line is:

1. Identify unstable or transition-like temporal segments.
2. Avoid letting those segments dominate prototype construction.
3. Use cleaner prototypes for cross-domain prototype adaptation.

This idea is mainly inspired by:

- `BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation`
- `Cross-Domain Detection via Graph-Induced Prototype Alignment`

The working hypothesis is:

> Not all temporal positions should contribute equally to prototype construction.  
> Some positions are important for learning, but too unstable to be reliable prototype evidence.

---

## Why Closed-Set Was Used

For this project stage, closed-set construction is necessary for a fair structural-method study.

The reason is not simply to obtain a higher score, but to make the problem definition cleaner.

Our current method family studies:

- prototype construction
- class-conditional alignment
- category structure stability

These ideas implicitly assume that source and target share the same semantic label space.

If `unknown` is retained, it mixes together several different phenomena:

- rare source classes
- classes existing in target but absent in source
- miscellaneous merged categories

In that setting, a degraded result becomes hard to interpret:

- did the structure method fail?
- or were open-set categories being forced into the wrong semantic structure?
- or was the `unknown` bucket contaminating prototype estimation?

Therefore, closed-set construction is currently used to answer a cleaner question:

> under a shared category space, can structure-aware prototype adaptation provide real gains?

### Important comparison note

Closed-set results should **not** be directly interpreted as proof that a method is better than the original open-set paper setting.

This is because closed-set changes the task itself:

- unknown categories are removed
- non-overlapping classes are excluded
- the adaptation problem becomes easier and cleaner

So a higher closed-set score mainly means:

- the task setting is more controlled
- and the structure method can be evaluated with less open-set interference

The proper comparison rule is:

- compare **closed-set method vs closed-set baseline**
- do **not** directly compare closed-set results to open-set literature numbers as if they were the same benchmark

In this project, the role of closed-set evaluation is:

1. study whether category-structure methods are intrinsically useful
2. reduce ambiguity in prototype-level analysis
3. provide a controlled environment before considering broader open-set comparisons

---

## Experiment RGPA-v1

### Design

First minimal implementation of refinement-guided prototype construction:

- Keep the existing TimeMatch + PRA training pipeline unchanged as much as possible.
- Only modify the prototype construction path inside PRA.
- Define temporal instability using adjacent feature differences:

\[
s_t = \|h_t - h_{t-1}\|_2
\]

- For each sample, keep only the lowest-instability temporal positions.
- Use the retained stable temporal positions to pool a refined sample feature.
- Build source/target batch prototypes from these refined features.

Implementation switches:

- `--pra_use_refined_prototypes true`
- `--pra_refinement_keep_ratio 0.7`

This means:

- refined prototype pooling is enabled
- only the most stable `70%` temporal positions are used to construct prototype features

---

## Run Configuration

### Task

- `FR1 -> DK1`

### Source weights

- closed-set source model: `outputs/pseltae_FR1_baseline`

### Command summary

Equivalent run configuration:

```bash
python train.py \
  -e timematch_FR1_to_DK1_refinedproto \
  --source france/30TXT/2017 \
  --target denmark/32VNH/2017 \
  --seed 111 \
  --overwrite_existing \
  timematch \
  --weights outputs/pseltae_FR1_baseline \
  --pseudo_threshold 0.9 \
  --trade_off 2.0 \
  --epochs 20 \
  --steps_per_epoch 500 \
  --use_pra \
  --pra_trade_off 0.005 \
  --pra_point_trade_off 0.001 \
  --pra_warmup_epochs 8 \
  --pra_min_samples_per_class 2 \
  --pra_bank_momentum 0.97 \
  --pra_normalize_geometry true \
  --pra_use_refined_prototypes true \
  --pra_refinement_keep_ratio 0.7
```

---

## Observations From Training Log

### Sanity check

The experiment did activate the intended refinement-guided PRA path correctly:

- before warmup ended:
  - `point=0.00000`
  - `edge=0.00000`
  - `enabled_batches=0/500`
- after warmup:
  - point and relation losses became non-zero
  - `enabled_batches=500/500`

So this was not a configuration failure. The refined prototype path was actually used.

### Teacher pseudo-label trend

Teacher pseudo-label F1 increased during training:

- early stage: about `0.49`
- later stage: about `0.56`

This suggests:

- training remained stable
- target pseudo labels did not collapse
- the refined prototype mechanism did not break optimization

### Validation trend

Best validation F1 observed:

- `0.5848`

This happened late in training and slightly improved over the earlier refined run states, but it still does not indicate a strong gain over the closed-set baseline reference.

---

## Final Result

### Refined prototype result

- `FR1 -> DK1`
- test accuracy: `0.9171`
- test macro-F1: `0.5999`

### Closed-set baseline reference

Previously recorded closed-set baseline:

- `FR1 -> DK1 = 0.6391047`

### Delta

\[
0.5999 - 0.6391 \approx -0.0392
\]

So this experiment is approximately:

- `-3.9` macro-F1 points below the closed-set baseline

---

## Interpretation

### What this result means

This first refinement-guided version did **not** improve performance.

The result suggests that the current definition of unstable temporal segments is too crude:

- large adjacent feature change does **not** necessarily mean the segment is harmful for prototype construction
- in time-series classification, high-variation segments may actually be highly discriminative

So the current rule:

> "remove temporally volatile positions from prototype construction"

is not sufficient.

### Important conclusion

This experiment does **not** invalidate the overall refinement-guided mainline.

Instead, it shows that:

- prototype quality is still important
- but `feature difference magnitude` is not a good enough proxy for unreliability

In other words:

> the issue is likely not whether refinement is useful,  
> but whether we are identifying the right kind of unstable segments

---

## What Was Learned

### Confirmed

- refinement-guided prototype construction is technically feasible in the current codebase
- the opt-in implementation runs stably
- PRA can still train under refined prototype pooling

### Rejected in current form

- using adjacent temporal feature differences as the main instability score
- hard-masking out the most varying `30%` temporal positions for prototype construction

### New insight

Temporal variability and prototype unreliability are not the same thing.

Likely better future definitions should be more task-aware, such as:

- uncertainty-based unstable segments
- pseudo-label instability across time
- attention inconsistency across time
- shift-sensitive temporal regions

---

## Next Steps

Most reasonable next actions:

1. Do **not** spend more time tuning only `pra_refinement_keep_ratio` for this exact version.
2. Treat this as a negative but informative experiment.
3. Shift the next method iteration toward one of these:
   - better instability definition
   - prototype contrastive adaptation
   - refinement + uncertainty-aware prototype construction

Current recommendation:

- keep `Refinement-Guided Prototype Adaptation` as an active mainline
- but move away from pure temporal-difference masking
- prioritize `Prototype Contrastive Adaptation` as the next implementation step

---

## Short Conclusion

`RGPA-v1` demonstrated that refinement-guided prototype construction can be integrated cleanly into TimeMatch + PRA, but the first temporal-difference-based stable-segment rule reduced performance on `FR1 -> DK1`. The main lesson is that useful refinement must distinguish truly unreliable temporal regions from merely high-variation but discriminative regions.
