# Prototype Contrastive Adaptation Experiments

## Purpose

This note tracks experiments specifically under the `Prototype Contrastive Adaptation` mainline.

The goal of this line is:

1. Treat prototypes as class-level semantic anchors.
2. Pull sample features toward their corresponding class prototypes.
3. Improve cross-domain category structure alignment through sample-to-prototype discrimination rather than only pointwise or relation MSE matching.

This line is motivated by the earlier structural analysis in this project:

- `Prototype Distance` showed the strongest correlation with target-domain macro-F1.
- `Relation Structure Distance` also mattered, but was weaker.

The working hypothesis is:

> If prototype-level semantic structure is the strongest explanatory signal for transfer quality,  
> then explicitly optimizing sample-to-prototype discrimination may be a more direct and effective adaptation mechanism.

---

## Why This Mainline Was Started

This line was prioritized after the first refinement-guided prototype experiment (`RGPA-v1`) underperformed.

`RGPA-v1` suggested that:

- prototype quality likely matters
- but simply removing temporally volatile positions is not enough

Instead of continuing to refine the instability definition immediately, the next step was to test a more direct prototype-centered objective:

- keep prototypes as the main semantic carrier
- use contrastive pressure to tighten class-wise alignment
- check whether this gives a cleaner positive signal than refinement-based masking

---

## Important Evaluation Setting

All experiments in this note are interpreted under the current **closed-set** setting.

This means:

- `unknown` is excluded
- only shared retained semantic classes are evaluated
- the correct comparison target is the **closed-set baseline**, not the original open-set paper numbers

For `FR1 -> DK1`, the current closed-set baseline reference is:

- macro-F1: `0.6391047`

This baseline is the primary comparison point for the experiments below.

---

## Experiment ProtoCon-v1

### Intended goal

The intended goal of `ProtoCon-v1` was to test a prototype-contrastive variant on `FR1 -> DK1`.

The initial expectation was:

- keep the existing prototype branch infrastructure
- add sample-to-prototype contrastive learning
- observe whether prototype-centric discrimination improves over the baseline

### What was actually run

This run was **not** a pure prototype-contrastive experiment.

From the training log:

- Epochs `1-8`:
  - `point=0.00000`
  - `edge=0.00000`
  - `proto_con=0.00000`
- Epoch `9+`:
  - `point > 0`
  - `edge > 0`
  - `proto_con > 0`

So after warmup, the training objective actually included:

- point alignment loss
- relation alignment loss
- prototype contrastive loss

This means the run should be interpreted as:

> `PRA + Prototype Contrastive` mixed version

and **not** as a pure prototype-contrastive ablation.

This distinction is critical for later comparison.

---

## Run Configuration

### Task

- `FR1 -> DK1`

### Source weights

- closed-set source model: `outputs/pseltae_FR1_baseline`

### Effective configuration summary

Equivalent run configuration:

```bash
python train.py \
  -e timematch_FR1_to_DK1_protocon_v1 \
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
  --disable_pra \
  --pra_warmup_epochs 8 \
  --pra_min_samples_per_class 2 \
  --pra_bank_momentum 0.97 \
  --pra_use_prototype_contrastive true \
  --pra_contrastive_trade_off 0.05 \
  --pra_contrastive_temperature 0.1
```

### Important interpretation note

Although `--disable_pra` was passed, the current implementation still entered the unified prototype branch once prototype contrastive was enabled, and the log confirms that point/relation losses were also active after warmup.

So the effective experiment was a **mixed branch experiment**, not a pure contrastive-only one.

---

## Observations From Training Log

### Sanity check

The prototype contrastive path did activate correctly:

- before warmup ended:
  - `proto_con=0.00000`
- after warmup:
  - `proto_con` became clearly non-zero

This confirms that the newly added contrastive term was actually participating in optimization.

### Mixed-loss activation

However, the log also confirms that:

- `point` became non-zero
- `edge` became non-zero

Therefore this run cannot isolate the effect of prototype contrastive by itself.

### Teacher pseudo-label trend

Teacher pseudo-label F1 improved during training:

- early stage: about `0.49`
- late stage: about `0.56`

So optimization remained stable and target pseudo-label quality did not collapse.

### Validation trend

Best validation F1 observed:

- `0.5875`

This is slightly above the best validation F1 seen in `RGPA-v1`, but still not enough to suggest a meaningful gain over the closed-set baseline.

---

## Final Result

### Mixed prototype-contrastive result

- `FR1 -> DK1`
- test accuracy: `0.9096`
- test macro-F1: `0.5934`

### Comparison to closed-set baseline

Closed-set baseline:

- `0.6391047`

Delta:

\[
0.5934 - 0.6391 \approx -0.0457
\]

So this run is approximately:

- `-4.6` macro-F1 points below the closed-set baseline

### Comparison to RGPA-v1

Previous `RGPA-v1` result:

- `0.5999`

Delta:

\[
0.5934 - 0.5999 \approx -0.0065
\]

So this mixed prototype-contrastive version is also slightly below `RGPA-v1`.

---

## Interpretation

### What this result does show

This result shows that the current **mixed** version:

- point alignment
- relation alignment
- prototype contrastive

does **not** outperform the closed-set baseline on `FR1 -> DK1`.

### What this result does not show

This result does **not** justify concluding that:

- prototype contrastive is inherently ineffective

because the experiment does not isolate prototype contrastive from the existing PRA losses.

So the correct interpretation is:

> the current mixed objective did not improve performance,  
> but the pure prototype-contrastive hypothesis remains untested

---

## What Was Learned

### Confirmed

- the prototype-contrastive branch is implemented and active
- its loss term enters training after warmup as intended

---

## Experiment ProtoCon-pure-v1

### Purpose

After `ProtoCon-v1`, the next step was to isolate whether **prototype contrastive alone** helps when the point and relation alignment losses are fully removed from optimization.

The key question became:

> If prototype-centered discrimination is the right structural bias,  
> can a pure sample-to-prototype contrastive objective recover the gap to the closed-set baseline?

### Effective configuration

This run disabled the PRA point/relation loss contributions:

- `pra_trade_off = 0.0`
- `pra_point_trade_off = 0.0`

while keeping:

- pseudo-label supervision
- prototype contrastive learning

### Important interpretation note

The training log still printed non-zero `point` and `edge` values in later epochs, but they were diagnostic-only and did **not** contribute to the total loss.

This can be verified directly from the logged scalar composition, where:

- `loss ≈ source + 2 * target + 0.05 * proto_con`

matches the reported total loss closely.

So this run should be interpreted as a valid **pure prototype-contrastive ablation**.

### Final result

- `FR1 -> DK1`
- test accuracy: `0.9183`
- test macro-F1: `0.5996`

### Comparison

Compared with the closed-set baseline:

\[
0.5996 - 0.6391 \approx -0.0395
\]

Compared with `RGPA-v1`:

\[
0.5996 - 0.5999 \approx -0.0003
\]

Compared with mixed `ProtoCon-v1`:

\[
0.5996 - 0.5934 \approx +0.0062
\]

### Interpretation

This result showed three things clearly:

1. the earlier negative result was **not only** caused by point/relation-loss interference
2. removing the mixed PRA terms makes the method slightly cleaner and slightly better
3. pure prototype contrastive still does **not** recover the closed-set baseline

So the correct conclusion is:

> simple sample-to-prototype contrastive pressure, by itself, is not sufficient to close the transfer gap on `FR1 -> DK1`

---

## Experiment MemProto-v1

### Purpose

After the pure prototype-contrastive ablation, the next question was whether the bottleneck came from **prototype instability** rather than from the contrastive objective itself.

This led to a memory-based variant:

- use EMA source/target prototype banks
- build invariant anchors from memory instead of relying only on current-batch prototype estimates

The working hypothesis was:

> If current prototypes are too noisy, then more stable memory-based anchors may make prototype contrastive learning more effective.

### Effective configuration

This run enabled:

- `pra_use_prototype_contrastive = true`
- `pra_use_memory_invariant_prototypes = true`

and kept:

- `pra_trade_off = 0.0`
- `pra_point_trade_off = 0.0`

So the intended objective was:

- source supervised loss
- target pseudo-label loss
- memory-based prototype contrastive loss

without point or relation alignment loss contributions.

### Log behavior

This run is cleaner than `ProtoCon-pure-v1` from a diagnostic perspective.

- before warmup:
  - `point=0`
  - `edge=0`
  - `proto_con=0`
- after warmup:
  - `point=0`
  - `edge=0`
  - `proto_con>0`

So this can be treated as a clean **memory-based prototype-contrastive** experiment.

### Final result

- `FR1 -> DK1`
- test accuracy: `0.9174`
- test macro-F1: `0.5989`

### Comparison

Compared with the closed-set baseline:

\[
0.5989 - 0.6391 \approx -0.0402
\]

Compared with `RGPA-v1`:

\[
0.5989 - 0.5999 \approx -0.0010
\]

Compared with `ProtoCon-pure-v1`:

\[
0.5989 - 0.5996 \approx -0.0007
\]

### Interpretation

This result suggests that simply replacing current-batch anchors with memory-based invariant anchors is **not enough** to change the overall outcome.

In other words:

- prototype stability was a reasonable hypothesis
- but the current memory-bank realization still lands at essentially the same performance level

So the main conclusion becomes stronger:

> three prototype-enhancement variants now cluster near `0.599`,  
> while the closed-set baseline remains `0.6391`

This means the problem likely sits deeper than a small loss-design detail.

---

## Cross-Experiment Conclusion

For `FR1 -> DK1`, the current prototype-centered structural variants are:

- `RGPA-v1`: `0.5999`
- `ProtoCon-v1` mixed: `0.5934`
- `ProtoCon-pure-v1`: `0.5996`
- `MemProto-v1`: `0.5989`

Reference closed-set baseline:

- `0.6391047`

### What this pattern means

The consistent pattern is that:

- prototype-based structure is still the strongest **analysis signal**
- but current prototype-side training objectives do **not** convert that signal into a performance gain

So a careful interpretation is:

> "prototype distance matters" is not equivalent to  
> "adding a prototype loss will improve adaptation"

There is likely a gap between:

- explanatory structure metrics
- and the form of optimization target that the model can actually benefit from

### Practical implication

At this stage, it is probably not efficient to keep making small local edits to:

- contrastive temperature
- contrastive weight
- bank momentum

Those may change the score slightly, but the current evidence does not suggest they will recover a meaningful gap of about `4` macro-F1 points.

The next stronger candidates should shift attention toward:

1. better target semantic assignment mechanisms
2. better temporal-structure definitions
3. or a new structure carrier beyond the current prototype-loss family
- the combined optimization remains stable

### Not yet answered

- whether **pure** prototype contrastive can help
- whether the negative result comes from contrastive itself
- or from interference between contrastive and the existing point/relation losses

### Current takeaway

At this stage, the safest conclusion is:

> the mixed `PRA + Prototype Contrastive` formulation is not a winning version on `FR1 -> DK1`

but the mainline should not be rejected until a clean pure-contrastive ablation is run.

---

## Recommended Next Step

The next proper experiment is:

- `ProtoCon-pure-v1`

with:

- `pra_point_trade_off = 0.0`
- `pra_trade_off = 0.0`
- `pra_use_prototype_contrastive = true`

The goal is to isolate:

- supervised source loss
- target pseudo-label loss
- prototype contrastive loss

and remove point/relation MSE interference.

Only after that run can this mainline be judged fairly.

---

## Experiment ProtoCon-pure-v1

### Goal

This run was designed as the clean ablation that `ProtoCon-v1` failed to provide.

The intended objective was:

- supervised source classification
- target pseudo-label supervision
- prototype contrastive loss

while removing the influence of point/relation PRA loss weights.

### Run configuration

### Task

- `FR1 -> DK1`

### Source weights

- closed-set source model: `outputs/pseltae_FR1_baseline`

### Effective configuration summary

```bash
python train.py \
  -e timematch_FR1_to_DK1_protocon_pure_v1 \
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
  --disable_pra \
  --pra_warmup_epochs 8 \
  --pra_min_samples_per_class 2 \
  --pra_bank_momentum 0.97 \
  --pra_point_trade_off 0.0 \
  --pra_trade_off 0.0 \
  --pra_use_prototype_contrastive true \
  --pra_contrastive_trade_off 0.05 \
  --pra_contrastive_temperature 0.1
```

### Interpretation note

Although `point` and `edge` were still printed in the log after warmup, the total loss numerically matches:

- source loss
- target pseudo loss
- prototype contrastive loss

with `pra_point_trade_off = 0.0` and `pra_trade_off = 0.0`.

So these point/relation terms were still being computed for diagnostics, but they were not contributing to optimization.

This means `ProtoCon-pure-v1` can be treated as an **effective pure prototype-contrastive ablation**.

---

## Observations From ProtoCon-pure-v1

### Sanity check

The pure contrastive path behaved as expected:

- Epochs `1-8`:
  - `proto_con=0.00000`
- Epoch `9+`:
  - `proto_con > 0`

So the prototype contrastive branch was active after warmup.

### Teacher pseudo-label trend

Teacher pseudo-label F1 again improved during training:

- early stage: about `0.49`
- late stage: about `0.56`

This indicates stable optimization and no pseudo-label collapse.

### Validation trend

Best validation F1 observed:

- `0.5847`

This is close to:

- `RGPA-v1`: `0.5848`
- mixed `ProtoCon-v1`: `0.5875`

but still not enough to suggest a meaningful structural gain over the baseline.

---

## Final Result For ProtoCon-pure-v1

### Pure prototype-contrastive result

- `FR1 -> DK1`
- test accuracy: `0.9183`
- test macro-F1: `0.5996`

### Comparison to closed-set baseline

Closed-set baseline:

- `0.6391047`

Delta:

\[
0.5996 - 0.6391 \approx -0.0395
\]

So this run is approximately:

- `-4.0` macro-F1 points below the closed-set baseline

### Comparison to mixed ProtoCon-v1

Mixed `ProtoCon-v1`:

- `0.5934`

Delta:

\[
0.5996 - 0.5934 \approx +0.0062
\]

So the pure version is slightly better than the mixed version.

### Comparison to RGPA-v1

`RGPA-v1`:

- `0.5999`

Delta:

\[
0.5996 - 0.5999 \approx -0.0003
\]

So the pure prototype-contrastive result is effectively tied with `RGPA-v1`.

---

## Updated Interpretation

### What this run answers

This run shows that the earlier negative result cannot be explained only by interference from point/relation losses.

Even after isolating the effective optimization objective down to prototype contrastive:

- performance still remained clearly below the closed-set baseline

### Main conclusion

At this stage, the evidence suggests:

> prototype importance as an analysis signal does not automatically translate into gains from a simple sample-to-prototype contrastive objective

In other words:

- prototype structure still appears important
- but the current contrastive formulation is not sufficient to exploit it

### Practical takeaway

The bottleneck now looks less like:

- "contrastive was mixed with the wrong losses"

and more like:

- "prototype construction and prototype stability are still not good enough"

---

## Current Mainline Status

### Rejected in current form

- mixed `PRA + Prototype Contrastive`
- simple pure `Prototype Contrastive` with current prototype construction

### Still plausible direction

- memory-based invariant prototypes
- more stable prototype anchors beyond single-batch estimates
- stronger prototype construction mechanisms before contrastive alignment

### Recommended next step

The next version worth testing is:

- `Memory-Based Invariant Prototype Adaptation`

The main reason is:

- the project's strongest empirical signal still points to prototype-level structure
- but both refinement-v1 and contrastive-v1 suggest that raw batch-level prototype construction is not yet reliable enough
