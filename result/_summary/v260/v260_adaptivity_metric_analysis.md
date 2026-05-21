# v2.6.0 Structure Metric Analysis: Toward Adaptivity

## 1. Audit Status

v2.6.0 has two complete audit views:

| audit view | tasks | role |
| --- | ---: | --- |
| raw input | 22/22 | input-level temporal mismatch portrait |
| PSE feature space | 22/22 | main evidence for source-structure adaptivity |

The PSE feature-space audit is the one that matters for v2.6, because our method shapes the feature trajectory between PSE and LTAE, not the raw input sequence.

## 2. Main Reading

The audit confirms a useful but important boundary:

> Current metrics are strong enough to guide structure strength, but not yet strong enough to directly select the full structure view.

The current `suggested_view` is not reliable. In PSE feature space it collapses almost entirely to `global`, while our empirical best table shows many tasks prefer segmented compactness and a few prefer dynamics-like views. This is a scoring-design issue: the current global score is structurally easier to become large than the segmented/dynamics scores.

Therefore, v2.6 should not immediately become automatic view selection. The safer first step is:

> v2.6.1 = target-aware structure strength adaptation under a fixed conservative view bank.

## 3. Metric Signals

### 3.1 Source Reliability

`source_reliability` measures how cleanly source class prototypes are separated relative to source compactness.

Useful interpretation:

- high reliability: source geometry is already clear, compactness can be strengthened moderately;
- low reliability: source geometry is fragile, strong structure loss may over-constrain or distort the source manifold.

Observed pattern:

- HAR / HHAR generally have higher source reliability;
- remote sensing tasks are much lower, especially DK1 as source;
- this supports weaker, more conservative structure factors for many remote tasks.

### 3.2 Target Margin

`target_margin_ratio` measures whether target samples are confidently closer to one source prototype than to alternatives.

Useful interpretation:

- high margin: target already falls near a source class geometry; stronger source compactness is less risky;
- low margin: target is ambiguous under source prototypes; overly stiff source structure may hurt alignment.

This is one of the most important target-aware signals for v2.6.1.

### 3.3 Temporal Mismatch CV

`temporal_mismatch_cv` measures how uneven the source-target prototype mismatch is across time.

Useful interpretation:

- low CV: mismatch is relatively global and smooth;
- high CV: mismatch is concentrated in certain temporal regions, so global strengthening is risky and local/windowed structure is more plausible.

Observed pattern:

- HAR / HHAR mostly have low CV;
- remote tasks often have high CV;
- remote tasks are therefore more likely to require local/segmented views, but also weaker global stiffness.

### 3.4 Segment Compression

`uniform_k*_adjusted_compression` measures whether a temporal partition can explain the mismatch curve by lowering within-segment variance.

Useful interpretation:

- high compression: mismatch has meaningful local structure; segmented/windowed constraints are theoretically justified;
- low compression: segmentation has little explanatory value; global or weak compactness is safer.

This is the strongest candidate for future window/view adaptation.

### 3.5 Dynamics Cosine

`dynamics_delta_cosine_mean` is not yet reliable as a selector.

Reason:

- known dynamics-benefiting tasks do not consistently show high positive dynamics cosine;
- in some cases negative or near-zero dynamics cosine may mean "dynamics mismatch exists", not "dynamics constraint will help";
- the current metric mixes reliability and mismatch magnitude.

For now, dynamics should stay in the candidate view bank, not become an automatic rule.

## 4. Practical Rule Suggested by v2.6.0

For v2.6.1, use a conservative target-aware strength factor:

$$
\lambda' = \lambda \cdot s
$$

where:

$$
s = clip(1 + \Delta_{src} + \Delta_{tgt} - \Delta_{mismatch}, s_{min}, s_{max})
$$

Recommended initial bounds:

$$
s_{min}=0.75,\quad s_{max}=1.20
$$

Suggested discrete mapping:

| signal | condition | factor contribution |
| --- | --- | ---: |
| source reliability | high | +0.05 |
| source reliability | low | -0.08 |
| target margin | high | +0.10 |
| target margin | low | -0.08 |
| temporal mismatch CV | high | -0.10 |
| temporal mismatch CV | low | +0.05 |
| segment compression | high | do not increase global strength; route to local/window later |

Important design choice:

> Segment compression should not simply increase all structure weights. It should later affect where the constraint is applied, not only how strong it is.

## 5. Why Strength Adaptation Comes Before View Adaptation

The empirical best table proves that different tasks prefer different views, but the audit does not yet provide a clean automatic view classifier.

However, the audit already gives stable signals about stiffness:

- high target margin and clean source geometry allow stronger compactness;
- high temporal mismatch or low target margin require weaker structure;
- remote sensing should often be more conservative than HAR / HHAR.

So v2.6.1 should be a low-risk adaptation layer:

> keep the view fixed, adapt only the structure weight scale.

Then v2.6.2 can use segment compression for window/view adaptation.

## 6. Proposed Version Split

### v2.6.1 Target-aware Strength Adaptation

Goal:

> adapt structure loss strength from source-target feature geometry.

Implementation:

- use a warm-up or baseline source checkpoint to extract PSE features;
- compute source reliability, target margin, temporal mismatch CV;
- map them to a conservative multiplier;
- multiply `intra`, `trend`, `segment_inter`, `boundary`, and optionally `prototype_dynamics` by this factor.

This is the next mainline.

### v2.6.2 Window / Scale Adaptation

Goal:

> decide whether structure should be global or local.

Use:

- segment compression;
- mismatch top-20 concentration;
- mismatch CV;
- choose global / K=3 / K=5 / K=8 or soft window weights.

Do not hard-code "the best K". Prefer:

$$
w_k \propto compression_k
$$

or choose the smallest K whose adjusted compression is close to the maximum.

### v2.6.3 Dynamics Reliability

Goal:

> make dynamics view meaningful rather than accidental.

Current dynamics cosine is insufficient. It needs to separate:

- dynamics reliability: are local changes stable within source classes?
- dynamics mismatch: is target different in local change pattern?
- dynamics usefulness: does constraining source dynamics improve target alignment?

Only after this should dynamics become adaptive.

## 7. Immediate Next Experiment

Run a small v2.6.1 probe on representative tasks:

| group | tasks | reason |
| --- | --- | --- |
| remote low-reliability | DK1->FR1, DK1->FR2 | tests whether weakening structure helps fragile source geometry |
| remote high-gain segmented | FR2->DK1, AT1->DK1 | tests whether conservative scaling preserves gains |
| HAR high-margin | HAR 2->11, HAR 9->18 | tests whether strengthening compactness helps clean target alignment |
| HHAR stable compact | HHAR 2->7, HHAR 3->8 | tests general time-series transfer |

Use the same structure view as the current per-task best only for the first probe. This isolates strength adaptation from view adaptation.

## 8. Current Decision

v2.6.0 supports the following mainline decision:

> Start adaptivity from target-aware stiffness / strength adaptation, not from automatic view selection.

The reason is simple: view preference is real, but current view metrics are not yet discriminative enough. Strength reliability is clearer, lower risk, and easier to validate quickly.

