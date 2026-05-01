# Structure Motivation And Metrics

## 1. Important Direction Correction

The **starting point of this work** comes from the self-attention paper's analysis section, and this point must stay explicit:

- the paper mainly emphasizes the **internal structure of each domain**
- especially the **source domain's own class variability, separability, and temporal richness**
- it does **not** mainly start from "how large the source-target structure gap is"

So the primary question inspired by the paper is:

> what properties of a source domain make it a good transferable source?

and **not**:

> how different are the source and target structures?

The source-target structure-gap analysis is something we added later for method design, but it is **not the original motivation** and should not dominate the narrative.

## 2. What The Self-Attention Paper Tells Us

The paper suggests that transferability is strongly related to **domain self-structure**, especially:

- how rich the source-domain class patterns are
- how separable the source-domain classes are
- how wide or narrow the source-domain class dynamics are
- how difficult a domain is as a target because of its own internal class-pattern structure

The most faithful reading is:

- a "good source" is not only a domain that is close to the target
- it is also a domain whose own internal class structure is favorable for supervised learning and transfer

## 3. What Our Additional Analysis Adds

We now separate the analysis into two layers:

### 3.1 Domain self-structure

These metrics describe the **source domain itself**:

- how large the source class variability is
- how separated classes are
- how rich the source temporal dynamics are

This layer is closest to the self-attention paper's analysis section.

### 3.2 Source-target structure gap

These metrics describe **what kind of structure mismatch exists between source and target**:

- prototype-level semantic gap
- phase-wise encoded temporal gap
- class-relative structure gap
- global temporal-shift-like gap

This layer is our extension for method design.

## 4. Current Structural Conclusion

The most stable conclusion so far is:

- the strongest signals for optimization still come from **encoded phase-wise temporal structure**
- but the original paper motivation should still be grounded in **domain self-structure**
- prototype-level gap is useful as an auxiliary view
- relation-only structure is weaker than phase/PSE structure

That leads to the current working view:

> the next method should first be motivated by **source-domain internal structure**, and then operationalized through **class-internal encoded temporal structure across phases**, with prototype-level class anchors kept as auxiliary structure.

## 5. Metrics And How To Reproduce Them

This section records how each currently-used structure-related metric is computed so it can be reproduced later.

All metrics are currently implemented in:

- [recompute_transfer_metrics.py](C:\Code\dev\PythonProject\timematch\analysis\recompute_transfer_metrics.py)

### 5.1 Shared setup

For each `source -> target` task:

1. load the corresponding trained TimeMatch checkpoint from `outputs/`
2. build source and target datasets using the task class set
3. run the classifier in eval mode
4. extract:
   - sample-level temporal features from `model.forward(..., return_feats=True)`
   - sequence-level spatial encoder outputs from `model.spatial_encoder(...)`
   - raw mean parcel curves from `pixels.mean(dim=-1)`
5. aggregate task-level metrics from those extracted quantities

For closed-set runs:

- class list = source classes with at least 200 samples
- then drop `unknown`

For open-set runs:

- class list = source classes with at least 200 samples
- keep `unknown`

### 5.2 Phase/PSE structure metrics

These are currently the strongest structure indicators.

#### `pse_early_curve_distance`

How to compute:

1. for each sample, take `model.spatial_encoder(...)` output over time
2. interpolate each sample curve to a fixed temporal grid (`temporal_grid_size`, currently 30)
3. for each class, average interpolated curves across samples
4. split the grid into 3 equal phases: early / mid / late
5. for each shared class, compute Euclidean distance between source and target class mean curves in the early segment
6. average over shared classes

#### `pse_mid_curve_distance`

Same as above, but use the middle third of the temporal grid.

#### `pse_late_curve_distance`

Same as above, but use the last third of the temporal grid.

#### `pse_trend_curve_distance`

How to compute:

1. take each class mean interpolated PSE curve
2. compute first differences over time
3. flatten the trend sequence
4. compute source-target Euclidean distance per class
5. average over shared classes

### 5.3 Prototype / relation metrics

#### `prototype_distance`

How to compute:

1. use `return_feats=True` to collect sample-level temporal features
2. for each shared class, compute source and target class mean feature vectors
3. compute Euclidean distance between the two class prototypes
4. average over shared classes

#### `relation_structure_distance`

How to compute:

1. compute source and target class prototypes
2. `L2`-normalize prototypes
3. build prototype similarity matrices with dot products
4. zero the diagonals
5. Frobenius-normalize each relation matrix
6. compute mean squared difference between source and target relation matrices

### 5.4 Source domain self-structure metrics

These are the metrics most directly aligned with the self-attention paper's analysis spirit.
They should be treated as the **primary motivation-side indicators**.

#### `source_curve_spread`

How to compute:

1. take source class mean interpolated PSE curves
2. flatten each class curve
3. compute pairwise Euclidean distances between class curves
4. take the mean pairwise distance

Interpretation:

- how spread apart source class temporal patterns are
- large spread may indicate richer class diversity, but by itself it does not say whether the structure is easy or hard to learn

#### `source_curve_activity_range`

How to compute:

1. take each source class mean interpolated PSE curve
2. for each feature dimension, compute `max - min` over time
3. compute the `L2` norm of that range vector
4. average across classes

Interpretation:

- how large the temporal activity amplitude is inside the source domain
- this is still a fairly coarse quantity; it says "how much things move" but not "whether the movement is structurally useful"

#### `source_fisher_ratio`

How to compute:

1. compute global feature mean over all source samples
2. for each class:
   - compute class mean
   - accumulate between-class scatter
   - accumulate within-class scatter
3. return `between / within`

Interpretation:

- source class separability in feature space
- this is much easier to connect to optimization, because larger separability suggests cleaner supervised source training

#### `source_ndvi_q90_range_mean`

How to compute:

1. from raw parcel curves, use Sentinel-2:
   - red = channel 2 (`B04`)
   - nir = channel 6 (`B08`)
2. compute `NDVI = (nir - red) / (nir + red + 1e-6)`
3. for each source class, collect all NDVI values over all samples and time steps
4. compute `q95 - q05` for that class
5. average across classes

Interpretation:

- source-domain class-specific vegetation variability

This is one of the most direct approximations to the paper's source-variability analysis.
It is also one of the most interpretable indicators.

#### `source_ndvi_q90_range_std`

How to compute:

1. compute the same per-class `q95 - q05` NDVI range as above
2. take the standard deviation across classes

Interpretation:

- how uneven the class variability is within the source domain
- useful as a secondary descriptor, but less directly actionable than the mean range itself

#### `source_class_curve_variance_mean`

How to compute:

1. take each source class mean raw curve
2. compute variance over time for each channel
3. average channel variances
4. average across classes

Interpretation:

- average temporal variance of source class raw curves
- a coarse temporal-richness proxy; useful, but not very specific about what kind of structure is beneficial

### 5.5 Source-target global temporal discrepancy metrics

These are closer to the paper's global temporal discrepancy discussion.

#### `global_curve_shift_mse`

How to compute:

1. average all raw source sample curves into one global source curve
2. average all raw target sample curves into one global target curve
3. compute the mean squared error between them

Interpretation:

- global domain-level raw temporal difference

#### `domain_bandwise_curve_mse`

How to compute:

1. compute the same global source and target raw curves
2. compute MSE per spectral band
3. average over bands

Interpretation:

- multi-band global temporal discrepancy

#### `best_shifted_curve_mse`

How to compute:

1. take global source and target raw curves
2. shift one curve against the other over a small discrete range (`max_shift_steps`, currently 5)
3. compute MSE for each shift
4. keep the minimum

Interpretation:

- how similar the domains become after allowing a small global linear temporal shift

#### `classwise_curve_shift_mse_mean`

How to compute:

1. take source and target class mean raw curves
2. for each shared class, compute raw-curve MSE
3. average across shared classes

Interpretation:

- class-specific global temporal mismatch

#### `domain_ndvi_curve_mse`

How to compute:

1. compute global source raw curve and global target raw curve
2. convert each to NDVI over time using:
   - red = channel 2 (`B04`)
   - nir = channel 6 (`B08`)
3. compute MSE between the two global NDVI curves

Interpretation:

- global vegetation-dynamics discrepancy between source and target

### 5.6 Class-relative raw-curve structure metrics

## 6. Source-Phase Self-Structure Experiments

This section records the phase-based **source domain self-structure** experiments that were run after we decided to stop drifting toward source-target gap analysis and return to the original motivation:

> which source-domain internal structures make a source more transferable?

The core principle is:

- phase metrics are computed on **feature curves**
- phase boundaries should preferably be driven by **class structure over time**
- the goal is not to find "more complex" sources in a vague sense
- the goal is to identify **which source-phase structures are most predictive of transferability**

### 6.1 What was run

We ran the closed-set source-phase self-structure analysis with the dedicated launcher:

- [run_recompute_source_phase_metrics_closed.sh](C:\Code\dev\PythonProject\timematch\launchers\analysis\run_recompute_source_phase_metrics_closed.sh)

#### Uniform 5-phase baseline

```bash
CUDA_VISIBLE_DEVICES=1 \
PHASE_PARTITION_MODE=uniform \
PHASE_COUNT=5 \
PHASE_OUTPUT_CSV=result/baseline_analysis/source_phase_self_structure_metrics_closed_uniform_p5.csv \
nohup bash launchers/analysis/run_recompute_source_phase_metrics_closed.sh \
> logs/source_phase_metrics_closed_uniform_p5.log 2>&1 &
```

#### Structure-driven 5-phase

```bash
CUDA_VISIBLE_DEVICES=1 \
PHASE_PARTITION_MODE=structure \
PHASE_COUNT=5 \
PHASE_OUTPUT_CSV=result/baseline_analysis/source_phase_self_structure_metrics_closed_structure_p5.csv \
nohup bash launchers/analysis/run_recompute_source_phase_metrics_closed.sh \
> logs/source_phase_metrics_closed_structure_p5.log 2>&1 &
```

#### Structure-driven 3-phase

```bash
CUDA_VISIBLE_DEVICES=2 \
PHASE_PARTITION_MODE=structure \
PHASE_COUNT=3 \
PHASE_OUTPUT_CSV=result/baseline_analysis/source_phase_self_structure_metrics_closed_structure.csv \
nohup bash launchers/analysis/run_recompute_source_phase_metrics_closed.sh \
> logs/source_phase_metrics_closed_structure_p3.log 2>&1 &
```

The resulting tables are:

- [source_phase_self_structure_metrics_closed_uniform_p5.csv](C:\Code\dev\PythonProject\timematch\result\baseline_analysis\source_phase_self_structure_metrics_closed_uniform_p5.csv)
- [source_phase_self_structure_metrics_closed_structure_p5.csv](C:\Code\dev\PythonProject\timematch\result\baseline_analysis\source_phase_self_structure_metrics_closed_structure_p5.csv)
- [source_phase_self_structure_metrics_closed_structure.csv](C:\Code\dev\PythonProject\timematch\result\baseline_analysis\source_phase_self_structure_metrics_closed_structure.csv)

### 6.2 How phase boundaries are defined

#### Uniform partition

Uniform partition simply splits the temporal grid into `K` contiguous equal-length chunks:

\[
\{1,\dots,T\} \rightarrow \mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_K
\]

using:

- `phase_count = K`
- equal partition over the interpolated temporal grid

#### Structure-driven partition

Structure-driven partition does **not** cut by calendar position directly.
Instead, it first computes a **per-time-step structure state** on the source domain.

For each time step \(t\), we compute:

1. time-step separability
\[
\mathrm{Sep}_t = \frac{B_t}{W_t + \varepsilon}
\]

2. time-step margin
\[
\mathrm{Margin}_t = \frac{1}{C}\sum_{c=1}^{C}
\frac{\min_{c' \neq c}\|\mu_c^{(t)} - \mu_{c'}^{(t)}\|_2}
{\sqrt{\frac{1}{N_c}\sum_i \|h_{c,i}^{(t)} - \mu_c^{(t)}\|_2^2} + \varepsilon}
\]

3. time-step compactness proxy
\[
\mathrm{Comp}_t = \frac{1}{W_t + \varepsilon}
\]

where:

- \(h_{c,i}^{(t)}\) is the source feature at time step \(t\)
- \(\mu_c^{(t)}\) is the class center at time step \(t\)
- \(B_t\) is between-class scatter
- \(W_t\) is within-class scatter

Then each curve is smoothed, standardized, and stacked into a state vector:

\[
u_t =
\big[
z(\mathrm{Sep}_t),
z(\mathrm{Margin}_t),
z(\mathrm{Comp}_t)
\big]
\]

Finally, we find `phase_count - 1` boundaries so that the sequence
\[
u_1, u_2, \dots, u_T
\]
is segmented into `phase_count` contiguous parts with minimum within-segment SSE.

So the structure-driven phases are:

- not equal-length
- not manually tied to early/mid/late semantics
- instead tied to **changes in source-domain class-structure state over time**

### 6.3 Metric definitions

Let the `k`-th phase be \(\mathcal{T}_k\).

#### `source_phase_separability_pk`

For phase \(k\), average the time-step separability over all time steps in that phase:

\[
\mathrm{source\_phase\_separability}_{p_k}
=
\frac{1}{|\mathcal{T}_k|}
\sum_{t \in \mathcal{T}_k}
\frac{B_t}{W_t + \varepsilon}
\]

Interpretation:

- how discriminative the source domain is in that phase overall

#### `source_phase_margin_pk`

For phase \(k\), average the time-step nearest-class margin:

\[
\mathrm{source\_phase\_margin}_{p_k}
=
\frac{1}{|\mathcal{T}_k|}
\sum_{t \in \mathcal{T}_k} \mathrm{Margin}_t
\]

Interpretation:

- how safe the class boundaries are in that phase

#### `source_phase_compactness_pk`

For phase \(k\), use the inverse within-class scatter:

\[
\mathrm{source\_phase\_compactness}_{p_k}
=
\frac{1}{\frac{1}{|\mathcal{T}_k|}
\sum_{t \in \mathcal{T}_k} W_t + \varepsilon}
\]

Interpretation:

- how stable and tight same-class features are in that phase

This metric is especially important because it is directly compatible with:

- phase prototype reliability
- source-side structural regularization
- phase-aware weighting

### 6.4 Results

#### Uniform 5-phase

Top correlations with `target_f1`:

- `source_phase_compactness_p1`: `+0.6727`
- `source_phase_compactness_p2`: `+0.6309`
- `source_phase_margin_p2`: `+0.5839`
- `source_phase_compactness_p3`: `+0.5788`
- `source_phase_compactness_p4`: `+0.4291`
- `source_phase_compactness_p5`: `+0.3991`

#### Structure-driven 5-phase

Top correlations with `target_f1`:

- `source_phase_compactness_p2`: `+0.6872`
- `source_phase_compactness_p1`: `+0.6259`
- `source_phase_margin_p3`: `+0.5612`
- `source_phase_compactness_p4`: `+0.5064`
- `source_phase_compactness_p3`: `+0.4978`
- `source_phase_compactness_p5`: `+0.4207`
- `source_phase_separability_p3`: `+0.3506`

Example boundaries from the structure-driven 5-phase run:

- `FR1 -> FR2`: `3, 6, 9, 19`
- `FR1 -> DK1`: `6, 11, 16, 19`
- `FR1 -> AT1`: `3, 6, 9, 18`

#### Structure-driven 3-phase

Top correlations with `target_f1`:

- `source_phase_compactness_p1`: `+0.6786`
- `source_phase_compactness_p2`: `+0.5203`
- `source_phase_margin_p2`: `+0.4964`
- `source_phase_separability_p3`: `+0.4045`
- `source_phase_separability_p2`: `+0.3753`
- `source_phase_margin_p3`: `+0.3720`

Example boundaries from the structure-driven 3-phase run:

- `FR1 -> FR2`: `6, 19`
- `FR1 -> DK1`: `5, 18`
- `FR1 -> AT1`: `6, 18`

### 6.5 What these results mean

The key conclusion is:

1. the source-domain self-structure line is valid
2. feature-curve-based phase metrics are much more actionable than coarse whole-domain descriptors
3. the most useful structure indicators are:
   - **phase compactness**
   - **phase margin**
4. separability has signal, but should currently be treated as secondary
5. structure-driven phase partition is worth keeping, because it produces meaningful non-uniform boundaries and slightly stronger top signals than the uniform 5-phase baseline

Most importantly, this changes how we should phrase the optimization goal.

The goal is **not**:

- "make the source domain more complex"

The goal is:

- make source representations **more compact within class** in informative phases
- make source representations **more margin-separated across classes** in informative phases

So "better source self-structure" should now be interpreted as:

- tighter phase-wise class structure
- safer phase-wise class boundaries
- not generic variability inflation

These correspond to the "class-relative structure" idea more directly.

#### `class_relative_curve_structure_mse`

How to compute:

1. take source class mean raw curves and flatten them
2. compute source class-to-class pairwise distance matrix
3. do the same for target
4. compute mean squared difference between the two matrices

Interpretation:

- how much the class-relative raw temporal geometry changes across domains

#### `class_relative_curve_structure_corr`

How to compute:

1. build source and target class-relative raw distance matrices
2. extract upper-triangular pairwise entries
3. compute correlation between the two vectors

Interpretation:

- how well source and target preserve the same class-relative structure ordering

### 5.7 Auxiliary legacy metrics

These are kept for completeness, but are no longer the main focus.

#### `mmd`

- RBF-kernel MMD between source and target feature samples

#### `coral`

- covariance alignment difference between source and target features

#### `acf_distance`

- mean absolute difference between source and target raw-curve autocorrelation summaries

## 6. Better Self-Structure Indicators For The Next Stage

The current self-structure metrics are useful, but some are still too coarse to directly guide adaptation design.
For the next stage, we should prioritize **interpretable, actionable self-structure indicators** built on feature curves.

### 6.1 Why feature curves are likely more effective

Raw curves are easy to interpret, but they mix:

- acquisition noise
- band-level redundancy
- irrelevant amplitude variation

Feature curves from the encoder are likely more useful because they are already shaped by the supervised objective and therefore better reflect class-discriminative temporal structure.

So the recommended principle is:

- keep **raw NDVI-based metrics** as interpretable anchors
- use **encoded feature-curve metrics** as the main optimization-facing structure indicators

### 6.2 Recommended next self-structure indicators

These are not all implemented yet, but they are the most promising directionally-correct replacements or refinements.

#### A. Source phase separability

Definition idea:

1. split each class feature curve into early / mid / late phases
2. compute class means per phase
3. compute between-class distance within each phase
4. divide by within-class phase dispersion

This would tell us:

- in which phase the source domain is actually discriminative

Why it helps optimization:

- it directly suggests where alignment should focus
- e.g. stronger adaptation on highly discriminative phases

#### B. Source class compactness over time

Definition idea:

1. for each class, measure the average distance from sample feature curves to the class mean feature curve
2. aggregate over time or by phase

This would tell us:

- whether source classes are stable and tight enough to define reliable temporal anchors

Why it helps optimization:

- compact classes support prototype-based or phase-anchor-based adaptation

#### C. Source phase margin

Definition idea:

1. for each phase, compute nearest-class distances between class mean feature curves
2. compare them with intra-class dispersion

This would tell us:

- whether the source domain has clear class margins in specific phases

Why it helps optimization:

- directly motivates phase-selective alignment or phase-weighted losses

#### D. Source temporal transition sharpness

Definition idea:

1. compute first differences of class mean feature curves
2. measure where strong transitions happen and how concentrated they are

This would tell us:

- whether source classes have sharp, informative phenological transitions

Why it helps optimization:

- can guide whether adaptation should emphasize transition regions or stable regions

### 6.3 What to keep vs. what to downgrade

Keep as primary self-structure metrics:

- `source_ndvi_q90_range_mean`
- `source_fisher_ratio`
- future `source phase separability`
- future `source phase margin`

Keep as supporting descriptors:

- `source_curve_spread`
- `source_curve_activity_range`
- `source_ndvi_q90_range_std`
- `source_class_curve_variance_mean`

### 6.4 Phase partition directions worth keeping

Uniform early/mid/late splitting is acceptable as a baseline, but it is too crude to become the long-term definition of temporal structure.  
For the next stage, the following three phase-partition directions should be kept as the main candidates.

#### Direction 2. Change-intensity-based phase partition

Core idea:

- use temporal change strength to define phase boundaries
- separate relatively stable segments from strong-transition segments

Typical signals:

- first-order difference magnitude
- second-order difference magnitude
- local variance / local energy
- peaks of temporal change

Why it matters:

- more general than crop-specific phenology
- can transfer to broader time-series classification tasks

Main limitation:

- sensitive to noise
- sample-specific boundaries can be hard to aggregate across a domain

#### Direction 3. Class-structure-driven phase partition

Core idea:

- do not split by time alone
- split by where the source domain shows different structural roles

Typical signals per time step:

- class separability
- nearest-class margin
- within-class compactness
- between-class scatter vs. within-class scatter

Typical partition result:

- highly discriminative phases
- weakly discriminative phases
- transition phases

Why it matters:

- directly aligned with the project goal of using source-domain internal structure to guide transfer
- more actionable than raw change-based partitioning

Current recommendation:

- this should be the priority phase-partition direction

#### Direction 4. Learned soft phase partition

Core idea:

- do not force hard boundaries
- assign each time step softly to one of several latent phases

Typical form:

- learn phase assignment weights from time-step features
- aggregate latent phase representations with weighted sums

Why it matters:

- flexible
- task-adaptive
- can potentially discover phases that fixed rules miss

Main limitation:

- less interpretable than explicit partitions
- better suited for later method design than for the very first analysis pass

Current recommendation:

- keep as a later-stage method direction after fixed-rule structure analysis is clearer

## 7. Practical Reproduction Notes

If we rerun these metrics later, the main things to keep fixed are:

- same baseline checkpoints
- same `closed_set` or `open_set` setting
- same class filtering rule (`count >= 200`)
- same `temporal_grid_size`
- same `max_shift_steps`
- same feature extractor (`pseltae` unless intentionally changed)

If we want continuity with current results, do **not** silently change:

- interpolation grid size
- NDVI channel indices
- class filtering threshold
- whether metrics are computed from raw curves or encoded PSE curves

## 8. Current Recommendation

For the next method-definition stage:

- keep **domain self-structure** as the main motivation line
- treat **source-target structure gap** only as a secondary design extension
- for future metric design, move from coarse domain-level variability toward **phase-aware feature-curve self-structure**
- prioritize structure definitions that are:
  - interpretable
  - phase-specific
  - directly convertible into optimization targets
