# Project Progress Summary

## Current scope
- Main baseline has moved from the original open-set setting to a **closed-set** setting.
- PRA remains a single-pair validation track centered on `FR1 -> DK1`.
- Multi-source closed-set TimeMatch baseline has now been rerun for all 12 transfer tasks.
- Structure analysis is now a separate analysis track based on the saved closed-set baseline checkpoints rather than new training runs.

## Why the project focus changed
- Earlier discussion started from the idea that "more complex source domains may transfer better".
- However, source-only complexity measures did not align well with actual adaptation outcomes.
- After switching to closed-set evaluation and computing task-level structure metrics, the project focus became:

> The key question is not which domain is more complex by itself, but which source domain provides **more stable and more useful transferable structure** for a target domain.

- This means the analysis target is now `source -> target` pairs rather than standalone source domains.

## Closed-set baseline overview

| Task | TimeMatch |
|---|---:|
| FR1 -> FR2 | 0.7998860 |
| FR1 -> DK1 | 0.6391047 |
| FR1 -> AT1 | 0.7395250 |
| FR2 -> FR1 | 0.7146287 |
| FR2 -> DK1 | 0.4652110 |
| FR2 -> AT1 | 0.6486893 |
| DK1 -> FR1 | 0.5917054 |
| DK1 -> FR2 | 0.5071109 |
| DK1 -> AT1 | 0.5264147 |
| AT1 -> FR1 | 0.7031468 |
| AT1 -> FR2 | 0.6226317 |
| AT1 -> DK1 | 0.7511958 |

## Closed-set vs previous open-set baseline
- Compared with the earlier open-set baseline table, the closed-set rerun improved in most tasks.
- Among the 11 tasks with directly comparable old open-set results:
  - 8 improved
  - 3 decreased
- Large gains appeared in:
  - `DK1 -> FR2`
  - `FR1 -> FR2`
  - `FR1 -> AT1`
- This confirms that removing the `unknown` bucket and enforcing a proper closed-set protocol materially changes the transfer landscape.

## Closed-set data loading change
- `unknown` is no longer kept in the active class list.
- Rare source classes and classes that do not belong to the source closed set are excluded during loading rather than collapsed into `unknown`.
- As a result:
  - target classes used in training/evaluation are always a subset of the source closed-set classes
  - source-only rare classes and target-only unseen classes are removed

## PRA status
- PRA has been tested mainly on `FR1 -> DK1`.
- Multiple variants were tried:
  - point/edge trade-off tuning
  - bank momentum tuning
  - pseudo-threshold tuning
  - pseudo-label filtering variants
  - geometry normalization before point/edge alignment
- The consistent outcome is:
  - PRA can train stably
  - PRA can get close to baseline
  - PRA has **not** shown a stable gain over the debugged TimeMatch baseline

## Structure analysis track

### Goal
- Before adding new structure losses, determine which structure metrics actually explain transfer performance.
- Reuse the 12 closed-set TimeMatch baseline checkpoints and compute task-level structure metrics for each `source -> target` pair.

### Metrics currently implemented
- `Target F1`
- `MMD`
- `CORAL`
- `Prototype Distance`
- `Relation Structure Distance`
- `ACF Distance`

### Current analysis table

| Source | Target | Target F1 | MMD | CORAL | Prototype Distance | Relation Structure Distance | ACF Distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| AT1 | DK1 | 0.7512 | 0.0849 | 0.001696 | 2.6984 | 0.000617 | 0.0581 |
| AT1 | FR1 | 0.7031 | 0.0646 | 0.000263 | 1.6566 | 0.000830 | 0.0662 |
| AT1 | FR2 | 0.6226 | 0.0697 | 0.000183 | 1.1330 | 0.000601 | 0.0678 |
| DK1 | AT1 | 0.5264 | 0.0612 | 0.001852 | 3.0822 | 0.000994 | 0.0581 |
| DK1 | FR1 | 0.5917 | 0.0884 | 0.001355 | 3.0653 | 0.000892 | 0.0444 |
| DK1 | FR2 | 0.5071 | 0.0902 | 0.000152 | 2.3737 | 0.001596 | 0.0670 |
| FR1 | AT1 | 0.7395 | 0.0458 | 0.000009 | 0.7958 | 0.001419 | 0.0665 |
| FR1 | DK1 | 0.6391 | 0.0933 | 0.000033 | 1.3149 | 0.001948 | 0.0441 |
| FR1 | FR2 | 0.7999 | 0.1047 | 0.000013 | 0.6940 | 0.001205 | 0.0766 |
| FR2 | AT1 | 0.6487 | 0.0648 | 0.003278 | 2.4584 | 0.000579 | 0.0696 |
| FR2 | DK1 | 0.4652 | 0.0572 | 0.001645 | 2.7916 | 0.001897 | 0.0667 |
| FR2 | FR1 | 0.7146 | 0.0976 | 0.000415 | 1.0041 | 0.000457 | 0.0766 |

### Correlation with target F1
- `prototype_distance`: `-0.6518`
- `relation_structure_distance`: `-0.4135`
- `coral`: `-0.2866`
- `acf_distance`: `+0.2818`
- `mmd`: `+0.2765`

## What the current analysis shows

### 1. Prototype distance is the strongest signal so far
- The clearest result is that **smaller source-target prototype distance tends to correspond to higher target F1**.
- This means that category-level structure stability is currently the most useful explanatory factor.

### 2. Relation structure matters, but is secondary
- `relation_structure_distance` also shows a useful negative relationship with target F1.
- However, it is clearly weaker than `prototype_distance`.
- This suggests:
  - point-level prototype alignment is likely more important than relation-only alignment
  - relation structure may still help, but it should not be the first design priority

### 3. Source-only complexity is not the main story
- Earlier source-only MSE-based "separability/complexity" did not explain the observed transfer results well.
- The current evidence points toward:

> Transfer performance depends more on **source-target category structure stability** than on source-only structural strength.

### 4. ACF / global discrepancy are not yet the main drivers
- `MMD`, `CORAL`, and `ACF Distance` currently show weaker explanatory power than prototype-based measures.
- This does not mean temporal structure is irrelevant, but it suggests the current implementation is not capturing the dominant transferable factor yet.

## Method-design implication
- If a new structure-based adaptation method is developed, the first priority should be:
  1. improve target prototype construction
  2. stabilize cross-domain prototype alignment
  3. only then consider stronger relation/graph modeling

- In other words, current evidence supports:
  - **prototype quality first**
  - **relation structure second**

## Pseudo labels vs clustering for target structure
- At this stage, the preferred mainline remains:
  - **high-confidence pseudo labels as the first main solution**
  - **clustering as an auxiliary or comparative solution**
- Reason:
  - the strongest current signal is category-level prototype stability
  - pseudo-label prototypes are more directly tied to semantic class structure
- Clustering is still worth studying, especially as:
  - a target-structure discovery tool
  - a comparison baseline
  - or a first-stage grouping method before semantic refinement

## What the latest prototype variants changed
- Three recent prototype-centered variants on `FR1 -> DK1` now give almost the same result:
  - `RGPA-v1`: `0.5999`
  - `ProtoCon-pure-v1`: `0.5996`
  - `MemProto-v1`: `0.5989`
- The closed-set baseline remains:
  - `0.6391047`
- This means:
  - prototype structure is still the strongest **explanatory** signal
  - but current prototype-side losses are **not yet** the right optimization mechanism

## What the v3 temporal analysis changed
- The new `v3` analysis compared:
  - raw temporal curves
  - phase/window variants
  - trend / detrended / frequency descriptors
  - and the same family of metrics on **PSE temporal feature curves**
- The strongest new correlations with `target_f1` are now:
  - `pse_trend_curve_distance`: `-0.6997`
  - `pse_early_curve_distance`: `-0.6855`
  - `pse_mid_curve_distance`: `-0.6845`
  - `pse_temporal_curve_distance`: `-0.6573`
  - `prototype_distance`: `-0.6518`
  - `pse_frequency_magnitude_distance`: `-0.6393`
- This is the first time a new metric family slightly exceeds `prototype_distance`.
- The main conclusion is:

> encoded temporal structure is more informative than raw temporal curve geometry

## What this means structurally
- Raw-curve whole-season distances remain weak or unstable.
- Simple seasonal three-way splitting did not add much signal.
- Approximate phenology splitting based only on curve peak was also weak.
- In contrast, **PSE feature trajectories** show:
  - stronger signal than raw curves
  - stronger signal than naive whole-curve comparison
  - and clear evidence that **early / mid season** matter more than late season

## Updated judgment on phase structure
- The project should now treat **phase-aware encoded temporal structure** as a top-priority mainline.
- The key unit is no longer just:
  - one sample
  - one pooled feature
  - one class prototype
- A better unit is now likely:
  - `class x phase`
  - or `sample phase -> class phase structure`

## Updated judgment on decomposition
- Trend-level structure is now strongly supported:
  - `pse_trend_curve_distance` is the strongest metric in `v3`
- Frequency structure is also useful on the encoded side:
  - `pse_frequency_magnitude_distance` remains competitive
- So future temporal structure design should not rely only on:
  - raw pointwise curve distance
- It should explicitly consider:
  - encoded trend
  - encoded local seasonal windows
  - encoded phase-specific semantics

## Updated judgment on clustering
- Clustering may still help, but it should **not** be assumed to be the next primary solution.
- Current evidence suggests:
  - if clustering is used, it should probably serve as a **supporting structure discovery mechanism**
  - rather than fully replacing semantic pseudo labels as the main supervision source
- The main risk of a clustering-first route is:
  - it may improve geometric compactness
  - while drifting away from the semantic categories that actually define evaluation performance

## Updated judgment on temporal structure
- Compared with purely spatial/category prototype structure, **encoded temporal structure is now a stronger next candidate than more prototype-loss tweaking**.
- Reason:
  - current prototype-distance analysis says category-level alignment matters
  - but the existing prototype methods still plateau around `0.599`
  - `v3` now suggests the missing transferable signal lies in **how category identity is expressed over encoded time structure**, not only in static feature geometry
- In other words:

> the model may already know "which class center to approach",  
> but still not represent "how that class evolves across the season" well enough across domains

## More precise next-step priority
- The next method-design priority should now be:
  1. preserve semantic supervision through pseudo labels
  2. enrich structure definition with **phase-aware PSE temporal signals**
  3. use clustering only as an auxiliary view of target organization

## Candidate temporal-structure directions
- Phase-aware PSE alignment:
  - align early / mid / late encoded trajectories rather than only pooled sample features
- Class-phase prototype modeling:
  - treat each class as a sequence of phase prototypes instead of a single static point
- Encoded trend alignment:
  - compare low-frequency semantic evolution across domains
- Encoded frequency alignment:
  - compare temporal spectrum magnitude after PSE encoding
- Teacher-consistency-over-phase:
  - prefer target samples whose phase-wise semantics remain stable under shift/augmentation

## Files added for analysis
- Closed-set transfer analysis script:
  - [analyze_closed_set_transfer.py](/C:/Code/dev/PythonProject/timematch/analyze_closed_set_transfer.py)
- Current analysis result table:
  - [closed_set_transfer_metrics.csv](/C:/Code/dev/PythonProject/timematch/result/baseline_analysis/closed_set_transfer_metrics.csv)
  - [closed_set_transfer_metrics_v2.csv](/C:/Code/dev/PythonProject/timematch/result/baseline_analysis/closed_set_transfer_metrics_v2.csv)
  - [closed_set_transfer_metrics_v3.csv](/C:/Code/dev/PythonProject/timematch/result/baseline_analysis/closed_set_transfer_metrics_v3.csv)

## Next steps
1. Use the `v3` result as the new decision point:
   - encoded temporal structure has enough signal to justify a training method
2. Design and implement a first **Phase-Aware PSE Structure Alignment** method:
   - phase-wise encoded features
   - class-phase prototypes
   - trend-sensitive structure loss
3. Keep clustering as a comparison or support track, including:
   - clustering-based phase discovery
   - clustering vs pseudo-label phase agreement
4. Only after the first phase-aware encoded method is tested, decide whether to add stronger decomposition or joint completion modules.

## Current mainline conclusion

> At the current stage, the most promising direction is not to keep expanding PRA variants or source-only complexity analysis, but to study **phase-aware encoded temporal structure**, especially how source and target classes align through PSE feature trajectories and class-phase semantics rather than only as static prototype points.
