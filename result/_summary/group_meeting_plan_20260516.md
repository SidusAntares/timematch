# Group Meeting Plan 2026-05-16

## Goal

Prepare a complete, defensible stage result for the Saturday morning group meeting.

The goal is not to finish every possible experiment. The goal is to present a coherent story and a near-complete framework:

> source-side temporal structure shaping improves TimeMatch transfer, and the remaining bottleneck is adaptive structural weighting / checkpoint choice.

By Friday night, the project should have:

- a stable best-result table,
- a clear method evolution story,
- one latest adaptive experiment result,
- a complete framework diagram / pipeline explanation,
- and a concrete next-step plan.

Because repeated runs have shown non-trivial variance, the final report should distinguish:

- **best run**: used to show the current potential / upper performance;
- **mean ± variance over repeats**: used to show reliability and avoid overclaiming.

For the group meeting, it is acceptable to highlight the best result, but the slide or backup table should also disclose that repeated training has measurable variance.

## Core Storyline

### 1. Problem

Cross-domain time-series crop mapping suffers from temporal structure mismatch.

The baseline TimeMatch adaptation can align domains, but source initialization quality and source temporal structure strongly affect final transfer.

### 2. Main Method Line

The project moved from simple source-only training to source-side temporal structure shaping:

- `v2.2.3`: stable comparison baseline.
- `v2.4.3b`: current best source-structure backbone.
- `v2.5.0`: checkpoint bank oracle shows different source checkpoints suit different targets.
- `v2.5.1-v2.5.4`: checkpoint selection attempts show target-aware selector is hard.
- `v2.5.5`: current source-side structure adaptivity line.

### 3. Main Claim

The best current contribution should be framed as:

> A source-side temporal-structure shaping method for TimeMatch, with evidence that structural weighting and checkpoint choice are transfer-sensitive.

Do not overclaim that the adaptive gate is solved.

## Must-Have Deliverables Before Friday Night

### A. Result Tables

Prepare these tables:

1. Baseline / `v2.2.3` / `v2.4.3b` 12-task comparison.
2. `v2.4.3a/b/c` comparison if already documented.
3. `v2.5.0` checkpoint bank oracle table.
4. `v2.5.1-v2.5.4` selector failure summary.
5. `v2.5.5b` source-weight ablation table.

The presentation should emphasize:

- `v2.4.3b` as current reliable method.
- `v2.5.0` as evidence that checkpoint/source training stage matters.
- `v2.5.5b` as evidence that source structural weights matter.

### B. Best Experimental Effect

For headline performance, use the strongest stable table available:

- Primary: `v2.4.3b` full 12-task result.
- Secondary: `v2.5.0` oracle checkpoint bank upper bound.
- Exploratory: `v2.5.5b` source-weight ablation improvements on representative pairs.

Avoid presenting `v2.5.1-v2.5.4` as successful algorithms. Present them as ablation / negative evidence that shaped the final direction.

### C. Written Summary

Create/update:

- `result/_summary/v2.4.3.md`
- `result/_summary/v2.5.0.md`
- `result/_summary/v2.5.1-v2.5.4_selector_lessons.md`
- `result/_summary/v2.5.5b_source_weight_ablation.md`
- `result/_summary/final_stage_summary_for_group_meeting.md`

## Timeline

### Thursday Night

Current running experiment:

- Let `v2.5.5b` finish.
- Do not start another large full experiment before reading it.

When logs return:

1. Parse `baseline`, `transition_light`, and `structure_light`.
2. Decide whether `structure_light` is useful.
3. Write a compact `v2.5.5b` summary.

Decision rules:

- If `transition_light` or `structure_light` improves most tasks but hurts one:
  - conclusion = source adaptivity is necessary, not one global setting.
- If one variant improves all or nearly all representative tasks:
  - consider a Friday daytime full-12 run for that variant.
- If results are noisy:
  - do not full-12; instead present it as exploratory evidence.

### Friday Daytime

Focus on analysis and documentation first.

Priority order:

1. Finish all result extraction.
2. Produce clean tables.
3. Write the stage summary.
4. Only then decide whether to launch another experiment.

Recommended experiment if `v2.5.5b` has clear signal:

- Run a small 2x2 decomposition on two pairs:
  - `30TXT -> 33UVP` as gain pair
  - `31TCJ -> 32VNH` as loss pair
- Compare:
  - baseline
  - boundary-light only
  - segment-inter-light only
  - both-light

Purpose:

> identify whether the observed gain/loss comes from boundary-window pressure or inter-segment pressure.

Do not start a full 12-task run unless the signal is already clean.

### Friday Night

Freeze the presentation version.

Minimum freeze criteria:

- all key tables written,
- final summary md written,
- latest `v2.5.5b` interpreted,
- complete framework figure / text ready,
- next-step plan written.

After freezing, optional overnight run:

- If one adaptive variant is clearly good:
  - run repeat experiments for that variant and baseline on representative tasks.
  - if time allows, run full 12 tasks for that variant.
- If no variant is clearly good:
  - run the 2x2 decomposition with one repeat on representative tasks.

The overnight run should be treated as performance polishing / reliability confirmation. The Saturday presentation must still be understandable without it.

### Saturday Morning

Use overnight results only if they are clean and easy to explain.

If overnight results are incomplete or contradictory:

- do not force them into the main story.
- mention them as "ongoing verification" only.

## Presentation Structure

### Slide 1: Task and Motivation

Cross-region / cross-time time-series UDA, with TimeMatch as the adaptation backbone.

### Slide 2: Core Hypothesis

Source temporal structure quality affects target adaptation.

### Slide 3: Method Evolution

Show version path:

`baseline -> v2.2.3 -> v2.4.3b -> v2.5.0 -> v2.5.5`

### Slide 4: Complete Framework

Explain:

- source-only structure shaping,
- checkpoint bank,
- target-aware checkpoint choice as the target-adaptivity interface,
- optional DA-stage feedback / controller as future completion.

### Slide 5: Source Structure Design

Explain:

- temporal segments,
- trend / residual / boundary / inter-segment components,
- source-only supervision.

### Slide 6: Main 12-Task Result

Baseline vs `v2.2.3` vs `v2.4.3b`.

### Slide 7: Checkpoint Bank Finding

`v2.5.0` shows source training stage is target-sensitive.

### Slide 8: Selector Attempts and Lesson

Show that short-warmup selectors are not reliable enough.

Message:

> target adaptivity is real, but unsupervised selection is hard.

### Slide 9: Source Weight Adaptivity

Show `v2.5.5b` representative-pair result.

Message:

> changing structural weights changes the transfer curve, supporting source-side adaptivity.

### Slide 10: Current Best and Reliability

Best stable method:

- `v2.4.3b`

Best upper-bound insight:

- checkpoint bank oracle / source-weight ablation.

Report:

- best run as potential;
- repeat mean / variance as reliability evidence when available.

Limitation:

- adaptive controller / selector is not solved yet.

### Slide 11: Next Step

Move from fixed structural weights to bounded adaptive weighting:

- source-side component selection,
- target-side checkpoint selection,
- DA-stage weak feedback.

## What Not To Do Before Group Meeting

- Do not start a completely new architecture line such as MoE backbone replacement.
- Do not spend time optimizing failed selector heuristics.
- Do not present `v2.5.1-v2.5.4` as final methods.
- Do not wait for overnight experiments before writing the main summary.
- Do not present only the best run without acknowledging run-to-run variance somewhere in the backup table.

## Immediate Next Actions

1. Wait for full `v2.5.5b` logs.
2. Parse and summarize `v2.5.5b`.
3. Write `v2.5.5b_source_weight_ablation.md`.
4. Write `final_stage_summary_for_group_meeting.md`.
5. Prepare a complete framework diagram / pipeline explanation.
6. Decide Friday daytime whether to run 2x2 decomposition, repeat experiments, or full-12 variant verification.
