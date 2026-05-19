# Gain Validation Launchers

This directory contains the active scripts for validating source-side temporal
structure shaping. Older selector, checkpoint-bank, GTW, and controller scripts
were removed from the active launcher path; use git history if a legacy
experiment needs to be recovered.

## Main Scripts

- `run_v243b_boundarywindow_repeats_50_70_5x.sh`
- `run_v243b_negative_task_structure_weight_sweep.sh`
- `run_v243b_remaining_negative_structure_weight_sweep.sh`
- `run_global_compact_fr1_fr2_sweep.sh`
- `run_v244a_pointwise_dynamics_12tasks.sh`
- `run_v244b_global_trajectory_12tasks.sh`
- `run_structure_component_short_probe.sh`
- `run_structure_component_overnight_probe.sh`
- `run_v25_theory_followup_probe.sh`
- `run_v25_boundary_transition_probe.sh`

## Structure Component Probes

`run_structure_component_short_probe.sh` is the fast check before a long run.
By default it probes `FR2 -> FR1`, a task where segmented structure has a strong
signal. It compares:

- `seg_full`: v2.4.3b-style segmented structure.
- `seg_no_dual`: same structure without dual-path cls/relation preservation.
- `seg_intra_only`: segmented intra compactness only.
- `global_compact`: no-segment global compactness.

`run_structure_component_overnight_probe.sh` is the longer unattended v2.5
partition-theory validation run. By default it probes `FR1 -> FR2`, `FR2 ->
FR1`, and `DK1 -> FR1` with:

- `doy_k5_full`
- `doy_k5_no_dual`
- `doy_k5_intra_only`
- `doy_k5_transition_only`
- `doy_k5_boundary_weighted`
- `uniform_k5_full`
- `uniform_k2_full`
- `uniform_k10_full`
- `global_k1_compact`
- `global_k1_dynamics`

These variants are intended to test whether gains come from compactness,
partition granularity, DOY-gap placement, boundary weighting, dynamics, or
preservation constraints. Random/permuted partition controls are not included
because the current training code does not expose a safe random partition mode.

`run_v25_theory_followup_probe.sh` is the focused follow-up after the
partition-theory validation run. It avoids boundary/dynamics-heavy variants and
tests the theory claims on additional high-information tasks:

- compactness as the core source-side structure objective;
- global / uniform / DOY-gap as alternative temporal structure views;
- dual-path relation as a semantic-preservation constraint, not a structure
  component.

`run_v25_boundary_transition_probe.sh` is the local-transition follow-up. It
keeps the same four theory tasks and tests whether the third v2.5 structure
view should be treated as boundary-centered transition shaping. Since the
current boundary-window term is a weighting modulator for adjacent
segment-inter transitions rather than an independent loss, the script uses a
clean factorial comparison:

- `uniform_k5_intra_ref`: phase compactness only.
- `uniform_k5_transition`: adds adjacent segment transition consistency.
- `uniform_k5_boundary02`: reweights transition consistency around boundaries.
- `uniform_k5_boundary05`: stronger boundary weighting sensitivity check.
- `doy_k5_transition`: same transition test under DOY-gap partitions.
- `doy_k5_boundary02`: DOY-gap boundary-weighted transition.

Trend is disabled in these variants so gains can be attributed to transition
and boundary weighting instead of the older smooth-trend regularizer.

## Shared Block

- `run_source_structure_block.sh`

The shared block owns the common source training plus TimeMatch DA call. Keep
new gain-validation scripts pointed at this block so parameter differences stay
visible in the top-level launcher.
