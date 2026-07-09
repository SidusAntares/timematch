# v2.8 Cleaned-Code Reproduction Plan

## Goal

Return to cleaned-code source-side structure shaping.

This is a plan only.  Do not run these experiments until the residual local-shift archive is complete.

## Planned Configs

```text
1. cleaned base TimeMatch
2. raw_global source structure + TimeMatch
3. smooth_k3 source structure + TimeMatch
4. elastic_r2 source structure + TimeMatch
```

## Initial Scope

```text
tasks:
  AT1->DK1
  FR1->FR2
  FR2->AT1
  FR2->FR1

seeds:
  1, 2, 3
```

## Purpose

```text
Verify that cleaned refactored code reproduces v2.8 conclusions.
Compare all settings within the same code version.
Do not rely on old historical runs as strong evidence.
If the result is consistent, then consider full12.
```

## Metrics

```text
source-on-target macro-F1
DA macro-F1
DA gain
runtime
shift statistics
pseudo confidence
pseudo ratio
```

## Stop Condition

If the cleaned-code baseline differs strongly from the old baseline, diagnose the baseline mismatch before running full12.

