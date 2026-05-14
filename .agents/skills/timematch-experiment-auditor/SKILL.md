---
name: timematch-experiment-auditor
description: Use this skill whenever working on the TimeMatch project and the user asks to analyze experiment logs, compare versions, summarize v2.x results, decide next experiments, inspect checkpoint-bank/source-weight ablations, or write/update result markdown files. This skill is project-specific and should trigger for TimeMatch logs under logs/, result/_summary/, launchers/, v2.4.x/v2.5.x experiments, sourcephasecompact, checkpoint selection, source-side structure adaptivity, or DA result comparisons.
---

# TimeMatch Experiment Auditor

This skill keeps TimeMatch experiment analysis consistent. Use it to avoid re-learning the project history every turn and to produce comparable, decision-oriented summaries.

## Project Context

The current mainline is source-only temporal structure shaping for time-series UDA.

Important anchors:

- `v2.4.3b` is the current validated source-structure backbone.
- `v2.5.0` checkpoint bank/oracle showed that different source checkpoints can be best for different source-target pairs.
- `v2.5.1` and much of `v2.5.3` showed that short-warmup target checkpoint selectors are unreliable.
- `v2.5.4b` offline selector audit concluded not to keep tuning static short-warmup selector heuristics as the main route.
- `v2.5.5` is the active source-side structure-adaptivity route.

Treat checkpoint selection as an analysis tool, oracle upper-bound reference, and possible later auxiliary module, but not the immediate mainline unless the user explicitly redirects.

## Default Analysis Workflow

When the user says logs have been transferred or asks for result analysis:

1. Locate the newest relevant log directory.
2. Extract structured rows rather than reading logs manually line by line.
3. Separate source-only results from DA results.
4. For checkpoint-bank experiments, group by variant, source tile, target tile, and checkpoint epoch.
5. For each group, report all checkpoint F1 values, best epoch, best F1, and difference vs baseline or previous best when available.
6. Check progress markers and errors: `V255B_DA_START`, `V255B_DA_DONE`, `SELECTION_RESULT`, `Traceback`, `RuntimeError`, `CUDA out of memory`, and `Killed`.
7. Make a decision-oriented conclusion: what is supported, what remains uncertain, whether to continue running, and what exact next experiment should be run.

Do not overclaim from partial logs. Use language like "intermediate signal" or "not final yet" when experiments are incomplete.

## Metrics Convention

Prefer macro F1 from the `Test result ... f1=` line when available.

If both sklearn table and compact result line appear:

- Use `Test result for ... accuracy=..., f1=...` as the primary value.
- Use the classwise table only for diagnosing which classes moved.

When producing tables:

- Show values as `0.7365` rather than percentages unless the user asks for percent.
- Bold the best checkpoint within a row or version table.
- Include an `avg` row for full 12-task summaries.

## Key Experiment Types

### v2.4.3b / Source Structure Backbone

Compare against baseline and v2.2.3 when requested.

Focus on source-only F1, final DA macro F1, and whether the structure loss improves transfer rather than only source accuracy.

### v2.5.0 / Checkpoint Bank Oracle

Use this to establish upper bound and pair-dependent checkpoint preference.

Report per-task F1 at each checkpoint, best checkpoint, and oracle average.

Do not call this a deployable selector.

### v2.5.1-v2.5.4 / Selector Attempts

Judge by hit rate against oracle best checkpoint, near-hit rate, selected F1 vs oracle F1, and whether the method collapses into source-global choice rather than pair-dependent choice.

Current project lesson:

> static / short-warmup selector heuristics are not reliable enough as the main route.

### v2.5.5 / Source-Side Structure Adaptivity

Current question:

> Does changing source structure-loss component weights improve the oracle checkpoint curve in a consistent, explainable way?

For source-weight ablations, report source-only F1 per variant, DA F1 per checkpoint, best checkpoint per variant, gain/loss of variant best vs baseline best, and whether the whole checkpoint curve shifts up or only one point spikes.

Interpretation rules:

- If a variant improves most tasks but hurts one, conclude it supports adaptivity, not a new global default.
- If improvement is only in a single checkpoint, flag possible run variance.
- If source-only F1 changes little but DA F1 changes a lot, emphasize transfer-sensitive structure rather than source classification quality.

## Recommended Quick Extraction Pattern

Use a small read-only Python snippet for ad hoc extraction. Do not create throwaway files unless the user wants a reusable script.

```python
import re
from pathlib import Path

log_dir = Path("logs/<experiment_dir>")
pat = re.compile(
    r"Test result for timematch_(?P<src>[^_]+)_to_(?P<tgt>[^_]+).*?"
    r"_v255b_(?P<variant>baseline|transition_light|structure_light)_allckpt.*?"
    r"_(?P=variant)_epoch_(?P<epoch>\\d+): accuracy=(?P<acc>[0-9.]+), f1=(?P<f1>[0-9.]+)"
)

rows = []
for path in log_dir.glob("*.log"):
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows.extend(m.groupdict() | {"file": path.name} for m in pat.finditer(text))
```

Adapt the regex to the version tag.

## Markdown Summary Template

When writing `result/_summary/vX.md`, use:

```markdown
# vX Summary

## Motivation

## Implementation

## Experiment Setup

## Results

| task | baseline | variant/checkpoint results | best | delta |
|---|---:|---:|---:|---:|

## Interpretation

## Decision

## Next Step
```

Keep the conclusion falsifiable. Prefer "supports source-side adaptivity", "does not justify a global default", or "needs component decomposition" over vague statements like "works" or "failed".

## MCP Usage Guidance

Use MCP review tools for design sanity checks, not as the primary log parser.

Good uses:

- ask whether an experiment isolates the right variable,
- ask what result would falsify the hypothesis,
- ask what minimal next experiment should be.

Poor uses:

- sending huge logs for extraction,
- asking the MCP to infer project history from raw files,
- using review output as final truth without checking local logs.

If MCP disagrees with local evidence, local parsed results win.

## Final Response Style

For partial logs, start with completion status, then provide the compact result table, then give the current decision.

For completed experiments, include the final table, compare against baseline/oracle, and recommend the next code/script change only if the evidence supports it.

Avoid long historical recaps unless the user explicitly asks.
