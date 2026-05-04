## V2 Checkpoint Manifest

This file records the current local evidence for the experiment family we have
been referring to as `V2`.

### Experiment definition used in notes

- Source-only pretraining: `sourcephasecompact`
- Adaptation stage: plain `timematch`
- Naming pattern:
  - source model: `pseltae_<SRC>_closedset_noshift_sourcephasecompact_p5`
  - DA model: `timematch_<SRC>_to_<TGT>_closedset_noshift_sourcephasecompact_p5`

### Local log evidence

- `logs/sourcephasecompact_sourceonly_A.log`
- `logs/sourcephasecompact_sourceonly_B.log`

These logs show:

1. source-only stage with `method='sourcephasecompact'`
2. optional direct transfer evaluation with `--eval`
3. adaptation stage with `method='timematch'` and
   `weights='outputs/pseltae_<SRC>_closedset_noshift_sourcephasecompact_p5'`

### Source-model output directories

- `outputs/pseltae_30TXT_closedset_noshift_sourcephasecompact_p5`
- `outputs/pseltae_31TCJ_closedset_noshift_sourcephasecompact_p5`
- `outputs/pseltae_32VNH_closedset_noshift_sourcephasecompact_p5`
- `outputs/pseltae_33UVP_closedset_noshift_sourcephasecompact_p5`

### DA output directories

- `outputs/timematch_30TXT_to_31TCJ_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_30TXT_to_32VNH_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_30TXT_to_33UVP_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_31TCJ_to_30TXT_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_31TCJ_to_32VNH_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_31TCJ_to_33UVP_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_32VNH_to_30TXT_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_32VNH_to_31TCJ_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_32VNH_to_33UVP_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_33UVP_to_30TXT_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_33UVP_to_31TCJ_closedset_noshift_sourcephasecompact_p5`
- `outputs/timematch_33UVP_to_32VNH_closedset_noshift_sourcephasecompact_p5`

### Current blocker

The local workspace currently preserves metrics artifacts (`json`, `txt`, `pkl`)
for this experiment family, but does **not** preserve `fold_*/model.pt` under
these output directories.

This matters because `analysis/recompute_transfer_metrics.py` currently expects:

- `outputs/<experiment_name>/fold_*/model.pt`

Without those files, we cannot run checkpoint-based shared-encoder feature
analysis directly on the old local V2 artifacts.

### Important code-history note

In the current local git history, `sourcephasecompact` appears in commit
`a909325` (`v4`) together with `SourcePhaseWeightTracker`.

So, from the local repository alone, we cannot yet prove that the currently
named `sourcephasecompact_p5` outputs came from a pure pre-dynamic-weight V2
implementation. The experiment naming and logs match the V2 definition used in
the notes, but exact source-code provenance still needs either:

1. recovery of the original training checkpoint/code snapshot, or
2. confirmation from an external archive / server copy.

