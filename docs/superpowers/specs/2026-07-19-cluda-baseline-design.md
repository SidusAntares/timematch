# CLUDA Baseline Design

## Scope

Keep the existing `plain_pseltae` result and add source-only `plain_psecludatcn` plus source-initialized `cluda_pse_full`. The local implementation is behavior-based because upstream commit `60e0b10f0e967fbada08ee6c164fc7c97932c57b` exposes no license file in its root.

## Architecture

`methods/cluda` owns input adaptation, the causal residual TCN, MLPs, nearest-neighbor lookup, four-view augmentation, five-loss composition, full CLUDA state, and the joint trainer. `CLUDATCNClassifier` implements the existing TimeMatch model call signature for source-only training and evaluation. `CLUDA` separately owns query/key encoders, queues, momentum, projector, predictor, discriminator, and GRL.

## Data contract

The adapter accepts `[B,T,C,S]` pixels, `[B,T,S]` validity, and `[B,T]` positions. Raw-pixel noise/dropout precede the existing TimeMatch PSE (`mean_std`, 128 outputs, no extras). Temporal cutout/crop masks the PSE output; the existing frozen sinusoid uses real positions and the temporal mask is applied again before the 128-input official TCN. The last valid TCN feature is normalized. No interpolation or date creation occurs.

## Training and selection

Full CLUDA uses source labels only and initializes from a strictly architecture- and task-matched source checkpoint. Target parcel counting/splitting and the target training dataset are label-agnostic; metadata labels are stripped and the joint step never reads the dummy label field. Target validation is not called for checkpoint selection or tuning. Checkpoints are chronological. Source-only PSE-CLUDA-TCN uses the existing supervised loop with normal PSE BN; full CLUDA freezes only the PSE running statistics and EMA-updates the complete key PSE+TCN.

## Verification

Tests cover TCN behavioral equivalence, input adaptation, momentum/frozen key encoder, wrapping queues, diagonal positives, exact current-batch NNCL, GRL, normalized prediction features, all loss terms and zero weights, independent views, target-label exclusion, and a CPU training step.
