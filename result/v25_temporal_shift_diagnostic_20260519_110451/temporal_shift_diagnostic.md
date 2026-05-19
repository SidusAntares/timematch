# v2.5 Temporal Structure Diagnostic

This report diagnoses raw source-target temporal mismatch. It is a guide for
choosing structure views, not a replacement for DA validation.

| task | distance | best partition | adj. compression | raw compression | peak ratio | top20 mass | variation | hint |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| DK1_to_FR1 | joint_z_l2 | doy_gap:5 | 0.1080 | 0.3054 | 2.5385 | 0.3571 | 0.3485 | segmented_compactness |
| DK1_to_FR1 | source_mad_l1 | doy_gap:5 | 0.1130 | 0.3197 | 2.6692 | 0.3681 | 0.3783 | segmented_compactness |
| FR1_to_FR2 | joint_z_l2 | uniform:10 | 0.0788 | 0.2490 | 2.6844 | 0.3817 | 0.4476 | trajectory_dynamics_probe |
| FR1_to_FR2 | source_mad_l1 | uniform:10 | 0.0792 | 0.2504 | 2.7619 | 0.3892 | 0.4704 | trajectory_dynamics_probe |
| FR2_to_FR1 | joint_z_l2 | uniform:10 | 0.0788 | 0.2490 | 2.6844 | 0.3817 | 0.4476 | trajectory_dynamics_probe |
| FR2_to_FR1 | source_mad_l1 | uniform:10 | 0.0845 | 0.2671 | 2.8433 | 0.3950 | 0.4796 | trajectory_dynamics_probe |

Interpretation:

- `global_compactness`: mismatch is diffuse; no-seg/global compactness is a plausible first view.
- `segmented_compactness`: mismatch is locally structured; segment-wise compactness has a diagnostic basis.
- `trajectory_dynamics_probe`: mismatch changes sharply over time; dynamics should be tested carefully.
- `source_mad_l1`: preserves source-target offsets after scaling channels by source-domain MAD.
- `joint_z_l2`: shape-only view; useful as a contrast because it removes class-wise scale/offset.
- `partition_compression`: how much a partition reduces within-segment variance of the mismatch curve.
- `adj. compression`: compression divided by sqrt(segment count), to reduce the trivial advantage of finer partitions.
