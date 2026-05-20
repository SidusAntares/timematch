# v2.5 HAR/HHAR Multi-View Summary

## Inputs

- `/data/user/timematch/logs/v25_har_hhar_multiview_20260520_115537_seed1`
- `/data/user/timematch/logs/v25_har_hhar_multiview_20260520_115537_seed2`
- `/data/user/timematch/logs/v25_har_hhar_multiview_20260520_115537_seed3`

## Aggregate

| dataset | tasks | runs | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 15 | 11/15 | 0.6420 | 0.6870 | 0.0450 | 9/15 | 0.6078 | 0.6317 | 0.0239 |
| HHAR_SA | 5 | 15 | 13/15 | 0.6598 | 0.7023 | 0.0425 | 12/15 | 0.6414 | 0.6902 | 0.0488 |
| ALL | 10 | 30 | 24/30 | 0.6509 | 0.6947 | 0.0437 | 21/30 | 0.6246 | 0.6609 | 0.0364 |

## Task View Preference

| dataset | task | runs | source gains | source base | source best | source delta | source best views | DA gains | DA base | DA best | DA delta | DA best views |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 3 | 2/3 | 0.5733 | 0.5993 | 0.0259 | compact_k3:3 | 1/3 | 0.5564 | 0.3710 | -0.1855 | compact_k8:1, compact_k5:1, dynamics_mse:1 |
| HAR | 2->11 | 3 | 2/3 | 0.6210 | 0.6504 | 0.0294 | intra_light_k5:1, compact_k5:1, compact_k3:1 | 3/3 | 0.5909 | 0.7068 | 0.1159 | compact_k5:1, intra_light_k5:1, compact_k8:1 |
| HAR | 6->23 | 3 | 2/3 | 0.6930 | 0.6935 | 0.0004 | compact_k3:2, compact_k5:1 | 2/3 | 0.5745 | 0.6902 | 0.1157 | dynamics_cosine:2, compact_k3:1 |
| HAR | 7->13 | 3 | 2/3 | 0.8669 | 0.8843 | 0.0175 | compact_k3:2, compact_k5:1 | 1/3 | 0.9025 | 0.8866 | -0.0159 | dynamics_cosine:1, compact_k8:1, noseg_global:1 |
| HAR | 9->18 | 3 | 3/3 | 0.4558 | 0.6074 | 0.1516 | compact_k5:2, dynamics_cosine:1 | 2/3 | 0.4147 | 0.5039 | 0.0892 | compact_k5:2, noseg_global:1 |
| HHAR_SA | 0->6 | 3 | 2/3 | 0.3114 | 0.3321 | 0.0207 | compact_k8:2, compact_k3:1 | 1/3 | 0.2722 | 0.2648 | -0.0073 | compact_k3:1, dynamics_mse:1, compact_k5:1 |
| HHAR_SA | 1->6 | 3 | 3/3 | 0.7929 | 0.8401 | 0.0471 | compact_k3:2, dynamics_mse:1 | 3/3 | 0.8390 | 0.8874 | 0.0484 | dynamics_mse:2, compact_k3:1 |
| HHAR_SA | 2->7 | 3 | 3/3 | 0.4374 | 0.5245 | 0.0871 | compact_k5:1, compact_k3:1, dynamics_mse:1 | 3/3 | 0.3383 | 0.4505 | 0.1122 | dynamics_mse:2, dynamics_cosine:1 |
| HHAR_SA | 3->8 | 3 | 2/3 | 0.8828 | 0.8910 | 0.0082 | intra_light_k5:2, dynamics_mse:1 | 2/3 | 0.9030 | 0.9098 | 0.0068 | compact_k5:1, intra_light_k5:1, compact_k8:1 |
| HHAR_SA | 4->5 | 3 | 3/3 | 0.8746 | 0.9241 | 0.0495 | compact_k5:1, dynamics_cosine:1, dynamics_mse:1 | 3/3 | 0.8544 | 0.9384 | 0.0840 | compact_k5:1, noseg_global:1, intra_light_k5:1 |

## Reading

- `best views` records which structure view wins within each log root. If several views appear across roots, the task has unstable or seed-sensitive structure preference.
- The goal of this table is not to claim a single fixed parameter is universal, but to test whether a multi-view source-structure bank contains useful views across generic time-series UDA tasks.
