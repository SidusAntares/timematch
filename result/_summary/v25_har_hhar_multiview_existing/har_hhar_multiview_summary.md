# v2.5 HAR/HHAR Multi-View Summary

## Inputs

- `logs\har_hhar_structure_overnight_20260519_231310`
- `logs\har_hhar_structure_overnight_har_hhar_compactk5_seed1_20260520_102422`
- `logs\har_hhar_structure_overnight_har_hhar_compactk5_seed2_20260520_102422`
- `logs\har_hhar_structure_overnight_har_hhar_compactk5_seed3_20260520_102422`

## Aggregate

| dataset | tasks | runs | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 20 | 16/20 | 0.6769 | 0.7173 | 0.0404 | 13/20 | 0.6177 | 0.6383 | 0.0206 |
| HHAR_SA | 5 | 20 | 18/20 | 0.6521 | 0.6802 | 0.0281 | 18/20 | 0.6118 | 0.6800 | 0.0681 |
| ALL | 10 | 40 | 34/40 | 0.6645 | 0.6988 | 0.0342 | 31/40 | 0.6148 | 0.6591 | 0.0444 |

## Task View Preference

| dataset | task | runs | source gains | source base | source best | source delta | source best views | DA gains | DA base | DA best | DA delta | DA best views |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 4 | 4/4 | 0.5698 | 0.5875 | 0.0177 | compact_boundary_light:4 | 0/4 | 0.5046 | 0.3871 | -0.1175 | compact_k8:3, compact_k3:1 |
| HAR | 2->11 | 4 | 0/4 | 0.7323 | 0.7117 | -0.0206 | compact_k5:4 | 4/4 | 0.6750 | 0.7559 | 0.0809 | compact_k5:4 |
| HAR | 6->23 | 4 | 4/4 | 0.7133 | 0.7219 | 0.0086 | compact_boundary_light:4 | 1/4 | 0.6591 | 0.6537 | -0.0054 | dynamics_mse:2, intra_light_k5:1, dynamics_cosine:1 |
| HAR | 7->13 | 4 | 4/4 | 0.8802 | 0.8967 | 0.0165 | compact_k5:4 | 4/4 | 0.8828 | 0.9016 | 0.0188 | compact_k5:4 |
| HAR | 9->18 | 4 | 4/4 | 0.4891 | 0.6687 | 0.1796 | compact_k5:4 | 4/4 | 0.3668 | 0.4932 | 0.1264 | compact_k5:4 |
| HHAR_SA | 0->6 | 4 | 2/4 | 0.3287 | 0.3026 | -0.0261 | compact_k3:3, compact_k5:1 | 4/4 | 0.2516 | 0.3587 | 0.1071 | compact_k3:2, compact_k5:1, noseg_global:1 |
| HHAR_SA | 1->6 | 4 | 4/4 | 0.7597 | 0.7907 | 0.0310 | compact_k3:2, compact_k5:1, compact_k8:1 | 4/4 | 0.8285 | 0.8361 | 0.0076 | compact_k3:2, compact_k5:1, compact_k8:1 |
| HHAR_SA | 2->7 | 4 | 4/4 | 0.4515 | 0.5527 | 0.1012 | compact_k5:3, compact_k3:1 | 4/4 | 0.2494 | 0.4680 | 0.2186 | compact_k5:3, compact_k3:1 |
| HHAR_SA | 3->8 | 4 | 4/4 | 0.8683 | 0.8785 | 0.0102 | compact_k8:2, compact_k5:2 | 4/4 | 0.8769 | 0.9022 | 0.0253 | compact_k5:2, intra_light_k5:1, compact_k8:1 |
| HHAR_SA | 4->5 | 4 | 4/4 | 0.8524 | 0.8767 | 0.0243 | compact_k5:2, compact_k3:2 | 2/4 | 0.8528 | 0.8350 | -0.0178 | compact_k5:3, compact_k3:1 |

## Reading

- `best views` records which structure view wins within each log root. If several views appear across roots, the task has unstable or seed-sensitive structure preference.
- The goal of this table is not to claim a single fixed parameter is universal, but to test whether a multi-view source-structure bank contains useful views across generic time-series UDA tasks.
