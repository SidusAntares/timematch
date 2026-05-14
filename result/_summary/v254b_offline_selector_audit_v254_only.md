# v2.5.4b Offline Selector Audit

## Scope

- logs: `logs\v254_metricbank_quickcheck_20260514_113252`
- candidate rows: `16`
- tasks: `4`
- near-hit threshold: `0.01` macro F1 gap

## Top Single Metrics

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `selection_max_class_fraction:min` | 2/4 | 2/4 | 0.6913 | 0.7258 | -0.0345 |
| 2 | `metric_bank_rank_score:max` | 2/4 | 2/4 | 0.6821 | 0.7258 | -0.0438 |
| 3 | `selection_temporal_perturbation_score:max` | 2/4 | 2/4 | 0.6821 | 0.7258 | -0.0438 |
| 4 | `selection_trajectory_base_score:max` | 2/4 | 2/4 | 0.6821 | 0.7258 | -0.0438 |
| 5 | `selection_trajectory_excess_late_gain:max` | 1/4 | 1/4 | 0.6695 | 0.7258 | -0.0563 |
| 6 | `selection_trajectory_late_gain_ratio:max` | 1/4 | 1/4 | 0.6695 | 0.7258 | -0.0563 |
| 7 | `selection_trajectory_multiplier:min` | 1/4 | 1/4 | 0.6695 | 0.7258 | -0.0563 |
| 8 | `selection_source_prior_js:min` | 2/4 | 2/4 | 0.6645 | 0.7258 | -0.0613 |
| 9 | `selection_trajectory_excess_late_gain:min` | 2/4 | 2/4 | 0.6645 | 0.7258 | -0.0613 |
| 10 | `selection_trajectory_late_gain_ratio:min` | 2/4 | 2/4 | 0.6645 | 0.7258 | -0.0613 |
| 11 | `selection_trajectory_multiplier:max` | 2/4 | 2/4 | 0.6645 | 0.7258 | -0.0613 |
| 12 | `selection_shift_stability:max` | 1/4 | 1/4 | 0.6615 | 0.7258 | -0.0643 |
| 13 | `selection_high_conf_class_entropy:min` | 2/4 | 2/4 | 0.6607 | 0.7258 | -0.0651 |
| 14 | `selection_coverage:min` | 1/4 | 1/4 | 0.6566 | 0.7258 | -0.0692 |
| 15 | `selection_mean_confidence:min` | 1/4 | 1/4 | 0.6566 | 0.7258 | -0.0692 |

## Metric Groups

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `metric_bank_existing` | 2/4 | 2/4 | 0.6821 | 0.7258 | -0.0438 |
| 2 | `robust_perturbation` | 2/4 | 2/4 | 0.6771 | 0.7258 | -0.0488 |
| 3 | `source_prior_shift` | 2/4 | 2/4 | 0.6679 | 0.7258 | -0.0579 |
| 4 | `distribution_health` | 1/4 | 1/4 | 0.6658 | 0.7258 | -0.0600 |
| 5 | `trajectory` | 2/4 | 2/4 | 0.6645 | 0.7258 | -0.0613 |
| 6 | `confidence_agreement` | 1/4 | 1/4 | 0.6055 | 0.7258 | -0.1203 |

## Leave-One-Task-Out

- held-out hit: `0/4` (0.000)
- held-out near-hit: `0/4` (0.000)
- held-out avg gap: `-0.1089`

| heldout task | chosen pipeline | selected | oracle | gap |
|---|---|---:|---:|---:|
| `30TXT->33UVP` | `selection_max_class_fraction:min` | `epoch30` | `epoch100` | -0.0651 |
| `31TCJ->32VNH` | `selection_high_conf_class_entropy:min` | `epoch70` | `epoch50` | -0.1953 |
| `32VNH->33UVP` | `metric_bank_rank_score:max` | `epoch70` | `epoch30` | -0.1020 |
| `33UVP->32VNH` | `selection_max_class_fraction:min` | `epoch70` | `epoch30` | -0.0731 |

## Variance Check

| task | candidates | metrics | low-variance metrics |
|---|---:|---:|---:|
| `30TXT->33UVP` | 4 | 22 | 3 |
| `31TCJ->32VNH` | 4 | 22 | 2 |
| `32VNH->33UVP` | 4 | 22 | 2 |
| `33UVP->32VNH` | 4 | 22 | 2 |

## Decision Rule

If leave-one-task-out hit rate is not clearly above `0.50`, do not launch another GPU selector run based only on these metrics.
