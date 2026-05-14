# v2.5.4b Offline Selector Audit

## Scope

- logs: `logs\v253h_blend_selection_only_12tasks_20260513_211223`
- candidate rows: `48`
- tasks: `12`
- near-hit threshold: `0.01` macro F1 gap

## Top Single Metrics

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `selection_score:max` | 5/12 | 7/12 | 0.6776 | 0.6891 | -0.0115 |
| 2 | `selection_max_class_fraction:min` | 5/12 | 6/12 | 0.6726 | 0.6891 | -0.0165 |
| 3 | `selection_temporal_perturbation_score:max` | 5/12 | 7/12 | 0.6708 | 0.6891 | -0.0183 |
| 4 | `selection_shift_stability:min` | 5/12 | 8/12 | 0.6623 | 0.6891 | -0.0268 |
| 5 | `selection_high_conf_class_entropy:max` | 4/12 | 6/12 | 0.6622 | 0.6891 | -0.0269 |
| 6 | `selection_legacy_score:max` | 3/12 | 6/12 | 0.6611 | 0.6891 | -0.0280 |
| 7 | `selection_coverage:min` | 4/12 | 4/12 | 0.6585 | 0.6891 | -0.0305 |
| 8 | `selection_class_entropy:max` | 2/12 | 3/12 | 0.6575 | 0.6891 | -0.0316 |
| 9 | `selection_source_prior_js:min` | 5/12 | 7/12 | 0.6566 | 0.6891 | -0.0325 |
| 10 | `selection_prediction_entropy:max` | 3/12 | 4/12 | 0.6550 | 0.6891 | -0.0341 |
| 11 | `selection_mean_confidence:min` | 3/12 | 4/12 | 0.6550 | 0.6891 | -0.0341 |
| 12 | `selection_shift_stability:max` | 1/12 | 1/12 | 0.6550 | 0.6891 | -0.0341 |
| 13 | `selection_trajectory_base_score:max` | 1/12 | 3/12 | 0.6547 | 0.6891 | -0.0343 |
| 14 | `selection_trajectory_base_score:min` | 1/12 | 3/12 | 0.6547 | 0.6891 | -0.0343 |
| 15 | `selection_trajectory_excess_late_gain:max` | 1/12 | 3/12 | 0.6547 | 0.6891 | -0.0343 |

## Metric Groups

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `distribution_health` | 7/12 | 8/12 | 0.6775 | 0.6891 | -0.0115 |
| 2 | `robust_perturbation` | 4/12 | 6/12 | 0.6671 | 0.6891 | -0.0220 |
| 3 | `source_prior_shift` | 3/12 | 3/12 | 0.6624 | 0.6891 | -0.0266 |
| 4 | `trajectory` | 1/12 | 3/12 | 0.6547 | 0.6891 | -0.0343 |
| 5 | `confidence_agreement` | 1/12 | 4/12 | 0.6360 | 0.6891 | -0.0531 |

## Leave-One-Task-Out

- held-out hit: `3/12` (0.250)
- held-out near-hit: `5/12` (0.417)
- held-out avg gap: `-0.0274`

| heldout task | chosen pipeline | selected | oracle | gap |
|---|---|---:|---:|---:|
| `30TXT->31TCJ` | `selection_score:max` | `epoch30` | `epoch50` | -0.0116 |
| `30TXT->32VNH` | `distribution_health` | `epoch100` | `epoch70` | -0.0126 |
| `30TXT->33UVP` | `selection_max_class_fraction:min` | `epoch30` | `epoch100` | -0.0651 |
| `31TCJ->30TXT` | `distribution_health` | `epoch100` | `epoch30` | -0.0338 |
| `31TCJ->32VNH` | `selection_score:max` | `epoch50` | `epoch50` | 0.0000 |
| `31TCJ->33UVP` | `selection_score:max` | `epoch30` | `epoch30` | 0.0000 |
| `32VNH->30TXT` | `selection_score:max` | `epoch30` | `epoch70` | -0.0071 |
| `32VNH->31TCJ` | `selection_score:max` | `epoch30` | `epoch50` | -0.0204 |
| `32VNH->33UVP` | `selection_temporal_perturbation_score:max` | `epoch70` | `epoch30` | -0.1020 |
| `33UVP->30TXT` | `selection_score:max` | `epoch100` | `epoch70` | -0.0035 |
| `33UVP->31TCJ` | `selection_score:max` | `epoch70` | `epoch70` | 0.0000 |
| `33UVP->32VNH` | `distribution_health` | `epoch70` | `epoch30` | -0.0731 |

## Variance Check

| task | candidates | metrics | low-variance metrics |
|---|---:|---:|---:|
| `30TXT->31TCJ` | 4 | 20 | 5 |
| `30TXT->32VNH` | 4 | 20 | 5 |
| `30TXT->33UVP` | 4 | 20 | 6 |
| `31TCJ->30TXT` | 4 | 20 | 5 |
| `31TCJ->32VNH` | 4 | 20 | 5 |
| `31TCJ->33UVP` | 4 | 20 | 5 |
| `32VNH->30TXT` | 4 | 20 | 6 |
| `32VNH->31TCJ` | 4 | 20 | 5 |
| `32VNH->33UVP` | 4 | 20 | 5 |
| `33UVP->30TXT` | 4 | 20 | 5 |
| `33UVP->31TCJ` | 4 | 20 | 5 |
| `33UVP->32VNH` | 4 | 20 | 5 |

## Decision Rule

If leave-one-task-out hit rate is not clearly above `0.50`, do not launch another GPU selector run based only on these metrics.
