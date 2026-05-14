# v2.5.4b Offline Selector Audit

## Scope

- logs: `logs\v253i_fullflow_12tasks_rerun_and_structure_20260513_231325`
- candidate rows: `48`
- tasks: `12`
- near-hit threshold: `0.01` macro F1 gap

## Top Single Metrics

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `selection_score:max` | 6/12 | 9/12 | 0.6822 | 0.6891 | -0.0069 |
| 2 | `selection_temporal_perturbation_score:max` | 6/12 | 9/12 | 0.6822 | 0.6891 | -0.0069 |
| 3 | `selection_high_conf_class_entropy:max` | 6/12 | 7/12 | 0.6759 | 0.6891 | -0.0131 |
| 4 | `selection_class_entropy:max` | 6/12 | 6/12 | 0.6759 | 0.6891 | -0.0132 |
| 5 | `selection_teacher_student_agreement:max` | 6/12 | 9/12 | 0.6753 | 0.6891 | -0.0137 |
| 6 | `selection_mean_confidence:min` | 5/12 | 6/12 | 0.6703 | 0.6891 | -0.0188 |
| 7 | `selection_prediction_entropy:max` | 4/12 | 5/12 | 0.6668 | 0.6891 | -0.0223 |
| 8 | `selection_source_prior_js:min` | 3/12 | 6/12 | 0.6665 | 0.6891 | -0.0226 |
| 9 | `selection_perturbation_label_agreement:min` | 4/12 | 6/12 | 0.6662 | 0.6891 | -0.0229 |
| 10 | `selection_perturbation_score:min` | 4/12 | 6/12 | 0.6662 | 0.6891 | -0.0229 |
| 11 | `selection_shift_stability:max` | 4/12 | 5/12 | 0.6634 | 0.6891 | -0.0257 |
| 12 | `selection_max_class_fraction:min` | 4/12 | 5/12 | 0.6627 | 0.6891 | -0.0264 |
| 13 | `selection_legacy_score:max` | 2/12 | 5/12 | 0.6619 | 0.6891 | -0.0271 |
| 14 | `selection_max_class_fraction:max` | 3/12 | 6/12 | 0.6618 | 0.6891 | -0.0273 |
| 15 | `selection_coverage:min` | 4/12 | 5/12 | 0.6617 | 0.6891 | -0.0274 |

## Metric Groups

| rank | selector | hit | near | selected avg | oracle avg | gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `source_prior_shift` | 5/12 | 7/12 | 0.6705 | 0.6891 | -0.0186 |
| 2 | `distribution_health` | 4/12 | 4/12 | 0.6620 | 0.6891 | -0.0270 |
| 3 | `trajectory` | 1/12 | 3/12 | 0.6547 | 0.6891 | -0.0343 |
| 4 | `robust_perturbation` | 3/12 | 6/12 | 0.6439 | 0.6891 | -0.0451 |
| 5 | `confidence_agreement` | 1/12 | 5/12 | 0.6427 | 0.6891 | -0.0464 |

## Leave-One-Task-Out

- held-out hit: `4/12` (0.333)
- held-out near-hit: `7/12` (0.583)
- held-out avg gap: `-0.0262`

| heldout task | chosen pipeline | selected | oracle | gap |
|---|---|---:|---:|---:|
| `30TXT->31TCJ` | `selection_score:max` | `epoch50` | `epoch50` | 0.0000 |
| `30TXT->32VNH` | `selection_score:max` | `epoch30` | `epoch70` | -0.0115 |
| `30TXT->33UVP` | `selection_score:max` | `epoch50` | `epoch100` | -0.0032 |
| `31TCJ->30TXT` | `selection_score:max` | `epoch50` | `epoch30` | -0.0119 |
| `31TCJ->32VNH` | `source_prior_shift` | `epoch100` | `epoch50` | -0.1431 |
| `31TCJ->33UVP` | `selection_score:max` | `epoch30` | `epoch30` | 0.0000 |
| `32VNH->30TXT` | `selection_score:max` | `epoch30` | `epoch70` | -0.0071 |
| `32VNH->31TCJ` | `selection_score:max` | `epoch50` | `epoch50` | 0.0000 |
| `32VNH->33UVP` | `selection_teacher_student_agreement:max` | `epoch100` | `epoch30` | -0.0885 |
| `33UVP->30TXT` | `selection_score:max` | `epoch100` | `epoch70` | -0.0035 |
| `33UVP->31TCJ` | `selection_score:max` | `epoch70` | `epoch70` | 0.0000 |
| `33UVP->32VNH` | `selection_score:max` | `epoch50` | `epoch30` | -0.0454 |

## Variance Check

| task | candidates | metrics | low-variance metrics |
|---|---:|---:|---:|
| `30TXT->31TCJ` | 4 | 20 | 5 |
| `30TXT->32VNH` | 4 | 20 | 5 |
| `30TXT->33UVP` | 4 | 20 | 6 |
| `31TCJ->30TXT` | 4 | 20 | 5 |
| `31TCJ->32VNH` | 4 | 20 | 6 |
| `31TCJ->33UVP` | 4 | 20 | 6 |
| `32VNH->30TXT` | 4 | 20 | 5 |
| `32VNH->31TCJ` | 4 | 20 | 6 |
| `32VNH->33UVP` | 4 | 20 | 5 |
| `33UVP->30TXT` | 4 | 20 | 5 |
| `33UVP->31TCJ` | 4 | 20 | 5 |
| `33UVP->32VNH` | 4 | 20 | 5 |

## Decision Rule

If leave-one-task-out hit rate is not clearly above `0.50`, do not launch another GPU selector run based only on these metrics.
