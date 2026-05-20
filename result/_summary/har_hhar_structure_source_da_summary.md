# HAR/HHAR Structure Overnight Summary

- Log root: `logs\har_hhar_structure_overnight_20260519_231310`
- Parsed tasks: 10
- Logs with explicit error patterns: 0

## Aggregate

| dataset | tasks | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 4/5 | 0.6769 | 0.7203 | 0.0433 | 3/5 | 0.6177 | 0.6457 | 0.0281 |
| HHAR_SA | 5 | 5/5 | 0.6521 | 0.6924 | 0.0403 | 5/5 | 0.6118 | 0.6741 | 0.0622 |
| ALL | 10 | 9/10 | 0.6645 | 0.7064 | 0.0418 | 8/10 | 0.6148 | 0.6599 | 0.0451 |

## Task-Level Results

| dataset | task | source base | source best view | source best | source delta | DA base | DA best view | DA best | DA delta | DA-source base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 0.5698 | compact_boundary_light | 0.5875 | 0.0177 | 0.5046 | compact_k8 | 0.4273 | -0.0773 | -0.0652 |
| HAR | 2->11 | 0.7323 | compact_k5 | 0.7117 | -0.0206 | 0.6750 | compact_k5 | 0.7826 | 0.1076 | -0.0573 |
| HAR | 6->23 | 0.7133 | compact_boundary_light | 0.7219 | 0.0086 | 0.6591 | intra_light_k5 | 0.6468 | -0.0123 | -0.0542 |
| HAR | 7->13 | 0.8802 | compact_k5 | 0.8919 | 0.0117 | 0.8828 | compact_k5 | 0.9118 | 0.0290 | 0.0026 |
| HAR | 9->18 | 0.4891 | compact_k5 | 0.6884 | 0.1993 | 0.3668 | compact_k5 | 0.4602 | 0.0934 | -0.1223 |
| HHAR_SA | 0->6 | 0.3287 | compact_k5 | 0.3404 | 0.0117 | 0.2516 | compact_k5 | 0.3821 | 0.1305 | -0.0771 |
| HHAR_SA | 1->6 | 0.7597 | compact_k5 | 0.7909 | 0.0312 | 0.8285 | compact_k5 | 0.8391 | 0.0106 | 0.0688 |
| HHAR_SA | 2->7 | 0.4515 | compact_k3 | 0.5738 | 0.1223 | 0.2494 | compact_k3 | 0.3666 | 0.1172 | -0.2021 |
| HHAR_SA | 3->8 | 0.8683 | compact_k8 | 0.8685 | 0.0002 | 0.8769 | intra_light_k5 | 0.8904 | 0.0135 | 0.0086 |
| HHAR_SA | 4->5 | 0.8524 | compact_k5 | 0.8885 | 0.0361 | 0.8528 | compact_k5 | 0.8921 | 0.0393 | 0.0004 |

## Reading

- `source delta` measures whether source-side structure training improves the source-trained checkpoint before TimeMatch.
- `DA delta` measures whether the same structure-trained source checkpoint improves the final TimeMatch result.
- `DA-source base` shows whether vanilla TimeMatch helps or hurts compared with vanilla source-only on the same task.
- A positive `source delta` but negative `DA delta` suggests the structure view helps representation but the DA stage may be misaligned for that task.
