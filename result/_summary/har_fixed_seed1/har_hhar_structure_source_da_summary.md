# HAR/HHAR Structure Overnight Summary

- Log root: `logs\har_hhar_structure_overnight_har_hhar_fixed_compactk5_seed1_20260520_111906`
- Parsed tasks: 10
- Logs with explicit error patterns: 0

## Aggregate

| dataset | tasks | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 4/5 | 0.6769 | 0.7068 | 0.0299 | 3/5 | 0.6177 | 0.5997 | -0.0179 |
| HHAR_SA | 5 | 2/5 | 0.6521 | 0.6712 | 0.0190 | 3/5 | 0.6118 | 0.6088 | -0.0030 |
| ALL | 10 | 6/10 | 0.6645 | 0.6890 | 0.0245 | 6/10 | 0.6148 | 0.6043 | -0.0105 |

## Task-Level Results

| dataset | task | source base | source best view | source best | source delta | DA base | DA best view | DA best | DA delta | DA-source base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 0.5698 | compact_k5 | 0.5875 | 0.0177 | 0.5046 | compact_k5 | 0.2168 | -0.2878 | -0.0652 |
| HAR | 2->11 | 0.7323 | compact_k5 | 0.7117 | -0.0206 | 0.6750 | compact_k5 | 0.7518 | 0.0768 | -0.0573 |
| HAR | 6->23 | 0.7133 | compact_k5 | 0.7219 | 0.0086 | 0.6591 | compact_k5 | 0.6054 | -0.0537 | -0.0542 |
| HAR | 7->13 | 0.8802 | compact_k5 | 0.8919 | 0.0117 | 0.8828 | compact_k5 | 0.9118 | 0.0290 | 0.0026 |
| HAR | 9->18 | 0.4891 | compact_k5 | 0.6210 | 0.1319 | 0.3668 | compact_k5 | 0.5128 | 0.1460 | -0.1223 |
| HHAR_SA | 0->6 | 0.3287 | compact_k5 | 0.2407 | -0.0880 | 0.2516 | compact_k5 | 0.1379 | -0.1137 | -0.0771 |
| HHAR_SA | 1->6 | 0.7597 | compact_k5 | 0.7458 | -0.0139 | 0.8285 | compact_k5 | 0.7010 | -0.1275 | 0.0688 |
| HHAR_SA | 2->7 | 0.4515 | compact_k5 | 0.5919 | 0.1404 | 0.2494 | compact_k5 | 0.3455 | 0.0961 | -0.2021 |
| HHAR_SA | 3->8 | 0.8683 | compact_k5 | 0.8501 | -0.0182 | 0.8769 | compact_k5 | 0.9153 | 0.0384 | 0.0086 |
| HHAR_SA | 4->5 | 0.8524 | compact_k5 | 0.9273 | 0.0749 | 0.8528 | compact_k5 | 0.9443 | 0.0915 | 0.0004 |

## Reading

- `source delta` measures whether source-side structure training improves the source-trained checkpoint before TimeMatch.
- `DA delta` measures whether the same structure-trained source checkpoint improves the final TimeMatch result.
- `DA-source base` shows whether vanilla TimeMatch helps or hurts compared with vanilla source-only on the same task.
- A positive `source delta` but negative `DA delta` suggests the structure view helps representation but the DA stage may be misaligned for that task.
