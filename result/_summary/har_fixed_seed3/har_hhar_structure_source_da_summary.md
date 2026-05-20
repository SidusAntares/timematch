# HAR/HHAR Structure Overnight Summary

- Log root: `logs\har_hhar_structure_overnight_har_hhar_fixed_compactk5_seed3_20260520_111906`
- Parsed tasks: 10
- Logs with explicit error patterns: 0

## Aggregate

| dataset | tasks | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 5/5 | 0.5999 | 0.6671 | 0.0673 | 2/5 | 0.5939 | 0.5182 | -0.0757 |
| HHAR_SA | 5 | 1/5 | 0.6648 | 0.6251 | -0.0397 | 1/5 | 0.6588 | 0.5936 | -0.0652 |
| ALL | 10 | 6/10 | 0.6323 | 0.6461 | 0.0138 | 3/10 | 0.6264 | 0.5559 | -0.0705 |

## Task-Level Results

| dataset | task | source base | source best view | source best | source delta | DA base | DA best view | DA best | DA delta | DA-source base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 0.5653 | compact_k5 | 0.6447 | 0.0794 | 0.5909 | compact_k5 | 0.0376 | -0.5533 | 0.0256 |
| HAR | 2->11 | 0.4105 | compact_k5 | 0.4826 | 0.0721 | 0.5337 | compact_k5 | 0.5880 | 0.0543 | 0.1232 |
| HAR | 6->23 | 0.6200 | compact_k5 | 0.6468 | 0.0268 | 0.3580 | compact_k5 | 0.5877 | 0.2297 | -0.2620 |
| HAR | 7->13 | 0.8518 | compact_k5 | 0.9126 | 0.0608 | 0.9235 | compact_k5 | 0.9033 | -0.0202 | 0.0717 |
| HAR | 9->18 | 0.5517 | compact_k5 | 0.6490 | 0.0973 | 0.5633 | compact_k5 | 0.4743 | -0.0890 | 0.0116 |
| HHAR_SA | 0->6 | 0.3104 | compact_k5 | 0.2408 | -0.0696 | 0.2467 | compact_k5 | 0.1462 | -0.1005 | -0.0637 |
| HHAR_SA | 1->6 | 0.7994 | compact_k5 | 0.8046 | 0.0052 | 0.8488 | compact_k5 | 0.8175 | -0.0313 | 0.0494 |
| HHAR_SA | 2->7 | 0.4315 | compact_k5 | 0.3712 | -0.0603 | 0.4114 | compact_k5 | 0.2695 | -0.1419 | -0.0201 |
| HHAR_SA | 3->8 | 0.9107 | compact_k5 | 0.8751 | -0.0356 | 0.9335 | compact_k5 | 0.8757 | -0.0578 | 0.0228 |
| HHAR_SA | 4->5 | 0.8719 | compact_k5 | 0.8338 | -0.0381 | 0.8538 | compact_k5 | 0.8592 | 0.0054 | -0.0181 |

## Reading

- `source delta` measures whether source-side structure training improves the source-trained checkpoint before TimeMatch.
- `DA delta` measures whether the same structure-trained source checkpoint improves the final TimeMatch result.
- `DA-source base` shows whether vanilla TimeMatch helps or hurts compared with vanilla source-only on the same task.
- A positive `source delta` but negative `DA delta` suggests the structure view helps representation but the DA stage may be misaligned for that task.
