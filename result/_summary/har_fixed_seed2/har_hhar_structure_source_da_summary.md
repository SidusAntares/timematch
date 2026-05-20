# HAR/HHAR Structure Overnight Summary

- Log root: `logs\har_hhar_structure_overnight_har_hhar_fixed_compactk5_seed2_20260520_111906`
- Parsed tasks: 10
- Logs with explicit error patterns: 0

## Aggregate

| dataset | tasks | source gains | source base | source best | source delta | DA gains | DA base | DA best | DA delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 5 | 1/5 | 0.6492 | 0.6488 | -0.0004 | 3/5 | 0.6119 | 0.6409 | 0.0290 |
| HHAR_SA | 5 | 0/5 | 0.6626 | 0.6160 | -0.0466 | 1/5 | 0.6534 | 0.5943 | -0.0591 |
| ALL | 10 | 1/10 | 0.6559 | 0.6324 | -0.0235 | 4/10 | 0.6326 | 0.6176 | -0.0150 |

## Task-Level Results

| dataset | task | source base | source best view | source best | source delta | DA base | DA best view | DA best | DA delta | DA-source base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | 0.5849 | compact_k5 | 0.5563 | -0.0286 | 0.5738 | compact_k5 | 0.7147 | 0.1409 | -0.0111 |
| HAR | 2->11 | 0.7203 | compact_k5 | 0.7189 | -0.0014 | 0.5639 | compact_k5 | 0.6518 | 0.0879 | -0.1564 |
| HAR | 6->23 | 0.7458 | compact_k5 | 0.7117 | -0.0341 | 0.7065 | compact_k5 | 0.7154 | 0.0089 | -0.0393 |
| HAR | 7->13 | 0.8686 | compact_k5 | 0.8389 | -0.0297 | 0.9013 | compact_k5 | 0.8393 | -0.0620 | 0.0327 |
| HAR | 9->18 | 0.3266 | compact_k5 | 0.4184 | 0.0918 | 0.3139 | compact_k5 | 0.2831 | -0.0308 | -0.0127 |
| HHAR_SA | 0->6 | 0.2950 | compact_k5 | 0.2157 | -0.0793 | 0.3182 | compact_k5 | 0.1388 | -0.1794 | 0.0232 |
| HHAR_SA | 1->6 | 0.8197 | compact_k5 | 0.7664 | -0.0533 | 0.8397 | compact_k5 | 0.7972 | -0.0425 | 0.0200 |
| HHAR_SA | 2->7 | 0.4293 | compact_k5 | 0.4001 | -0.0292 | 0.3540 | compact_k5 | 0.3058 | -0.0482 | -0.0753 |
| HHAR_SA | 3->8 | 0.8695 | compact_k5 | 0.8344 | -0.0351 | 0.8985 | compact_k5 | 0.8453 | -0.0532 | 0.0290 |
| HHAR_SA | 4->5 | 0.8994 | compact_k5 | 0.8635 | -0.0359 | 0.8565 | compact_k5 | 0.8844 | 0.0279 | -0.0429 |

## Reading

- `source delta` measures whether source-side structure training improves the source-trained checkpoint before TimeMatch.
- `DA delta` measures whether the same structure-trained source checkpoint improves the final TimeMatch result.
- `DA-source base` shows whether vanilla TimeMatch helps or hurts compared with vanilla source-only on the same task.
- A positive `source delta` but negative `DA delta` suggests the structure view helps representation but the DA stage may be misaligned for that task.
