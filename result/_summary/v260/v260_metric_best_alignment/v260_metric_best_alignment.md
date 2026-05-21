# v2.6.0b Metric-Best Alignment

## Conclusion

The PSE feature-space audit is useful for strength adaptation, but the current view heuristic is not reliable enough for automatic view selection.

- `suggested_view` collapses toward `global`, while empirical best views are mostly segmented and partly dynamics.
- `source_reliability`, `target_margin_ratio`, and `temporal_mismatch_cv` are usable safety signals for conservative strength adaptation, but none of them is a standalone strength predictor.
- `uniform_k*_adjusted_compression` is better interpreted as a future window/locality signal, not as a global strength multiplier.
- `dynamics_delta_cosine_mean` is not sufficient to select dynamics by itself.

## Best-View Distribution

| source | global | segmented | dynamics |
| --- | ---: | ---: | ---: |
| all | 2 | 17 | 3 |
| HAR | 0 | 4 | 1 |
| HHAR_SA | 1 | 4 | 0 |
| remote | 1 | 9 | 2 |

## Metric Correlation With Best Strength

Strength is encoded as `light=0.25`, `medium=0.5`, `medium_high=0.75`, `strong=1.0`. This is only a coarse ordinal target.

| metric | Pearson r | reading |
| --- | ---: | --- |
| `source_reliability` | 0.0899 | higher source reliability can support stronger compactness, but dataset effects are strong |
| `target_margin_ratio` | -0.0457 | useful as a safety threshold, but not a linear standalone strength predictor |
| `temporal_mismatch_cv` | -0.2016 | high locality/mismatch variation argues against globally stronger stiffness |
| `uniform_k5_adjusted_compression` | -0.3435 | locality/window signal, not a direct strength signal |
| `uniform_k8_adjusted_compression` | -0.2679 | locality/window signal, useful for v2.6.2 |
| `dynamics_delta_cosine_mean` | 0.0696 | not reliable as a dynamics selector |

## Group Means By Empirical Best View

| best view | n | src_rel | tgt_margin | mismatch_cv | k5_comp | k8_comp | dyn_cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global | 2 | 0.9998 | 0.0880 | 0.1281 | 0.0675 | 0.0838 | 0.0161 |
| segmented | 17 | 1.9902 | 0.1445 | 0.1917 | 0.1076 | 0.1277 | 0.0485 |
| dynamics | 3 | 1.7360 | 0.1493 | 0.2237 | 0.1096 | 0.1048 | -0.0458 |

## Group Means By Empirical Best Strength

| strength | n | src_rel | tgt_margin | mismatch_cv | k5_comp | factor_old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| light | 5 | 1.8369 | 0.1528 | 0.2255 | 0.1194 | 0.9551 |
| medium | 14 | 1.6739 | 0.1312 | 0.1965 | 0.1073 | 0.9370 |
| medium_high | 2 | 3.8004 | 0.2048 | 0.0439 | 0.0722 | 1.0123 |
| strong | 1 | 0.8209 | 0.0719 | 0.2192 | 0.0506 | 0.8855 |

## Task Alignment Table

| dataset | task | best view | strength | gain | src_rel | tgt_margin | mismatch_cv | k5_comp | k8_comp | dyn_cos | old suggested | old factor |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| HAR | 12->16 | segmented | medium_high | -0.1175 | 2.5711 | 0.2662 | 0.0550 | 0.1083 | 0.1525 | 0.0282 | global | 1.0014 |
| HAR | 2->11 | segmented | medium | 0.0809 | 4.4351 | 0.3991 | 0.0696 | 0.1356 | 0.1658 | 0.0243 | global | 1.2000 |
| HAR | 6->23 | dynamics | light | -0.0054 | 3.7691 | 0.2878 | 0.0992 | 0.1355 | 0.1401 | 0.0024 | global | 1.0936 |
| HAR | 7->13 | segmented | medium | 0.0188 | 3.2693 | 0.2730 | 0.0914 | 0.0632 | 0.1072 | 0.0429 | global | 1.0336 |
| HAR | 9->18 | segmented | medium | 0.1264 | 2.3524 | 0.0532 | 0.0525 | 0.1052 | 0.1384 | 0.0056 | global | 0.9012 |
| HHAR_SA | 0->6 | global | medium | 0.1071 | 1.1787 | 0.1042 | 0.0370 | 0.0843 | 0.1195 | 0.0261 | global | 0.9026 |
| HHAR_SA | 1->6 | segmented | medium_high | 0.0076 | 5.0296 | 0.1433 | 0.0329 | 0.0361 | 0.0195 | -0.0268 | global | 1.0232 |
| HHAR_SA | 2->7 | segmented | medium | 0.2186 | 1.2658 | 0.0698 | 0.0486 | 0.1688 | 0.1695 | 0.0467 | global | 0.8952 |
| HHAR_SA | 3->8 | segmented | light | 0.0253 | 3.2922 | 0.2706 | 0.0795 | 0.1008 | 0.0747 | 0.0006 | global | 1.0324 |
| HHAR_SA | 4->5 | segmented | medium | -0.0178 | 3.6644 | 0.2214 | 0.0342 | 0.1285 | 0.1711 | 0.0272 | global | 1.0697 |
| remote | AT1->DK1 | segmented | medium | 0.1140 | 1.1129 | 0.0570 | 0.2730 | 0.0799 | 0.0950 | 0.1203 | global | 0.8867 |
| remote | AT1->FR1 | segmented | medium | 0.0087 | 1.0620 | 0.1573 | 0.2961 | 0.0978 | 0.1568 | -0.0047 | global | 0.9038 |
| remote | AT1->FR2 | segmented | medium | 0.0109 | 1.0488 | 0.0658 | 0.2000 | 0.0033 | 0.1032 | 0.1736 | global | 0.8892 |
| remote | DK1->AT1 | segmented | light | 0.0113 | 0.6843 | 0.0452 | 0.3771 | 0.1672 | 0.1667 | 0.1131 | global | 0.8804 |
| remote | DK1->FR1 | dynamics | light | 0.0676 | 0.5221 | 0.0884 | 0.3307 | 0.1057 | 0.0860 | -0.0421 | global | 0.8832 |
| remote | DK1->FR2 | segmented | medium | 0.0559 | 0.5122 | 0.0285 | 0.2433 | 0.2107 | 0.1547 | 0.0970 | global | 0.8772 |
| remote | FR1->AT1 | segmented | medium | 0.0075 | 0.8692 | 0.1497 | 0.3804 | 0.0962 | 0.1480 | -0.0247 | global | 0.8971 |
| remote | FR1->DK1 | segmented | medium | 0.0396 | 0.8551 | 0.0888 | 0.3936 | 0.1308 | 0.1321 | -0.0296 | global | 0.8883 |
| remote | FR1->FR2 | global | strong | 0.0155 | 0.8209 | 0.0719 | 0.2192 | 0.0506 | 0.0482 | 0.0060 | global | 0.8855 |
| remote | FR2->AT1 | segmented | medium | 0.0327 | 0.8966 | 0.1167 | 0.4077 | 0.0562 | 0.1107 | 0.1433 | global | 0.8901 |
| remote | FR2->DK1 | segmented | medium | 0.1706 | 0.9127 | 0.0517 | 0.2239 | 0.1410 | 0.1057 | 0.0873 | global | 0.8830 |
| remote | FR2->FR1 | dynamics | light | 0.1444 | 0.9168 | 0.0717 | 0.2412 | 0.0876 | 0.0884 | -0.0977 | global | 0.8859 |

## v2.6 Decision

Proceed to v2.6.1, but only after adopting this narrower definition:

> v2.6.1 should adapt the strength of intra compactness first, using target-aware feature-geometry signals. It should not adapt view/window/component weights yet.

Recommended first factor inputs:

- `source_reliability`
- `target_margin_ratio` as a safety threshold
- `temporal_mismatch_cv`

Use `uniform_k*_adjusted_compression` later for v2.6.2 window adaptation.
