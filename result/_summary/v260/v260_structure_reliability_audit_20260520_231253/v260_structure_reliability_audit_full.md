# v2.6.0 Structure Reliability Audit

| dataset | task | feature | src_rel | tgt_margin | mismatch_cv | segK5_adj | dyn_cos | suggested | factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | 12->16 | raw_input | 2.3788 | 0.1065 | 0.2652 | 0.3169 | 0.1004 | global | 0.8750 |
| HAR | 2->11 | raw_input | 3.9169 | 0.2629 | 0.0580 | 0.0227 | -0.0442 | global | 0.8750 |
| HAR | 6->23 | raw_input | 3.5241 | 0.2070 | 0.3551 | 0.1313 | -0.0845 | global | 0.8750 |
| HAR | 7->13 | raw_input | 3.0992 | 0.2698 | 0.2012 | 0.0686 | -0.0452 | global | 0.8750 |
| HAR | 9->18 | raw_input | 2.3604 | 0.1538 | 0.2171 | 0.0084 | -0.0484 | global | 0.8750 |
| HHAR_SA | 0->6 | raw_input | 0.5168 | 0.0781 | 0.2810 | 0.0606 | -0.0340 | global | 0.8750 |
| HHAR_SA | 1->6 | raw_input | 4.7140 | 0.2271 | 0.4010 | 0.1579 | 0.0702 | global | 0.8750 |
| HHAR_SA | 2->7 | raw_input | 0.5155 | 0.0317 | 0.3056 | 0.0590 | 0.0834 | global | 0.8750 |
| HHAR_SA | 3->8 | raw_input | 3.2641 | 0.3466 | 0.4014 | 0.0091 | -0.0440 | global | 0.8750 |
| HHAR_SA | 4->5 | raw_input | 4.1605 | 0.2249 | 0.3551 | 0.0288 | -0.0155 | global | 0.8750 |
| remote | AT1->DK1 | raw_input | 0.6926 | 0.1546 | 0.5882 | 0.0395 | 0.0014 | global | 0.8750 |
| remote | AT1->FR1 | raw_input | 0.6109 | 0.1112 | 0.6380 | 0.0714 | -0.1858 | global | 0.8750 |
| remote | AT1->FR2 | raw_input | 0.6547 | 0.0445 | 0.6758 | 0.0570 | 0.4161 | dynamics | 0.8750 |
| remote | DK1->AT1 | raw_input | 0.3805 | 0.2673 | 0.5884 | 0.0383 | 0.0012 | global | 0.8750 |
| remote | DK1->FR1 | raw_input | 0.3482 | 0.1673 | 0.5976 | 0.0702 | -0.0957 | global | 0.8750 |
| remote | DK1->FR2 | raw_input | 0.3110 | 0.2015 | 0.5100 | 0.0047 | -0.1068 | global | 0.8750 |
| remote | FR1->AT1 | raw_input | 0.5905 | 0.3576 | 0.6350 | 0.0664 | -0.1884 | global | 0.8750 |
| remote | FR1->DK1 | raw_input | 0.5147 | 0.1805 | 0.6111 | 0.0689 | -0.0998 | global | 0.8750 |
| remote | FR1->FR2 | raw_input | 0.4734 | 0.1981 | 0.5937 | 0.0594 | -0.1065 | global | 0.8750 |
| remote | FR2->AT1 | raw_input | 0.5755 | 0.0761 | 0.7139 | 0.0563 | 0.3815 | dynamics | 0.8750 |
| remote | FR2->DK1 | raw_input | 0.5320 | 0.0340 | 0.5293 | 0.0132 | -0.0775 | global | 0.8750 |
| remote | FR2->FR1 | raw_input | 0.5597 | 0.0818 | 0.5852 | 0.0564 | -0.1204 | global | 0.8750 |
