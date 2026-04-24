# Result Organization Notes

## Current scope
- PRA is only tested on `FR1 -> DK1`.
- All other transfer pairs should still be treated as TimeMatch baseline results.
- Unless PRA first exceeds the `FR1 -> DK1` baseline, there is no strong reason to expand PRA to other domain adaptation pairs yet.

## Baseline overview
- `austria/33UVP/2017 -> denmark/32VNH/2017`: source `0.6170`, TimeMatch `0.6227`, delta `+0.0057`
- `austria/33UVP/2017 -> france/30TXT/2017`: source `0.6156`, TimeMatch `0.6564`, delta `+0.0408`
- `austria/33UVP/2017 -> france/31TCJ/2017`: source `0.5751`, TimeMatch `0.5934`, delta `+0.0183`
- `denmark/32VNH/2017 -> austria/33UVP/2017`: source `0.5346`, TimeMatch `0.5482`, delta `+0.0136`
- `denmark/32VNH/2017 -> france/30TXT/2017`: source `0.4248`, TimeMatch `0.5322`, delta `+0.1074`
- `denmark/32VNH/2017 -> france/31TCJ/2017`: source `0.4039`, TimeMatch `0.3873`, delta `-0.0166`
- `france/30TXT/2017 -> austria/33UVP/2017`: source `0.6244`, TimeMatch `0.6337`, delta `+0.0093`
- `france/30TXT/2017 -> denmark/32VNH/2017`: source `0.5303`, TimeMatch `0.5608`, delta `+0.0305`
- `france/30TXT/2017 -> france/31TCJ/2017`: source `0.7206`, TimeMatch `0.6864`, delta `-0.0342`
- `france/31TCJ/2017 -> austria/33UVP/2017`: source `0.5302`, TimeMatch `0.5648`, delta `+0.0346`
- `france/31TCJ/2017 -> denmark/32VNH/2017`: source `0.4052`, TimeMatch `0.4819`, delta `+0.0767`
- `france/31TCJ/2017 -> france/30TXT/2017`: source `0.6785`, TimeMatch `0.7375`, delta `+0.0590`

## FR1 -> DK1 PRA overview
- `20260424_085816`: PRA `0.5413`, vs TimeMatch `-0.0195`, vs source `+0.0111`
- `20260424_114127_t0005_w8_m8_pt095`: PRA `0.5340`, vs TimeMatch `-0.0268`, vs source `+0.0037`
- `20260424_115418_t001_w8_m8_pt095`: PRA `0.5340`, vs TimeMatch `-0.0268`, vs source `+0.0037`
- `20260424_124912_t0005_w8_m2_pt090`: PRA `0.5591`, vs TimeMatch `-0.0017`, vs source `+0.0288`
- `20260424_130315_t001_w8_m2_pt090`: PRA `0.5588`, vs TimeMatch `-0.0020`, vs source `+0.0286`
- `20260424_170957_p0005_e0005_w8_m2_pt090_b095`: PRA `0.5601`, vs TimeMatch `-0.0007`, vs source `+0.0299`
- `20260424_172346_p0005_e001_w8_m2_pt090_b095`: PRA `0.5598`, vs TimeMatch `-0.0010`, vs source `+0.0295`

## Recommended next step
- Stop expanding PRA to other source-target pairs for now.
- Keep `FR1 -> DK1` as the single PRA validation track until PRA clearly beats the TimeMatch baseline `0.5608` macro-F1.
- The next useful experiments should focus on prototype quality and scheduling, not on adding more transfer pairs.
- Suggested order:
  1. tune point alignment weight downward from `0.005`
  2. tune bank momentum upward from `0.95`
  3. if needed, retune pseudo-threshold around `0.90-0.93`
