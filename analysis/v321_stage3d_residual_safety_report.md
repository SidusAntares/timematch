# v3.2.1 Stage 3d Residual Safety Report

## 1. Experiment Setup

Stage 3d tested whether safer residual local-shift variants can recover a useful signal after Stage 3c fixed the equivalence problem.

Settings:

```text
log_dir: logs/v321_stage3d_residual_safety_20260709_110531
tasks: AT1->DK1, FR2->FR1
seed: 1
epochs: 20
steps_per_epoch: 500
with_shift_aug: False
source checkpoint: cleaned smooth_k3 source checkpoint used by the Stage 3c/3d runs
control: global_only
```

Compared configurations:

```text
global_only
residual_raw
residual_zero_mean
residual_scaled_zero_mean_alpha05
residual_gated_scaled_zero_mean_alpha05_top065
```

All 10 jobs completed successfully.

## 2. Full Results

| task | config | val F1 | test F1 | delta vs global_only | global shift | local shift abs mean | clip frac | gate keep | pseudo conf | pseudo ratio | runtime s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AT1->DK1 | global_only | 0.7835 | 0.7874 | 0.0000 | -12 | 0.0000 | 0.0000 | 0.0000 | 0.8305 | 0.5304 | 699 |
| AT1->DK1 | residual_raw | 0.7621 | 0.7713 | -0.0161 | -13 | 11.3162 | 0.2445 | 1.0000 | 0.8110 | 0.4288 | 2016 |
| AT1->DK1 | residual_zero_mean | 0.7711 | 0.7782 | -0.0092 | -13 | 10.5337 | 0.1915 | 1.0000 | 0.8181 | 0.4544 | 2024 |
| AT1->DK1 | residual_scaled_zero_mean_alpha05 | 0.7596 | 0.7654 | -0.0220 | -13 | 5.8054 | 0.0141 | 1.0000 | 0.8258 | 0.4808 | 1989 |
| AT1->DK1 | residual_gated_scaled_zero_mean_alpha05_top065 | 0.7030 | 0.7157 | -0.0717 | -12 | 4.7067 | 0.0019 | 0.3571 | 0.8332 | 0.5203 | 2050 |
| FR2->FR1 | global_only | 0.7254 | 0.7200 | 0.0000 | -4 | 0.0000 | 0.0000 | 0.0000 | 0.9503 | 0.9108 | 754 |
| FR2->FR1 | residual_raw | 0.6987 | 0.7075 | -0.0125 | -4 | 8.8327 | 0.1310 | 1.0000 | 0.9491 | 0.9103 | 2210 |
| FR2->FR1 | residual_zero_mean | 0.7026 | 0.7060 | -0.0140 | -4 | 8.2713 | 0.0779 | 1.0000 | 0.9485 | 0.9089 | 2207 |
| FR2->FR1 | residual_scaled_zero_mean_alpha05 | 0.7141 | 0.7054 | -0.0146 | -4 | 4.3697 | 0.0060 | 1.0000 | 0.9510 | 0.9162 | 2041 |
| FR2->FR1 | residual_gated_scaled_zero_mean_alpha05_top065 | 0.7245 | 0.7120 | -0.0080 | -4 | 4.8245 | 0.0107 | 0.5234 | 0.9480 | 0.9090 | 2044 |

## 3. Best Residual vs Global-Only

| task | global_only test F1 | best residual config | best residual test F1 | best residual - global_only |
|---|---:|---|---:|---:|
| AT1->DK1 | 0.7874 | residual_zero_mean | 0.7782 | -0.0092 |
| FR2->FR1 | 0.7200 | residual_gated_scaled_zero_mean_alpha05_top065 | 0.7120 | -0.0080 |

No residual variant improved over `global_only`.

## 4. Runtime

| task | global_only runtime | residual runtime range | ratio |
|---|---:|---:|---:|
| AT1->DK1 | 699s | 1989-2050s | 2.8-2.9x |
| FR2->FR1 | 754s | 2041-2210s | 2.7-2.9x |

Residual local shift remains much slower than the valid scalar-shift control.

## 5. Interpretation

Stage 3c fixed the main control problem:

```text
base_equiv ~= smooth_base
global_forward == base_equiv
global_only == global_forward
```

Therefore `global_only` is a meaningful control for Stage 3d. Under this corrected control, all residual local-shift variants underperformed.

The negative result is meaningful because it is no longer explained by the earlier invalid control path. Directly converting stage correspondence into target-position residuals is too disruptive for the current TimeMatch training path.

## 6. Decision

Archive v3.2.1 stage-wise residual local shift as a negative boundary experiment.

Do not proceed to full12 for this method.

Do not continue tuning local-shift residual hyperparameters.

Do not add stage contrast on top of this path.

The following utilities remain useful as reusable diagnostic components:

```text
methods/local_shift/source_reference.py
methods/local_shift/target_partition.py
methods/local_shift/soft_alignment.py
methods/local_shift/local_position.py
```
