# CLUDA provenance and PSE input adaptation audit

Audit date: 2026-07-19. Upstream: <https://github.com/oezyurty/CLUDA>, branch `main`, commit `60e0b10f0e967fbada08ee6c164fc7c97932c57b`.

## Commit and license

The external checkout was fetched, switched to `main`, fast-forwarded, and resolved to the commit above. No `LICENSE*` or `COPYING*` file exists in the checked-out repository. This repository therefore does not copy upstream source files; the CLUDA components are behavior-based reimplementations backed by a direct same-state numerical TCN equivalence test.

## Module and line-logic mapping

| Local module | Audited upstream behavior |
|---|---|
| `model.py` | `main/models/cluda.py` lines 19-232: GRL, independent q/k encoders, initialization/frozen key parameters, whole-encoder EMA, projector/predictor/discriminator, normalized q/k, independent queues, diagonal source/target positives, `NN(k_t, q_s.detach())`, `q_t @ q_s.detach().T`, domain and source prediction paths |
| `tcn.py` | `utils/tcn_no_norm.py` lines 1-64: two causal Conv1d/chomp/ReLU/dropout stages, residual/1x1 downsample, dilation by level, normal(0,.01) convolution initialization |
| `mlp.py` | `utils/mlp.py` lines 1-36: Linear, optional BatchNorm, ReLU, Linear, sigmoid only for scalar output |
| `nearest_neighbor.py` | `utils/nearest_neighbor.py` lines 1-38: cosine similarity matrix and top-k current-candidate lookup |
| `augmentations.py` | `utils/augmentations.py` lines 1-164: cutout 4/.5, crop .5/.5, truncated Gaussian .1, channel dropout .1, synchronized history masks |
| `losses.py` | `utils/loss.py` plus `main/algorithms.py` CLUDA step: source/target/NN CE, domain BCE, source prediction CE and weighted sum |
| `trainer.py` | `main/algorithms.py` and `main/train.py`: four independent views, joint Adam with betas (.5,.99), GRL schedule, joint step/logging/checkpoint flow |
| PSE adapter in `model.py` | Local TimeMatch `models/pse.py` (`PixelSetEncoder`) and `models/tae.py` lines 165-171 (frozen sinusoid); this is necessarily local because upstream expects an already dense time series |

## Defaults

Official model-constructor defaults are K=24576, m=.999, T=.07, kernel=2, dilation=2 and dropout=.2. Official `main/train.py` CLI defaults are K=98304, m=.99, five 64-channel layers, kernel=3, hidden MLP=256, lr=5e-5, dropout=0, 20 epochs, 1,000 total default steps and five weights all equal to 1. Local CLI defaults preserve these, expressing 1,000 steps as 20x50. The reviewed screen launcher deliberately uses the requested 20x500 schedule while leaving the other official CLI defaults unchanged. Augmentation defaults remain 4/.5/.5/.5/.1/.1.

## Intentional TimeMatch input adaptation

The official experiment path retains `PixelSetData`, `RandomSamplePixels`, `RandomSampleTimeSteps`, `Normalize`, `ToTensor`, splits, class mapping and evaluation. It differs from upstream only because TimeMatch supplies pixel sets rather than dense time-series vectors:

1. `PixelSetEncoder(input_dim=10, mlp1=[10,32,64], pooling=mean_std, mlp2=[128,128], with_extra=False)` maps each observed date to 128 features.
2. The existing frozen TimeMatch sinusoid is indexed by the real `positions`; no scalar `positions/365` channel is appended.
3. Gaussian noise and spectral channel dropout operate on raw spectral pixels. PSE then runs, history crop/cutout masks its date features, the sinusoid is added, and the history mask is applied again before TCN.
4. The official causal residual TCN receives 128 channels and the normalized representation is gathered at the last valid date.
5. Full CLUDA counts/splits target parcels without inspecting labels and loads target training parcels with `ignore_labels=True`, stripping metadata labels and emitting only dummy `-1` labels that the step never reads. No interpolation, forward fill, new date, regular calendar grid, parcel extra feature, target label, or target-validation checkpoint selection is used.

The legacy masked-mean adapter remains callable only for internal regression/audit coverage. It is not used by either `plain_psecludatcn`, `cluda_pse_full`, or the official launcher.

## PSE BatchNorm and initialization

Source-only PSE BatchNorm runs normally. Full CLUDA freezes q/k PSE running statistics after every `train()` call while leaving q affine parameters trainable; all key parameters are gradient-free. Source initialization verifies task/seed/mapping flags and the exact ordered class list, strictly loads the source classifier, then strictly loads its encoder and predictor into full CLUDA and copies q to k. Projector, discriminator and queues retain their random initialization.

The official queue write does not handle a batch crossing the tail. The local queue retains FIFO behavior but uses modular indices so a crossing batch wraps correctly, as required by the audit tests.
