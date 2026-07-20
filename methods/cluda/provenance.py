"""Machine-readable provenance for the CLUDA behavioral reimplementation."""

UPSTREAM_REPOSITORY = "https://github.com/oezyurty/CLUDA"
UPSTREAM_BRANCH = "main"
UPSTREAM_COMMIT = "60e0b10f0e967fbada08ee6c164fc7c97932c57b"
LICENSE_STATUS = "no LICENSE/COPYING file exposed in the upstream root; all-rights-reserved fallback"

UPSTREAM_MAPPING = {
    "model.py": "main/models/cluda.py:19-232 GRL, q/k encoders, projector, predictor, discriminator, queues, momentum, MoCo and current-batch NNCL; local models/pse.py and models/tae.py:165-171 provide the TimeMatch PSE/sinusoid adapter",
    "tcn.py": "utils/tcn_no_norm.py: two causal dilated Conv1d operations, chomp, residual/downsample, ReLU, normal(0,.01) weights",
    "mlp.py": "utils/mlp.py: Linear-(BatchNorm)-ReLU-Linear with sigmoid only for scalar output",
    "nearest_neighbor.py": "utils/nearest_neighbor.py: cosine matrix and top-k current-candidate lookup",
    "augmentations.py": "utils/augmentations.py: history cutout/crop, truncated Gaussian noise, channel dropout, mask updates",
    "losses.py": "utils/loss.py and main/algorithms.py: CE/BCE components and weighted sum",
    "trainer.py": "main/algorithms.py and main/train.py: joint source/target steps, Adam betas=(.5,.99), logging/checkpoint flow",
}

OFFICIAL_DEFAULTS = {
    "model_constructor": {"momentum": .999, "queue_size": 24576, "temperature": .07,
                          "kernel_size": 2, "dilation_factor": 2, "dropout": .2, "num_neighbors": 1},
    "main_train_cli": {"momentum": .99, "queue_size": 98304, "epochs": 20,
                       "steps": 1000, "lr": 5e-5, "channels": "64-64-64-64-64",
                       "kernel_size": 3, "dilation_factor": 2, "hidden_dim": 256,
                       "dropout": 0., "all_five_loss_weights": 1.},
    "augmenter": {"cutout_length": 4, "cutout_prob": .5, "crop_min_history": .5,
                  "crop_prob": .5, "gaussian_std": .1, "channel_dropout_prob": .1},
    "readme_wisdm": {"momentum": .99, "queue_size": 8192, "lr": 1e-4,
                     "channels": "32-32-32-32-32-32", "kernel_size": 3,
                     "dilation_factor": 2, "hidden_dim": 128, "dropout": .2,
                     "weight_src_contrastive": .1, "weight_trg_contrastive": .1,
                     "weight_cross_domain_nn": .2, "weight_domain": 1., "weight_prediction": 1.},
}

INPUT_ADAPTATIONS = (
    "PixelSetEncoder(mean_std, with_extra=False) maps raw TimeMatch pixel sets to 128 features per observed date",
    "the existing frozen TimeMatch sinusoid is added using real positions; no scalar date channel is concatenated",
    "noise/dropout run on raw spectral pixels; crop/cutout mask PSE features and are reapplied after position addition",
    "the TCN output is gathered at the last valid date and L2-normalized",
    "full CLUDA target counting/loading is label-agnostic; no interpolation, fill, new date, calendar resampling, target label, or target-validation selection",
)
