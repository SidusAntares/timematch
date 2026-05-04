import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTemporalConvReshaper(nn.Module):
    """
    Source-only feature reshaper applied on top of PSE outputs.

    The module is intentionally conservative:
    - temporal depthwise convolution to reorganize local phase structure
    - pointwise mixing to recompose channel interactions
    - residual connection so the downstream LTAE still sees PSE-like features
    """

    def __init__(self, feature_dim, strength=0.10, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.feature_dim = feature_dim
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=feature_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)
        self.gate = nn.Parameter(torch.tensor(float(strength)))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.depthwise.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, spatial_feats):
        x = self.pre_norm(spatial_feats).transpose(1, 2)
        delta = self.pointwise(torch.tanh(self.depthwise(x))).transpose(1, 2)
        return spatial_feats + torch.tanh(self.gate) * delta


def build_source_feature_reshaper(kind, feature_dim, strength=0.10, kernel_size=3):
    if kind == "none":
        return None
    if kind == "residual_temporal_conv":
        return ResidualTemporalConvReshaper(
            feature_dim=feature_dim,
            strength=strength,
            kernel_size=kernel_size,
        )
    raise ValueError(f"Unknown source_feature_reshaper kind: {kind}")


def forward_with_optional_source_reshaper(
    model,
    pixels,
    mask,
    positions,
    extra,
    source_feature_reshaper=None,
    apply_source_feature_reshaper=False,
    return_feats=False,
):
    spatial_feats = model.spatial_encoder(pixels, mask, extra)
    if apply_source_feature_reshaper and source_feature_reshaper is not None:
        spatial_feats = source_feature_reshaper(spatial_feats)
    temporal_feats = model.temporal_encoder(spatial_feats, positions)
    logits = model.decoder(temporal_feats)
    if return_feats:
        return logits, temporal_feats, spatial_feats
    return logits


def compute_source_feature_reshaper_regularization(
    input_feats,
    output_feats,
    eps=1e-6,
):
    diff = output_feats - input_feats
    identity_loss = diff.pow(2).mean()

    input_mean = input_feats.mean(dim=1)
    output_mean = output_feats.mean(dim=1)
    mean_loss = F.mse_loss(output_mean, input_mean)

    input_std = input_feats.std(dim=1, unbiased=False)
    output_std = output_feats.std(dim=1, unbiased=False)
    std_loss = F.mse_loss(output_std, input_std)

    input_energy = input_feats.pow(2).mean(dim=(1, 2))
    output_energy = output_feats.pow(2).mean(dim=(1, 2))
    energy_loss = F.mse_loss(output_energy, input_energy)

    input_flat = input_feats.reshape(input_feats.shape[0], -1)
    output_flat = output_feats.reshape(output_feats.shape[0], -1)
    cosine = F.cosine_similarity(input_flat, output_flat, dim=1, eps=eps)
    cosine_loss = (1.0 - cosine).mean()

    total = identity_loss + 0.5 * mean_loss + 0.5 * std_loss + 0.5 * energy_loss + 0.25 * cosine_loss
    logs = {
        "source_reshaper_reg_loss": float(total.detach().item()),
        "source_reshaper_identity_loss": float(identity_loss.detach().item()),
        "source_reshaper_mean_loss": float(mean_loss.detach().item()),
        "source_reshaper_std_loss": float(std_loss.detach().item()),
        "source_reshaper_energy_loss": float(energy_loss.detach().item()),
        "source_reshaper_cosine_loss": float(cosine_loss.detach().item()),
        "source_reshaper_delta_norm": float(diff.detach().pow(2).mean().sqrt().item()),
    }
    return total, logs
