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

    def forward(self, spatial_feats, positions=None, labels=None):
        x = self.pre_norm(spatial_feats).transpose(1, 2)
        delta = self.pointwise(torch.tanh(self.depthwise(x))).transpose(1, 2)
        return spatial_feats + torch.tanh(self.gate) * delta


def build_source_feature_reshaper(kind, feature_dim, strength=0.10, kernel_size=3, phase_count=5):
    if kind in (None, "none"):
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
    labels=None,
    source_feature_reshaper=None,
    apply_source_feature_reshaper=False,
):
    spatial_feats = model.spatial_encoder(pixels, mask, extra)
    if source_feature_reshaper is not None and apply_source_feature_reshaper:
        spatial_feats = source_feature_reshaper(spatial_feats, positions=positions, labels=labels)
    temporal_feats = model.temporal_encoder(spatial_feats, positions)
    outputs = model.decoder(temporal_feats)
    return outputs


def compute_source_feature_reshaper_regularization(raw_spatial_feats, reshaped_spatial_feats):
    """
    Keep the source-only reshaper near-identity so downstream modules still
    recognize the transformed features as PSE outputs.
    """
    identity_loss = F.mse_loss(reshaped_spatial_feats, raw_spatial_feats)

    raw_mean = raw_spatial_feats.mean(dim=1)
    reshaped_mean = reshaped_spatial_feats.mean(dim=1)
    mean_loss = F.mse_loss(reshaped_mean, raw_mean)

    raw_std = raw_spatial_feats.std(dim=1, unbiased=False)
    reshaped_std = reshaped_spatial_feats.std(dim=1, unbiased=False)
    std_loss = F.mse_loss(reshaped_std, raw_std)

    raw_energy = raw_spatial_feats.pow(2).mean(dim=(1, 2))
    reshaped_energy = reshaped_spatial_feats.pow(2).mean(dim=(1, 2))
    energy_loss = F.mse_loss(reshaped_energy, raw_energy)

    raw_flat = raw_spatial_feats.reshape(raw_spatial_feats.shape[0], -1)
    reshaped_flat = reshaped_spatial_feats.reshape(reshaped_spatial_feats.shape[0], -1)
    cosine_sim = F.cosine_similarity(raw_flat, reshaped_flat, dim=1)
    cosine_loss = (1.0 - cosine_sim).mean()

    total = identity_loss + 0.5 * mean_loss + 0.5 * std_loss + 0.5 * energy_loss + 0.25 * cosine_loss

    logs = {
        "source_reshaper_reg_loss": float(total.detach().item()),
        "source_reshaper_identity_loss": float(identity_loss.detach().item()),
        "source_reshaper_mean_loss": float(mean_loss.detach().item()),
        "source_reshaper_std_loss": float(std_loss.detach().item()),
        "source_reshaper_energy_loss": float(energy_loss.detach().item()),
        "source_reshaper_cosine_loss": float(cosine_loss.detach().item()),
    }
    return total, logs


def compute_dual_path_relation_regularization(
    raw_logits,
    reshaped_logits,
    raw_temporal_feats=None,
    reshaped_temporal_feats=None,
):
    """
    Encourage the reshaped source path to preserve the predictive semantics of the
    raw PSE path while still allowing small structure-inducing changes.
    """
    with torch.no_grad():
        raw_probs = F.softmax(raw_logits, dim=1)
    logit_consistency = F.kl_div(
        F.log_softmax(reshaped_logits, dim=1),
        raw_probs,
        reduction="batchmean",
    )

    temporal_consistency = raw_logits.sum() * 0.0
    if raw_temporal_feats is not None and reshaped_temporal_feats is not None:
        temporal_consistency = F.mse_loss(reshaped_temporal_feats, raw_temporal_feats.detach())

    total = logit_consistency + temporal_consistency
    logs = {
        "source_dual_relation_loss": float(total.detach().item()),
        "source_dual_logit_consistency": float(logit_consistency.detach().item()),
        "source_dual_temporal_consistency": float(temporal_consistency.detach().item()),
    }
    return total, logs
