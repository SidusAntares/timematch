import torch
import torch.nn as nn
import torch.nn.functional as F


def _sort_by_positions(spatial_feats, positions):
    if positions is None:
        return spatial_feats
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    return torch.gather(spatial_feats, dim=1, index=expanded)


def _sort_with_indices(spatial_feats, positions):
    if positions is None:
        batch_size, seq_len = spatial_feats.shape[:2]
        identity = torch.arange(seq_len, device=spatial_feats.device).unsqueeze(0).expand(batch_size, -1)
        return spatial_feats, identity, identity
    sort_indices = torch.argsort(positions, dim=1)
    expanded = sort_indices.unsqueeze(-1).expand(-1, -1, spatial_feats.shape[-1])
    sorted_feats = torch.gather(spatial_feats, dim=1, index=expanded)
    inverse_indices = torch.argsort(sort_indices, dim=1)
    return sorted_feats, sort_indices, inverse_indices


def _restore_original_order(sorted_feats, inverse_indices):
    expanded = inverse_indices.unsqueeze(-1).expand(-1, -1, sorted_feats.shape[-1])
    return torch.gather(sorted_feats, dim=1, index=expanded)


def _compute_batch_domain_signature(spatial_feats, positions=None, labels=None, eps=1e-6, phase_count=5):
    ordered_feats = _sort_by_positions(spatial_feats, positions)
    temporal_mean = ordered_feats.mean(dim=1, keepdim=True)
    spread = (ordered_feats - temporal_mean).pow(2).mean(dim=(1, 2)).sqrt().mean()

    sequence_length = ordered_feats.shape[1]
    phase_slices = torch.tensor_split(
        torch.arange(sequence_length, device=ordered_feats.device, dtype=torch.long),
        min(max(phase_count, 1), sequence_length),
    )
    phase_activities = []
    for phase_indices in phase_slices:
        if phase_indices.numel() == 0:
            phase_activities.append(ordered_feats.new_tensor(0.0))
            continue
        phase_feats = ordered_feats[:, phase_indices, :]
        phase_activity = (phase_feats - phase_feats.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().mean()
        phase_activities.append(phase_activity)
    phase_activities = torch.stack(phase_activities)
    phase_contrast = phase_activities.std(unbiased=False)

    discriminability = ordered_feats.new_tensor(0.0)
    if labels is not None:
        phase_repr = ordered_feats.mean(dim=1)
        centers = []
        within_terms = []
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            if int(class_mask.sum().item()) < 2:
                continue
            class_feats = phase_repr[class_mask]
            class_center = class_feats.mean(dim=0, keepdim=True)
            centers.append(class_center.squeeze(0))
            within_terms.append((class_feats - class_center).pow(2).sum(dim=1).mean())
        if len(centers) >= 2:
            centers = torch.stack(centers, dim=0)
            between = torch.pdist(centers, p=2).mean()
            within = torch.stack(within_terms).mean()
            discriminability = between / (within.sqrt() + eps)

    signature = torch.stack([spread, phase_contrast, discriminability], dim=0)
    logs = {
        "source_domain_signature_spread": float(spread.detach().item()),
        "source_domain_signature_phase_contrast": float(phase_contrast.detach().item()),
        "source_domain_signature_discriminability": float(discriminability.detach().item()),
    }
    return signature, logs


def _compute_phase_centered_feats(spatial_feats, positions=None, phase_count=5):
    ordered_feats, _, inverse_indices = _sort_with_indices(spatial_feats, positions)
    sequence_length = ordered_feats.shape[1]
    phase_slices = torch.tensor_split(
        torch.arange(sequence_length, device=ordered_feats.device, dtype=torch.long),
        min(max(phase_count, 1), sequence_length),
    )
    centered = ordered_feats.new_zeros(ordered_feats.shape)
    for phase_indices in phase_slices:
        if phase_indices.numel() == 0:
            continue
        phase_feats = ordered_feats[:, phase_indices, :]
        centered[:, phase_indices, :] = phase_feats - phase_feats.mean(dim=1, keepdim=True)
    return _restore_original_order(centered, inverse_indices)


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
        self.last_logs = {
            "source_metric_family_weight_raw_shape": 1.0,
            "source_metric_family_weight_raw_phase": 0.0,
            "source_metric_family_weight_encoded_discriminability": 0.0,
        }
        return spatial_feats + torch.tanh(self.gate) * delta


class AdaptiveResidualTemporalConvReshaper(nn.Module):
    """
    Domain-sensitive reshaper with metric-family routing.

    Three expert branches roughly correspond to:
    - raw-shape
    - raw-phase
    - encoded-discriminability
    """

    FAMILY_NAMES = (
        "raw_shape",
        "raw_phase",
        "encoded_discriminability",
    )

    def __init__(self, feature_dim, strength=0.10, kernel_size=3, phase_count=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.feature_dim = feature_dim
        self.phase_count = phase_count
        self.pre_norm = nn.LayerNorm(feature_dim)
        expert_kernels = [kernel_size, kernel_size + 2, kernel_size + 4]
        self.expert_depthwise = nn.ModuleList([
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=k,
                padding=k // 2,
                groups=feature_dim,
                bias=False,
            )
            for k in expert_kernels
        ])
        self.expert_pointwise = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)
            for _ in expert_kernels
        ])
        self.router = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 3),
        )
        self.gate = nn.Parameter(torch.tensor(float(strength)))
        self.last_logs = {}
        self._reset_parameters()

    def _reset_parameters(self):
        for depthwise in self.expert_depthwise:
            nn.init.normal_(depthwise.weight, mean=0.0, std=1e-3)
        for pointwise in self.expert_pointwise:
            nn.init.normal_(pointwise.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(pointwise.bias)
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
                nn.init.zeros_(module.bias)

    def forward(self, spatial_feats, positions=None, labels=None):
        signature, signature_logs = _compute_batch_domain_signature(
            spatial_feats,
            positions=positions,
            labels=labels,
            phase_count=self.phase_count,
        )
        router_weights = torch.softmax(self.router(signature.unsqueeze(0)).squeeze(0), dim=0)

        x = self.pre_norm(spatial_feats).transpose(1, 2)
        expert_deltas = []
        for depthwise, pointwise in zip(self.expert_depthwise, self.expert_pointwise):
            expert_deltas.append(pointwise(torch.tanh(depthwise(x))).transpose(1, 2))
        mixed_delta = spatial_feats.new_zeros(spatial_feats.shape)
        for weight, delta in zip(router_weights, expert_deltas):
            mixed_delta = mixed_delta + weight * delta

        output = spatial_feats + torch.tanh(self.gate) * mixed_delta
        self.last_logs = dict(signature_logs)
        for family_name, weight in zip(self.FAMILY_NAMES, router_weights):
            self.last_logs[f"source_metric_family_weight_{family_name}"] = float(weight.detach().item())
        self.last_logs["source_metric_family_gate_strength"] = float(torch.tanh(self.gate).detach().item())
        return output


class ComponentizedResidualTemporalConvReshaper(nn.Module):
    """
    Interpretable source reshaper with explicit structural components.

    Instead of routing to opaque expert branches, this module decomposes the
    residual update into three named components:
    - shape: global curve spread / dynamic range adjustment
    - phase: phase-local structure adjustment
    - disc: encoded discriminability adjustment

    The batch domain signature directly determines the component coefficients,
    which makes the induced structure easier to inspect and reason about.
    """

    COMPONENT_NAMES = (
        "shape",
        "phase",
        "disc",
    )

    def __init__(self, feature_dim, strength=0.10, kernel_size=3, phase_count=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.feature_dim = feature_dim
        self.phase_count = phase_count
        self.pre_norm = nn.LayerNorm(feature_dim)

        shape_kernel = kernel_size + 2
        if shape_kernel % 2 == 0:
            shape_kernel += 1

        self.shape_depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=shape_kernel,
            padding=shape_kernel // 2,
            groups=feature_dim,
            bias=False,
        )
        self.shape_pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)

        self.phase_depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=feature_dim,
            bias=False,
        )
        self.phase_pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)

        self.disc_pointwise_in = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)
        self.disc_depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=feature_dim,
            bias=False,
        )
        self.disc_pointwise_out = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)

        self.gate = nn.Parameter(torch.tensor(float(strength)))
        self.last_logs = {}
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.shape_depthwise.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.shape_pointwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.shape_pointwise.bias)

        nn.init.normal_(self.phase_depthwise.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.phase_pointwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.phase_pointwise.bias)

        nn.init.normal_(self.disc_pointwise_in.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.disc_pointwise_in.bias)
        nn.init.normal_(self.disc_depthwise.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.disc_pointwise_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.disc_pointwise_out.bias)

    def _compute_component_alphas(self, signature, eps=1e-6):
        scores = torch.clamp(signature.detach(), min=0.0)
        score_sum = scores.sum()
        if float(score_sum.item()) <= eps:
            return torch.full_like(scores, 1.0 / scores.numel())
        return scores / (score_sum + eps)

    def forward(self, spatial_feats, positions=None, labels=None):
        signature, signature_logs = _compute_batch_domain_signature(
            spatial_feats,
            positions=positions,
            labels=labels,
            phase_count=self.phase_count,
        )
        component_alphas = self._compute_component_alphas(signature)

        normalized = self.pre_norm(spatial_feats)
        x = normalized.transpose(1, 2)

        shape_input = (normalized - normalized.mean(dim=1, keepdim=True)).transpose(1, 2)
        delta_shape = self.shape_pointwise(torch.tanh(self.shape_depthwise(shape_input))).transpose(1, 2)

        phase_input = _compute_phase_centered_feats(normalized, positions=positions, phase_count=self.phase_count).transpose(1, 2)
        delta_phase = self.phase_pointwise(torch.tanh(self.phase_depthwise(phase_input))).transpose(1, 2)

        disc_hidden = torch.tanh(self.disc_pointwise_in(x))
        delta_disc = self.disc_pointwise_out(torch.tanh(self.disc_depthwise(disc_hidden))).transpose(1, 2)

        mixed_delta = (
            component_alphas[0] * delta_shape
            + component_alphas[1] * delta_phase
            + component_alphas[2] * delta_disc
        )
        output = spatial_feats + torch.tanh(self.gate) * mixed_delta

        self.last_logs = dict(signature_logs)
        self.last_logs["source_component_alpha_shape"] = float(component_alphas[0].detach().item())
        self.last_logs["source_component_alpha_phase"] = float(component_alphas[1].detach().item())
        self.last_logs["source_component_alpha_disc"] = float(component_alphas[2].detach().item())
        self.last_logs["source_component_delta_shape_norm"] = float(delta_shape.detach().pow(2).mean().sqrt().item())
        self.last_logs["source_component_delta_phase_norm"] = float(delta_phase.detach().pow(2).mean().sqrt().item())
        self.last_logs["source_component_delta_disc_norm"] = float(delta_disc.detach().pow(2).mean().sqrt().item())
        self.last_logs["source_component_gate_strength"] = float(torch.tanh(self.gate).detach().item())
        return output


def build_source_feature_reshaper(kind, feature_dim, strength=0.10, kernel_size=3, phase_count=5):
    if kind == "none":
        return None
    if kind == "residual_temporal_conv":
        return ResidualTemporalConvReshaper(
            feature_dim=feature_dim,
            strength=strength,
            kernel_size=kernel_size,
        )
    if kind == "adaptive_residual_temporal_conv":
        return AdaptiveResidualTemporalConvReshaper(
            feature_dim=feature_dim,
            strength=strength,
            kernel_size=kernel_size,
            phase_count=phase_count,
        )
    if kind == "componentized_residual_temporal_conv":
        return ComponentizedResidualTemporalConvReshaper(
            feature_dim=feature_dim,
            strength=strength,
            kernel_size=kernel_size,
            phase_count=phase_count,
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
    return_feats=False,
):
    spatial_feats = model.spatial_encoder(pixels, mask, extra)
    if apply_source_feature_reshaper and source_feature_reshaper is not None:
        spatial_feats = source_feature_reshaper(spatial_feats, positions=positions, labels=labels)
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


def compute_dual_path_relation_regularization(
    raw_logits,
    reshaped_logits,
    raw_temporal_feats=None,
    reshaped_temporal_feats=None,
):
    """
    Tie the reshaped source path to the raw PSE path without forcing them to be identical.

    - logit consistency keeps downstream predictions aligned
    - optional temporal feature consistency keeps the reshaped path near the raw path
      at the LTAE input/output level, which is especially helpful when target data
      bypasses the reshaper entirely.
    """

    raw_probs = F.softmax(raw_logits.detach(), dim=1)
    reshaped_log_probs = F.log_softmax(reshaped_logits, dim=1)
    logit_consistency_loss = F.kl_div(
        reshaped_log_probs,
        raw_probs,
        reduction="batchmean",
    )

    temporal_consistency_loss = raw_logits.sum() * 0.0
    if raw_temporal_feats is not None and reshaped_temporal_feats is not None:
        temporal_consistency_loss = F.mse_loss(
            reshaped_temporal_feats,
            raw_temporal_feats.detach(),
        )

    total = logit_consistency_loss + temporal_consistency_loss
    logs = {
        "source_dual_relation_loss": float(total.detach().item()),
        "source_dual_logit_consistency_loss": float(logit_consistency_loss.detach().item()),
        "source_dual_temporal_consistency_loss": float(temporal_consistency_loss.detach().item()),
    }
    return total, logs
