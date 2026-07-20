from dataclasses import dataclass
import copy

import torch
from torch import nn
import torch.nn.functional as F

from models.pse import PixelSetEncoder
from models.tae import get_positional_encoding

from .augmentations import CLUDAView
from .losses import gradient_reverse
from .mlp import MLP
from .nearest_neighbor import nearest_neighbor_indices
from .tcn import TemporalConvNet


PSE_OUTPUT_DIM = 128


class PSECLUDAEncoder(nn.Module):
    """TimeMatch PSE + frozen TimeMatch sinusoid + official CLUDA TCN."""

    def __init__(self, input_dim, channels=(64, 64, 64, 64, 64), kernel_size=3,
                 stride=1, dilation_factor=2, dropout=0., max_position=365,
                 max_temporal_shift=100, positional_period=1000):
        super().__init__()
        self.max_temporal_shift = int(max_temporal_shift)
        self.pse = PixelSetEncoder(
            input_dim=input_dim,
            mlp1=[input_dim, 32, 64],
            pooling="mean_std",
            mlp2=[128, 128],
            with_extra=False,
        )
        self.position_embedding = nn.Embedding.from_pretrained(
            get_positional_encoding(
                max_position + 2 * self.max_temporal_shift,
                PSE_OUTPUT_DIM,
                T=positional_period,
            ),
            freeze=True,
        )
        self.tcn = TemporalConvNet(
            PSE_OUTPUT_DIM, channels, kernel_size, stride,
            dilation_factor, dropout,
        )

    def encode_pixels(self, pixels, valid_pixels):
        date_valid = valid_pixels.bool().any(dim=-1)
        if bool(date_valid.all()):
            return self.pse(pixels, valid_pixels, extra=None)
        features = pixels.new_zeros((*pixels.shape[:2], PSE_OUTPUT_DIM))
        if bool(date_valid.any()):
            valid_features = self.pse(
                pixels[date_valid].unsqueeze(1),
                valid_pixels[date_valid].unsqueeze(1),
                extra=None,
            ).squeeze(1)
            features[date_valid] = valid_features
        return features

    def forward(self, pixels, valid_pixels, positions, time_mask=None, return_temporal=False):
        features = self.encode_pixels(pixels, valid_pixels)
        date_valid = valid_pixels.bool().any(dim=-1)
        active = date_valid if time_mask is None else date_valid & time_mask.bool()

        # History crop/cutout is applied to PSE output, then positions are
        # added, then the same mask is applied again to remove position traces.
        active_values = active.unsqueeze(-1).to(features.dtype)
        features = features * active_values
        position_indices = positions.to(torch.long) + self.max_temporal_shift
        if position_indices.numel() and (
            int(position_indices.min()) < 0
            or int(position_indices.max()) >= self.position_embedding.num_embeddings
        ):
            raise ValueError("positions exceed the frozen TimeMatch positional embedding range")
        temporal = (features + self.position_embedding(position_indices)) * active_values

        tcn_output = self.tcn(temporal.transpose(1, 2))
        indices = torch.arange(active.shape[1], device=active.device).unsqueeze(0).expand_as(active)
        last_valid = torch.where(active, indices, torch.zeros_like(indices)).max(dim=1).values
        selected = tcn_output.gather(
            2, last_valid[:, None, None].expand(-1, tcn_output.shape[1], 1)
        ).squeeze(2)
        selected = selected * active.any(dim=1, keepdim=True).to(selected.dtype)
        encoded = F.normalize(selected, dim=1)
        return (encoded, temporal) if return_temporal else encoded


def _adapt_timematch_mean_debug(pixels, valid_pixels, positions, position_scale=365.0):
    """Legacy masked-mean diagnostic adapter; not used by official experiments."""
    if pixels.ndim != 4 or valid_pixels.ndim != 3 or positions.ndim != 2:
        raise ValueError("expected pixels [B,T,C,S], valid_pixels [B,T,S], positions [B,T]")
    weights = valid_pixels.to(dtype=pixels.dtype).unsqueeze(2)
    count = weights.sum(dim=-1)
    spectral_mean = (pixels * weights).sum(dim=-1) / count.clamp_min(1)
    date_valid = count.squeeze(2) > 0
    time_channel = positions.to(dtype=pixels.dtype).unsqueeze(-1) / float(position_scale)
    sequence = torch.cat((spectral_mean, time_channel), dim=2)
    sequence_mask = date_valid.unsqueeze(-1).expand_as(sequence).to(dtype=pixels.dtype)
    return sequence * sequence_mask, sequence_mask


class CLUDATCNClassifier(nn.Module):
    """Source-only PSE-CLUDA-TCN using the standard TimeMatch model API."""

    def __init__(self, input_dim, num_classes, channels=(64, 64, 64, 64, 64),
                 hidden_dim=256, kernel_size=3, stride=1, dilation_factor=2,
                 dropout=0., use_batch_norm=True, max_temporal_shift=100):
        super().__init__()
        self.encoder = PSECLUDAEncoder(
            input_dim, channels, kernel_size, stride, dilation_factor, dropout,
            max_temporal_shift=max_temporal_shift,
        )
        self.predictor = MLP(channels[-1], hidden_dim, num_classes, use_batch_norm)

    def forward(self, pixels, mask, positions, extra=None, return_feats=False, **_):
        del extra
        features = self.encoder(pixels, mask, positions)
        logits = self.predictor(features)
        return (logits, features) if return_feats else logits


@dataclass
class CLUDAOutput:
    logits_source: torch.Tensor
    labels_source: torch.Tensor
    logits_target: torch.Tensor
    labels_target: torch.Tensor
    logits_nn: torch.Tensor
    labels_nn: torch.Tensor
    domain_prediction: torch.Tensor
    domain_labels: torch.Tensor
    source_prediction: torch.Tensor
    q_source: torch.Tensor
    q_target: torch.Tensor
    k_source: torch.Tensor
    k_target: torch.Tensor
    source_positive_similarity: torch.Tensor
    target_positive_similarity: torch.Tensor


class CLUDA(nn.Module):
    def __init__(self, num_inputs, num_classes, channels=(64, 64, 64, 64, 64),
                 hidden_dim=256, use_batch_norm=True, num_neighbors=1, kernel_size=2,
                 stride=1, dilation_factor=2, dropout=.2, queue_size=24576,
                 momentum=.999, temperature=.07, max_temporal_shift=100):
        super().__init__()
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.num_neighbors = int(num_neighbors)
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")

        self.encoder_q = PSECLUDAEncoder(
            num_inputs, channels, kernel_size, stride, dilation_factor, dropout,
            max_temporal_shift=max_temporal_shift,
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad = False
        feature_dim = channels[-1]
        self.projector = MLP(feature_dim, hidden_dim, feature_dim, use_batch_norm)
        self.predictor = MLP(feature_dim, hidden_dim, num_classes, use_batch_norm)
        self.discriminator = MLP(feature_dim, hidden_dim, 1, use_batch_norm)
        self.register_buffer("queue_s", F.normalize(torch.randn(feature_dim, self.queue_size), dim=0))
        self.register_buffer("queue_t", F.normalize(torch.randn(feature_dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.last_prediction_input = None
        self._freeze_pse_batch_norm_stats()

    @staticmethod
    def _pse_batch_norm_eval(encoder):
        for module in encoder.pse.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _freeze_pse_batch_norm_stats(self):
        self._pse_batch_norm_eval(self.encoder_q)
        self._pse_batch_norm_eval(self.encoder_k)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_pse_batch_norm_stats()
        return self

    def _encode_query(self, view):
        return self.encoder_q(view.pixels, view.valid_pixels, view.positions, view.time_mask)

    @torch.no_grad()
    def _encode_key(self, view):
        return self.encoder_k(view.pixels, view.valid_pixels, view.positions, view.time_mask)

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        if self.training:
            for query, key in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                key.mul_(self.momentum).add_(query, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, source_keys, target_keys):
        if not self.training:
            return
        if source_keys.shape != target_keys.shape:
            raise ValueError("source and target key batches must have equal shape")
        batch_size = source_keys.shape[0]
        if batch_size > self.queue_size:
            raise ValueError("batch size cannot exceed queue size")
        pointer = int(self.queue_ptr.item())
        indices = (torch.arange(batch_size, device=self.queue_ptr.device) + pointer) % self.queue_size
        self.queue_s[:, indices] = source_keys.T
        self.queue_t[:, indices] = target_keys.T
        self.queue_ptr[0] = (pointer + batch_size) % self.queue_size

    @staticmethod
    def contrastive_logits(projected_query, key, queue, temperature):
        batch_logits = projected_query @ key.T
        queue_logits = projected_query @ queue.detach()
        labels = torch.arange(projected_query.shape[0], device=projected_query.device)
        return torch.cat((batch_logits, queue_logits), dim=1) / temperature, labels, batch_logits.diag()

    @staticmethod
    def nncl_logits(q_target, k_target, q_source, temperature, num_neighbors=1):
        source = q_source.detach()
        labels = nearest_neighbor_indices(k_target, source, num_neighbors)
        if num_neighbors != 1:
            raise ValueError("faithful first-version NNCL requires num_neighbors=1")
        return q_target @ source.T / temperature, labels.squeeze(1)

    def forward(self, source_q, source_k, target_q, target_k, alpha):
        q_source = self._encode_query(source_q)
        q_target = self._encode_query(target_q)
        projected_source = F.normalize(self.projector(q_source), dim=1)
        projected_target = F.normalize(self.projector(q_target), dim=1)

        with torch.no_grad():
            self.momentum_update_key_encoder()
            k_source = self._encode_key(source_k)
            k_target = self._encode_key(target_k)

        logits_source, labels_source, positive_source = self.contrastive_logits(
            projected_source, k_source, self.queue_s.clone(), self.temperature)
        logits_target, labels_target, positive_target = self.contrastive_logits(
            projected_target, k_target, self.queue_t.clone(), self.temperature)
        logits_nn, labels_nn = self.nncl_logits(
            q_target, k_target, q_source, self.temperature, self.num_neighbors)

        domain_input = torch.cat((gradient_reverse(q_source, alpha), gradient_reverse(q_target, alpha)), dim=0)
        domain_prediction = self.discriminator(domain_input)
        domain_labels = torch.cat((
            torch.ones(q_source.shape[0], 1, device=q_source.device),
            torch.zeros(q_target.shape[0], 1, device=q_target.device),
        ), dim=0)
        self.last_prediction_input = q_source
        source_prediction = self.predictor(q_source)
        self.dequeue_and_enqueue(k_source, k_target)
        return CLUDAOutput(
            logits_source, labels_source, logits_target, labels_target,
            logits_nn, labels_nn, domain_prediction, domain_labels,
            source_prediction, q_source, q_target, k_source, k_target,
            positive_source, positive_target,
        )

    def get_encoding(self, pixels, valid_pixels, positions, time_mask=None):
        return self.encoder_q(pixels, valid_pixels, positions, time_mask)

    def predict(self, pixels, valid_pixels, positions, time_mask=None):
        return self.predictor(self.get_encoding(pixels, valid_pixels, positions, time_mask))
