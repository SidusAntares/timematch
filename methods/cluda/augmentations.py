from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CLUDAView:
    pixels: torch.Tensor
    valid_pixels: torch.Tensor
    positions: torch.Tensor
    time_mask: torch.Tensor


class CLUDAAugmenter:
    """Build a raw-pixel view and its later PSE-history mask.

    Gaussian noise and channel dropout are applied before PSE. History cutout
    and crop only produce a temporal mask; the encoder applies that mask after
    PSE and again after adding positional encodings.
    """

    def __init__(self, cutout_length=4, cutout_prob=.5, crop_min_history=.5,
                 crop_prob=.5, gaussian_std=.1, channel_dropout_prob=.1):
        self.cutout_length = int(cutout_length)
        self.cutout_prob = float(cutout_prob)
        self.crop_min_history = float(crop_min_history)
        self.crop_prob = float(crop_prob)
        self.gaussian_std = float(gaussian_std)
        self.channel_dropout_prob = float(channel_dropout_prob)

    @classmethod
    def from_config(cls, config):
        return cls(config.cutout_length, config.cutout_prob, config.crop_min_history,
                   config.crop_prob, config.gaussian_std, config.channel_dropout_prob)

    def gaussian_noise(self, pixels, valid_pixels):
        if self.gaussian_std == 0:
            return pixels
        noise = nn.init.trunc_normal_(
            torch.empty_like(pixels), std=self.gaussian_std,
            a=-2 * self.gaussian_std, b=2 * self.gaussian_std,
        )
        return pixels + noise * valid_pixels.unsqueeze(2).to(pixels.dtype)

    def spectral_dropout(self, pixels):
        if self.channel_dropout_prob == 0:
            return pixels
        keep = (
            torch.rand(pixels.shape[0], 1, pixels.shape[2], 1, device=pixels.device)
            > self.channel_dropout_prob
        ).to(pixels.dtype)
        return pixels * keep

    def history_cutout_mask(self, date_valid):
        batch, length = date_valid.shape
        if length <= 1 or self.cutout_length <= 0:
            return date_valid
        width = min(self.cutout_length, length - 1)
        starts = torch.randint(0, max(length - width, 1), (batch, 1), device=date_valid.device)
        indices = torch.arange(length, device=date_valid.device).unsqueeze(0)
        keep = ~((indices >= starts) & (indices < starts + width))
        selected = torch.rand(batch, 1, device=date_valid.device) < self.cutout_prob
        keep = torch.where(selected, keep, torch.ones_like(keep))
        keep = self._protect_last_valid(keep, date_valid)
        return date_valid & keep

    @staticmethod
    def _protect_last_valid(keep, date_valid):
        indices = torch.arange(date_valid.shape[1], device=date_valid.device).unsqueeze(0)
        last = torch.where(date_valid, indices, torch.full_like(indices, -1)).max(dim=1).values
        has_date = last >= 0
        if bool(has_date.any()):
            rows = torch.arange(date_valid.shape[0], device=date_valid.device)[has_date]
            keep[rows, last[has_date]] = True
        return keep

    def history_crop_mask(self, date_valid):
        batch, length = date_valid.shape
        has_date = date_valid.any(dim=1)
        first = torch.where(
            has_date,
            date_valid.to(torch.int64).argmax(dim=1),
            torch.zeros(batch, dtype=torch.long, device=date_valid.device),
        )
        random_fraction = torch.rand(batch, device=date_valid.device) * self.crop_min_history
        starts = first + ((length - first).to(torch.float32) * random_fraction).to(torch.long)
        indices = torch.arange(length, device=date_valid.device).unsqueeze(0)
        keep = indices >= starts.unsqueeze(1)
        selected = torch.rand(batch, 1, device=date_valid.device) < self.crop_prob
        keep = torch.where(selected, keep, torch.ones_like(keep))
        keep = self._protect_last_valid(keep, date_valid)
        return date_valid & keep

    def __call__(self, pixels, valid_pixels, positions):
        if pixels.ndim != 4 or valid_pixels.ndim != 3 or positions.ndim != 2:
            raise ValueError("expected pixels [B,T,C,S], valid_pixels [B,T,S], positions [B,T]")
        augmented = self.gaussian_noise(pixels.clone(), valid_pixels)
        augmented = self.spectral_dropout(augmented)
        time_mask = valid_pixels.bool().any(dim=-1)
        time_mask = self.history_cutout_mask(time_mask)
        time_mask = self.history_crop_mask(time_mask)
        return CLUDAView(augmented, valid_pixels.clone(), positions.clone(), time_mask)


def make_four_views(source_pixels, source_valid, source_positions,
                    target_pixels, target_valid, target_positions, augmenter):
    return (
        augmenter(source_pixels, source_valid, source_positions),
        augmenter(source_pixels, source_valid, source_positions),
        augmenter(target_pixels, target_valid, target_positions),
        augmenter(target_pixels, target_valid, target_positions),
    )
