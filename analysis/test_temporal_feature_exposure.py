"""Temporal feature exposure checks for v3.2.1.

These tests intentionally use a tiny PseLTae model and synthetic inputs.  They
verify API compatibility, not model quality.
"""

import torch

from models.stclassifier import PseLTae


def make_model():
    return PseLTae(
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[12, 8],
        with_extra=True,
        extra_size=4,
        n_head=2,
        d_k=4,
        d_model=8,
        mlp3=[8, 8],
        mlp4=[8, 4],
        num_classes=5,
        max_temporal_shift=5,
    )


def make_batch(batch=2, steps=6, channels=3, pixels=7):
    pixel_values = torch.randn(batch, steps, channels, pixels)
    valid_pixels = torch.ones(batch, steps, pixels)
    positions = torch.arange(steps).unsqueeze(0).repeat(batch, 1).long()
    extra = torch.randn(batch, 4)
    return pixel_values, valid_pixels, positions, extra


def main():
    torch.manual_seed(11)
    model = make_model()
    model.train()
    pixels, mask, positions, extra = make_batch()

    logits = model(pixels, mask, positions, extra)
    assert logits.shape == (2, 5)

    logits_with_feats, pooled = model(pixels, mask, positions, extra, return_feats=True)
    assert logits_with_feats.shape == logits.shape
    assert pooled.ndim == 2 and pooled.shape[0] == 2

    logits_with_temporal, temporal = model(
        pixels,
        mask,
        positions,
        extra,
        return_temporal_features=True,
    )
    assert logits_with_temporal.shape == logits.shape
    assert temporal.shape == (2, 6, 8)
    assert temporal.requires_grad

    logits_all, pooled_all, temporal_all = model(
        pixels,
        mask,
        positions,
        extra,
        return_feats=True,
        return_temporal_features=True,
    )
    assert logits_all.shape == logits.shape
    assert pooled_all.shape == pooled.shape
    assert temporal_all.shape == temporal.shape
    assert temporal_all.requires_grad
    print("temporal feature exposure checks passed")


if __name__ == "__main__":
    main()
