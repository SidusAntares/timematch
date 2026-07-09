import torch

from models.stclassifier import PseLTae


def main():
    torch.manual_seed(7)
    model = PseLTae(
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean",
        mlp2=[4, 8],
        with_extra=False,
        n_head=2,
        d_k=4,
        d_model=8,
        mlp3=[8, 8],
        mlp4=[8, 5],
        num_classes=4,
        max_temporal_shift=5,
        T=64,
    )
    model.train()
    batch, steps, channels, pixels_n = 2, 6, 3, 5
    pixels = torch.randn(batch, steps, channels, pixels_n)
    valid_pixels = torch.ones(batch, steps, pixels_n)
    positions = torch.arange(steps).view(1, steps).expand(batch, -1)

    logits = model(pixels, valid_pixels, positions, None)
    logits_tf, temporal_features = model(
        pixels,
        valid_pixels,
        positions,
        None,
        return_temporal_features=True,
    )
    logits_from_temporal = model.forward_from_temporal_features(temporal_features, positions)
    logits_from_temporal_feats, pooled = model.forward_from_temporal_features(
        temporal_features,
        positions,
        return_feats=True,
    )

    assert logits.shape == (batch, 4)
    assert logits_tf.shape == logits.shape
    assert temporal_features.shape == (batch, steps, 8)
    assert logits_from_temporal.shape == logits.shape
    assert logits_from_temporal_feats.shape == logits.shape
    assert pooled.shape[0] == batch

    detached = temporal_features.detach().clone().requires_grad_(True)
    loss = model.forward_from_temporal_features(detached, positions).sum()
    loss.backward()
    assert detached.grad is not None
    assert float(detached.grad.abs().sum()) > 0.0

    print("PASS|test_forward_from_temporal_features")


if __name__ == "__main__":
    main()
