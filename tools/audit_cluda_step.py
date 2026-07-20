"""Run a deterministic CPU CLUDA step and print all required audit scalars."""

import torch

from methods.cluda.augmentations import CLUDAAugmenter
from methods.cluda.config import CLUDAConfig
from methods.cluda.model import CLUDA
from methods.cluda.trainer import cluda_training_step


def _sample(labels):
    return {
        "pixels": torch.randn(4, 8, 10, 3),
        "valid_pixels": torch.ones(4, 8, 3),
        "positions": torch.arange(8).repeat(4, 1),
        "label": torch.tensor(labels),
    }


def main():
    torch.manual_seed(20260719)
    config = CLUDAConfig(channels=(6,), hidden_dim=8, queue_size=11, gaussian_std=.01)
    model = CLUDA(10, 3, channels=config.channels, hidden_dim=config.hidden_dim,
                  queue_size=config.queue_size, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = cluda_training_step(
        model,
        optimizer,
        _sample([0, 1, 2, 1]),
        _sample([2, 0, 1, 2]),
        config,
        global_step=0,
        total_steps=10,
        augmenter=CLUDAAugmenter.from_config(config),
    )
    for name, value in metrics.items():
        print(f"{name}={value:.12g}" if isinstance(value, float) else f"{name}={value}")


if __name__ == "__main__":
    main()
