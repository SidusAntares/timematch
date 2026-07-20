from dataclasses import dataclass
import torch
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, value, alpha):
        ctx.alpha = float(alpha)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def gradient_reverse(value, alpha):
    return _GradientReverse.apply(value, alpha)


@dataclass(frozen=True)
class CLUDALossWeights:
    source_contrastive: float = 1.0
    target_contrastive: float = 1.0
    cross_domain_nn: float = 1.0
    domain: float = 1.0
    prediction: float = 1.0


def compose_losses(losses, weights):
    order = ("source_contrastive", "target_contrastive", "cross_domain_nn", "domain", "prediction")
    if set(losses) != set(order):
        raise ValueError(f"loss keys must be exactly {sorted(order)}")
    # Multiplication remains in the graph at zero so the corresponding gradient is exactly zero.
    return sum(losses[name] * getattr(weights, name) for name in order)
