import torch
from methods.cluda.losses import CLUDALossWeights, compose_losses, gradient_reverse


def test_five_losses_all_enter_total_and_zero_weight_removes_only_term():
    values = {name: torch.tensor(float(i), requires_grad=True) for i, name in enumerate(
        ["source_contrastive", "target_contrastive", "cross_domain_nn", "domain", "prediction"], 1)}
    weights = CLUDALossWeights(2, 3, 4, 5, 6)
    total = compose_losses(values, weights)
    assert total.item() == 2*1 + 3*2 + 4*3 + 5*4 + 6*5
    total.backward()
    assert [values[k].grad.item() for k in values] == [2, 3, 4, 5, 6]
    names = list(values)
    base_weights = [2, 3, 4, 5, 6]
    for zero_name in names:
        probe = {k: torch.tensor(v.detach().item(), requires_grad=True) for k, v in values.items()}
        zero_weights = base_weights.copy()
        zero_weights[names.index(zero_name)] = 0
        compose_losses(probe, CLUDALossWeights(*zero_weights)).backward()
        assert probe[zero_name].grad.item() == 0
        assert all(probe[k].grad.item() != 0 for k in names if k != zero_name)


def test_grl_reverses_gradient_with_alpha():
    x = torch.tensor([2.], requires_grad=True)
    (gradient_reverse(x, .25) * 4).sum().backward()
    assert x.grad.item() == -1


def test_loss_composition_uses_documented_deterministic_order():
    losses = {
        "source_contrastive": torch.tensor(1e20),
        "target_contrastive": torch.tensor(-1e20),
        "cross_domain_nn": torch.tensor(3.),
        "domain": torch.tensor(4.),
        "prediction": torch.tensor(5.),
    }
    assert compose_losses(losses, CLUDALossWeights()) == 12
