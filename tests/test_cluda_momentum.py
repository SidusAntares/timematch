import torch
from methods.cluda.model import CLUDA


def test_key_encoder_starts_equal_and_is_frozen():
    model = CLUDA(3, 2, channels=(4,), queue_size=8)
    for q, k in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        assert torch.equal(q, k)
        assert not k.requires_grad


def test_momentum_formula_is_exact():
    model = CLUDA(3, 2, channels=(4,), queue_size=8, momentum=.75)
    old = [p.detach().clone() for p in model.encoder_k.parameters()]
    with torch.no_grad():
        for p in model.encoder_q.parameters():
            p.add_(2)
        expected = [.75 * k + .25 * q for k, q in zip(old, model.encoder_q.parameters())]
        model.momentum_update_key_encoder()
    for actual, wanted in zip(model.encoder_k.parameters(), expected):
        assert torch.allclose(actual, wanted)
