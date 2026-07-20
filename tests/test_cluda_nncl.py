import torch
from methods.cluda.model import CLUDA
from methods.cluda.nearest_neighbor import nearest_neighbor_indices


def test_nncl_uses_key_target_to_detached_current_source_query():
    q_s = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    q_t = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    k_t = torch.tensor([[.1, .9], [.9, .1]])
    logits, labels = CLUDA.nncl_logits(q_t, k_t, q_s, temperature=.5, num_neighbors=1)
    assert torch.equal(labels, nearest_neighbor_indices(k_t, q_s.detach(), 1).squeeze(1))
    assert torch.allclose(logits, q_t @ q_s.detach().T / .5)
    logits.sum().backward()
    assert q_s.grad is None
    assert q_t.grad is not None


def test_batch_contrastive_positive_labels_are_diagonal():
    q = torch.eye(3)
    k = torch.eye(3)
    queue = torch.randn(3, 4)
    _, labels, positive = CLUDA.contrastive_logits(q, k, queue, .2)
    assert torch.equal(labels, torch.arange(3))
    assert torch.allclose(positive, torch.ones(3))
