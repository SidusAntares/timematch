import torch
from methods.cluda.model import CLUDA


def test_source_target_queues_are_independent_and_wrap():
    model = CLUDA(3, 2, channels=(2,), queue_size=5)
    model.train()
    model.queue_ptr[0] = 4
    source = torch.tensor([[1., 0.], [2., 0.], [3., 0.]])
    target = torch.tensor([[0., 1.], [0., 2.], [0., 3.]])
    model.dequeue_and_enqueue(source, target)
    assert model.queue_ptr.item() == 2
    assert torch.equal(model.queue_s[:, [4, 0, 1]], source.T)
    assert torch.equal(model.queue_t[:, [4, 0, 1]], target.T)
    assert not torch.equal(model.queue_s, model.queue_t)
