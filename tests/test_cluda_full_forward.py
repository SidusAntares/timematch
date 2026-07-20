import torch

from methods.cluda.augmentations import CLUDAAugmenter, make_four_views
from methods.cluda.model import CLUDA, _adapt_timematch_mean_debug


def test_adapter_uses_only_valid_pixels_and_never_adds_dates():
    pixels = torch.tensor([[[[2., 100.], [4., 200.]], [[8., 9.], [6., 7.]]]])
    valid = torch.tensor([[[1., 0.], [0., 0.]]])
    positions = torch.tensor([[10., 30.]])
    seq, mask = _adapt_timematch_mean_debug(pixels, valid, positions, position_scale=10.)
    assert seq.shape == mask.shape == (1, 2, 3)
    assert torch.allclose(seq[0, 0], torch.tensor([2., 4., 1.]))
    assert mask[0, 0].tolist() == [1., 1., 1.]
    assert mask[0, 1].tolist() == [0., 0., 0.]
    assert torch.count_nonzero(seq[0, 1]) == 0


def test_full_forward_has_official_outputs_and_normalized_prediction_query():
    torch.manual_seed(1)
    model = CLUDA(3, 4, channels=(5,), hidden_dim=7, queue_size=9, dropout=0).eval()
    pixels = torch.randn(3, 6, 3, 4)
    valid = torch.ones(3, 6, 4)
    positions = torch.arange(6).repeat(3, 1)
    augmenter = CLUDAAugmenter(cutout_prob=0, crop_prob=0, gaussian_std=0,
                               channel_dropout_prob=0)
    views = make_four_views(pixels, valid, positions, pixels + .02, valid, positions, augmenter)
    out = model(*views, alpha=.5)
    assert out.logits_source.shape == (3, 12)
    assert out.logits_target.shape == (3, 12)
    assert out.logits_nn.shape == (3, 3)
    assert out.source_prediction.shape == (3, 4)
    assert torch.allclose(out.q_source.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(model.last_prediction_input, out.q_source)
