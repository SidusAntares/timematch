import torch
from methods.cluda.augmentations import CLUDAAugmenter, make_four_views


def test_noise_and_spectral_dropout_only_change_raw_valid_pixels():
    torch.manual_seed(2)
    pixels = torch.ones(4, 12, 3, 5)
    valid = torch.ones(4, 12, 5)
    valid[:, :, -1] = 0
    positions = torch.arange(12).repeat(4, 1)
    aug = CLUDAAugmenter(cutout_prob=0, crop_prob=0, gaussian_std=.5, channel_dropout_prob=1)
    view = aug(pixels, valid, positions)
    assert torch.equal(view.positions, positions)
    assert torch.equal(view.valid_pixels, valid)
    assert torch.count_nonzero(view.pixels) == 0


def test_history_ops_keep_last_date_and_four_raw_views_are_independent():
    torch.manual_seed(5)
    pixels = torch.randn(8, 16, 10, 4)
    valid = torch.ones(8, 16, 4)
    positions = torch.arange(16).repeat(8, 1)
    aug = CLUDAAugmenter(cutout_length=4, cutout_prob=1, crop_prob=1, gaussian_std=.1, channel_dropout_prob=.1)
    views = make_four_views(pixels, valid, positions, pixels.clone(), valid.clone(), positions.clone(), aug)
    assert len(views) == 4
    assert any(not torch.equal(views[0].pixels, other.pixels) for other in views[1:])
    for view in views:
        assert torch.all(view.time_mask[:, -1])
        assert view.pixels.shape == pixels.shape
        assert view.positions.shape == positions.shape


def test_history_cutout_protects_last_valid_date_not_physical_last_column():
    torch.manual_seed(1)
    date_valid = torch.tensor([[True, True, True, False]])
    aug = CLUDAAugmenter(cutout_length=3, cutout_prob=1, crop_prob=0)
    result = aug.history_cutout_mask(date_valid)
    assert result[0, 2]
    assert not result[0, 3]
