import copy
import json

import pytest
import torch
from torch import nn

from methods.cluda.augmentations import CLUDAAugmenter, make_four_views
from methods.cluda.model import CLUDA, CLUDATCNClassifier, PSECLUDAEncoder
from methods.cluda.trainer import load_source_initialization
from tools.validate_cluda_source_checkpoint import validate


def _raw(batch=4, length=8, channels=10, pixels=5):
    return (
        torch.randn(batch, length, channels, pixels),
        torch.ones(batch, length, pixels),
        torch.arange(length).repeat(batch, 1),
    )


def test_pse_configuration_and_output_shape():
    encoder = PSECLUDAEncoder(10, channels=(12,), dropout=0)
    pixels, valid, positions = _raw()
    pse_features = encoder.encode_pixels(pixels, valid)
    assert pse_features.shape == (4, 8, 128)
    assert encoder.pse.with_extra is False
    assert encoder.pse.pooling == "mean_std"
    assert encoder.pse.mlp1_dim == [10, 32, 64]
    assert encoder.pse.mlp2_dim == [128, 128]
    assert encoder.tcn.network[0].conv1.in_channels == 128


def test_query_and_key_are_equal_but_independent_complete_encoders():
    model = CLUDA(10, 3, channels=(12,), hidden_dim=8, queue_size=9, dropout=0)
    assert model.encoder_q is not model.encoder_k
    assert model.encoder_q.pse is not model.encoder_k.pse
    for query, key in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        assert torch.equal(query, key)
        assert query is not key
        assert not key.requires_grad


def test_momentum_update_covers_pse_and_tcn():
    model = CLUDA(10, 3, channels=(12,), hidden_dim=8, queue_size=9, momentum=.75)
    with torch.no_grad():
        model.encoder_q.pse.mlp1[0].linear.weight.add_(2)
        model.encoder_q.tcn.network[0].conv1.weight.add_(2)
    pairs = (
        (model.encoder_q.pse.mlp1[0].linear.weight, model.encoder_k.pse.mlp1[0].linear.weight),
        (model.encoder_q.tcn.network[0].conv1.weight, model.encoder_k.tcn.network[0].conv1.weight),
    )
    expected = [.75 * k.detach().clone() + .25 * q for q, k in pairs]
    model.momentum_update_key_encoder()
    for (_, k), wanted in zip(pairs, expected):
        assert torch.allclose(k, wanted)


def test_positions_change_features_and_masked_dates_stay_zero_after_position_addition():
    torch.manual_seed(4)
    encoder = PSECLUDAEncoder(10, channels=(12,), dropout=0).eval()
    pixels, valid, positions = _raw(batch=2)
    time_mask = torch.ones(2, 8, dtype=torch.bool)
    time_mask[:, 2:4] = False
    encoded_a, temporal_a = encoder(pixels, valid, positions, time_mask, return_temporal=True)
    encoded_b = encoder(pixels, valid, positions + 1, time_mask)
    assert not torch.allclose(encoded_a, encoded_b)
    masked_pse = encoder.encode_pixels(pixels, valid) * time_mask.unsqueeze(-1)
    assert torch.count_nonzero(masked_pse[:, 2:4]) == 0
    assert torch.count_nonzero(temporal_a[:, 2:4]) == 0


def test_source_only_pse_batch_norm_updates_normally():
    classifier = CLUDATCNClassifier(10, 3, channels=(12,), hidden_dim=8, dropout=0).train()
    batch_norm = next(m for m in classifier.encoder.pse.modules() if isinstance(m, nn.BatchNorm1d))
    before = batch_norm.running_mean.clone()
    pixels, valid, positions = _raw()
    classifier(pixels, valid, positions)
    assert batch_norm.training
    assert not torch.equal(batch_norm.running_mean, before)


def test_empty_dates_do_not_enter_source_pse_batch_norm_statistics():
    torch.manual_seed(6)
    with_empty = PSECLUDAEncoder(10, channels=(12,), dropout=0).train()
    without_empty = copy.deepcopy(with_empty).train()
    pixels, valid, _ = _raw(batch=2, length=4)
    valid[:, 1] = 0
    with_empty.encode_pixels(pixels, valid)
    without_empty.encode_pixels(pixels[:, [0, 2, 3]], valid[:, [0, 2, 3]])
    bns_a = [m for m in with_empty.pse.modules() if isinstance(m, nn.BatchNorm1d)]
    bns_b = [m for m in without_empty.pse.modules() if isinstance(m, nn.BatchNorm1d)]
    for a, b in zip(bns_a, bns_b):
        assert torch.allclose(a.running_mean, b.running_mean)
        assert torch.allclose(a.running_var, b.running_var)


def test_full_cluda_freezes_pse_running_statistics_but_not_query_affine():
    model = CLUDA(10, 3, channels=(12,), hidden_dim=8, queue_size=9, dropout=0)
    model.train()
    query_bn = next(m for m in model.encoder_q.pse.modules() if isinstance(m, nn.BatchNorm1d))
    key_bn = next(m for m in model.encoder_k.pse.modules() if isinstance(m, nn.BatchNorm1d))
    assert not query_bn.training and not key_bn.training
    assert query_bn.weight.requires_grad
    before_q = (query_bn.running_mean.clone(), query_bn.running_var.clone())
    before_k = (key_bn.running_mean.clone(), key_bn.running_var.clone())
    pixels, valid, positions = _raw()
    view = make_four_views(pixels, valid, positions, pixels, valid, positions, CLUDAAugmenter())[0]
    model.encoder_q(view.pixels, view.valid_pixels, view.positions, view.time_mask)
    model._encode_key(view)
    assert torch.equal(query_bn.running_mean, before_q[0])
    assert torch.equal(query_bn.running_var, before_q[1])
    assert torch.equal(key_bn.running_mean, before_k[0])
    assert torch.equal(key_bn.running_var, before_k[1])


def test_source_checkpoint_strictly_loads_query_encoder_and_predictor(tmp_path):
    classifier = CLUDATCNClassifier(10, 3, channels=(12,), hidden_dim=8, dropout=0)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": copy.deepcopy(classifier.state_dict())}, checkpoint)
    full = CLUDA(10, 3, channels=(12,), hidden_dim=8, queue_size=9, dropout=0, kernel_size=3)
    load_source_initialization(full, classifier, str(checkpoint), "cpu")
    for expected, actual in zip(classifier.encoder.parameters(), full.encoder_q.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(classifier.predictor.parameters(), full.predictor.parameters()):
        assert torch.equal(expected, actual)
    for query, key in zip(full.encoder_q.parameters(), full.encoder_k.parameters()):
        assert torch.equal(query, key)


def test_source_initialization_rejects_ordered_class_mapping_mismatch(tmp_path):
    run_dir = tmp_path / "source"
    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    classifier = CLUDATCNClassifier(10, 3, channels=(12,), hidden_dim=8, dropout=0)
    torch.save({"state_dict": classifier.state_dict()}, fold_dir / "model.pt")
    (run_dir / "train_config.json").write_text(
        json.dumps({"classes": ["crop_a", "crop_b", "crop_c"]}), encoding="utf-8"
    )
    full = CLUDA(10, 3, channels=(12,), hidden_dim=8, queue_size=9,
                 dropout=0, kernel_size=3)
    load_source_initialization(
        full, classifier, str(fold_dir / "model.pt"), "cpu",
        expected_classes=["crop_a", "crop_b", "crop_c"],
    )
    with pytest.raises(ValueError, match="ordered class mapping"):
        load_source_initialization(
            full, classifier, str(fold_dir / "model.pt"), "cpu",
            expected_classes=["crop_b", "crop_a", "crop_c"],
        )


def test_launcher_preflight_rejects_architecture_or_task_mismatch(tmp_path):
    run_dir = tmp_path / "source"
    (run_dir / "fold_0").mkdir(parents=True)
    config = {
        "model": "cludatcn", "input_dim": 10, "with_extra": False,
        "with_shift_aug": False, "cluda_channels": "64-64-64-64-64",
        "cluda_hidden_dim": 256, "cluda_kernel_size": 3,
        "cluda_dilation_factor": 2, "cluda_dropout": 0.0,
        "cluda_max_temporal_shift": 100,
        "source": "austria/33UVP/2017", "target": "denmark/32VNH/2017", "seed": 1,
        "closed_set": True, "combine_spring_and_winter": False,
        "classes": ["crop_a", "crop_b", "crop_c"],
    }
    (run_dir / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
    classifier = CLUDATCNClassifier(10, 3)
    torch.save({"state_dict": classifier.state_dict()}, run_dir / "fold_0" / "model.pt")
    validate(run_dir, config["source"], config["target"], config["seed"])

    with pytest.raises(ValueError, match="source checkpoint identity mismatch"):
        validate(run_dir, "denmark/32VNH/2017", config["target"], config["seed"])

    config["with_extra"] = True
    (run_dir / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate(run_dir, config["source"], config["target"], config["seed"])
