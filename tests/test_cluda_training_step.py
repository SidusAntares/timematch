import copy
import inspect
from pathlib import Path
import torch
from methods.cluda.augmentations import CLUDAAugmenter
from methods.cluda.config import CLUDAConfig
from methods.cluda.model import CLUDA
from methods.cluda.trainer import cluda_training_step, grl_alpha


def _sample(with_label=True):
    sample = {
        "pixels": torch.randn(4, 8, 2, 3),
        "valid_pixels": torch.ones(4, 8, 3),
        "positions": torch.arange(8).repeat(4, 1),
    }
    if with_label:
        sample["label"] = torch.tensor([0, 1, 2, 1])
    return sample


def test_cpu_training_step_has_five_finite_losses_and_ignores_target_labels():
    torch.manual_seed(9)
    cfg = CLUDAConfig(channels=(6,), hidden_dim=8, queue_size=11, gaussian_std=.01)
    model = CLUDA(2, 3, channels=cfg.channels, hidden_dim=cfg.hidden_dim, queue_size=cfg.queue_size, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target_a = _sample(); target_b = {k: v.clone() for k, v in target_a.items()}; target_b["label"] = torch.tensor([2, 2, 2, 2])
    state = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state()
    metrics_a = cluda_training_step(model, optimizer, _sample(), target_a, cfg, 0, 10, CLUDAAugmenter.from_config(cfg))
    model.load_state_dict(state); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); torch.random.set_rng_state(rng)
    metrics_b = cluda_training_step(model, optimizer, _sample(), target_b, cfg, 0, 10, CLUDAAugmenter.from_config(cfg))
    for key in ["loss_source_contrastive", "loss_target_contrastive", "loss_cross_domain_nn", "loss_domain", "loss_prediction", "loss_total"]:
        assert torch.isfinite(torch.tensor(metrics_a[key]))
    assert metrics_a["loss_total"] == metrics_b["loss_total"]
    assert model.queue_ptr.item() == 4


def test_full_trainer_never_uses_target_validation_for_checkpointing():
    from methods.cluda.trainer import train_cluda_full
    source = inspect.getsource(train_cluda_full)
    assert "validation(" not in source
    assert "target_sample[\"label\"]" not in inspect.getsource(cluda_training_step)
    assert "ignore_labels=True" in source
    train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")
    assert "ignore_labels=config.method == 'cluda'" in train_source


def test_official_grl_schedule_uses_one_thousand_step_denominator():
    expected = 2.0 / (1.0 + torch.exp(torch.tensor(-1.0))).item() - 1.0
    assert abs(grl_alpha(100, 1000, "official") - expected) < 1e-7


def test_train_entry_routes_cluda_directly_and_launcher_only_names_pse_screen_groups():
    root = Path(__file__).resolve().parents[1]
    train_source = (root / "train.py").read_text(encoding="utf-8")
    assert "elif config.method == 'cluda':" in train_source
    assert "train_cluda_full(model, config" in train_source
    assert "choices=['psetae', 'pseltae', 'psetcnn', 'psegru', 'cludatcn']" in train_source
    launcher = (root / "launchers" / "baselines" / "run_cluda_pse_screen.sh").read_text(encoding="utf-8")
    for name in ("plain_psecludatcn", "cluda_pse_full"):
        assert name in launcher
    for excluded in ("plain_cludatcn", "cluda_full_random", "masked_mean"):
        assert excluded not in launcher
    for task in ("AT1_to_DK1", "DK1_to_FR2", "FR2_to_FR1"):
        assert task in launcher
    assert "tools/validate_cluda_source_checkpoint.py" in launcher
    assert "--cluda_init source_weights" in launcher
    assert launcher.index('source_name="plain_psecludatcn') < launcher.index('full_name="cluda_pse_full')
