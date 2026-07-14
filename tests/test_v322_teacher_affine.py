import hashlib
import json
from pathlib import Path

import pytest
import torch

from methods.temporal_alignment.teacher_affine import (
    TeacherAffineSpec,
    load_teacher_affine_spec,
    resolve_target_teacher_positions,
)


def _spec(*, stretch=1.0, global_shift=7):
    return TeacherAffineSpec(
        task="AT1_to_FR2",
        source="austria/33UVP/2017",
        target="france/31TCJ/2017",
        seed=1,
        checkpoint_sha256="a" * 64,
        repository_branch="exp/v322-affine-temporal-alignment",
        repository_commit="12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
        global_shift=global_shift,
        stretch=stretch,
        anchor=179.0,
    )


def _write_formal_json(tmp_path, checkpoint, **task_overrides):
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    task = {
        "name": "AT1_to_FR2",
        "source": "austria/33UVP/2017",
        "target": "france/31TCJ/2017",
        "seed": 1,
    }
    task.update(task_overrides)
    document = {
        "schema_version": "v322-stretch-estimate-v1",
        "repository": {
            "branch": "exp/v322-affine-temporal-alignment",
            "commit": "12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
        },
        "task": task,
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "position": {"anchor": 179.0},
        "global_shift": {"selected": 26},
        "identity_audit": {"passed": True},
        "stretch": {
            "selected": 1.05,
            "candidates": [
                {
                    "stretch": 1.05,
                    "valid": True,
                    "embedding_range_valid": True,
                }
            ],
        },
    }
    path = tmp_path / "stretch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, digest


def test_affine_stretch_one_is_exactly_global_only():
    positions = torch.tensor([[10, 20, 30], [15, 25, 35]], dtype=torch.long)
    global_only = resolve_target_teacher_positions(positions, 7, None)
    affine_one = resolve_target_teacher_positions(positions, 7, _spec(stretch=1.0))

    assert torch.equal(global_only, positions + 7)
    assert torch.equal(affine_one, global_only)
    assert affine_one.dtype == positions.dtype


def test_affine_uses_fixed_stretch_and_anchor():
    positions = torch.tensor([[100, 179, 260]], dtype=torch.long)
    transformed = resolve_target_teacher_positions(
        positions, 7, _spec(stretch=1.1, global_shift=7)
    )
    expected = torch.round(1.1 * (positions.float() - 179.0) + 179.0 + 7).long()
    assert torch.equal(transformed, expected)


def test_formal_json_validates_task_seed_domains_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    json_path, digest = _write_formal_json(tmp_path, checkpoint)

    spec = load_teacher_affine_spec(
        json_path,
        checkpoint_path=checkpoint,
        expected_task="AT1_to_FR2",
        expected_source="austria/33UVP/2017",
        expected_target="france/31TCJ/2017",
        expected_seed=1,
        expected_repository_branch="exp/v322-affine-temporal-alignment",
        expected_repository_commit="12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
    )
    assert spec.checkpoint_sha256 == digest
    assert spec.global_shift == 26
    assert spec.stretch == pytest.approx(1.05)
    assert spec.anchor == pytest.approx(179.0)

    for field, value, message in (
        ("expected_task", "wrong", "task"),
        ("expected_source", "wrong", "source"),
        ("expected_target", "wrong", "target"),
        ("expected_seed", 2, "seed"),
    ):
        kwargs = {
            "checkpoint_path": checkpoint,
            "expected_task": "AT1_to_FR2",
            "expected_source": "austria/33UVP/2017",
            "expected_target": "france/31TCJ/2017",
            "expected_seed": 1,
            "expected_repository_branch": "exp/v322-affine-temporal-alignment",
            "expected_repository_commit": "12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            load_teacher_affine_spec(json_path, **kwargs)

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_teacher_affine_spec(
            json_path,
            checkpoint_path=checkpoint,
            expected_task="AT1_to_FR2",
            expected_source="austria/33UVP/2017",
            expected_target="france/31TCJ/2017",
            expected_seed=1,
            expected_repository_branch="exp/v322-affine-temporal-alignment",
            expected_repository_commit="12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
        )


def test_formal_json_validates_repository_identity(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    json_path, _ = _write_formal_json(tmp_path, checkpoint)
    kwargs = {
        "checkpoint_path": checkpoint,
        "expected_task": "AT1_to_FR2",
        "expected_source": "austria/33UVP/2017",
        "expected_target": "france/31TCJ/2017",
        "expected_seed": 1,
        "expected_repository_branch": "exp/v322-affine-temporal-alignment",
        "expected_repository_commit": "12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
    }
    spec = load_teacher_affine_spec(json_path, **kwargs)
    assert spec.repository_branch == kwargs["expected_repository_branch"]
    assert spec.repository_commit == kwargs["expected_repository_commit"]

    with pytest.raises(ValueError, match="repository branch"):
        load_teacher_affine_spec(
            json_path, **{**kwargs, "expected_repository_branch": "wrong"}
        )
    with pytest.raises(ValueError, match="repository commit"):
        load_teacher_affine_spec(
            json_path, **{**kwargs, "expected_repository_commit": "a" * 40}
        )


def test_formal_json_requires_passed_identity_and_valid_selected_candidate(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    json_path, _ = _write_formal_json(tmp_path, checkpoint)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["identity_audit"]["passed"] = False
    json_path.write_text(json.dumps(document), encoding="utf-8")

    kwargs = {
        "checkpoint_path": checkpoint,
        "expected_task": "AT1_to_FR2",
        "expected_source": "austria/33UVP/2017",
        "expected_target": "france/31TCJ/2017",
        "expected_seed": 1,
        "expected_repository_branch": "exp/v322-affine-temporal-alignment",
        "expected_repository_commit": "12a9374eba844485e6113e88e1b85a2fa9bc2ff7",
    }
    with pytest.raises(ValueError, match="identity audit"):
        load_teacher_affine_spec(json_path, **kwargs)

    document["identity_audit"]["passed"] = True
    document["stretch"]["candidates"][0]["valid"] = False
    json_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="selected stretch candidate"):
        load_teacher_affine_spec(json_path, **kwargs)


def test_runtime_global_shift_must_match_formal_json():
    spec = _spec(global_shift=26)
    spec.validate_global_shift(26)
    with pytest.raises(ValueError, match="global shift mismatch"):
        spec.validate_global_shift(25)


def test_affine_stretch_one_preserves_one_step_trace():
    torch.manual_seed(7)
    initial = torch.nn.Linear(3, 2)
    global_model = torch.nn.Linear(3, 2)
    affine_model = torch.nn.Linear(3, 2)
    global_model.load_state_dict(initial.state_dict())
    affine_model.load_state_dict(initial.state_dict())
    positions = torch.tensor([[10, 20, 30], [15, 25, 35]], dtype=torch.long)
    target = torch.tensor([0, 1])

    def one_step(model, transformed_positions):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer.zero_grad()
        logits = model(transformed_positions.float())
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        return (
            loss.detach(),
            gradients,
            [parameter.detach().clone() for parameter in model.parameters()],
        )

    global_trace = one_step(
        global_model, resolve_target_teacher_positions(positions, 7, None)
    )
    affine_trace = one_step(
        affine_model,
        resolve_target_teacher_positions(positions, 7, _spec(stretch=1.0)),
    )
    assert torch.equal(global_trace[0], affine_trace[0])
    for global_tensor, affine_tensor in zip(global_trace[1], affine_trace[1]):
        assert torch.equal(global_tensor, affine_tensor)
    for global_tensor, affine_tensor in zip(global_trace[2], affine_trace[2]):
        assert torch.equal(global_tensor, affine_tensor)


def test_timematch_cli_defaults_to_global_only():
    source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
        encoding="utf-8"
    )
    assert '"--timematch_target_teacher_position_mode"' in source
    assert 'default="global_only"' in source
    assert 'choices=["global_only", "affine"]' in source
    assert '"--timematch_affine_stretch_json"' in source
    assert '"--timematch_affine_task"' in source
    assert '"--timematch_affine_repository_branch"' in source
    assert '"--timematch_affine_repository_commit"' in source


def test_timematch_affine_hook_is_confined_to_teacher_positions():
    source = (
        Path(__file__).resolve().parents[1]
        / "methods"
        / "timematch_base"
        / "train_loop.py"
    ).read_text(encoding="utf-8")
    assert "teacher_affine_spec = load_teacher_affine_spec(" in source
    assert "teacher_positions = resolve_target_teacher_positions(" in source
    assert "affine_spec=teacher_affine_spec" in source
    assert "position_t[pseudo_mask]" in source
    assert "position_s + source_to_target_shift" in source
