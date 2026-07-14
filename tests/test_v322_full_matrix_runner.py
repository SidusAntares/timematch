import hashlib
import json
from pathlib import Path

import pytest

from tools.run_v322_full_matrix import (
    FIXED_STRETCHES,
    MatrixUnit,
    PreflightReport,
    RunnerConfig,
    UnitLock,
    atomic_write_json,
    build_step_command,
    build_units,
    first_required_step,
    main,
    next_attempt_dir,
    preflight,
    resolve_checkpoint,
    run_matrix,
    run_unit,
    summarize_matrix,
    unit_paths,
    validate_unit,
)


BRANCH = "exp/v322-affine-temporal-alignment"
COMMIT = "a" * 40


def _config(tmp_path, *, patterns=None):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tools").mkdir()
    for name in ("audit_v322_temporal_ranges.py", "estimate_v322_stretch.py"):
        (root / "tools" / name).write_text("print('ok')", encoding="utf-8")
    (root / "train.py").write_text("print('ok')", encoding="utf-8")
    return RunnerConfig(
        repository_root=root,
        data_root=tmp_path / "data",
        output_root=tmp_path / "matrix",
        repository_branch=BRANCH,
        repository_commit=COMMIT,
        gpus=(0, 1, 2, 3),
        max_workers=4,
        python_executable="python",
        checkpoint_patterns=patterns
        or ("v28_cleaned_source_{source}_smooth_k3_seed{seed}/fold_0/model.pt",),
    )


def _checkpoint(config, unit):
    path = (
        config.repository_root
        / "outputs"
        / f"v28_cleaned_source_{unit.source}_smooth_k3_seed{unit.seed}"
        / "fold_0"
        / "model.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{unit.source}-{unit.seed}".encode())
    return path


def _write_valid_artifacts(config, unit, *, done=False):
    checkpoint = _checkpoint(config, unit)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    paths = unit_paths(config, unit)
    paths.audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "v322-temporal-range-audit-v1",
        "repository": {"branch": BRANCH, "commit": COMMIT},
        "embedding": {"embedding_index_min": 0, "embedding_index_max": 565},
        "source": {"domain": unit.source_dataset},
        "target": {"domain": unit.target_dataset},
        "checkpoint": {
            "sha256": digest,
            "source_domain": unit.source_dataset,
            "source_seed": unit.seed,
        },
    }
    paths.audit_json.write_text(json.dumps(audit), encoding="utf-8")
    paths.audit_tsv.write_text("domain\tvalue\nsource\t1\n", encoding="utf-8")
    paths.stretch_json.parent.mkdir(parents=True, exist_ok=True)
    stretch = {
        "schema_version": "v322-stretch-estimate-v1",
        "repository": {"branch": BRANCH, "commit": COMMIT},
        "task": {
            "name": unit.name,
            "source": unit.source_dataset,
            "target": unit.target_dataset,
            "seed": unit.seed,
        },
        "checkpoint": {"sha256": digest},
        "position": {"anchor": 179.0},
        "global_shift": {"selected": 3, "epsilon": 1e-5},
        "identity_audit": {"passed": True},
        "stretch": {
            "selected": 1.05,
            "margin": 0.01,
            "at_boundary": False,
            "candidates": [
                {
                    "stretch": value,
                    "valid": True,
                    "embedding_range_valid": True,
                }
                for value in FIXED_STRETCHES
            ],
        },
    }
    paths.stretch_json.write_text(json.dumps(stretch), encoding="utf-8")
    paths.stretch_tsv.write_text("stretch\tscore\n1.05\t1.0\n", encoding="utf-8")

    outputs = {}
    for mode, score in (("global_only", 0.70), ("affine", 0.72)):
        attempt = paths.base / mode / "attempt_001" / "run"
        fold = attempt / "fold_0"
        fold.mkdir(parents=True, exist_ok=True)
        train_config = {
            "source": unit.source_dataset,
            "target": unit.target_dataset,
            "seed": unit.seed,
            "weights": str(checkpoint.parent.parent),
            "timematch_shift_policy": "fixed_initial_shift",
            "timematch_shift_score_epsilon": 1e-5,
            "timematch_target_teacher_position_mode": mode,
        }
        if mode == "affine":
            train_config.update(
                timematch_affine_stretch_json=str(paths.stretch_json),
                timematch_affine_task=unit.name,
                timematch_affine_repository_branch=BRANCH,
                timematch_affine_repository_commit=COMMIT,
            )
        (attempt / "train_config.json").write_text(
            json.dumps(train_config), encoding="utf-8"
        )
        metric_name = unit.target_dataset.replace("/", "_")
        (fold / f"test_metrics_{metric_name}.json").write_text(
            json.dumps({"macro_f1": score}), encoding="utf-8"
        )
        (fold / "model.pt").write_bytes(b"student")
        outputs[mode] = str(attempt)
    atomic_write_json(
        paths.status,
        {
            "task": unit.name,
            "seed": unit.seed,
            "repository_branch": BRANCH,
            "repository_commit": COMMIT,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "audit_path": str(paths.audit_json),
            "stretch_path": str(paths.stretch_json),
            "global_output_path": outputs["global_only"],
            "affine_output_path": outputs["affine"],
            "state": "DONE" if done else "RUNNING_AFFINE",
        },
    )
    if done:
        paths.done.write_text("DONE\n", encoding="ascii")
    return paths


def test_builds_exactly_36_stably_ordered_units():
    units = build_units()
    assert len(units) == 36
    assert [(unit.name, unit.seed) for unit in units[:4]] == [
        ("AT1_to_DK1", 1),
        ("AT1_to_DK1", 2),
        ("AT1_to_DK1", 3),
        ("AT1_to_FR1", 1),
    ]
    assert all(unit.source != unit.target for unit in units)


def test_unit_directories_are_unique(tmp_path):
    config = _config(tmp_path)
    bases = [unit_paths(config, unit).base for unit in build_units()]
    assert len(bases) == len(set(bases)) == 36


def test_checkpoint_mapping_uses_matching_seed(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 2)
    expected = _checkpoint(config, unit)
    assert resolve_checkpoint(config, unit) == expected.resolve()
    assert "seed2" in str(resolve_checkpoint(config, unit))


def test_missing_checkpoint_fails_preflight_and_lists_all_missing(tmp_path):
    config = _config(tmp_path)
    report = preflight(config, check_cli=False)
    assert not report.passed
    assert len(report.missing_checkpoints) == 36


def test_command_order_and_identity_are_explicit(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    checkpoint = _checkpoint(config, unit)
    paths = unit_paths(config, unit)
    commands = [
        build_step_command(config, unit, paths, step, checkpoint=checkpoint)
        for step in ("audit", "stretch", "global_only", "affine")
    ]
    assert "audit_v322_temporal_ranges.py" in commands[0][1]
    assert "estimate_v322_stretch.py" in commands[1][1]
    assert commands[2][-1] != "affine"
    assert "--timematch_affine_stretch_json" not in commands[2]
    assert commands[3][commands[3].index("--timematch_affine_stretch_json") + 1] == str(
        paths.stretch_json
    )
    for command in commands[:2] + commands[3:]:
        assert COMMIT in command
        assert BRANCH in command


def test_audit_and_stretch_commands_do_not_use_target_labels(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("FR2", "FR1", 1)
    checkpoint = _checkpoint(config, unit)
    paths = unit_paths(config, unit)
    for step in ("audit", "stretch"):
        command = build_step_command(config, unit, paths, step, checkpoint=checkpoint)
        joined = " ".join(command).lower()
        assert "target-label" not in joined
        assert "true-label" not in joined


def test_next_attempt_never_overwrites(tmp_path):
    root = tmp_path / "global"
    (root / "attempt_001").mkdir(parents=True)
    assert next_attempt_dir(root).name == "attempt_002"


def test_atomic_status_write_is_complete(tmp_path):
    target = tmp_path / "status.json"
    atomic_write_json(target, {"state": "RUNNING_AUDIT", "task": "AT1_to_FR2"})
    assert json.loads(target.read_text(encoding="utf-8"))["state"] == "RUNNING_AUDIT"
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_prevents_duplicate_worker(tmp_path):
    lock = tmp_path / "unit.lock"
    with UnitLock(lock):
        with pytest.raises(FileExistsError):
            with UnitLock(lock):
                pass
    assert not lock.exists()


def test_locked_unit_does_not_mutate_owner_status(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    _checkpoint(config, unit)
    paths = unit_paths(config, unit)
    paths.base.mkdir(parents=True)
    atomic_write_json(paths.status, {"state": "RUNNING_STRETCH", "owner": 123})
    with UnitLock(paths.lock):
        assert not run_unit(config, unit, 1)
        assert json.loads(paths.status.read_text()) == {
            "state": "RUNNING_STRETCH",
            "owner": 123,
        }


def test_partial_completion_resumes_at_first_invalid_step(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    paths = _write_valid_artifacts(config, unit)
    Path(json.loads(paths.status.read_text())["global_output_path"]).joinpath(
        "fold_0", "model.pt"
    ).unlink()
    assert first_required_step(config, unit) == "global_only"


def test_done_unit_is_skipped_only_when_all_artifacts_validate(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    paths = _write_valid_artifacts(config, unit, done=True)
    assert first_required_step(config, unit) is None
    paths.stretch_json.write_text("{}", encoding="utf-8")
    assert first_required_step(config, unit) == "stretch"


def test_validate_unit_reports_pair_delta_and_identity(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    _write_valid_artifacts(config, unit, done=True)
    result = validate_unit(config, unit)
    assert result.done
    assert result.global_macro_f1 == pytest.approx(0.70)
    assert result.affine_macro_f1 == pytest.approx(0.72)
    assert result.delta_macro_f1 == pytest.approx(0.02)
    assert result.selected_stretch == pytest.approx(1.05)


@pytest.mark.parametrize(
    "mutation,expected_step",
    [
        (lambda document: document["repository"].update(commit="b" * 40), "audit"),
        (lambda document: document["checkpoint"].update(sha256="b" * 64), "audit"),
    ],
)
def test_identity_or_checkpoint_mismatch_refuses_reuse(
    tmp_path, mutation, expected_step
):
    config = _config(tmp_path)
    unit = MatrixUnit("FR2", "FR1", 1)
    paths = _write_valid_artifacts(config, unit, done=True)
    document = json.loads(paths.audit_json.read_text())
    mutation(document)
    paths.audit_json.write_text(json.dumps(document), encoding="utf-8")
    assert first_required_step(config, unit) == expected_step


def test_summary_is_stably_sorted_and_does_not_run_subprocess(tmp_path, monkeypatch):
    config = _config(tmp_path)
    for unit in (MatrixUnit("FR2", "FR1", 1), MatrixUnit("AT1", "FR2", 1)):
        _write_valid_artifacts(config, unit, done=True)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    summary = summarize_matrix(config, units=build_units(seeds=(1,)))
    rows = summary["rows"]
    assert rows[0]["task"] == "AT1_to_DK1"
    assert (config.output_root / "matrix_status.json").is_file()
    assert (config.output_root / "matrix_results.tsv").is_file()


def test_fixed_stretch_candidates_are_exact():
    assert FIXED_STRETCHES == (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)


def test_failed_step_stops_unit_and_records_failure(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    _checkpoint(config, unit)
    calls = []

    def fail_audit(command, **kwargs):
        calls.append(command)
        return 123, 7

    assert not run_unit(config, unit, 0, command_runner=fail_audit)
    assert len(calls) == 1
    status = json.loads(unit_paths(config, unit).status.read_text())
    assert status["state"] == "FAILED"
    assert status["exit_codes"] == {"audit": 7}
    assert "audit exited" in status["failure_message"]


def test_failed_unit_requires_explicit_restart(tmp_path):
    config = _config(tmp_path)
    unit = MatrixUnit("AT1", "FR2", 1)
    _checkpoint(config, unit)
    paths = unit_paths(config, unit)
    paths.base.mkdir(parents=True)
    atomic_write_json(
        paths.status,
        {
            "state": "FAILED",
            "repository_branch": BRANCH,
            "repository_commit": COMMIT,
            "checkpoint_sha256": hashlib.sha256(
                resolve_checkpoint(config, unit).read_bytes()
            ).hexdigest(),
        },
    )
    calls = []
    assert not run_unit(
        config, unit, 0, command_runner=lambda *args, **kwargs: calls.append(args)
    )
    assert calls == []


def test_restart_failed_reenters_pipeline(tmp_path):
    base = _config(tmp_path)
    config = RunnerConfig(**{**base.__dict__, "restart_failed": True})
    unit = MatrixUnit("AT1", "FR2", 1)
    checkpoint = _checkpoint(config, unit)
    paths = unit_paths(config, unit)
    paths.base.mkdir(parents=True)
    atomic_write_json(
        paths.status,
        {
            "state": "FAILED",
            "repository_branch": BRANCH,
            "repository_commit": COMMIT,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "commands": {},
            "processes": {},
            "exit_codes": {},
        },
    )
    calls = []

    def fail_again(command, **kwargs):
        calls.append(command)
        return 456, 9

    assert not run_unit(config, unit, 0, command_runner=fail_again)
    assert len(calls) == 1


def test_matrix_failure_does_not_block_other_units_and_returns_false(tmp_path):
    base = _config(tmp_path)
    config = RunnerConfig(**{**base.__dict__, "gpus": (0, 1), "max_workers": 2})
    units = [
        MatrixUnit("AT1", "DK1", 1),
        MatrixUnit("AT1", "FR1", 1),
        MatrixUnit("AT1", "FR2", 1),
        MatrixUnit("DK1", "AT1", 1),
    ]
    for unit in units:
        _checkpoint(config, unit)
    calls = []
    lock = __import__("threading").Lock()

    def fake_runner(_config, unit, gpu):
        with lock:
            calls.append((unit.name, gpu))
        return unit.name != "AT1_to_DK1"

    assert not run_matrix(
        config, units, unit_runner=fake_runner, check_cli=False
    )
    assert {task for task, _ in calls} == {unit.name for unit in units}
    assert dict(calls)["AT1_to_DK1"] == 0
    assert dict(calls)["AT1_to_FR2"] == 0
    assert dict(calls)["AT1_to_FR1"] == 1
    assert dict(calls)["DK1_to_AT1"] == 1


def test_single_gpu_assigns_every_unit_to_same_gpu(tmp_path):
    base = _config(tmp_path)
    config = RunnerConfig(**{**base.__dict__, "gpus": (3,), "max_workers": 1})
    units = [MatrixUnit("FR2", "FR1", 1), MatrixUnit("AT1", "FR2", 1)]
    for unit in units:
        _checkpoint(config, unit)
    assignments = []

    def fake_runner(_config, unit, gpu):
        assignments.append((unit.name, gpu))
        return True

    assert run_matrix(config, units, unit_runner=fake_runner, check_cli=False)
    assert assignments == [("FR2_to_FR1", 3), ("AT1_to_FR2", 3)]


def test_plan_mode_never_starts_training(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "tools.run_v322_full_matrix.preflight",
        lambda *args, **kwargs: PreflightReport(passed=True),
    )
    monkeypatch.setattr(
        "tools.run_v322_full_matrix.run_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan must not run training")
        ),
    )
    assert (
        main(
            [
                "plan",
                "--repository-root",
                str(config.repository_root),
                "--data-root",
                str(config.data_root),
                "--output-root",
                str(config.output_root),
                "--repository-branch",
                BRANCH,
                "--repository-commit",
                COMMIT,
                "--gpus",
                "0,1,2,3",
            ]
        )
        == 0
    )
