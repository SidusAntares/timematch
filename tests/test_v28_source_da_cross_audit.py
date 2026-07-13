from pathlib import Path
import csv
from types import SimpleNamespace
import tempfile
import unittest

from tools.v28_source_da_cross_audit import (
    common_f1_value,
    effect_scale,
    factorial_effects,
    summarize_cross,
    lookup_source_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class V28SourceDACrossAuditTest(unittest.TestCase):
    def test_gate_reads_common_macro_f1_from_existing_da_table(self):
        self.assertEqual(common_f1_value({"common_macro_f1": "0.75"}), "0.75")
        self.assertEqual(common_f1_value({"common_test_f1": "0.50"}), "0.50")
        self.assertEqual(common_f1_value({"macro_f1": "0.25"}), "0.25")

    def test_factorial_effect_orientation(self):
        effects = factorial_effects(a=1.0, b=8.0, c=3.0, d=5.0)
        self.assertEqual(effects["da_effect_under_old_source_C_minus_A"], 2.0)
        self.assertEqual(effects["da_effect_under_cleaned_source_B_minus_D"], 3.0)
        self.assertEqual(effects["source_effect_under_old_da_D_minus_A"], 4.0)
        self.assertEqual(effects["source_effect_under_cleaned_da_B_minus_C"], 5.0)
        self.assertEqual(effects["interaction_B_plus_A_minus_C_minus_D"], 1.0)

    def test_cross_launcher_requires_first_round_gate(self):
        text = (ROOT / "launchers/v28/launch_v28_source_da_cross_audit_4gpu.sh").read_text(encoding="utf-8")
        self.assertIn("FIRST_ROUND_PASSED", text)
        self.assertIn("jobs+=(\"${task}|${method}|old|cleaned\")", text)
        self.assertIn("jobs+=(\"${task}|${method}|cleaned|old\")", text)
        self.assertIn("[[ \"${#jobs[@]}\" -eq 8 ]]", text)
        self.assertIn("source_da_2x2_effects.tsv", text)
        self.assertIn("cross_status", text)

    def test_first_round_has_expected_evaluation_count(self):
        text = (ROOT / "tools/v28_source_da_cross_audit.py").read_text(encoding="utf-8")
        self.assertIn('check("source_common_evaluations", 144', text)
        self.assertIn('check("existing_da_common_evaluations", 8', text)
        for field in ("check_name", "completed", "failed", "status", "details"):
            self.assertIn(field, text)

    def test_existing_da_common_eval_has_auditable_load_fields(self):
        evaluator = (ROOT / "tools/evaluate_common_timematch_checkpoint.py").read_text(
            encoding="utf-8"
        )
        audit = (ROOT / "tools/v28_source_da_cross_audit.py").read_text(encoding="utf-8")
        for field in ("load_status", "missing_keys", "unexpected_keys"):
            self.assertIn(field, evaluator)
            self.assertIn(field, audit)
        for field in (
            "checkpoint_sha256",
            "common_accuracy",
            "common_macro_f1",
            "common_weighted_f1",
            "common_kappa",
            "delta_common_minus_native",
        ):
            self.assertIn(field, audit)

    def test_resume_mode_requires_all_source_results_before_existing_da_only(self):
        text = (
            ROOT / "launchers/v28/launch_v28_source_checkpoint_first_round_4gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("EXISTING_DA_ONLY", text)
        self.assertIn("SOURCE_RESULT_COUNT", text)
        self.assertIn('[[ "${SOURCE_RESULT_COUNT}" -eq 144 ]]', text)

    def test_final_launcher_stops_before_cross_when_existing_da_gate_fails(self):
        text = (
            ROOT / "launchers/v28/launch_v28_source_da_final_causal_audit_4gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("EXISTING_DA_ONLY=True", text)
        self.assertIn("FIRST_ROUND_PASSED", text)
        self.assertIn("exit 2", text)
        self.assertIn("launch_v28_source_da_cross_audit_4gpu.sh", text)

    def test_cross_outputs_separate_common_eval_results_and_effects(self):
        launcher = (
            ROOT / "launchers/v28/launch_v28_source_da_cross_audit_4gpu.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "cross_da_common_eval.tsv",
            "source_da_2x2_results.tsv",
            "source_da_2x2_effects.tsv",
        ):
            self.assertIn(name, launcher)
        for field in (
            "dominant_effect",
            "interpretation",
            "da_effect_under_old_source",
            "source_effect_under_old_da",
        ):
            self.assertIn(field, (ROOT / "tools/v28_source_da_cross_audit.py").read_text(encoding="utf-8"))

    def test_cross_launcher_records_environment_and_explicit_recipe(self):
        text = (
            ROOT / "launchers/v28/launch_v28_source_da_cross_audit_4gpu.sh"
        ).read_text(encoding="utf-8")
        for token in (
            "start_time",
            "end_time",
            "command",
            "git_commit",
            "python_version",
            "torch_version",
            "cuda_version",
            "--domain_specific_bn True",
            "--shift_estimator AM",
            "--sample_size 100",
            "--max_temporal_shift 60",
            "--balance_source True",
            "--use_focal_loss True",
            "--shift_source True",
            "refresh_status",
        ):
            self.assertIn(token, text)

    def test_effect_scale_uses_declared_thresholds(self):
        self.assertEqual(effect_scale(0.0001), "strict_numerical_equivalence")
        self.assertEqual(effect_scale(0.002), "practical_near_equivalence")
        self.assertEqual(effect_scale(0.01), "small_effect")
        self.assertEqual(effect_scale(0.01001), "material_effect")

    def test_summarize_cross_accepts_existing_common_macro_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.tsv"
            cross = root / "cross.tsv"
            results = root / "results.tsv"
            effects = root / "effects.tsv"
            with existing.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["task", "config", "source_version", "da_version", "common_macro_f1"],
                    delimiter="\t",
                )
                writer.writeheader()
                for task in ("AT1_to_FR2", "FR2_to_FR1"):
                    for method in ("base", "smooth_k3"):
                        writer.writerow({"task": task, "config": method, "source_version": "old", "da_version": "old", "common_macro_f1": 0.1})
                        writer.writerow({"task": task, "config": method, "source_version": "cleaned", "da_version": "cleaned", "common_macro_f1": 0.4})
            with cross.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["task", "method", "source_version", "da_version", "common_test_f1"],
                    delimiter="\t",
                )
                writer.writeheader()
                for task in ("AT1_to_FR2", "FR2_to_FR1"):
                    for method in ("base", "smooth_k3"):
                        writer.writerow({"task": task, "method": method, "source_version": "old", "da_version": "cleaned", "common_test_f1": 0.2})
                        writer.writerow({"task": task, "method": method, "source_version": "cleaned", "da_version": "old", "common_test_f1": 0.3})
            with self.assertRaises(SystemExit) as outcome:
                summarize_cross(SimpleNamespace(existing_da_eval=existing, cross_summary=cross, results_output=results, output=effects))
            self.assertEqual(outcome.exception.code, 0)
            self.assertEqual(len(results.read_text(encoding="utf-8").splitlines()), 5)
            self.assertIn("dominant_effect", effects.read_text(encoding="utf-8"))

    def test_source_checkpoint_lookup_returns_one_manifest_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "source.tsv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source_domain", "config", "seed", "old_checkpoint_path", "cleaned_checkpoint_path"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"source_domain": "AT1", "config": "base", "seed": 1, "old_checkpoint_path": "/old/model.pt", "cleaned_checkpoint_path": "/clean/model.pt"})
            self.assertEqual(lookup_source_checkpoint(manifest, "AT1", "base", "old", 1), "/old/model.pt")
            self.assertEqual(lookup_source_checkpoint(manifest, "AT1", "base", "cleaned", 1), "/clean/model.pt")

    def test_cross_launcher_preflights_all_eight_checkpoints_before_training(self):
        text = (ROOT / "launchers/v28/launch_v28_source_da_cross_audit_4gpu.sh").read_text(encoding="utf-8")
        self.assertIn("lookup-source-checkpoint", text)
        self.assertIn("CROSS_PREFLIGHT_OK|jobs=8", text)
        self.assertLess(text.index("CROSS_PREFLIGHT_OK|jobs=8"), text.index("run_job()"))
        self.assertIn("REUSE_EXISTING_CROSS", text)
        self.assertIn("REUSED_CROSS", text)


if __name__ == "__main__":
    unittest.main()
