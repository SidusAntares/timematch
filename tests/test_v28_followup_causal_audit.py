from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V28FollowupCausalAuditTest(unittest.TestCase):
    def test_recovery_flow_is_bounded_and_reparses_matches(self):
        tool = ROOT / "tools/recover_v28_old_artifacts.py"
        launcher = ROOT / "launchers/v28/launch_v28_old_artifact_recovery.sh"
        self.assertTrue(tool.is_file())
        self.assertTrue(launcher.is_file())
        text = tool.read_text(encoding="utf-8")
        self.assertIn("--search_root", text)
        self.assertIn("--max_depth", text)
        self.assertIn("recovered_results.tsv", text)
        self.assertNotIn("Path('/').rglob", text)

    def test_source_task_specific_smoke_has_exact_four_cells(self):
        tool = ROOT / "tools/trace_v28_source_task_smoke.py"
        launcher = ROOT / "launchers/v28/launch_v28_source_task_specific_smoke_4gpu.sh"
        self.assertTrue(tool.is_file())
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("FR1_to_FR2,FR1_to_DK1", text)
        self.assertIn("base,smooth_k3", text)
        self.assertIn("SOURCE_EPOCHS=\"${SOURCE_EPOCHS:-1}\"", text)
        self.assertIn("source_task_specific_smoke.tsv", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertNotIn("FR2_to_FR1", text)

    def test_full_da_audit_is_only_at1_fr2_four_jobs(self):
        tool = ROOT / "tools/run_v28_full_da_audit.py"
        launcher = ROOT / "launchers/v28/launch_v28_at1_fr2_full_da_audit_4gpu.sh"
        self.assertTrue(tool.is_file())
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("TASK=\"${TASK:-AT1_to_FR2}\"", text)
        self.assertIn("METHODS=\"${METHODS:-base,smooth_k3}\"", text)
        self.assertIn("IMPLEMENTATIONS=\"${IMPLEMENTATIONS:-old,cleaned}\"", text)
        self.assertIn("--epochs 20", text)
        self.assertIn("--steps_per_epoch 500", text)
        self.assertIn("--timematch_shift_score_epsilon 1e-5", text)
        self.assertNotIn("FR2_to_FR1", text)

    def test_sync_exports_source_and_da_commits_separately(self):
        text = (ROOT / "sync_v28_old_da_to_server.sh").read_text(encoding="utf-8")
        self.assertIn("timematch_old_da_f04e1e0", text)
        self.assertIn("timematch_old_source_89d9df4", text)
        self.assertIn("89d9df4e52744cb955168b0d203a2ddd61c3199e", text)


if __name__ == "__main__":
    unittest.main()
