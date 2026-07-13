import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V28CausalAuditStaticTest(unittest.TestCase):
    def test_shift_epsilon_is_explicit_and_defaults_to_cleaned_value(self):
        train_text = (ROOT / "train.py").read_text(encoding="utf-8")
        loop_text = (ROOT / "methods/timematch_base/train_loop.py").read_text(encoding="utf-8-sig")

        self.assertIn("--timematch_shift_score_epsilon", train_text)
        self.assertIn("default=1e-12", train_text)
        self.assertIn("shift_score_epsilon=1e-12", loop_text)
        self.assertIn("shift_score_epsilon=shift_score_epsilon", loop_text)

    def test_score_function_uses_parameter_for_probability_logs(self):
        path = ROOT / "methods/timematch_base/train_loop.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        score_fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "score_shift_softmaxes"
        )
        argument_names = [arg.arg for arg in score_fn.args.args]
        self.assertIn("shift_score_epsilon", argument_names)
        source = ast.get_source_segment(path.read_text(encoding="utf-8-sig"), score_fn)
        self.assertIn("+ shift_score_epsilon", source)

    def test_audit_tools_and_launchers_are_present(self):
        required = [
            "tools/compare_v28_smooth_loss.py",
            "tools/trace_timematch_da_step.py",
            "tools/evaluate_common_timematch_checkpoint.py",
            "tools/summarize_v28_da_causal_audit.py",
            "launchers/v28/launch_v28_shift_epsilon_audit_4gpu.sh",
            "launchers/v28/launch_v28_old_vs_cleaned_da_4gpu.sh",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_launchers_limit_scope_and_track_failures(self):
        epsilon = (ROOT / "launchers/v28/launch_v28_shift_epsilon_audit_4gpu.sh").read_text(
            encoding="utf-8"
        )
        comparison = (ROOT / "launchers/v28/launch_v28_old_vs_cleaned_da_4gpu.sh").read_text(
            encoding="utf-8"
        )
        for text in (epsilon, comparison):
            self.assertIn("AT1_to_FR2", text)
            self.assertIn("FR2_to_FR1", text)
            self.assertIn("base,smooth_k3", text)
            self.assertIn("job_status.tsv", text)
            self.assertIn("CUDA_VISIBLE_DEVICES", text)
            self.assertNotIn("FR1_to_DK1", text)
        self.assertIn("1e-5,1e-12", epsilon)
        self.assertIn("f04e1e06805270d4e98db688ae869fbdeb6493b6", comparison)

    def test_server_audit_does_not_require_cleaned_tree_git_metadata(self):
        compare = (ROOT / "tools/compare_v28_smooth_loss.py").read_text(encoding="utf-8")
        trace_launcher = (ROOT / "launchers/v28/launch_v28_da_trace_smoke.sh").read_text(
            encoding="utf-8"
        )
        comparison_launcher = (
            ROOT / "launchers/v28/launch_v28_old_vs_cleaned_da_4gpu.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--old_source_file", compare)
        self.assertIn("sync_v28_old_da_to_server.sh", trace_launcher)
        self.assertNotIn("worktree add", trace_launcher)
        self.assertNotIn("worktree add", comparison_launcher)


if __name__ == "__main__":
    unittest.main()
