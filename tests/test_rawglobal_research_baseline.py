import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
class ActiveBaselineStaticTest(unittest.TestCase):
    def test_cli_exposes_only_off_and_raw_global(self):
        train_text = (ROOT / "train.py").read_text(encoding="utf-8")
        config_text = (ROOT / "methods/source_structure/configs.py").read_text(encoding="utf-8")
        self.assertIn("add_source_structure_args(parser)", train_text)
        self.assertIn("--source_structure_mode", config_text)
        self.assertIn("--source_structure_weight", config_text)
        self.assertNotIn("--source_structure_loss_version", train_text)
        self.assertNotIn("--source_structure_time_smooth_kernel_size", train_text)
        self.assertNotIn("--source_structure_elastic_radius", train_text)

        from methods.source_structure.configs import (
            add_source_structure_args,
            validate_source_structure_config,
        )

        parser = add_source_structure_args(argparse.ArgumentParser())
        defaults = parser.parse_args([])
        self.assertEqual(defaults.source_structure_mode, "off")
        self.assertEqual(defaults.source_structure_weight, 0.0)
        self.assertEqual(
            parser.parse_args(["--source_structure_mode", "raw_global"]).source_structure_mode,
            "raw_global",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source_structure_mode", "smooth_k3"])

        validate_source_structure_config(
            SimpleNamespace(
                method="source_structure",
                source_structure_mode="off",
                source_structure_weight=0.0,
            )
        )
        validate_source_structure_config(
            SimpleNamespace(
                method="source_structure",
                source_structure_mode="raw_global",
                source_structure_weight=1.0,
            )
        )
        invalid = (
            SimpleNamespace(
                method=None,
                source_structure_mode="raw_global",
                source_structure_weight=1.0,
            ),
            SimpleNamespace(
                method="source_structure",
                source_structure_mode="raw_global",
                source_structure_weight=0.0,
            ),
            SimpleNamespace(
                method="source_structure",
                source_structure_mode="off",
                source_structure_weight=1.0,
            ),
        )
        for config in invalid:
            with self.assertRaises(ValueError):
                validate_source_structure_config(config)

        plain_launcher = (ROOT / "launchers/rawglobal/run_plain_source.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--source_structure_mode off", plain_launcher)
        self.assertTrue(plain_launcher.rstrip().endswith("source_structure"))

        dispatch = train_text[
            train_text.index("elif config.method == 'source_structure':") :
            train_text.index("else:\n                train_supervised", train_text.index("elif config.method == 'source_structure':"))
        ]
        self.assertIn("if config.source_structure_mode == 'off':", dispatch)
        self.assertIn("train_supervised(", dispatch)
        self.assertNotIn("--source_checkpoint_epochs", train_text)

    def test_da_loop_does_not_import_or_compute_source_structure(self):
        da_text = (ROOT / "methods/timematch_base/train_loop.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertNotIn("source_structure", da_text)
        self.assertNotIn("raw_global", da_text)

        train_text = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("train_timematch_local_shift", train_text)
        self.assertNotIn("subparsers.add_parser('timematch_local_shift'", train_text)
        self.assertNotIn("sourcephasecompact", train_text)

    def test_active_launchers_contain_no_historical_modes(self):
        launcher_root = ROOT / "launchers/rawglobal"
        expected = {
            "run_plain_source.sh",
            "run_rawglobal_source.sh",
            "run_da_from_source_checkpoint.sh",
        }
        self.assertEqual({path.name for path in launcher_root.glob("*.sh")}, expected)
        forbidden = (
            "smooth",
            "timepoint",
            "elastic",
            "umsc",
            "routing",
            "selector",
            "affine",
            "stretch",
            "v322",
        )
        text = "\n".join(path.read_text(encoding="utf-8").lower() for path in launcher_root.glob("*.sh"))
        for token in forbidden:
            self.assertNotIn(token, text)

        da_launcher = (launcher_root / "run_da_from_source_checkpoint.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("SOURCE_EXPERIMENT_DIR", da_launcher)
        self.assertIn("${SOURCE_EXPERIMENT_DIR}/fold_0/model.pt", da_launcher)


class RawGlobalTensorTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        from methods.source_structure.raw_global import compute_raw_global_compactness

        self.compute = compute_raw_global_compactness

    def test_raw_global_api_does_not_accept_positions(self):
        self.assertEqual(
            tuple(inspect.signature(self.compute).parameters),
            ("temporal_features", "labels"),
        )

    def test_matches_manual_equal_class_mean(self):
        torch = self.torch
        h = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[3.0, 4.0], [5.0, 6.0]],
                [[2.0, 0.0], [4.0, 2.0]],
                [[6.0, 2.0], [8.0, 4.0]],
                [[10.0, 4.0], [12.0, 6.0]],
                [[100.0, 100.0], [100.0, 100.0]],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 1, 2])

        result = self.compute(h, labels)
        pooled = h.mean(dim=1)
        expected_losses = []
        for class_id in (0, 1):
            class_feats = pooled[labels == class_id]
            center = class_feats.mean(dim=0)
            expected_losses.append(((class_feats - center) ** 2).sum(dim=-1).mean())
        expected = torch.stack(expected_losses).mean()

        torch.testing.assert_close(result.loss, expected)
        self.assertEqual(result.valid_class_count, 2)
        self.assertEqual(result.samples_per_class, {0: 2, 1: 3, 2: 1})
        self.assertEqual(result.skipped_class_ids, (2,))

    def test_no_valid_class_returns_differentiable_zero(self):
        torch = self.torch
        h = torch.randn(3, 4, 5, requires_grad=True)
        result = self.compute(h, torch.tensor([0, 1, 2]))
        self.assertEqual(result.loss.item(), 0.0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertEqual(torch.count_nonzero(h.grad).item(), 0)

    def test_raw_loss_updates_spatial_path_not_temporal_path(self):
        torch = self.torch
        spatial = torch.nn.Linear(3, 4, bias=False)
        temporal = torch.nn.Linear(4, 2, bias=False)
        inputs = torch.randn(6, 5, 3)
        h = spatial(inputs)
        _ = temporal(h.mean(dim=1))

        result = self.compute(h, torch.tensor([0, 0, 0, 1, 1, 1]))
        result.loss.backward()

        self.assertGreater(spatial.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(temporal.weight.grad)

    def test_raw_loss_updates_real_pse_but_not_ltae_or_classifier(self):
        torch = self.torch
        torch.manual_seed(11)
        torch.set_num_threads(1)
        from models.stclassifier import PseLTae

        model = PseLTae(input_dim=10, num_classes=3, with_extra=False)
        pixels = torch.randn(4, 3, 10, 6)
        mask = torch.ones(4, 3, 6)
        extra = torch.zeros(4, 4)
        h = model.spatial_encoder(pixels, mask, extra)
        raw_loss = self.compute(h, torch.tensor([0, 0, 1, 1])).loss
        raw_loss.backward()

        pse_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.spatial_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(pse_grad, 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.temporal_encoder.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))

    def test_float32_and_float64_are_preserved(self):
        torch = self.torch
        labels = torch.tensor([0, 0, 1, 1])
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                h = torch.randn(4, 3, 2, dtype=dtype, requires_grad=True)
                result = self.compute(h, labels)
                self.assertEqual(result.loss.dtype, dtype)
                self.assertEqual(result.loss.device, h.device)

    def test_plain_and_zero_weight_paths_keep_logits_and_classification_loss(self):
        torch = self.torch
        torch.manual_seed(7)
        torch.set_num_threads(1)
        from models.stclassifier import PseLTae

        model = PseLTae(input_dim=10, num_classes=3, with_extra=False).eval()
        pixels = torch.randn(2, 5, 10, 8)
        mask = torch.ones(2, 5, 8)
        positions = torch.tensor(
            [[5, 21, 47, 83, 120], [5, 21, 47, 83, 120]], dtype=torch.long
        )
        positions_before = positions.clone()
        extra = torch.zeros(2, 4)
        labels = torch.tensor([0, 1])

        with torch.no_grad():
            plain_logits = model(pixels, mask, positions, extra)
            temporal_logits, h = model(
                pixels,
                mask,
                positions,
                extra,
                return_temporal_features=True,
            )
            replay_logits = model.forward_from_temporal_features(h, positions)

        torch.testing.assert_close(plain_logits, temporal_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(plain_logits, replay_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(positions, positions_before, rtol=0.0, atol=0.0)
        classification_loss = torch.nn.functional.cross_entropy(plain_logits, labels)
        raw_loss = self.compute(h, labels).loss
        zero_weight_total = classification_loss + 0.0 * raw_loss
        torch.testing.assert_close(
            classification_loss,
            zero_weight_total,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
