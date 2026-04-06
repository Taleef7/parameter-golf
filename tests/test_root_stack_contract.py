from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import audit_ablation_evidence as ablation_audit

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train_gpt.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_gpt_root_stack_test", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HyperparametersContractTests(unittest.TestCase):
    def test_hyperparameters_parse_root_stack_knobs_from_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "MLP_NEGATIVE_SLOPE": "0.5",
                "EMA_ENABLED": "1",
                "EMA_DECAY": "0.995",
                "EVAL_STRIDE": "64",
            },
            clear=False,
        ):
            module = load_train_module()
            args = module.Hyperparameters()

        self.assertAlmostEqual(args.mlp_negative_slope, 0.5)
        self.assertTrue(args.ema_enabled)
        self.assertAlmostEqual(args.ema_decay, 0.995)
        self.assertEqual(args.eval_stride, 64)


class MLPContractTests(unittest.TestCase):
    def test_zero_negative_slope_preserves_relu_squared_behavior(self) -> None:
        module = load_train_module()
        mlp = module.MLP(dim=1, mlp_mult=1, negative_slope=0.0)
        with torch.no_grad():
            mlp.fc.weight.fill_(1.0)
            mlp.proj.weight.fill_(1.0)

        x = torch.tensor([[[-2.0], [3.0]]], dtype=torch.float32)
        out = mlp(x)
        expected = torch.tensor([[[0.0], [9.0]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))

    def test_positive_negative_slope_uses_leaky_relu_squared_behavior(self) -> None:
        module = load_train_module()
        mlp = module.MLP(dim=1, mlp_mult=1, negative_slope=0.5)
        with torch.no_grad():
            mlp.fc.weight.fill_(1.0)
            mlp.proj.weight.fill_(1.0)

        x = torch.tensor([[[-2.0], [3.0]]], dtype=torch.float32)
        out = mlp(x)
        expected = torch.tensor([[[1.0], [9.0]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))


class EMAContractTests(unittest.TestCase):
    def test_ema_updates_and_applies_shadow_parameters(self) -> None:
        module = load_train_module()
        param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        model = torch.nn.Module()
        model.register_parameter("weight", param)

        ema = module.ExponentialMovingAverage(decay=0.5)
        ema.update(model)
        self.assertAlmostEqual(float(ema.shadow["weight"].item()), 1.0)

        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update(model)
        self.assertAlmostEqual(float(ema.shadow["weight"].item()), 2.0)

        with torch.no_grad():
            model.weight.fill_(10.0)
        ema.apply_to(model)
        self.assertAlmostEqual(float(model.weight.item()), 2.0)


class SlidingWindowContractTests(unittest.TestCase):
    def test_window_plan_scores_each_token_once(self) -> None:
        module = load_train_module()
        plan = module.build_sliding_window_plan(total_tokens=10, seq_len=4, stride=2)
        coverage = [0] * 10
        for window in plan:
            for token_idx in range(window.scored_start, window.end):
                coverage[token_idx] += 1

        self.assertEqual([window.start for window in plan], [0, 2, 4, 6])
        self.assertEqual(coverage, [1] * 10)

    def test_window_plan_rejects_non_positive_stride(self) -> None:
        module = load_train_module()
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            module.build_sliding_window_plan(total_tokens=10, seq_len=4, stride=0)


class MetricSurfaceContractTests(unittest.TestCase):
    def test_root_sliding_metric_is_accepted_by_ablation_metric_extractor(self) -> None:
        metric, value = ablation_audit.extract_metric_from_text(
            "\n".join(
                [
                    "final_int8_zlib_roundtrip_exact val_loss:1.50000000 val_bpb:1.30000000",
                    "final_int8_zlib_sliding_window_exact val_loss:1.40000000 val_bpb:1.25000000",
                ]
            )
        )
        self.assertEqual(metric, "final_int8_zlib_sliding_window")
        self.assertTrue(math.isclose(value, 1.25))


if __name__ == "__main__":
    unittest.main()
