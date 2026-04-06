from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = ROOT / "train_gpt.py"


def load_train_gpt_module():
    spec = importlib.util.spec_from_file_location("train_gpt_taec_test", TRAIN_GPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {TRAIN_GPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenAwareEmbeddingCalibrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_gpt = load_train_gpt_module()

    def test_piece_bucket_classifies_special_byte_boundary_and_continuation(self) -> None:
        module = self.train_gpt
        self.assertEqual(
            module.classify_token_piece("<s>", is_control=True, is_unknown=False, is_unused=False, is_byte=False),
            module.TOKEN_CLASS_SPECIAL,
        )
        self.assertEqual(
            module.classify_token_piece("<0x41>", is_control=False, is_unknown=False, is_unused=False, is_byte=True),
            module.TOKEN_CLASS_RAW_BYTE,
        )
        self.assertEqual(
            module.classify_token_piece("▁the", is_control=False, is_unknown=False, is_unused=False, is_byte=False),
            module.TOKEN_CLASS_BOUNDARY_PIECE,
        )
        self.assertEqual(
            module.classify_token_piece("ing", is_control=False, is_unknown=False, is_unused=False, is_byte=False),
            module.TOKEN_CLASS_CONTINUATION_PIECE,
        )

    def test_byte_bucket_counts_piece_bytes_after_boundary_marker_is_removed(self) -> None:
        module = self.train_gpt
        self.assertEqual(module.classify_token_piece_byte_bucket("<s>", is_control=True, is_byte=False), 0)
        self.assertEqual(module.classify_token_piece_byte_bucket("<0x41>", is_control=False, is_byte=True), 1)
        self.assertEqual(module.classify_token_piece_byte_bucket("▁a", is_control=False, is_byte=False), 1)
        self.assertEqual(module.classify_token_piece_byte_bucket("▁the", is_control=False, is_byte=False), 3)
        self.assertEqual(module.classify_token_piece_byte_bucket("hello", is_control=False, is_byte=False), 4)

    def test_token_aware_embedding_calibration_is_identity_with_zero_initialized_gains(self) -> None:
        module = self.train_gpt
        calibrator = module.TokenAwareEmbeddingCalibration(model_dim=3, max_scale=0.1)
        embeddings = torch.tensor([[[1.0, -2.0, 0.5], [0.25, 0.5, -1.0]]], dtype=torch.float32)
        token_class_ids = torch.tensor([[module.TOKEN_CLASS_BOUNDARY_PIECE, module.TOKEN_CLASS_RAW_BYTE]], dtype=torch.int64)
        token_byte_ids = torch.tensor([[3, 1]], dtype=torch.int64)

        calibrated = calibrator(embeddings, token_class_ids, token_byte_ids)
        self.assertTrue(torch.equal(calibrated, embeddings))

    def test_token_aware_embedding_calibration_scales_tokens_from_class_and_byte_tables(self) -> None:
        module = self.train_gpt
        calibrator = module.TokenAwareEmbeddingCalibration(model_dim=2, max_scale=0.1)
        with torch.no_grad():
            calibrator.class_gain[module.TOKEN_CLASS_BOUNDARY_PIECE].fill_(1.0)
            calibrator.byte_gain[3].fill_(0.5)

        embeddings = torch.tensor([[[2.0, -4.0], [2.0, -4.0]]], dtype=torch.float32)
        token_class_ids = torch.tensor([[module.TOKEN_CLASS_BOUNDARY_PIECE, module.TOKEN_CLASS_CONTINUATION_PIECE]], dtype=torch.int64)
        token_byte_ids = torch.tensor([[3, 3]], dtype=torch.int64)

        calibrated = calibrator(embeddings, token_class_ids, token_byte_ids)
        expected_scale_hit = 1.0 + 0.1 * torch.tanh(torch.tensor(1.5))
        expected_scale_miss = 1.0 + 0.1 * torch.tanh(torch.tensor(0.5))
        self.assertTrue(torch.allclose(calibrated[0, 0], embeddings[0, 0] * expected_scale_hit))
        self.assertTrue(torch.allclose(calibrated[0, 1], embeddings[0, 1] * expected_scale_miss))

    def test_curriculum_sequence_schedule_returns_short_before_switch_and_long_after(self) -> None:
        module = self.train_gpt
        args = SimpleNamespace(
            curriculum_seq_enabled=True,
            curriculum_seq_short=512,
            train_seq_len=1024,
            curriculum_seq_switch_seconds=180.0,
        )
        self.assertEqual(module.resolve_train_seq_len(args, elapsed_seconds=0.0), 512)
        self.assertEqual(module.resolve_train_seq_len(args, elapsed_seconds=179.9), 512)
        self.assertEqual(module.resolve_train_seq_len(args, elapsed_seconds=180.0), 1024)
        self.assertEqual(module.resolve_train_seq_len(args, elapsed_seconds=240.0), 1024)

    def test_curriculum_sequence_schedule_returns_train_seq_len_when_disabled(self) -> None:
        module = self.train_gpt
        args = SimpleNamespace(
            curriculum_seq_enabled=False,
            curriculum_seq_short=512,
            train_seq_len=1024,
            curriculum_seq_switch_seconds=180.0,
        )
        self.assertEqual(module.resolve_train_seq_len(args, elapsed_seconds=0.0), 1024)


if __name__ == "__main__":
    unittest.main()
