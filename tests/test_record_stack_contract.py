from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from experiments import audit_submission_package as audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STACK = ROOT / "experiments" / "train_gpt_stack.py"
BASELINE_SCRIPT = ROOT / "train_gpt.py"
RECORD_DIR = ROOT / "records" / "track_10min_16mb" / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
PROMOTED_SCRIPT = RECORD_DIR / "train_gpt.py"
README = RECORD_DIR / "README.md"
SUBMISSION_JSON = RECORD_DIR / "submission.json"
TRAIN_LOG = RECORD_DIR / "train.log"
SEED_LOGS = [
    RECORD_DIR / "train_seed1337.log",
    RECORD_DIR / "train_seed42.log",
    RECORD_DIR / "train_seed2025.log",
]

REQUIRED_RECORD_FILES = [
    PROMOTED_SCRIPT,
    README,
    SUBMISSION_JSON,
    TRAIN_LOG,
    *SEED_LOGS,
]

REQUIRED_README_SNIPPETS = [
    "legal_ttt",
    "final_int6_sliding_window_s64",
    "TTT_ENABLED=1",
    "TTT_ENABLED=0",
    "MAX_WALLCLOCK_SECONDS=600",
    "ITERATIONS=9000",
    "EVAL_STRIDE=64",
    "train.log` - reviewer-friendly alias for `train_seed2025.log`",
    "train_seed1337.log",
    "train_seed42.log",
    "train_seed2025.log",
    "submission.json",
    "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon",
    "byte-identical",
    "16,000,000",
    "Stack Integration + Legal TTT + Parallel Muon",
    "2026-03-28_RandomMapAdapters_Stack/` is a separate non-record package",
    "not part of the submission evidence",
    "python experiments/audit_submission_package.py",
]


class RecordStackContractTests(unittest.TestCase):
    def test_record_folder_contains_complete_submission_package(self) -> None:
        self.assertTrue(RECORD_DIR.is_dir(), f"missing record dir: {RECORD_DIR}")
        for path in REQUIRED_RECORD_FILES:
            self.assertTrue(path.is_file(), f"missing required record artifact: {path}")

    def test_promoted_script_matches_integrated_source_script(self) -> None:
        self.assertEqual(
            PROMOTED_SCRIPT.read_text(encoding="utf-8"),
            SOURCE_STACK.read_text(encoding="utf-8"),
            "promoted record script drifted from experiments/train_gpt_stack.py",
        )

    def test_repository_root_baseline_remains_distinct_newcomer_entrypoint(self) -> None:
        baseline_text = BASELINE_SCRIPT.read_text(encoding="utf-8")
        promoted_text = PROMOTED_SCRIPT.read_text(encoding="utf-8")
        self.assertNotEqual(
            baseline_text,
            promoted_text,
            "repository-root train_gpt.py was overwritten by the integrated stack",
        )
        self.assertIn("good launching-off points for new participants", baseline_text)
        self.assertNotIn("TTT_ENABLED", baseline_text)
        self.assertNotIn("flash_attn_interface", baseline_text)
        self.assertIn("TTT_ENABLED", promoted_text)
        self.assertIn("flash_attn_interface", promoted_text)

    def test_record_readme_documents_required_submission_contract(self) -> None:
        text = README.read_text(encoding="utf-8")
        for snippet in REQUIRED_README_SNIPPETS:
            self.assertIn(snippet, text)

    def test_record_readme_only_claims_knobs_supported_by_promoted_script(self) -> None:
        supported = set(re.findall(r'os\.environ\.get\("([A-Z0-9_]+)"', PROMOTED_SCRIPT.read_text(encoding="utf-8")))
        text = README.read_text(encoding="utf-8")
        for env_var in [
            "MAX_WALLCLOCK_SECONDS",
            "ITERATIONS",
            "EVAL_STRIDE",
            "TTT_ENABLED",
        ]:
            self.assertIn(env_var, supported, f"README documents unsupported env var: {env_var}")
            self.assertIn(env_var, text)

    def test_record_readme_describes_both_ttt_boundary_conditions_unambiguously(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertRegex(text, r"TTT_ENABLED=1[\s\S]*`legal_ttt`|`legal_ttt`[\s\S]*TTT_ENABLED=1")
        self.assertRegex(
            text,
            r"TTT_ENABLED=0[\s\S]*`final_int6_sliding_window_s64`|`final_int6_sliding_window_s64`[\s\S]*TTT_ENABLED=0",
        )

    def test_train_log_is_byte_identical_alias_of_seed2025(self) -> None:
        self.assertEqual(
            TRAIN_LOG.read_bytes(),
            (RECORD_DIR / "train_seed2025.log").read_bytes(),
            "train.log should remain the canonical alias of train_seed2025.log",
        )

    def test_submission_json_matches_audited_seed_evidence(self) -> None:
        metadata = json.loads(SUBMISSION_JSON.read_text(encoding="utf-8"))
        payload = audit.run_audit(SEED_LOGS)
        aggregate = payload["aggregate"]

        self.assertEqual(metadata["name"], "Stack Integration + Legal TTT + Parallel Muon")
        self.assertEqual(metadata["seed_count"], 3)
        self.assertEqual(metadata["seeds"], [1337, 42, 2025])
        self.assertEqual(metadata["train_log_alias"], "train.log -> train_seed2025.log")
        self.assertAlmostEqual(metadata["val_bpb"], aggregate["mean_val_bpb"])
        self.assertAlmostEqual(metadata["val_bpb_std"], aggregate["std_val_bpb"])
        self.assertEqual(metadata["bytes_total"], aggregate["max_total_submission_bytes"])
        self.assertGreater(metadata["bytes_code"], 0)
        self.assertIn("byte-identical", metadata["blurb"])
        self.assertIn("non-record work", metadata["blurb"])


if __name__ == "__main__":
    unittest.main()
