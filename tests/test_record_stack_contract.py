from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STACK = ROOT / "experiments" / "train_gpt_stack.py"
BASELINE_SCRIPT = ROOT / "train_gpt.py"
RECORD_DIR = ROOT / "records" / "track_10min_16mb" / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
PROMOTED_SCRIPT = RECORD_DIR / "train_gpt.py"
README = RECORD_DIR / "README.md"

REQUIRED_README_SNIPPETS = [
    "legal_ttt",
    "final_int6_sliding_window_s64",
    "TTT_ENABLED=1",
    "TTT_ENABLED=0",
    "MAX_WALLCLOCK_SECONDS=600",
    "ITERATIONS=9000",
    "EVAL_STRIDE=64",
    "python records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py",
    "records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log",
    "Total submission size int6+lzma",
    "16,000,000",
    "repository-root `train_gpt.py` newcomer baseline",
]


class RecordStackContractTests(unittest.TestCase):
    def test_record_folder_contains_promoted_script_and_readme(self) -> None:
        self.assertTrue(RECORD_DIR.is_dir(), f"missing record dir: {RECORD_DIR}")
        self.assertTrue(PROMOTED_SCRIPT.is_file(), f"missing promoted script: {PROMOTED_SCRIPT}")
        self.assertTrue(README.is_file(), f"missing record README: {README}")

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

    def test_record_readme_documents_required_launcher_and_metric_contract(self) -> None:
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
        self.assertRegex(text, r"TTT_ENABLED=1[^\n]*`legal_ttt`|`legal_ttt` when `TTT_ENABLED=1`")
        self.assertRegex(
            text,
            r"TTT_ENABLED=0[^\n]*`final_int6_sliding_window_s64`|`final_int6_sliding_window_s64` when `TTT_ENABLED=0`",
        )


if __name__ == "__main__":
    unittest.main()
