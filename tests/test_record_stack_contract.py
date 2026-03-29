from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from experiments import audit_submission_launchability as launchability_audit
from experiments import audit_submission_package as package_audit

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
    "inherited 3-seed reproducibility",
    "`world_size:8`",
    "accepted post-step wallclock overshoot is `128 ms` above the `600000 ms` cap",
    "audited max `legal_ttt` eval time is `410268 ms` under the `600000 ms` cap",
    "audited non-TTT fallback surface is `final_int6_sliding_window` with `stride:64` (future-compatible with `final_int6_sliding_window_s64`)",
    "audited max non-TTT eval time is `97749 ms` under the `600000 ms` cap",
    "`DATA_PATH` default `./data/datasets/fineweb10B_sp1024`",
    "`TOKENIZER_PATH` default `./data/tokenizers/fineweb_1024_bpe.model`",
    "Hugging Face download logic lives in `data/cached_challenge_fineweb.py`, not in the promoted submission script",
    "no-network proof is script-specific",
    "launchability_contract",
    "submission.json",
    "train.log` - reviewer-friendly alias for `train_seed2025.log`",
    "train_seed1337.log",
    "train_seed42.log",
    "train_seed2025.log",
    "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon",
    "byte-identical",
    "15,990,006 bytes",
    "16,000,000",
    "Stack Integration + Legal TTT + Parallel Muon",
    "2026-03-28_RandomMapAdapters_Stack/` is a separate non-record package",
    "not part of the submission evidence",
    "python experiments/audit_submission_package.py",
    "python experiments/audit_submission_launchability.py --readme records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/README.md --submission records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/submission.json --script records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py --logs records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed1337.log records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed42.log records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed2025.log",
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
            "DATA_PATH",
            "TOKENIZER_PATH",
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
        self.assertIn("accepted post-step wallclock overshoot is `128 ms` above the `600000 ms` cap", text)
        self.assertIn("audited max non-TTT eval time is `97749 ms` under the `600000 ms` cap", text)

    def test_train_log_is_byte_identical_alias_of_seed2025(self) -> None:
        self.assertEqual(
            TRAIN_LOG.read_bytes(),
            (RECORD_DIR / "train_seed2025.log").read_bytes(),
            "train.log should remain the canonical alias of train_seed2025.log",
        )

    def test_submission_json_matches_audited_seed_and_launchability_evidence(self) -> None:
        metadata = json.loads(SUBMISSION_JSON.read_text(encoding="utf-8"))
        package_payload = package_audit.run_audit(SEED_LOGS)
        launchability_payload = launchability_audit.run_audit(
            readme=README,
            submission=SUBMISSION_JSON,
            script=PROMOTED_SCRIPT,
            logs=SEED_LOGS,
        )
        aggregate = package_payload["aggregate"]
        launchability = launchability_payload["launchability_aggregate"]
        script_launchability = launchability_payload["script_launchability"]

        self.assertEqual(metadata["name"], "Stack Integration + Legal TTT + Parallel Muon")
        self.assertEqual(metadata["seed_count"], 3)
        self.assertEqual(metadata["seeds"], [1337, 42, 2025])
        self.assertEqual(metadata["train_log_alias"], "train.log -> train_seed2025.log")
        self.assertAlmostEqual(metadata["val_bpb"], aggregate["mean_val_bpb"])
        self.assertAlmostEqual(metadata["val_bpb_std"], aggregate["std_val_bpb"])
        self.assertEqual(metadata["bytes_total"], aggregate["max_total_submission_bytes"])
        self.assertGreater(metadata["bytes_code"], 0)
        self.assertIn("byte-identical", metadata["blurb"])
        self.assertIn("world_size:8", metadata["blurb"])
        self.assertIn("data/cached_challenge_fineweb.py", metadata["blurb"])
        self.assertIn("non-record work", metadata["blurb"])

        self.assertEqual(metadata["launchability_contract"]["world_size"], launchability["expected_world_size"])
        self.assertEqual(metadata["launchability_contract"]["train_time_cap_ms"], launchability["train_time_cap_ms"])
        self.assertEqual(
            metadata["launchability_contract"]["max_train_time_overshoot_ms"],
            launchability["max_train_time_overshoot_ms"],
        )
        self.assertEqual(
            metadata["launchability_contract"]["max_legal_ttt_eval_time_ms"],
            launchability["max_legal_ttt_eval_time_ms"],
        )
        self.assertEqual(
            metadata["launchability_contract"]["max_non_ttt_eval_time_ms"],
            launchability["max_sliding_eval_time_ms"],
        )
        self.assertEqual(metadata["launchability_contract"]["data_path_default"], script_launchability["data_path_default"])
        self.assertEqual(metadata["launchability_contract"]["tokenizer_path_default"], script_launchability["tokenizer_path_default"])
        self.assertTrue(metadata["launchability_contract"]["no_network_proof"])
        self.assertEqual(
            metadata["launchability_contract"]["promoted_script_sha256"],
            package_payload["provenance"]["promoted_sha256"],
        )
        self.assertEqual(
            metadata["launchability_contract"]["provenance_status"],
            "byte-identical promoted/proven train_gpt.py",
        )


if __name__ == "__main__":
    unittest.main()
