from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import audit_submission_launchability as audit

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "records" / "track_10min_16mb" / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
README = RECORD_DIR / "README.md"
SUBMISSION = RECORD_DIR / "submission.json"
SCRIPT = RECORD_DIR / "train_gpt.py"
LOGS = [
    RECORD_DIR / "train_seed1337.log",
    RECORD_DIR / "train_seed42.log",
    RECORD_DIR / "train_seed2025.log",
]


class LaunchabilityDocsContractTests(unittest.TestCase):
    def test_canonical_readme_and_submission_surfaces_pass_audit(self) -> None:
        payload = audit.run_audit(
            readme=README,
            submission=SUBMISSION,
            script=SCRIPT,
            logs=LOGS,
        )

        self.assertIn("docs_contract", payload)
        self.assertTrue(payload["docs_contract"]["readme_contract_verified"])
        self.assertTrue(payload["docs_contract"]["submission_contract_verified"])
        self.assertIn("--readme", payload["docs_contract"]["canonical_audit_command"])
        self.assertIn("--submission", payload["docs_contract"]["canonical_audit_command"])

    def test_missing_world_size_snippet_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(
                README.read_text(encoding="utf-8").replace(
                    "- every audited seed log records `world_size:8`",
                    "- audited seed logs keep the expected distributed launch setting",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "README launchability contract missing snippet"):
                audit.run_audit(readme=readme, submission=SUBMISSION, script=SCRIPT, logs=LOGS)

    def test_missing_non_ttt_boundary_snippet_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(
                README.read_text(encoding="utf-8").replace(
                    "audited non-TTT fallback surface is `final_int6_sliding_window` with `stride:64` (future-compatible with `final_int6_sliding_window_s64`)",
                    "non-TTT fallback exists",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "README launchability contract missing snippet"):
                audit.run_audit(readme=readme, submission=SUBMISSION, script=SCRIPT, logs=LOGS)

    def test_submission_seed_count_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            metadata = json.loads(SUBMISSION.read_text(encoding="utf-8"))
            metadata["seed_count"] = 2
            submission.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "submission metadata drift for seed_count"):
                audit.run_audit(readme=README, submission=submission, script=SCRIPT, logs=LOGS)

    def test_submission_launchability_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            metadata = json.loads(SUBMISSION.read_text(encoding="utf-8"))
            metadata["launchability_contract"]["no_network_proof"] = False
            submission.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "submission launchability drift for no_network_proof"):
                audit.run_audit(readme=README, submission=submission, script=SCRIPT, logs=LOGS)


if __name__ == "__main__":
    unittest.main()
