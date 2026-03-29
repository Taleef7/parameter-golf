from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import audit_submission_package as audit


class SubmissionPackageContractTests(unittest.TestCase):
    def make_log(
        self,
        directory: Path,
        *,
        metric_line: str | None,
        size_bytes: int | None = 15_900_006,
        extra_lines: list[str] | None = None,
        name: str = "seed.log",
    ) -> Path:
        lines = ["header"]
        if extra_lines:
            lines.extend(extra_lines)
        if metric_line is not None:
            lines.append(metric_line)
        if size_bytes is not None:
            lines.append(f"Total submission size int6+lzma: {size_bytes} bytes")
        path = directory / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_audit_seed_log_uses_canonical_verify_run_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(Path(tmp), metric_line="legal_ttt val_bpb:1.1192")
            with mock.patch.object(audit, "extract_metric", return_value=("legal_ttt", "1.1192")) as patched:
                seed = audit.audit_seed_log(log_path)
        patched.assert_called_once_with(log_path)
        self.assertEqual(seed.chosen_metric, "legal_ttt")
        self.assertAlmostEqual(seed.val_bpb, 1.1192)
        self.assertEqual(seed.total_submission_bytes, 15_900_006)

    def test_extract_submission_size_bytes_fails_for_missing_log(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "log file not found"):
            audit.extract_submission_size_bytes(Path("does-not-exist.log"))

    def test_audit_seed_log_fails_for_empty_log_without_accepted_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(Path(tmp), metric_line=None, size_bytes=15_900_006)
            with self.assertRaisesRegex(ValueError, "no accepted metric found"):
                audit.audit_seed_log(log_path)

    def test_audit_seed_log_fails_when_only_fallback_metric_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(
                Path(tmp),
                metric_line="final_int6_sliding_window_s64 val_bpb:1.1226",
            )
            with self.assertRaisesRegex(audit.SubmissionAuditError, "expected chosen_metric legal_ttt"):
                audit.audit_seed_log(log_path)

    def test_extract_submission_size_bytes_fails_when_size_line_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(Path(tmp), metric_line="legal_ttt val_bpb:1.1192", size_bytes=None)
            with self.assertRaisesRegex(ValueError, "Total submission size int6\\+lzma"):
                audit.extract_submission_size_bytes(log_path)

    def test_summarize_seeds_reports_mean_std_and_conservative_max(self) -> None:
        seeds = [
            audit.SeedAudit("seed1337.log", "legal_ttt", 1.1192, 15_977_386),
            audit.SeedAudit("seed42.log", "legal_ttt", 1.1200, 15_876_510),
            audit.SeedAudit("seed2025.log", "legal_ttt", 1.1189, 15_990_006),
        ]
        summary = audit.summarize_seeds(seeds)
        self.assertEqual(summary.seed_count, 3)
        self.assertAlmostEqual(summary.mean_val_bpb, 1.1193666666666666)
        self.assertAlmostEqual(summary.std_val_bpb, 0.00046427960923947035)
        self.assertEqual(summary.max_total_submission_bytes, 15_990_006)
        self.assertTrue(summary.within_size_limit)

    def test_summarize_seeds_rejects_size_overflow(self) -> None:
        seeds = [
            audit.SeedAudit("seed1337.log", "legal_ttt", 1.1192, 15_999_999),
            audit.SeedAudit("seed42.log", "legal_ttt", 1.1200, 16_000_001),
            audit.SeedAudit("seed2025.log", "legal_ttt", 1.1189, 15_990_006),
        ]
        with self.assertRaisesRegex(audit.SubmissionAuditError, "submission size exceeds limit"):
            audit.summarize_seeds(seeds)

    def test_audit_script_provenance_accepts_byte_identical_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            proven = tmpdir / "proven.py"
            promoted = tmpdir / "promoted.py"
            body = "print('same')\n"
            proven.write_text(body, encoding="utf-8")
            promoted.write_text(body, encoding="utf-8")
            provenance = audit.audit_script_provenance(promoted, proven)
        self.assertTrue(provenance.scripts_match)
        self.assertEqual(provenance.promoted_sha256, provenance.proven_sha256)

    def test_audit_script_provenance_rejects_script_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            proven = tmpdir / "proven.py"
            promoted = tmpdir / "promoted.py"
            proven.write_text("print('proven')\n", encoding="utf-8")
            promoted.write_text("print('promoted')\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.SubmissionAuditError, "promoted script drifted"):
                audit.audit_script_provenance(promoted, proven)

    def test_run_audit_returns_machine_checkable_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            logs = [
                self.make_log(tmpdir, metric_line=f"legal_ttt val_bpb:{value}", size_bytes=size, name=name)
                for name, value, size in [
                    ("seed1337.log", "1.1192", 15_977_386),
                    ("seed42.log", "1.1200", 15_876_510),
                    ("seed2025.log", "1.1189", 15_990_006),
                ]
            ]
            proven = tmpdir / "proven.py"
            promoted = tmpdir / "promoted.py"
            body = "print('same')\n"
            proven.write_text(body, encoding="utf-8")
            promoted.write_text(body, encoding="utf-8")

            payload = audit.run_audit(logs, promoted_script=promoted, proven_script=proven)

        encoded = json.dumps(payload, sort_keys=True)
        self.assertIn('"accepted_metric_fallbacks"', encoded)
        self.assertIn('"seed_audits"', encoded)
        self.assertIn('"aggregate"', encoded)
        self.assertIn('"provenance"', encoded)
        self.assertEqual(payload["aggregate"]["max_total_submission_bytes"], 15_990_006)
        self.assertTrue(payload["provenance"]["scripts_match"])


if __name__ == "__main__":
    unittest.main()
