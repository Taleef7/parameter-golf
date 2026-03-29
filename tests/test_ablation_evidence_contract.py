from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import audit_ablation_evidence as audit


class AblationEvidenceContractTests(unittest.TestCase):
    def make_log(
        self,
        directory: Path,
        *,
        metric_lines: list[str] | None = None,
        size_lines: list[str] | None = None,
        extra_lines: list[str] | None = None,
        name: str = "train.log",
    ) -> Path:
        lines: list[str] = []
        if extra_lines:
            lines.extend(extra_lines)
        if metric_lines:
            lines.extend(metric_lines)
        if size_lines:
            lines.extend(size_lines)
        path = directory / name
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def make_candidate(
        self,
        record_dir: Path,
        *,
        family_id: str = "family-a",
        log_name: str = "train.log",
        counts_toward_target: bool = True,
        inherited_from: str | None = None,
        note: str | None = None,
    ) -> audit.CandidateRecord:
        return audit.CandidateRecord(
            record_dir=record_dir.as_posix(),
            log_name=log_name,
            family_id=family_id,
            counts_toward_target=counts_toward_target,
            inherited_from=inherited_from,
            note=note,
        )

    def test_extract_metric_prefers_newer_historical_metric_eras_in_precedence_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(
                Path(tmp),
                metric_lines=[
                    "final_int8_zlib_roundtrip val_bpb:1.3000",
                    "final_int6_roundtrip val_bpb:1.2000",
                    "final_int6_sliding_window val_bpb:1.1500",
                ],
                size_lines=["Total submission size int6+zstd: 15500000 bytes"],
            )
            metric, value = audit.extract_metric(log_path)
        self.assertEqual(metric, "final_int6_sliding_window")
        self.assertAlmostEqual(value, 1.15)

    def test_extract_metric_rejects_stale_exact_alias_without_supported_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(
                Path(tmp),
                metric_lines=["final_int8_zlib_roundtrip_exact val_bpb:1.1227"],
                size_lines=["Total submission size int8+zlib: 15800000 bytes"],
            )
            with self.assertRaisesRegex(ValueError, "no accepted metric found"):
                audit.extract_metric(log_path)

    def test_extract_metric_fails_for_missing_log(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "log file not found"):
            audit.extract_metric(Path("does-not-exist.log"))

    def test_extract_metric_fails_for_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(Path(tmp))
            with self.assertRaisesRegex(ValueError, "log is empty"):
                audit.extract_metric(log_path)

    def test_extract_submission_size_uses_metric_compatible_size_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(
                Path(tmp),
                metric_lines=["final_int6_sliding_window val_bpb:1.1248"],
                size_lines=[
                    "Total submission size int8+zlib: 15555017 bytes",
                    "Total submission size int6+zstd: 15512308 bytes",
                ],
            )
            label, value = audit.extract_submission_size(log_path, "final_int6_sliding_window")
        self.assertEqual(label, "int6+zstd")
        self.assertEqual(value, 15_512_308)

    def test_extract_submission_size_fails_when_size_line_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = self.make_log(
                Path(tmp),
                metric_lines=["final_int8_zlib_roundtrip val_bpb:1.2244"],
            )
            with self.assertRaisesRegex(ValueError, "missing compatible size line"):
                audit.extract_submission_size(log_path, "final_int8_zlib_roundtrip")

    def test_audit_candidate_rejects_placeholder_non_record_folder(self) -> None:
        candidate = audit.CandidateRecord(
            record_dir="records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack",
            log_name="train.log",
            family_id="placeholder",
        )
        with self.assertRaisesRegex(audit.AblationEvidenceAuditError, "placeholder/non-record folder"):
            audit.audit_candidate(candidate)

    def test_audit_candidate_rejects_size_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record_dir = Path(tmp) / "records" / "track_10min_16mb" / "2026-03-20_Overflow"
            record_dir.mkdir(parents=True)
            self.make_log(
                record_dir,
                metric_lines=["final_int8_zlib_roundtrip val_bpb:1.2244"],
                size_lines=["Total submission size int8+zlib: 16000001 bytes"],
            )
            candidate = self.make_candidate(record_dir)
            with self.assertRaisesRegex(audit.AblationEvidenceAuditError, "exceeds size limit"):
                audit.audit_candidate(candidate)

    def test_summarize_rows_rejects_duplicate_counted_families(self) -> None:
        rows = [
            audit.AuditRow("shared", "a", "a/train.log", "legal_ttt", 1.11, 15_900_000, "int6+lzma", True, None, None),
            audit.AuditRow("shared", "b", "b/train.log", "legal_ttt", 1.12, 15_900_000, "int6+lzma", True, None, None),
        ]
        with self.assertRaisesRegex(audit.DuplicateEvidenceFamilyError, "family_id=shared"):
            audit.summarize_rows(rows)

    def test_run_audit_treats_promoted_folder_as_inherited_not_second_counted_family(self) -> None:
        payload = audit.run_audit()
        rows = payload["rows"]
        counted = [row for row in rows if row["counts_toward_target"]]
        inherited = [row for row in rows if not row["counts_toward_target"]]
        promoted = next(row for row in inherited if row["record_dir"].endswith("2026-03-28_StackIntegration_LegalTTT_ParallelMuon"))
        legal_counted = [row for row in counted if row["family_id"] == "legal_ttt_parallel_muon"]

        self.assertEqual(len(legal_counted), 1)
        self.assertTrue(promoted["inherited_from"].endswith("2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed2025.log"))
        self.assertGreaterEqual(payload["aggregate"]["counted_row_count"], 5)
        self.assertTrue(payload["aggregate"]["minimum_real_rows_satisfied"])

    def test_run_audit_returns_machine_checkable_json_payload(self) -> None:
        payload = audit.run_audit()
        encoded = json.dumps(payload, sort_keys=True)
        self.assertIn('"accepted_metric_fallbacks"', encoded)
        self.assertIn('"rows"', encoded)
        self.assertIn('"aggregate"', encoded)

    def test_mixed_metric_eras_across_candidates_are_audited_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "records" / "track_10min_16mb"
            first = base / "2026-03-18_OldEra"
            second = base / "2026-03-23_NewEra"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            self.make_log(
                first,
                metric_lines=["final_int8_zlib_roundtrip val_bpb:1.2100"],
                size_lines=["Total submission size int8+zlib: 15800000 bytes"],
            )
            self.make_log(
                second,
                metric_lines=[
                    "final_int6_roundtrip val_bpb:1.1450",
                    "legal_ttt val_bpb:1.1190",
                ],
                size_lines=["Total submission size int6+lzma: 15990000 bytes"],
            )
            payload = audit.run_audit(
                candidates=[
                    self.make_candidate(first, family_id="old-era"),
                    self.make_candidate(second, family_id="new-era"),
                ],
                minimum_real_rows=2,
            )
        chosen = {row["family_id"]: row["chosen_metric"] for row in payload["rows"]}
        self.assertEqual(chosen["old-era"], "final_int8_zlib_roundtrip")
        self.assertEqual(chosen["new-era"], "legal_ttt")
        self.assertEqual(payload["aggregate"]["counted_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
