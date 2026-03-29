from __future__ import annotations

import json
import unittest
from pathlib import Path


class S06EvidenceDocsContractTests(unittest.TestCase):
    maxDiff = None

    @staticmethod
    def _load_audit_payload() -> dict:
        return json.loads(Path("experiments/ablation_evidence.json").read_text(encoding="utf-8"))

    @staticmethod
    def _parse_markdown_table(markdown: str, heading: str) -> list[dict[str, str]]:
        marker = f"## {heading}"
        try:
            after_heading = markdown.split(marker, 1)[1]
        except IndexError as exc:
            raise AssertionError(f"missing heading: {marker}") from exc

        lines = after_heading.splitlines()
        table_lines: list[str] = []
        started = False
        for line in lines:
            if line.startswith("## ") and started:
                break
            if line.startswith("|"):
                table_lines.append(line)
                started = True
            elif started and not line.strip():
                break

        if len(table_lines) < 2:
            raise AssertionError(f"no markdown table found under heading: {marker}")

        headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for raw in table_lines[2:]:
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            if len(cells) != len(headers):
                raise AssertionError(f"table row has {len(cells)} cells but expected {len(headers)}: {raw}")
            rows.append(dict(zip(headers, cells, strict=True)))
        return rows

    def test_counted_ledger_rows_match_audited_json(self) -> None:
        payload = self._load_audit_payload()
        counted_json = [row for row in payload["rows"] if row.get("counts_toward_target")]
        counted_by_family = {row["family_id"]: row for row in counted_json}

        markdown = Path("EXPERIMENTS.md").read_text(encoding="utf-8")
        counted_rows = self._parse_markdown_table(markdown, "Counted audited ablations")

        self.assertGreaterEqual(len(counted_rows), 5)
        self.assertEqual(len(counted_rows), len(counted_json))
        self.assertEqual(
            {row["Family ID"] for row in counted_rows},
            set(counted_by_family),
        )

        for row in counted_rows:
            audited = counted_by_family[row["Family ID"]]
            self.assertEqual(row["Record Folder"], f"`{audited['record_dir']}`")
            self.assertEqual(row["Log"], f"`{Path(audited['log_path']).name}`")
            self.assertEqual(row["Metric"], f"`{audited['chosen_metric']}`")
            self.assertEqual(float(row["val_bpb"]), audited["val_bpb"])
            self.assertEqual(row["Size Label"], f"`{audited['size_label']}`")
            self.assertEqual(int(row["Total Bytes"]), audited["total_submission_bytes"])
            self.assertEqual(row["Counts Toward 5+ Target"], "Yes")
            self.assertLessEqual(int(row["Total Bytes"]), payload["aggregate"]["size_limit_bytes"])

        self.assertNotIn(
            "`records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon`",
            {row["Record Folder"] for row in counted_rows},
        )

    def test_excluded_rows_call_out_inherited_and_placeholder_artifacts(self) -> None:
        markdown = Path("EXPERIMENTS.md").read_text(encoding="utf-8")
        excluded_rows = self._parse_markdown_table(markdown, "Explicitly excluded / non-counting artifacts")
        by_artifact = {row["Artifact"]: row for row in excluded_rows}

        promoted = by_artifact["Promoted legal-TTT folder"]
        self.assertEqual(promoted["Counts Toward 5+ Target"], "No")
        self.assertIn("already-counted `legal_ttt_parallel_muon` family", promoted["Why excluded"])
        self.assertIn("2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log", promoted["Source"])

        placeholder = by_artifact["Random-map adapter comparison"]
        self.assertEqual(placeholder["Counts Toward 5+ Target"], "No")
        self.assertIn("fixture-backed", placeholder["Why excluded"])
        self.assertIn("not for satisfying the real historical ablation threshold", placeholder["Why excluded"])
        self.assertIn("track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/README.md", placeholder["Source"])

    def test_grant_application_records_submitted_status_without_dropping_rationale(self) -> None:
        grant = Path("GRANT_APPLICATION.md").read_text(encoding="utf-8")
        self.assertIn("## Submission Status", grant)
        self.assertRegex(grant, r"(?i)\bstatus:\*\* submitted\b|\bsubmitted\b")
        self.assertIn("## Project Description", grant)
        self.assertIn("## Technical Approach", grant)
        self.assertIn("## Why These Choices", grant)
        self.assertIn("## Expected Outcome", grant)


if __name__ == "__main__":
    unittest.main()
