from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path


SMOKE_LOG_PATH = Path("logs/smoke_complete.txt")
AUDIT_JSON_PATH = Path("experiments/ablation_evidence.json")
EXPERIMENTS_PATH = Path("EXPERIMENTS.md")
GRANT_PATH = Path("GRANT_APPLICATION.md")

SIZE_LINE_RE = re.compile(r"Total submission size int8\+zlib:\s*(\d+)\s+bytes")
FINAL_METRIC_RE = re.compile(r"final_int8_zlib_roundtrip val_loss:([0-9]+(?:\.[0-9]+)?) val_bpb:([0-9]+(?:\.[0-9]+)?)")
SOURCE_TEMPLATE_RE = re.compile(r"final_int8_zlib_roundtrip val_loss:\{q_val_loss:\.4f\} val_bpb:\{q_val_bpb:\.4f\}")


class S06ClosureContractTests(unittest.TestCase):
    maxDiff = None

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @classmethod
    def _load_smoke_text(cls) -> str:
        if not SMOKE_LOG_PATH.exists():
            raise AssertionError(f"missing smoke artifact: {SMOKE_LOG_PATH}")
        return cls._read_text(SMOKE_LOG_PATH)

    @classmethod
    def _assert_completed_smoke_text(cls, text: str) -> tuple[int, float]:
        if SOURCE_TEMPLATE_RE.search(text):
            raise AssertionError("smoke log still contains the source-template final metric instead of a numeric runtime line")

        size_match = SIZE_LINE_RE.search(text)
        if not size_match:
            raise AssertionError("missing completed int8+zlib size line")

        metric_match = FINAL_METRIC_RE.search(text)
        if not metric_match:
            raise AssertionError("missing numeric final_int8_zlib_roundtrip val_bpb line")

        size_bytes = int(size_match.group(1))
        val_bpb = float(metric_match.group(2))

        if size_bytes <= 0:
            raise AssertionError(f"invalid int8+zlib size: {size_bytes}")
        if val_bpb <= 0.0:
            raise AssertionError(f"invalid final val_bpb: {val_bpb}")

        return size_bytes, val_bpb

    @staticmethod
    def _counted_rows(payload: dict) -> list[dict]:
        return [row for row in payload["rows"] if row.get("counts_toward_target")]

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

    def test_smoke_artifact_contains_completed_numeric_runtime_lines(self) -> None:
        text = self._load_smoke_text()
        size_bytes, val_bpb = self._assert_completed_smoke_text(text)

        self.assertGreater(size_bytes, 0)
        self.assertGreater(val_bpb, 0.0)
        self.assertNotIn("stopping_early: wallclock_cap", text)

    def test_counted_ledger_still_has_five_plus_real_rows(self) -> None:
        payload = json.loads(AUDIT_JSON_PATH.read_text(encoding="utf-8"))
        counted_rows = self._counted_rows(payload)
        self.assertGreaterEqual(len(counted_rows), 5)
        self.assertTrue(payload["aggregate"]["minimum_real_rows_satisfied"])
        self.assertEqual(payload["aggregate"]["counted_row_count"], len(counted_rows))

        experiments_md = EXPERIMENTS_PATH.read_text(encoding="utf-8")
        counted_table = self._parse_markdown_table(experiments_md, "Counted audited ablations")
        self.assertEqual(len(counted_table), len(counted_rows))

    def test_grant_doc_still_records_submitted_status(self) -> None:
        grant = GRANT_PATH.read_text(encoding="utf-8")
        self.assertIn("## Submission Status", grant)
        self.assertRegex(grant, r"(?i)\bstatus:\*\* submitted\b|\bsubmitted\b")

    def test_incomplete_smoke_log_is_rejected(self) -> None:
        incomplete = "step:1/1 train_loss:6.1234\n"
        with self.assertRaisesRegex(AssertionError, r"missing completed int8\+zlib size line"):
            self._assert_completed_smoke_text(incomplete)

    def test_template_false_positive_is_rejected(self) -> None:
        fake = (
            'Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes\n'
            'final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}\n'
        )
        with self.assertRaisesRegex(AssertionError, "source-template final metric"):
            self._assert_completed_smoke_text(fake)

    def test_missing_final_metric_is_rejected_even_when_size_line_exists(self) -> None:
        missing_metric = "Total submission size int8+zlib: 123456 bytes\nstep:1/1 train_loss:5.4321\n"
        with self.assertRaisesRegex(AssertionError, "missing numeric final_int8_zlib_roundtrip val_bpb line"):
            self._assert_completed_smoke_text(missing_metric)


if __name__ == "__main__":
    unittest.main()
