from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import audit_random_map_runtime_proof as audit

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = ROOT / "experiments" / "run_configs.md"
RUNPOD_GUIDE = ROOT / "experiments" / "runpod_guide.md"


class RandomMapRuntimeProofContractTests(unittest.TestCase):
    def make_log(
        self,
        directory: Path,
        *,
        label: str,
        enabled: bool,
        metric_value: float = 1.1500,
        include_size: bool = True,
        include_metric: bool = True,
        extra_lines: list[str] | None = None,
    ) -> Path:
        config_line = audit.EXPECTED_CONFIG_LINES[label]
        path = directory / f"{label}.log"
        lines = [
            "logs/random-map-runtime-proof.log",
            config_line,
            "random_map_adapter_params:16384 random_map_buffer_params:8192" if enabled else "random_map_adapter_params:0 random_map_buffer_params:0",
            "step:9000/9000 val_loss:1.9800 val_bpb:1.1500 train_time:598500ms step_avg:66.50ms",
        ]
        if extra_lines:
            lines.extend(extra_lines)
        if include_size:
            lines.append("Total submission size int6+lzma: 15680000 bytes")
        if include_metric:
            lines.extend(
                [
                    "final_int6_roundtrip val_loss:0.8200 val_bpb:1.1700 eval_time:101ms",
                    "final_int6_sliding_window val_loss:0.8150 val_bpb:1.1600 stride:32 eval_time:145ms",
                    f"final_int6_sliding_window_s64 val_loss:0.8100 val_bpb:{metric_value:.4f} stride:64 eval_time:133ms",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_runtime_shaped_pair_passes_and_reports_signed_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = self.make_log(tmpdir, label="baseline", enabled=False, metric_value=1.1500)
            adapter = self.make_log(tmpdir, label="adapter", enabled=True, metric_value=1.1400)

            payload = audit.run_audit(baseline_log=baseline, adapter_log=adapter)

        self.assertEqual(payload["pair"]["baseline"]["chosen_metric"], audit.EXPECTED_METRIC)
        self.assertEqual(payload["pair"]["adapter"]["chosen_metric"], audit.EXPECTED_METRIC)
        self.assertAlmostEqual(payload["pair"]["adapter_minus_baseline_bpb_delta"], -0.01)

    def test_missing_log_path_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(FileNotFoundError, "file not found"):
                audit.run_audit(baseline_log=tmpdir / "missing.log", adapter_log=adapter)

    def test_placeholder_markers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = self.make_log(
                tmpdir,
                label="baseline",
                enabled=False,
                extra_lines=["# preserved_windows_host_note: placeholder"],
            )
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "placeholder marker: preserved_windows_host_note"):
                audit.run_audit(baseline_log=baseline, adapter_log=adapter)

    def test_cmd_exe_failure_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = self.make_log(
                tmpdir,
                label="baseline",
                enabled=False,
                extra_lines=[audit.CMD_FAILURE_HEADER],
            )
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "cmd.exe failure header"):
                audit.run_audit(baseline_log=baseline, adapter_log=adapter)

    def test_missing_size_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = self.make_log(tmpdir, label="baseline", enabled=False, include_size=False)
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "Total submission size int6\\+lzma"):
                audit.run_audit(baseline_log=baseline, adapter_log=adapter)

    def test_missing_expected_config_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = tmpdir / "baseline.log"
            baseline.write_text(
                "\n".join(
                    [
                        "random_map_adapter:enabled=False rank=0 layers=[] targets=[] seed=999 scale_init=0.0000",
                        "Total submission size int6+lzma: 15600000 bytes",
                        "final_int6_sliding_window_s64 val_loss:0.8100 val_bpb:1.1500 stride:64 eval_time:133ms",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "missing expected config line"):
                audit.run_audit(baseline_log=baseline, adapter_log=adapter)

    def test_wrong_chosen_metric_is_rejected_via_shared_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = self.make_log(
                tmpdir,
                label="baseline",
                enabled=False,
                extra_lines=["legal_ttt val_loss:0.8000 val_bpb:1.1200"],
            )
            adapter = self.make_log(tmpdir, label="adapter", enabled=True)
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "expected final_int6_sliding_window_s64"):
                audit.run_audit(baseline_log=baseline, adapter_log=adapter)

    def test_doc_audit_requires_fixed_paths_audit_command_and_placeholder_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            doc = tmpdir / "doc.md"
            doc.write_text("\n".join(audit.REQUIRED_DOC_SNIPPETS) + "\n", encoding="utf-8")
            audited = audit.audit_doc(doc)
            self.assertEqual(audited.path, doc.as_posix())

            doc.write_text("python experiments/audit_random_map_runtime_proof.py\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.RandomMapRuntimeAuditError, "doc contract drift"):
                audit.audit_doc(doc)

    def test_real_operator_docs_include_capability_gate_and_copy_back_contract(self) -> None:
        for path in (RUN_CONFIGS, RUNPOD_GUIDE):
            text = path.read_text(encoding="utf-8")
            for snippet in audit.REQUIRED_DOC_SNIPPETS:
                self.assertIn(snippet, text, f"missing {snippet!r} in {path}")


if __name__ == "__main__":
    unittest.main()
