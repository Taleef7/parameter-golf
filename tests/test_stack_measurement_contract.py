from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY_RUN = ROOT / "experiments" / "verify_run.py"
TRAIN_SCRIPT = ROOT / "experiments" / "train_gpt_stack.py"
RUN_CONFIGS = ROOT / "experiments" / "run_configs.md"
RUNPOD_GUIDE = ROOT / "experiments" / "runpod_guide.md"
FIXTURE_LOG = ROOT / "tests" / "fixtures" / "sample_stack_eval.log"

REPO_DOCS = (RUN_CONFIGS, RUNPOD_GUIDE)
REQUIRED_DOC_VARS = {
    "TTT_ENABLED",
    "EVAL_STRIDE",
    "ITERATIONS",
    "MAX_WALLCLOCK_SECONDS",
    "NUM_LAYERS",
    "MLP_MULT",
    "XSA_LAST_N",
    "VE_ENABLED",
    "VE_DIM",
    "VE_LAYERS",
    "MUON_WD",
    "ADAM_WD",
    "SWA_ENABLED",
    "SWA_EVERY",
    "TTT_LR",
    "TTT_EPOCHS",
    "TTT_CHUNK_TOKENS",
    "TTT_FREEZE_BLOCKS",
    "TTT_MOMENTUM",
    "TTT_BATCH_SEQS",
    "TTT_GRAD_CLIP",
}
FORBIDDEN_STALE_VARS = {
    "GPTQ_INT6",
    "LEAKY_SLOPE",
    "EMA_ENABLED",
    "EMA_DECAY",
    "LATE_QAT",
}
ACCEPTED_METRICS = [
    "legal_ttt",
    "final_int6_sliding_window_s64",
    "final_int6_sliding_window",
    "final_int6_roundtrip",
]


def run_verify(log_path: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(VERIFY_RUN), str(log_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def extract_supported_env_vars() -> set[str]:
    text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    return set(re.findall(r'os\.environ\.get\("([A-Z0-9_]+)"', text))


def extract_doc_env_vars() -> set[str]:
    tokens: set[str] = set()
    for doc_path in REPO_DOCS:
        text = doc_path.read_text(encoding="utf-8")
        tokens.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    return tokens


class VerifyRunContractTests(unittest.TestCase):
    def test_fixture_prefers_legal_ttt_over_fallback_metrics(self) -> None:
        result = run_verify(FIXTURE_LOG)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("chosen_metric: legal_ttt", result.stdout)
        self.assertIn("val_bpb: 1.1400", result.stdout)
        self.assertIn(
            "accepted_fallbacks: legal_ttt, final_int6_sliding_window_s64, final_int6_sliding_window, final_int6_roundtrip",
            result.stdout,
        )

    def test_metric_precedence_uses_s64_when_ttt_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "s64.log"
            log_path.write_text(
                "\n".join(
                    [
                        "final_int6_roundtrip val_loss:0.8300 val_bpb:1.1700",
                        "final_int6_sliding_window val_loss:0.8200 val_bpb:1.1600",
                        "final_int6_sliding_window_s64 val_loss:0.8100 val_bpb:1.1500",
                    ]
                ),
                encoding="utf-8",
            )
            result = run_verify(log_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("chosen_metric: final_int6_sliding_window_s64", result.stdout)
        self.assertIn("val_bpb: 1.1500", result.stdout)

    def test_missing_log_path_exits_non_zero(self) -> None:
        result = run_verify(ROOT / "tests" / "fixtures" / "does_not_exist.log")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("log file not found", result.stderr)

    def test_empty_log_exits_non_zero_with_accepted_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "empty.log"
            log_path.write_text("", encoding="utf-8")
            result = run_verify(log_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no accepted metric found", result.stderr)
        self.assertIn("legal_ttt", result.stderr)
        self.assertIn("final_int6_roundtrip", result.stderr)

    def test_legacy_int8_alias_only_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "legacy.log"
            log_path.write_text(
                "final_int8_zlib_roundtrip_exact val_loss:0.8100 val_bpb:1.1500\n",
                encoding="utf-8",
            )
            result = run_verify(log_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no accepted metric found", result.stderr)

    def test_partial_metric_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "partial.log"
            log_path.write_text(
                "legal_ttt val_loss:0.8000\nfinal_int6_sliding_window_s64 val_loss:0.8100\n",
                encoding="utf-8",
            )
            result = run_verify(log_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no accepted metric found", result.stderr)


class DocumentationContractTests(unittest.TestCase):
    def test_docs_only_reference_real_env_vars(self) -> None:
        supported = extract_supported_env_vars()
        mentioned = extract_doc_env_vars()
        unsupported = sorted((mentioned - supported) & (REQUIRED_DOC_VARS | FORBIDDEN_STALE_VARS))
        self.assertEqual(unsupported, [], f"unsupported documented env vars: {unsupported}")

    def test_docs_include_required_stack_knobs(self) -> None:
        mentioned = extract_doc_env_vars()
        missing = sorted(REQUIRED_DOC_VARS - mentioned)
        self.assertEqual(missing, [], f"missing required doc vars: {missing}")

    def test_docs_do_not_mention_known_stale_toggles(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in REPO_DOCS)
        for env_var in sorted(FORBIDDEN_STALE_VARS):
            self.assertNotRegex(text, rf"\b{re.escape(env_var)}\b", f"stale env var still documented: {env_var}")

    def test_docs_describe_metric_precedence_contract(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in REPO_DOCS)
        for metric in ACCEPTED_METRICS:
            self.assertIn(metric, text)
        self.assertIn("legal_ttt", text)
        self.assertIn("final_int6_sliding_window_s64", text)


if __name__ == "__main__":
    unittest.main()
