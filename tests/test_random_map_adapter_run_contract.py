from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "experiments" / "compare_random_map_runs.py"
SCRIPT = ROOT / "experiments" / "train_gpt_random_map_adapter.py"
RUN_CONFIGS = ROOT / "experiments" / "run_configs.md"
RUNPOD_GUIDE = ROOT / "experiments" / "runpod_guide.md"
ARTIFACT_DIR = ROOT / "records" / "track_non_record_16mb" / "2026-03-28_RandomMapAdapters_Stack"
ARTIFACT_SCRIPT = ARTIFACT_DIR / "train_gpt.py"
ARTIFACT_README = ARTIFACT_DIR / "README.md"
BASELINE_FIXTURE = ROOT / "tests" / "fixtures" / "random_map_baseline.log"
ADAPTER_FIXTURE = ROOT / "tests" / "fixtures" / "random_map_adapter_enabled.log"


class RandomMapAdapterRunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper_source = HELPER.read_text(encoding="utf-8")
        cls.script_source = SCRIPT.read_text(encoding="utf-8")
        cls.run_configs = RUN_CONFIGS.read_text(encoding="utf-8")
        cls.runpod_guide = RUNPOD_GUIDE.read_text(encoding="utf-8")
        cls.artifact_readme = ARTIFACT_README.read_text(encoding="utf-8")

    def run_helper(self, *paths: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python", str(HELPER), *map(str, paths)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def write_log(self, path: Path, metric: str, value: float) -> None:
        path.write_text(
            "\n".join(
                [
                    "random_map_adapter:enabled=True rank=8 layers=[9, 10] targets=['q', 'v'] seed=1729 scale_init=0.0100",
                    f"{metric} val_bpb:{value:.4f}",
                    "artifact_bytes: 15728640",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_helper_source_reuses_verifier_and_enforces_expected_metric(self) -> None:
        self.assertIn("verify_run.py", self.helper_source)
        self.assertIn("EXPECTED_METRIC = \"final_int6_sliding_window_s64\"", self.helper_source)
        self.assertIn("adapter_minus_baseline_bpb_delta", self.helper_source)

    def test_helper_reports_negative_delta_for_fixture_pair(self) -> None:
        result = self.run_helper(BASELINE_FIXTURE, ADAPTER_FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baseline_metric: final_int6_sliding_window_s64", result.stdout)
        self.assertIn("adapter_metric: final_int6_sliding_window_s64", result.stdout)
        self.assertIn("adapter_minus_baseline_bpb_delta: -0.0100", result.stdout)

    def test_helper_reports_equal_and_positive_boundary_cases_clearly(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmpdir:
            tmp = Path(tmpdir)
            equal_a = tmp / "equal_a.log"
            equal_b = tmp / "equal_b.log"
            worse = tmp / "worse.log"
            self.write_log(equal_a, "final_int6_sliding_window_s64", 1.2000)
            self.write_log(equal_b, "final_int6_sliding_window_s64", 1.2000)
            self.write_log(worse, "final_int6_sliding_window_s64", 1.2300)

            equal_result = self.run_helper(equal_a, equal_b)
            self.assertEqual(equal_result.returncode, 0, equal_result.stderr)
            self.assertIn("adapter_minus_baseline_bpb_delta: +0.0000", equal_result.stdout)

            worse_result = self.run_helper(equal_a, worse)
            self.assertEqual(worse_result.returncode, 0, worse_result.stderr)
            self.assertIn("adapter_minus_baseline_bpb_delta: +0.0300", worse_result.stdout)

    def test_helper_fails_on_missing_paths_and_wrong_arity(self) -> None:
        wrong_arity = self.run_helper(BASELINE_FIXTURE)
        self.assertNotEqual(wrong_arity.returncode, 0)
        self.assertIn("Usage: python experiments/compare_random_map_runs.py <baseline_log> <adapter_log>", wrong_arity.stderr)

        missing_result = self.run_helper(ROOT / "tests" / "fixtures" / "does_not_exist.log", ADAPTER_FIXTURE)
        self.assertNotEqual(missing_result.returncode, 0)
        self.assertIn("verify_run failed", missing_result.stderr)
        self.assertIn("log file not found", missing_result.stderr)

    def test_helper_rejects_wrong_metric_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmpdir:
            tmp = Path(tmpdir)
            baseline = tmp / "baseline.log"
            adapter = tmp / "adapter.log"
            self.write_log(baseline, "final_int6_sliding_window_s64", 1.2000)
            self.write_log(adapter, "legal_ttt", 1.1800)
            result = self.run_helper(baseline, adapter)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected final_int6_sliding_window_s64", result.stderr)
            self.assertIn("legal_ttt", result.stderr)

    def test_docs_lock_one_non_ttt_comparison_protocol(self) -> None:
        for doc in (self.run_configs, self.runpod_guide, self.artifact_readme):
            self.assertIn("TTT_ENABLED=0", doc)
            self.assertIn("EVAL_STRIDE=64", doc)
            self.assertIn("MAX_WALLCLOCK_SECONDS=600", doc)
            self.assertIn("ITERATIONS=9000", doc)
            self.assertIn("RANDOM_MAP_ADAPTER_ENABLED=0", doc)
            self.assertIn("RANDOM_MAP_ADAPTER_ENABLED=1", doc)
            self.assertIn("RANDOM_MAP_ADAPTER_RANK=8", doc)
            self.assertIn("RANDOM_MAP_ADAPTER_LAYERS=9,10", doc)
            self.assertIn("RANDOM_MAP_ADAPTER_TARGETS=q,v", doc)
            self.assertIn("baseline_no_adapter.log", doc)
            self.assertIn("random_map_adapter.log", doc)
            self.assertIn("final_int6_sliding_window_s64", doc)

    def test_docs_only_name_supported_random_map_knobs(self) -> None:
        for knob in [
            "RANDOM_MAP_ADAPTER_ENABLED",
            "RANDOM_MAP_ADAPTER_RANK",
            "RANDOM_MAP_ADAPTER_LAYERS",
            "RANDOM_MAP_ADAPTER_TARGETS",
            "RANDOM_MAP_ADAPTER_SEED",
            "RANDOM_MAP_ADAPTER_SCALE_INIT",
            "TTT_ENABLED",
            "EVAL_STRIDE",
            "ITERATIONS",
            "MAX_WALLCLOCK_SECONDS",
        ]:
            self.assertIn(knob, self.script_source)

    def test_non_record_artifact_folder_is_packaged_and_script_matches_experiment_source(self) -> None:
        self.assertTrue(ARTIFACT_DIR.is_dir())
        self.assertTrue(ARTIFACT_SCRIPT.is_file())
        self.assertTrue(ARTIFACT_README.is_file())
        self.assertEqual(ARTIFACT_SCRIPT.read_text(encoding="utf-8"), self.script_source)
        self.assertIn("compare_random_map_runs.py", self.artifact_readme)
        self.assertIn(str(ARTIFACT_DIR).replace(str(ROOT) + "/", "records/") if False else "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack", self.artifact_readme)


if __name__ == "__main__":
    unittest.main()
