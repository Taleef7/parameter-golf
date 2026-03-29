from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = ROOT / "experiments" / "run_configs.md"
RUNPOD_GUIDE = ROOT / "experiments" / "runpod_guide.md"
ARTIFACT_README = ROOT / "records" / "track_non_record_16mb" / "2026-03-28_RandomMapAdapters_Stack" / "README.md"


class RuntimeProofDocsContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_configs = RUN_CONFIGS.read_text(encoding="utf-8")
        cls.runpod_guide = RUNPOD_GUIDE.read_text(encoding="utf-8")
        cls.artifact_readme = ARTIFACT_README.read_text(encoding="utf-8")

    def test_operator_docs_name_the_fixed_paths_and_audit_command(self) -> None:
        for doc in (self.run_configs, self.runpod_guide, self.artifact_readme):
            self.assertIn("records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log", doc)
            self.assertIn("records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log", doc)
            self.assertIn("python experiments/audit_random_map_runtime_proof.py", doc)

    def test_operator_docs_name_required_runtime_signals_and_placeholder_markers(self) -> None:
        for doc in (self.run_configs, self.runpod_guide, self.artifact_readme):
            self.assertIn("final_int6_sliding_window_s64", doc)
            self.assertIn("Total submission size int6+lzma:", doc)
            self.assertIn("preserved_windows_host_note", doc)
            self.assertIn("appended_contract_fixture", doc)


if __name__ == "__main__":
    unittest.main()
