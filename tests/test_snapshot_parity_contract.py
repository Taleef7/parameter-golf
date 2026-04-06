from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = ROOT / "experiments" / "check_snapshot_parity.py"


def load_snapshot_module():
    spec = importlib.util.spec_from_file_location("check_snapshot_parity_test", CHECK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {CHECK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SnapshotParityContractTests(unittest.TestCase):
    def test_run_check_reports_known_snapshot_pairs(self) -> None:
        module = load_snapshot_module()
        payload = module.run_check()
        pair_ids = {pair["id"] for pair in payload["pairs"]}
        self.assertIn("stack_promoted_snapshot", pair_ids)
        self.assertIn("random_map_adapter_snapshot", pair_ids)
        self.assertTrue(payload["all_match"])
        self.assertEqual(payload["refreshed"], [])

    def test_cli_returns_machine_readable_json(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("pairs", payload)
        self.assertIn("all_match", payload)

    def test_run_check_can_report_mismatch_for_custom_pair(self) -> None:
        module = load_snapshot_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "src.py"
            snap = tmpdir / "snap.py"
            src.write_text("print('src')\n", encoding="utf-8")
            snap.write_text("print('snap')\n", encoding="utf-8")
            payload = module.run_check(
                pairs=[module.SnapshotPair("custom", src, snap)]
            )
        self.assertFalse(payload["all_match"])
        self.assertFalse(payload["pairs"][0]["match"])

    def test_run_check_can_refresh_custom_pair_before_reporting(self) -> None:
        module = load_snapshot_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "src.py"
            snap = tmpdir / "snap.py"
            src.write_text("print('src')\n", encoding="utf-8")
            snap.write_text("print('snap')\n", encoding="utf-8")
            payload = module.run_check(
                pairs=[module.SnapshotPair("custom", src, snap)],
                refresh=True,
            )
            self.assertEqual(snap.read_text(encoding="utf-8"), src.read_text(encoding="utf-8"))
        self.assertTrue(payload["all_match"])
        self.assertEqual(payload["refreshed"][0]["id"], "custom")


if __name__ == "__main__":
    unittest.main()
