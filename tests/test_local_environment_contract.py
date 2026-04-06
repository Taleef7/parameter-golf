from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"
CHECK_SCRIPT = ROOT / "experiments" / "check_local_environment.py"


def load_check_module():
    spec = importlib.util.spec_from_file_location("check_local_environment_test", CHECK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {CHECK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalEnvironmentContractTests(unittest.TestCase):
    def test_requirements_dev_declares_pytest(self) -> None:
        self.assertTrue(REQUIREMENTS_DEV.exists(), f"missing dev requirements file: {REQUIREMENTS_DEV}")
        text = REQUIREMENTS_DEV.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?im)^pytest(?:[<>=!~].*)?$")

    def test_check_module_builds_machine_readable_report(self) -> None:
        module = load_check_module()
        report = module.build_report(["json", "definitely_missing_parameter_golf_pkg"])
        self.assertEqual(report["python"], sys.executable)
        self.assertEqual(report["checks"][0]["module"], "json")
        self.assertTrue(report["checks"][0]["available"])
        self.assertEqual(report["checks"][1]["module"], "definitely_missing_parameter_golf_pkg")
        self.assertFalse(report["checks"][1]["available"])

    def test_cli_exits_non_zero_for_missing_required_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["LOCAL_ENV_REQUIRED_MODULES"] = "json,definitely_missing_parameter_golf_pkg"
            result = subprocess.run(
                [sys.executable, str(CHECK_SCRIPT)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("definitely_missing_parameter_golf_pkg", result.stdout)


if __name__ == "__main__":
    unittest.main()
