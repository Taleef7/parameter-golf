#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys


def required_modules() -> list[str]:
    raw = os.environ.get("LOCAL_ENV_REQUIRED_MODULES", "sentencepiece,pytest")
    return [token.strip() for token in raw.split(",") if token.strip()]


def build_report(modules: list[str]) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for module in modules:
        checks.append(
            {
                "module": module,
                "available": importlib.util.find_spec(module) is not None,
            }
        )
    return {"python": sys.executable, "checks": checks}


def main() -> int:
    report = build_report(required_modules())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(check["available"] for check in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
