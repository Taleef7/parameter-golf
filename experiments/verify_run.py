#!/usr/bin/env python3
"""Extract the canonical integrated-stack val_bpb from a training log.

Usage: python experiments/verify_run.py <log_file_path>

Accepted metrics, in precedence order:
1. legal_ttt
2. final_int6_sliding_window_s64
3. final_int6_sliding_window
4. final_int6_roundtrip

The verifier scans the log once, records the latest value seen for each accepted
metric, then prints the highest-precedence metric that was present. Legacy
aliases such as ``final_int8_zlib_roundtrip_exact`` are intentionally ignored so
stale logs fail loudly instead of being misclassified as the current S03 stack.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ACCEPTED_METRICS: tuple[str, ...] = (
    "legal_ttt",
    "final_int6_sliding_window_s64",
    "final_int6_sliding_window",
    "final_int6_roundtrip",
)

METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    metric: re.compile(rf"\b{re.escape(metric)}\b.*?\bval_bpb:([0-9]+(?:\.[0-9]+)?)")
    for metric in ACCEPTED_METRICS
}


def extract_metric(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    found: dict[str, str] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for metric, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    found[metric] = match.group(1)

    for metric in ACCEPTED_METRICS:
        if metric in found:
            return metric, found[metric]

    accepted = ", ".join(ACCEPTED_METRICS)
    raise ValueError(
        "no accepted metric found in log; expected one of: "
        f"{accepted}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python experiments/verify_run.py <log_file_path>", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    try:
        metric, value = extract_metric(log_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"chosen_metric: {metric}")
    print(f"accepted_fallbacks: {', '.join(ACCEPTED_METRICS)}")
    print(f"val_bpb: {value}")


if __name__ == "__main__":
    main()
