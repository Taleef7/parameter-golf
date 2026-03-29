#!/usr/bin/env python3
"""Compare two S04 random-map-adapter logs via the shared verifier contract.

Usage:
    python experiments/compare_random_map_runs.py <baseline_log> <adapter_log>

The helper intentionally stays thin: it shells through experiments/verify_run.py for
both logs, requires both to resolve to the non-TTT fallback metric
``final_int6_sliding_window_s64``, and prints the signed adapter-minus-baseline BPB
delta so operators do not hand-compare markdown or improvised env-var mixes.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXPECTED_METRIC = "final_int6_sliding_window_s64"
VERIFY_SCRIPT = Path(__file__).with_name("verify_run.py")
VAL_BPB_PATTERN = re.compile(r"^val_bpb:\s*([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)
CHOSEN_METRIC_PATTERN = re.compile(r"^chosen_metric:\s*(\S+)$", re.MULTILINE)


def run_verifier(log_path: Path) -> tuple[str, float]:
    command = [sys.executable, str(VERIFY_SCRIPT), str(log_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "verifier produced no output"
        raise RuntimeError(f"verify_run failed for {log_path}: {detail}")

    stdout = result.stdout
    metric_match = CHOSEN_METRIC_PATTERN.search(stdout)
    value_match = VAL_BPB_PATTERN.search(stdout)
    if metric_match is None:
        raise RuntimeError(f"verify_run output for {log_path} is missing chosen_metric")
    if value_match is None:
        raise RuntimeError(f"verify_run output for {log_path} is missing val_bpb")

    metric = metric_match.group(1)
    if metric != EXPECTED_METRIC:
        raise RuntimeError(
            f"{log_path} chose metric {metric}, expected {EXPECTED_METRIC} for the first S04 non-TTT comparison"
        )
    return metric, float(value_match.group(1))


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python experiments/compare_random_map_runs.py <baseline_log> <adapter_log>",
            file=sys.stderr,
        )
        return 1

    baseline_path = Path(sys.argv[1])
    adapter_path = Path(sys.argv[2])

    try:
        baseline_metric, baseline_bpb = run_verifier(baseline_path)
        adapter_metric, adapter_bpb = run_verifier(adapter_path)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    delta = adapter_bpb - baseline_bpb
    print(f"baseline_log: {baseline_path}")
    print(f"baseline_metric: {baseline_metric}")
    print(f"baseline_val_bpb: {baseline_bpb:.4f}")
    print(f"adapter_log: {adapter_path}")
    print(f"adapter_metric: {adapter_metric}")
    print(f"adapter_val_bpb: {adapter_bpb:.4f}")
    print(f"adapter_minus_baseline_bpb_delta: {delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
