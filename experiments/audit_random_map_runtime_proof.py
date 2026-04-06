#!/usr/bin/env python3
"""Mechanical runtime-proof audit for the fixed S07 random-map adapter evidence pair.

This helper intentionally stays slice-specific instead of widening ``experiments/verify_run.py``.
It validates the fixed baseline/adapter log contract, rejects known placeholder markers,
and can optionally assert that docs describe the same operator contract.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_random_map_runs import (
    EXPECTED_METRIC,
    FALLBACK_STRIDE64_METRIC,
    metric_satisfies_non_ttt_contract,
    run_verifier,
)
ARTIFACT_DIR = ROOT / "records" / "track_non_record_16mb" / "2026-03-28_RandomMapAdapters_Stack"
DEFAULT_BASELINE_LOG = ARTIFACT_DIR / "baseline_no_adapter.log"
DEFAULT_ADAPTER_LOG = ARTIFACT_DIR / "random_map_adapter.log"
DEFAULT_README = ARTIFACT_DIR / "README.md"
DEFAULT_EXPERIMENTS = ROOT / "EXPERIMENTS.md"
REQUIRED_SIZE_PREFIX = "Total submission size int6+lzma:"
CMD_FAILURE_HEADER = "'TTT_ENABLED' is not recognized as an internal or external command"
PLACEHOLDER_MARKERS: dict[str, str] = {
    "preserved_windows_host_note": "preserved_windows_host_note",
    "appended_contract_fixture": "appended_contract_fixture",
    "cmd.exe failure header": CMD_FAILURE_HEADER,
}
EXPECTED_CONFIG_LINES: dict[str, str] = {
    "baseline": "random_map_adapter:enabled=False rank=0 layers=[] targets=[] seed=1729 scale_init=0.0000",
    "adapter": "random_map_adapter:enabled=True rank=8 layers=[9, 10] targets=['q', 'v'] seed=1729 scale_init=0.0100",
}
SIZE_PATTERN = re.compile(r"Total submission size int6\+lzma:\s*([0-9]+)\s*bytes")
REQUIRED_DOC_SNIPPETS: tuple[str, ...] = (
    "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log",
    "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log",
    "python experiments/audit_random_map_runtime_proof.py",
    "final_int6_sliding_window_s64",
    "final_int6_sliding_window",
    "stride:64",
    "Total submission size int6+lzma:",
    "preserved_windows_host_note",
    "appended_contract_fixture",
    "flash_attn_interface",
    "RUNPOD_API_KEY",
    "ssh <remote-host>",
    "scp <remote-host>",
    "Do not rerun or overwrite the fixed logs until one of those control paths is proven from this workspace.",
)
EXPERIMENTS_DOC_SNIPPETS: tuple[str, ...] = (
    "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/README.md",
    "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log",
    "records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log",
    "final_int6_sliding_window",
    "2.2096",
    "2.2804",
    "+0.0708",
    "non-record",
)


class RandomMapRuntimeAuditError(RuntimeError):
    """Raised when the S07 runtime-proof contract is violated."""


@dataclass(frozen=True)
class LogAudit:
    label: str
    log_path: str
    chosen_metric: str
    val_bpb: float
    total_submission_bytes: int
    expected_config_line: str


@dataclass(frozen=True)
class PairAudit:
    baseline: LogAudit
    adapter: LogAudit
    adapter_minus_baseline_bpb_delta: float


@dataclass(frozen=True)
class DocAudit:
    path: str
    checked_snippets: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeProofAudit:
    pair: PairAudit
    docs: tuple[DocAudit, ...]


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_size_bytes(text: str, *, log_path: Path) -> int:
    match = SIZE_PATTERN.search(text)
    if match is None:
        raise RandomMapRuntimeAuditError(
            f"missing '{REQUIRED_SIZE_PREFIX}' line in log: {log_path}"
        )
    return int(match.group(1))


def audit_log(log_path: Path, *, label: str, expected_config_line: str) -> LogAudit:
    text = _read_text(log_path)
    for marker_name, marker_text in PLACEHOLDER_MARKERS.items():
        if marker_text in text:
            raise RandomMapRuntimeAuditError(
                f"{label} log {log_path} contains placeholder marker: {marker_name}"
            )
    if expected_config_line not in text:
        raise RandomMapRuntimeAuditError(
            f"{label} log {log_path} is missing expected config line: {expected_config_line}"
        )
    if not metric_satisfies_non_ttt_contract(text, EXPECTED_METRIC if EXPECTED_METRIC in text else FALLBACK_STRIDE64_METRIC):
        raise RandomMapRuntimeAuditError(
            f"{label} log {log_path} is missing a required stride-64 non-TTT metric line"
        )

    total_submission_bytes = _extract_size_bytes(text, log_path=log_path)
    try:
        chosen_metric, val_bpb = run_verifier(log_path)
    except RuntimeError as exc:
        raise RandomMapRuntimeAuditError(str(exc)) from exc

    return LogAudit(
        label=label,
        log_path=log_path.as_posix(),
        chosen_metric=chosen_metric,
        val_bpb=val_bpb,
        total_submission_bytes=total_submission_bytes,
        expected_config_line=expected_config_line,
    )


def audit_pair(baseline_log: Path, adapter_log: Path) -> PairAudit:
    baseline = audit_log(
        baseline_log,
        label="baseline",
        expected_config_line=EXPECTED_CONFIG_LINES["baseline"],
    )
    adapter = audit_log(
        adapter_log,
        label="adapter",
        expected_config_line=EXPECTED_CONFIG_LINES["adapter"],
    )
    return PairAudit(
        baseline=baseline,
        adapter=adapter,
        adapter_minus_baseline_bpb_delta=adapter.val_bpb - baseline.val_bpb,
    )


def audit_doc(path: Path, required_snippets: Iterable[str] = REQUIRED_DOC_SNIPPETS) -> DocAudit:
    text = _read_text(path)
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise RandomMapRuntimeAuditError(
            f"doc contract drift in {path}: missing snippets {missing}"
        )
    return DocAudit(path=path.as_posix(), checked_snippets=tuple(required_snippets))


def run_audit(
    *,
    baseline_log: Path = DEFAULT_BASELINE_LOG,
    adapter_log: Path = DEFAULT_ADAPTER_LOG,
    readme: Path | None = None,
    experiments: Path | None = None,
) -> dict[str, object]:
    pair = audit_pair(baseline_log, adapter_log)
    docs: list[DocAudit] = []
    if readme is not None:
        docs.append(audit_doc(readme, REQUIRED_DOC_SNIPPETS))
    if experiments is not None:
        docs.append(audit_doc(experiments, EXPERIMENTS_DOC_SNIPPETS))
    payload = RuntimeProofAudit(pair=pair, docs=tuple(docs))
    return asdict(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_LOG, help="Baseline log path.")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_LOG, help="Adapter log path.")
    parser.add_argument("--readme", type=Path, default=None, help="Optional README path to audit.")
    parser.add_argument("--experiments", type=Path, default=None, help="Optional EXPERIMENTS.md path to audit.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        payload = run_audit(
            baseline_log=args.baseline,
            adapter_log=args.adapter,
            readme=args.readme,
            experiments=args.experiments,
        )
    except (FileNotFoundError, RandomMapRuntimeAuditError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
