#!/usr/bin/env python3
"""Mechanical audit for the S05 submission package.

This audit intentionally reuses ``experiments.verify_run`` for metric selection so
package checks stay aligned with the canonical verifier contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.verify_run import ACCEPTED_METRICS, extract_metric  # noqa: E402

SIZE_LIMIT_BYTES = 16_000_000
SIZE_PATTERN = re.compile(r"Total submission size int6\+lzma:\s*([0-9]+)\s*bytes")
DEFAULT_PROVEN_SCRIPT = (
    ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
    / "train_gpt.py"
)
DEFAULT_PROMOTED_SCRIPT = (
    ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
    / "train_gpt.py"
)


@dataclass(frozen=True)
class SeedAudit:
    log_path: str
    chosen_metric: str
    val_bpb: float
    total_submission_bytes: int


@dataclass(frozen=True)
class ProvenanceAudit:
    promoted_script: str
    proven_script: str
    scripts_match: bool
    promoted_sha256: str
    proven_sha256: str


@dataclass(frozen=True)
class AggregateAudit:
    seed_count: int
    mean_val_bpb: float
    std_val_bpb: float
    max_total_submission_bytes: int
    size_limit_bytes: int
    within_size_limit: bool


class SubmissionAuditError(RuntimeError):
    """Raised when the submission evidence or provenance contract fails."""


def extract_submission_size_bytes(log_path: Path) -> int:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = SIZE_PATTERN.search(line)
        if match:
            return int(match.group(1))

    raise ValueError(
        "missing 'Total submission size int6+lzma' line in log: "
        f"{log_path}"
    )


def audit_seed_log(log_path: Path) -> SeedAudit:
    chosen_metric, value = extract_metric(log_path)
    if chosen_metric != "legal_ttt":
        raise SubmissionAuditError(
            "expected chosen_metric legal_ttt for submission evidence, "
            f"got {chosen_metric} in {log_path}"
        )

    return SeedAudit(
        log_path=str(log_path),
        chosen_metric=chosen_metric,
        val_bpb=float(value),
        total_submission_bytes=extract_submission_size_bytes(log_path),
    )


def summarize_seeds(seed_audits: list[SeedAudit], size_limit_bytes: int = SIZE_LIMIT_BYTES) -> AggregateAudit:
    if not seed_audits:
        raise ValueError("at least one seed audit is required")

    values = [seed.val_bpb for seed in seed_audits]
    max_bytes = max(seed.total_submission_bytes for seed in seed_audits)
    summary = AggregateAudit(
        seed_count=len(seed_audits),
        mean_val_bpb=statistics.fmean(values),
        std_val_bpb=statistics.pstdev(values),
        max_total_submission_bytes=max_bytes,
        size_limit_bytes=size_limit_bytes,
        within_size_limit=max_bytes <= size_limit_bytes,
    )
    if not summary.within_size_limit:
        raise SubmissionAuditError(
            "submission size exceeds limit: "
            f"max_total_submission_bytes={summary.max_total_submission_bytes} "
            f"size_limit_bytes={summary.size_limit_bytes}"
        )
    return summary


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_script_provenance(promoted_script: Path, proven_script: Path) -> ProvenanceAudit:
    if not promoted_script.exists():
        raise FileNotFoundError(f"promoted script not found: {promoted_script}")
    if not proven_script.exists():
        raise FileNotFoundError(f"proven script not found: {proven_script}")

    promoted_bytes = promoted_script.read_bytes()
    proven_bytes = proven_script.read_bytes()
    promoted_sha = hashlib.sha256(promoted_bytes).hexdigest()
    proven_sha = hashlib.sha256(proven_bytes).hexdigest()
    match = promoted_bytes == proven_bytes
    if not match:
        raise SubmissionAuditError(
            "promoted script drifted from proven script: "
            f"{promoted_script} != {proven_script}"
        )
    return ProvenanceAudit(
        promoted_script=str(promoted_script),
        proven_script=str(proven_script),
        scripts_match=match,
        promoted_sha256=promoted_sha,
        proven_sha256=proven_sha,
    )


def run_audit(
    log_paths: list[Path],
    promoted_script: Path = DEFAULT_PROMOTED_SCRIPT,
    proven_script: Path = DEFAULT_PROVEN_SCRIPT,
    size_limit_bytes: int = SIZE_LIMIT_BYTES,
) -> dict[str, object]:
    seed_audits = [audit_seed_log(path) for path in log_paths]
    aggregate = summarize_seeds(seed_audits, size_limit_bytes=size_limit_bytes)
    provenance = audit_script_provenance(promoted_script, proven_script)
    return {
        "accepted_metric_fallbacks": list(ACCEPTED_METRICS),
        "seed_audits": [asdict(seed) for seed in seed_audits],
        "aggregate": asdict(aggregate),
        "provenance": asdict(provenance),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_paths", nargs="+", type=Path, help="Canonical seed logs to audit")
    parser.add_argument("--promoted-script", type=Path, default=DEFAULT_PROMOTED_SCRIPT)
    parser.add_argument("--proven-script", type=Path, default=DEFAULT_PROVEN_SCRIPT)
    parser.add_argument("--size-limit-bytes", type=int, default=SIZE_LIMIT_BYTES)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        payload = run_audit(
            log_paths=args.log_paths,
            promoted_script=args.promoted_script,
            proven_script=args.proven_script,
            size_limit_bytes=args.size_limit_bytes,
        )
    except (FileNotFoundError, ValueError, SubmissionAuditError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
