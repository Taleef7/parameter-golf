#!/usr/bin/env python3
"""Mechanical audit for historical ablation evidence used by S06.

This parser is intentionally separate from ``experiments.verify_run`` so the
submission verifier keeps its narrow S03/S05 contract while S06 can audit older
metric eras without widening the promoted proof surface.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SIZE_LIMIT_BYTES = 16_000_000
MIN_REAL_ROWS = 5
ACCEPTED_METRICS: tuple[str, ...] = (
    "legal_ttt",
    "final_int6_sliding_window_s64",
    "final_int6_sliding_window",
    "final_int6_roundtrip",
    "final_int8_zlib_sliding_window",
    "final_int8_zlib_roundtrip",
)
METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    metric: re.compile(rf"\b{re.escape(metric)}(?:_exact)?\b.*?\bval_bpb:([0-9]+(?:\.[0-9]+)?)")
    for metric in ACCEPTED_METRICS
}
SIZE_PATTERNS: dict[str, re.Pattern[str]] = {
    "int6+lzma": re.compile(r"Total submission size int6\+lzma:\s*([0-9]+)\s*bytes"),
    "int6+zstd": re.compile(r"Total submission size int6\+zstd:\s*([0-9]+)\s*bytes"),
    "int8+zlib": re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes"),
}
SIZE_LABELS_BY_METRIC: dict[str, tuple[str, ...]] = {
    "legal_ttt": ("int6+lzma", "int6+zstd"),
    "final_int6_sliding_window_s64": ("int6+lzma", "int6+zstd"),
    "final_int6_sliding_window": ("int6+lzma", "int6+zstd"),
    "final_int6_roundtrip": ("int6+lzma", "int6+zstd"),
    "final_int8_zlib_sliding_window": ("int8+zlib",),
    "final_int8_zlib_roundtrip": ("int8+zlib",),
}


@dataclass(frozen=True)
class CandidateRecord:
    record_dir: str
    log_name: str
    family_id: str
    counts_toward_target: bool = True
    inherited_from: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class AuditRow:
    family_id: str
    record_dir: str
    log_path: str
    chosen_metric: str
    val_bpb: float
    total_submission_bytes: int
    size_label: str
    counts_toward_target: bool
    inherited_from: str | None
    note: str | None


@dataclass(frozen=True)
class AggregateAudit:
    row_count: int
    counted_row_count: int
    inherited_row_count: int
    distinct_counted_families: int
    size_limit_bytes: int
    minimum_real_rows: int
    minimum_real_rows_satisfied: bool


class AblationEvidenceAuditError(RuntimeError):
    """Raised when a historical evidence contract fails."""


class DuplicateEvidenceFamilyError(AblationEvidenceAuditError):
    """Raised when more than one counted row claims the same proof family."""


DEFAULT_CANDIDATES: tuple[CandidateRecord, ...] = (
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-17_NaiveBaseline",
        log_name="train.log",
        family_id="2026-03-17_naive_baseline",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-18_FP16Embed_WD3600",
        log_name="train.log",
        family_id="2026-03-18_fp16_embed_wd3600",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-18_LongContextSeq2048",
        log_name="train.log",
        family_id="2026-03-18_long_context_seq2048",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-19_10L_MixedPrecision",
        log_name="train.log",
        family_id="2026-03-19_10l_mixed_precision",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-19_WarmdownQuantization",
        log_name="train.log",
        family_id="2026-03-19_warmdown_quantization",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120",
        log_name="train.log",
        family_id="2026-03-20_11l_efficient_partial_xsa_fa3_swa120",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271",
        log_name="train.log",
        family_id="2026-03-20_11l_xsa4_ema_int6_mlp3x_wd04",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248",
        log_name="train.log",
        family_id="2026-03-21_11l_xsa4_ema_partial_rope_late_qat",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233",
        log_name="train.log",
        family_id="2026-03-22_11l_ema_gptq_lite_warmdown3500_qat015",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon",
        log_name="train_seed2025.log",
        family_id="legal_ttt_parallel_muon",
    ),
    CandidateRecord(
        record_dir="records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon",
        log_name="train.log",
        family_id="legal_ttt_parallel_muon",
        counts_toward_target=False,
        inherited_from="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed2025.log",
        note="Promoted folder inherits the already-counted legal_ttt proof family instead of adding a second ablation row.",
    ),
)


def _normalize(path: Path | str) -> str:
    return Path(path).as_posix()


def _validate_record_dir(record_dir: Path) -> None:
    if not record_dir.exists():
        raise FileNotFoundError(f"record folder not found: {record_dir}")
    if not record_dir.is_dir():
        raise AblationEvidenceAuditError(f"record folder is not a directory: {record_dir}")
    normalized = _normalize(record_dir)
    wrapped = f"/{normalized}/"
    if "/records/track_non_record_16mb/" in wrapped:
        raise AblationEvidenceAuditError(
            "placeholder/non-record folder is not valid ablation evidence: "
            f"{record_dir}"
        )
    if "/records/track_10min_16mb/" not in wrapped:
        raise AblationEvidenceAuditError(
            "record folder must live under records/track_10min_16mb: "
            f"{record_dir}"
        )


def extract_metric(log_path: Path) -> tuple[str, float]:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return extract_metric_from_text(text, log_path=log_path)


def extract_metric_from_text(text: str, *, log_path: Path | str = "<memory>") -> tuple[str, float]:
    if text == "":
        raise ValueError(f"log is empty: {log_path}")

    found: dict[str, float] = {}
    for metric, pattern in METRIC_PATTERNS.items():
        for match in pattern.finditer(text):
            found[metric] = float(match.group(1))

    for metric in ACCEPTED_METRICS:
        if metric in found:
            return metric, found[metric]

    accepted = ", ".join(ACCEPTED_METRICS)
    raise ValueError(
        "no accepted metric found in log; expected one of: "
        f"{accepted}; log={log_path}"
    )


def extract_submission_size(log_path: Path, chosen_metric: str) -> tuple[str, int]:
    found_sizes: dict[str, int] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for label, pattern in SIZE_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    found_sizes[label] = int(match.group(1))

    for label in SIZE_LABELS_BY_METRIC[chosen_metric]:
        if label in found_sizes:
            return label, found_sizes[label]

    accepted = ", ".join(SIZE_LABELS_BY_METRIC[chosen_metric])
    raise ValueError(
        "missing compatible size line for chosen metric "
        f"{chosen_metric}; expected one of [{accepted}] in log: {log_path}"
    )


def audit_candidate(candidate: CandidateRecord, size_limit_bytes: int = SIZE_LIMIT_BYTES) -> AuditRow:
    record_dir = Path(candidate.record_dir)
    _validate_record_dir(record_dir)
    log_path = record_dir / candidate.log_name
    chosen_metric, value = extract_metric(log_path)
    size_label, total_submission_bytes = extract_submission_size(log_path, chosen_metric)
    if total_submission_bytes > size_limit_bytes:
        raise AblationEvidenceAuditError(
            "historical ablation exceeds size limit: "
            f"family_id={candidate.family_id} log={log_path} "
            f"total_submission_bytes={total_submission_bytes} "
            f"size_limit_bytes={size_limit_bytes}"
        )
    return AuditRow(
        family_id=candidate.family_id,
        record_dir=_normalize(record_dir),
        log_path=_normalize(log_path),
        chosen_metric=chosen_metric,
        val_bpb=float(value),
        total_submission_bytes=total_submission_bytes,
        size_label=size_label,
        counts_toward_target=candidate.counts_toward_target,
        inherited_from=candidate.inherited_from,
        note=candidate.note,
    )


def _reject_duplicate_counted_families(rows: Sequence[AuditRow]) -> None:
    families: dict[str, list[str]] = {}
    for row in rows:
        if row.counts_toward_target:
            families.setdefault(row.family_id, []).append(row.log_path)
    duplicates = {family: paths for family, paths in families.items() if len(paths) > 1}
    if duplicates:
        family, paths = next(iter(duplicates.items()))
        raise DuplicateEvidenceFamilyError(
            "duplicate counted evidence family detected: "
            f"family_id={family} paths={paths}"
        )


def summarize_rows(
    rows: Sequence[AuditRow],
    *,
    size_limit_bytes: int = SIZE_LIMIT_BYTES,
    minimum_real_rows: int = MIN_REAL_ROWS,
) -> AggregateAudit:
    counted_rows = [row for row in rows if row.counts_toward_target]
    inherited_rows = [row for row in rows if not row.counts_toward_target]
    _reject_duplicate_counted_families(rows)
    return AggregateAudit(
        row_count=len(rows),
        counted_row_count=len(counted_rows),
        inherited_row_count=len(inherited_rows),
        distinct_counted_families=len({row.family_id for row in counted_rows}),
        size_limit_bytes=size_limit_bytes,
        minimum_real_rows=minimum_real_rows,
        minimum_real_rows_satisfied=len(counted_rows) >= minimum_real_rows,
    )


def run_audit(
    candidates: Iterable[CandidateRecord] = DEFAULT_CANDIDATES,
    *,
    size_limit_bytes: int = SIZE_LIMIT_BYTES,
    minimum_real_rows: int = MIN_REAL_ROWS,
) -> dict[str, object]:
    rows = [audit_candidate(candidate, size_limit_bytes=size_limit_bytes) for candidate in candidates]
    aggregate = summarize_rows(
        rows,
        size_limit_bytes=size_limit_bytes,
        minimum_real_rows=minimum_real_rows,
    )
    return {
        "accepted_metric_fallbacks": list(ACCEPTED_METRICS),
        "rows": [asdict(row) for row in rows],
        "aggregate": asdict(aggregate),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the machine-checkable audit payload.",
    )
    parser.add_argument(
        "--size-limit-bytes",
        type=int,
        default=SIZE_LIMIT_BYTES,
        help="Maximum allowed total submission bytes for any counted row.",
    )
    parser.add_argument(
        "--minimum-real-rows",
        type=int,
        default=MIN_REAL_ROWS,
        help="Minimum number of counted real rows required by downstream docs.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        payload = run_audit(
            size_limit_bytes=args.size_limit_bytes,
            minimum_real_rows=args.minimum_real_rows,
        )
    except (FileNotFoundError, ValueError, AblationEvidenceAuditError) as exc:
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
