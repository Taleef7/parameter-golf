#!/usr/bin/env python3
"""Mechanical launchability audit for the promoted S08 submission package.

This helper intentionally stays slice-specific instead of widening
``experiments.audit_submission_package`` or ``experiments.verify_run``. It reuses
``experiments.audit_submission_package.run_audit(...)`` for provenance, seed
statistics, and package-size continuity, then layers the missing launchability
checks on top of the promoted script and inherited seed logs.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import audit_submission_package as package_audit  # noqa: E402

TRAIN_TIME_CAP_MS = 600_000
TRAIN_TIME_OVERSHOOT_TOLERANCE_MS = 1_000
EVAL_TIME_CAP_MS = 600_000
EXPECTED_WORLD_SIZE = 8
DEFAULT_SCRIPT = (
    ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
    / "train_gpt.py"
)
DEFAULT_README = DEFAULT_SCRIPT.parent / "README.md"
DEFAULT_SUBMISSION = DEFAULT_SCRIPT.parent / "submission.json"
DEFAULT_LOGS = [
    ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon"
    / name
    for name in ("train_seed1337.log", "train_seed42.log", "train_seed2025.log")
]
DEFAULT_DOWNLOADER_SCRIPT = ROOT / "data" / "cached_challenge_fineweb.py"
NETWORK_IMPORT_ROOTS = frozenset(
    {
        "aiohttp",
        "datasets",
        "ftplib",
        "http",
        "httplib",
        "httpx",
        "huggingface_hub",
        "requests",
        "socket",
        "urllib",
        "urllib3",
        "webbrowser",
    }
)
DATA_PATH_PATTERN = re.compile(r'os\.environ\.get\("DATA_PATH",\s*"([^"]+)"\)')
TOKENIZER_PATH_PATTERN = re.compile(r'os\.environ\.get\("TOKENIZER_PATH",\s*"([^"]+)"\)')
WORLD_SIZE_PATTERN = re.compile(r"\bworld_size:(\d+)\b")
TOKENIZER_LOG_PATTERN = re.compile(r"\btokenizer_path=([^\s]+)")
VAL_LOADER_PATTERN = re.compile(r"\bval_loader:shards pattern=([^\s]+)")
STOPPING_PATTERN = re.compile(r"stopping_early:\s*([^\s]+)\s+train_time:(\d+)ms")
LEGAL_TTT_PATTERN = re.compile(r"\blegal_ttt\b.*?\beval_time:(\d+)ms")
SLIDING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "final_int6_sliding_window_s64",
        re.compile(r"\bfinal_int6_sliding_window_s64\b.*?\beval_time:(\d+)ms"),
    ),
    (
        "final_int6_sliding_window",
        re.compile(r"\bfinal_int6_sliding_window\b.*?\bstride:64\b.*?\beval_time:(\d+)ms"),
    ),
)


class SubmissionLaunchabilityAuditError(RuntimeError):
    """Raised when the promoted submission no longer satisfies launchability."""


@dataclass(frozen=True)
class LaunchabilitySeedAudit:
    log_path: str
    world_size: int
    observed_dataset_path: str
    observed_tokenizer_path: str
    stopping_reason: str
    train_time_ms: int
    train_time_cap_ms: int
    train_time_overshoot_ms: int
    train_time_within_tolerance: bool
    legal_ttt_eval_time_ms: int
    sliding_eval_metric: str
    sliding_eval_time_ms: int
    eval_time_cap_ms: int
    total_submission_bytes: int


@dataclass(frozen=True)
class ScriptLaunchabilityAudit:
    script_path: str
    data_path_default: str
    tokenizer_path_default: str
    downloader_control_path: str
    unsupported_network_imports: tuple[str, ...]
    downloader_network_imports: tuple[str, ...]
    local_only_paths_verified: bool
    no_network_proof: bool


@dataclass(frozen=True)
class LaunchabilityAggregate:
    seed_count: int
    expected_world_size: int
    train_time_cap_ms: int
    train_time_overshoot_tolerance_ms: int
    eval_time_cap_ms: int
    max_train_time_ms: int
    max_train_time_overshoot_ms: int
    max_legal_ttt_eval_time_ms: int
    max_sliding_eval_time_ms: int


@dataclass(frozen=True)
class DocsContractAudit:
    readme_path: str
    submission_path: str
    canonical_audit_command: str
    readme_contract_verified: bool
    submission_contract_verified: bool


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def _require_local_path(path_value: str, *, label: str, source: Path) -> str:
    normalized = path_value.replace("\\", "/")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized):
        raise SubmissionLaunchabilityAuditError(
            f"{label} must be local-only in {source}, got {path_value}"
        )
    if "/data/" not in normalized and not normalized.startswith("./data/") and not normalized.startswith("data/"):
        raise SubmissionLaunchabilityAuditError(
            f"{label} must point inside local data/ in {source}, got {path_value}"
        )
    return path_value


def _extract_path_default(pattern: re.Pattern[str], *, env_name: str, script_path: Path, source: str) -> str:
    match = pattern.search(source)
    if match is None:
        raise SubmissionLaunchabilityAuditError(
            f"missing local {env_name} default in script: {script_path}"
        )
    return _require_local_path(match.group(1), label=env_name, source=script_path)


def _collect_network_imports(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in NETWORK_IMPORT_ROOTS:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in NETWORK_IMPORT_ROOTS:
                found.add(root)
    return tuple(sorted(found))


def audit_script_local_only(script_path: Path, downloader_control_path: Path = DEFAULT_DOWNLOADER_SCRIPT) -> ScriptLaunchabilityAudit:
    source = _read_text(script_path)
    control_source = _read_text(downloader_control_path)
    data_path_default = _extract_path_default(
        DATA_PATH_PATTERN,
        env_name="DATA_PATH",
        script_path=script_path,
        source=source,
    )
    tokenizer_path_default = _extract_path_default(
        TOKENIZER_PATH_PATTERN,
        env_name="TOKENIZER_PATH",
        script_path=script_path,
        source=source,
    )

    script_network_imports = _collect_network_imports(source)
    if script_network_imports:
        raise SubmissionLaunchabilityAuditError(
            f"unsupported network imports in promoted script {script_path}: {', '.join(script_network_imports)}"
        )

    downloader_network_imports = _collect_network_imports(control_source)
    if not downloader_network_imports:
        raise SubmissionLaunchabilityAuditError(
            f"ambiguous no-network proof: downloader control {downloader_control_path} exposes no network imports"
        )

    return ScriptLaunchabilityAudit(
        script_path=script_path.as_posix(),
        data_path_default=data_path_default,
        tokenizer_path_default=tokenizer_path_default,
        downloader_control_path=downloader_control_path.as_posix(),
        unsupported_network_imports=script_network_imports,
        downloader_network_imports=downloader_network_imports,
        local_only_paths_verified=True,
        no_network_proof=True,
    )


def _extract_numeric_match(pattern: re.Pattern[str], text: str, *, missing_message: str, malformed_message: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is not None:
        return match
    if missing_message != malformed_message and re.search(pattern.pattern.replace(r"(\\d+)", r"[^\\s]+"), text):
        raise SubmissionLaunchabilityAuditError(malformed_message)
    raise SubmissionLaunchabilityAuditError(missing_message)


def _extract_world_size(text: str, *, log_path: Path) -> int:
    match = WORLD_SIZE_PATTERN.search(text)
    if match is None:
        raise SubmissionLaunchabilityAuditError(f"missing world_size line in log: {log_path}")
    world_size = int(match.group(1))
    if world_size != EXPECTED_WORLD_SIZE:
        raise SubmissionLaunchabilityAuditError(
            f"expected world_size:{EXPECTED_WORLD_SIZE} in log {log_path}, got world_size:{world_size}"
        )
    return world_size


def _extract_local_log_path(pattern: re.Pattern[str], text: str, *, label: str, log_path: Path) -> str:
    match = pattern.search(text)
    if match is None:
        raise SubmissionLaunchabilityAuditError(f"missing {label} line in log: {log_path}")
    return _require_local_path(match.group(1), label=label, source=log_path)


def _extract_stop_reason_and_train_time(text: str, *, log_path: Path) -> tuple[str, int]:
    match = STOPPING_PATTERN.search(text)
    if match is None:
        if "stopping_early:" in text and "train_time:" in text:
            raise SubmissionLaunchabilityAuditError(
                f"missing numeric stopping_early train_time line in log: {log_path}"
            )
        raise SubmissionLaunchabilityAuditError(f"missing stopping_early line in log: {log_path}")
    reason = match.group(1)
    train_time_ms = int(match.group(2))
    if reason != "wallclock_cap":
        raise SubmissionLaunchabilityAuditError(
            f"expected stopping_early: wallclock_cap in log {log_path}, got {reason}"
        )
    overshoot_ms = max(0, train_time_ms - TRAIN_TIME_CAP_MS)
    if overshoot_ms > TRAIN_TIME_OVERSHOOT_TOLERANCE_MS:
        raise SubmissionLaunchabilityAuditError(
            "wallclock overshoot exceeds tolerance in "
            f"{log_path}: train_time_ms={train_time_ms} "
            f"cap_ms={TRAIN_TIME_CAP_MS} tolerance_ms={TRAIN_TIME_OVERSHOOT_TOLERANCE_MS}"
        )
    return reason, train_time_ms


def _extract_eval_time(pattern: re.Pattern[str], text: str, *, metric_name: str, log_path: Path) -> int:
    match = pattern.search(text)
    if match is not None:
        return int(match.group(1))
    if metric_name in text:
        raise SubmissionLaunchabilityAuditError(
            f"missing numeric {metric_name} eval_time line in log: {log_path}"
        )
    raise SubmissionLaunchabilityAuditError(
        f"missing {metric_name} eval_time line in log: {log_path}"
    )


def _extract_sliding_eval_time(text: str, *, log_path: Path) -> tuple[str, int]:
    for metric_name, pattern in SLIDING_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return metric_name, int(match.group(1))
    for metric_name, _pattern in SLIDING_PATTERNS:
        if metric_name in text or (metric_name == "final_int6_sliding_window" and "final_int6_sliding_window" in text):
            raise SubmissionLaunchabilityAuditError(
                f"missing numeric {metric_name} eval_time line in log: {log_path}"
            )
    raise SubmissionLaunchabilityAuditError(
        f"missing final_int6_sliding_window_s64 eval_time line in log: {log_path}"
    )


def audit_seed_launchability(log_path: Path, *, package_seed_audit: dict[str, object]) -> LaunchabilitySeedAudit:
    text = _read_text(log_path)
    world_size = _extract_world_size(text, log_path=log_path)
    observed_tokenizer_path = _extract_local_log_path(
        TOKENIZER_LOG_PATTERN,
        text,
        label="tokenizer_path",
        log_path=log_path,
    )
    observed_dataset_path = _extract_local_log_path(
        VAL_LOADER_PATTERN,
        text,
        label="val_loader path",
        log_path=log_path,
    )
    stopping_reason, train_time_ms = _extract_stop_reason_and_train_time(text, log_path=log_path)
    legal_ttt_eval_time_ms = _extract_eval_time(
        LEGAL_TTT_PATTERN,
        text,
        metric_name="legal_ttt",
        log_path=log_path,
    )
    sliding_eval_metric, sliding_eval_time_ms = _extract_sliding_eval_time(text, log_path=log_path)
    if legal_ttt_eval_time_ms > EVAL_TIME_CAP_MS:
        raise SubmissionLaunchabilityAuditError(
            f"legal_ttt eval_time exceeds cap in {log_path}: {legal_ttt_eval_time_ms}ms > {EVAL_TIME_CAP_MS}ms"
        )
    if sliding_eval_time_ms > EVAL_TIME_CAP_MS:
        raise SubmissionLaunchabilityAuditError(
            f"{sliding_eval_metric} eval_time exceeds cap in {log_path}: {sliding_eval_time_ms}ms > {EVAL_TIME_CAP_MS}ms"
        )

    package_log_path = package_seed_audit.get("log_path")
    if package_log_path != str(log_path):
        raise SubmissionLaunchabilityAuditError(
            f"package audit log ordering drifted for {log_path}: got {package_log_path}"
        )
    total_submission_bytes = int(package_seed_audit["total_submission_bytes"])
    return LaunchabilitySeedAudit(
        log_path=log_path.as_posix(),
        world_size=world_size,
        observed_dataset_path=observed_dataset_path,
        observed_tokenizer_path=observed_tokenizer_path,
        stopping_reason=stopping_reason,
        train_time_ms=train_time_ms,
        train_time_cap_ms=TRAIN_TIME_CAP_MS,
        train_time_overshoot_ms=max(0, train_time_ms - TRAIN_TIME_CAP_MS),
        train_time_within_tolerance=True,
        legal_ttt_eval_time_ms=legal_ttt_eval_time_ms,
        sliding_eval_metric=sliding_eval_metric,
        sliding_eval_time_ms=sliding_eval_time_ms,
        eval_time_cap_ms=EVAL_TIME_CAP_MS,
        total_submission_bytes=total_submission_bytes,
    )


def summarize_launchability(seed_audits: list[LaunchabilitySeedAudit]) -> LaunchabilityAggregate:
    if not seed_audits:
        raise ValueError("at least one launchability seed audit is required")
    return LaunchabilityAggregate(
        seed_count=len(seed_audits),
        expected_world_size=EXPECTED_WORLD_SIZE,
        train_time_cap_ms=TRAIN_TIME_CAP_MS,
        train_time_overshoot_tolerance_ms=TRAIN_TIME_OVERSHOOT_TOLERANCE_MS,
        eval_time_cap_ms=EVAL_TIME_CAP_MS,
        max_train_time_ms=max(seed.train_time_ms for seed in seed_audits),
        max_train_time_overshoot_ms=max(seed.train_time_overshoot_ms for seed in seed_audits),
        max_legal_ttt_eval_time_ms=max(seed.legal_ttt_eval_time_ms for seed in seed_audits),
        max_sliding_eval_time_ms=max(seed.sliding_eval_time_ms for seed in seed_audits),
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_canonical_audit_command(
    *,
    readme: Path,
    submission: Path,
    script: Path,
    logs: list[Path] | tuple[Path, ...],
) -> str:
    log_args = " ".join(_display_path(path) for path in logs)
    return (
        "python experiments/audit_submission_launchability.py "
        f"--readme {_display_path(readme)} "
        f"--submission {_display_path(submission)} "
        f"--script {_display_path(script)} "
        f"--logs {log_args}"
    )


def expected_readme_snippets(
    *,
    aggregate: LaunchabilityAggregate,
    package_payload: dict[str, object],
    script_audit: ScriptLaunchabilityAudit,
) -> list[str]:
    package_aggregate = package_payload["aggregate"]
    max_total_submission_bytes = int(package_aggregate["max_total_submission_bytes"])
    return [
        "inherited 3-seed reproducibility",
        "- every audited seed log records `world_size:8`",
        (
            f"accepted post-step wallclock overshoot is `{aggregate.max_train_time_overshoot_ms} ms` "
            f"above the `{aggregate.train_time_cap_ms} ms` cap"
        ),
        (
            f"audited max `legal_ttt` eval time is `{aggregate.max_legal_ttt_eval_time_ms} ms` "
            f"under the `{aggregate.eval_time_cap_ms} ms` cap"
        ),
        (
            "audited non-TTT fallback surface is `final_int6_sliding_window` with `stride:64` "
            "(future-compatible with `final_int6_sliding_window_s64`)"
        ),
        (
            f"audited max non-TTT eval time is `{aggregate.max_sliding_eval_time_ms} ms` "
            f"under the `{aggregate.eval_time_cap_ms} ms` cap"
        ),
        f"`DATA_PATH` default `{script_audit.data_path_default}`",
        f"`TOKENIZER_PATH` default `{script_audit.tokenizer_path_default}`",
        "Hugging Face download logic lives in `data/cached_challenge_fineweb.py`, not in the promoted submission script",
        "no-network proof is script-specific",
        "15,990,006 bytes",
        "separate non-record package",
        build_canonical_audit_command(
            readme=DEFAULT_README,
            submission=DEFAULT_SUBMISSION,
            script=DEFAULT_SCRIPT,
            logs=DEFAULT_LOGS,
        ),
    ]


def validate_readme_contract(
    readme_path: Path,
    *,
    aggregate: LaunchabilityAggregate,
    package_payload: dict[str, object],
    script_audit: ScriptLaunchabilityAudit,
) -> None:
    text = _read_text(readme_path)
    for snippet in expected_readme_snippets(
        aggregate=aggregate,
        package_payload=package_payload,
        script_audit=script_audit,
    ):
        if snippet not in text:
            raise SubmissionLaunchabilityAuditError(
                f"README launchability contract missing snippet in {readme_path}: {snippet}"
            )
    if "no network access occurs anywhere" in text.lower():
        raise SubmissionLaunchabilityAuditError(
            f"README claims stronger network isolation than the audit can prove: {readme_path}"
        )


def validate_submission_contract(
    submission_path: Path,
    *,
    aggregate: LaunchabilityAggregate,
    package_payload: dict[str, object],
    script_audit: ScriptLaunchabilityAudit,
) -> None:
    metadata = json.loads(_read_text(submission_path))
    package_aggregate = package_payload["aggregate"]
    provenance = package_payload["provenance"]
    launchability = metadata.get("launchability_contract")
    if not isinstance(launchability, dict):
        raise SubmissionLaunchabilityAuditError(
            f"submission metadata missing launchability_contract object: {submission_path}"
        )

    expected_pairs = {
        "seed_count": aggregate.seed_count,
        "seeds": [1337, 42, 2025],
        "bytes_total": int(package_aggregate["max_total_submission_bytes"]),
        "val_bpb": float(package_aggregate["mean_val_bpb"]),
        "val_bpb_std": float(package_aggregate["std_val_bpb"]),
    }
    for key, expected in expected_pairs.items():
        actual = metadata.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise SubmissionLaunchabilityAuditError(
                    f"submission metadata drift for {key} in {submission_path}: {actual} != {expected}"
                )
        elif actual != expected:
            raise SubmissionLaunchabilityAuditError(
                f"submission metadata drift for {key} in {submission_path}: {actual} != {expected}"
            )

    launchability_expected = {
        "world_size": aggregate.expected_world_size,
        "train_time_cap_ms": aggregate.train_time_cap_ms,
        "train_time_overshoot_tolerance_ms": aggregate.train_time_overshoot_tolerance_ms,
        "max_train_time_ms": aggregate.max_train_time_ms,
        "max_train_time_overshoot_ms": aggregate.max_train_time_overshoot_ms,
        "eval_time_cap_ms": aggregate.eval_time_cap_ms,
        "max_legal_ttt_eval_time_ms": aggregate.max_legal_ttt_eval_time_ms,
        "max_non_ttt_eval_time_ms": aggregate.max_sliding_eval_time_ms,
        "non_ttt_metric": "final_int6_sliding_window",
        "non_ttt_stride": 64,
        "data_path_default": script_audit.data_path_default,
        "tokenizer_path_default": script_audit.tokenizer_path_default,
        "downloader_control": "data/cached_challenge_fineweb.py",
        "no_network_proof": True,
        "promoted_script_sha256": provenance["promoted_sha256"],
        "provenance_status": "byte-identical promoted/proven train_gpt.py",
    }
    for key, expected in launchability_expected.items():
        actual = launchability.get(key)
        if actual != expected:
            raise SubmissionLaunchabilityAuditError(
                f"submission launchability drift for {key} in {submission_path}: {actual} != {expected}"
            )


def run_audit(
    *,
    script: Path = DEFAULT_SCRIPT,
    logs: list[Path] | tuple[Path, ...] = tuple(DEFAULT_LOGS),
    proven_script: Path = package_audit.DEFAULT_PROVEN_SCRIPT,
    downloader_control: Path = DEFAULT_DOWNLOADER_SCRIPT,
    readme: Path | None = None,
    submission: Path | None = None,
) -> dict[str, object]:
    package_payload = package_audit.run_audit(
        list(logs),
        promoted_script=script,
        proven_script=proven_script,
    )
    if "aggregate" not in package_payload or "provenance" not in package_payload or "seed_audits" not in package_payload:
        raise SubmissionLaunchabilityAuditError(
            "package audit payload is missing required aggregate/provenance/seed_audits fields"
        )

    package_seed_audits = package_payload["seed_audits"]
    if not isinstance(package_seed_audits, list) or len(package_seed_audits) != len(logs):
        raise SubmissionLaunchabilityAuditError(
            "package audit payload has mismatched seed_audits count"
        )

    script_audit = audit_script_local_only(script, downloader_control)
    seed_audits = [
        audit_seed_launchability(log_path, package_seed_audit=seed_payload)
        for log_path, seed_payload in zip(logs, package_seed_audits)
    ]
    aggregate = summarize_launchability(seed_audits)
    payload = {
        "package_audit": package_payload,
        "script_launchability": asdict(script_audit),
        "seed_launchability": [asdict(seed) for seed in seed_audits],
        "launchability_aggregate": asdict(aggregate),
    }
    if readme is not None or submission is not None:
        if readme is None or submission is None:
            raise SubmissionLaunchabilityAuditError(
                "readme and submission must be provided together when validating reviewer docs"
            )
        validate_readme_contract(
            readme,
            aggregate=aggregate,
            package_payload=package_payload,
            script_audit=script_audit,
        )
        validate_submission_contract(
            submission,
            aggregate=aggregate,
            package_payload=package_payload,
            script_audit=script_audit,
        )
        payload["docs_contract"] = asdict(
            DocsContractAudit(
                readme_path=readme.as_posix(),
                submission_path=submission.as_posix(),
                canonical_audit_command=build_canonical_audit_command(
                    readme=readme,
                    submission=submission,
                    script=script,
                    logs=logs,
                ),
                readme_contract_verified=True,
                submission_contract_verified=True,
            )
        )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help="Promoted train_gpt.py entrypoint to audit.",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=DEFAULT_LOGS,
        help="Seed logs to audit.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Optional README.md to validate against the audited launchability contract.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="Optional submission.json to validate against the audited launchability contract.",
    )
    parser.add_argument(
        "--proven-script",
        type=Path,
        default=package_audit.DEFAULT_PROVEN_SCRIPT,
        help="Optional proven script path for provenance reuse.",
    )
    parser.add_argument(
        "--downloader-control",
        type=Path,
        default=DEFAULT_DOWNLOADER_SCRIPT,
        help="Downloader helper used as the negative-control network surface.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        payload = run_audit(
            script=args.script,
            logs=args.logs,
            proven_script=args.proven_script,
            downloader_control=args.downloader_control,
            readme=args.readme,
            submission=args.submission,
        )
    except (FileNotFoundError, ValueError, package_audit.SubmissionAuditError, SubmissionLaunchabilityAuditError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
