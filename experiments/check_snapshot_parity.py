#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SnapshotPair:
    id: str
    source: Path
    snapshot: Path


DEFAULT_PAIRS: tuple[SnapshotPair, ...] = (
    SnapshotPair(
        "stack_promoted_snapshot",
        ROOT / "experiments" / "train_gpt_stack.py",
        ROOT / "records" / "track_10min_16mb" / "2026-03-28_StackIntegration_LegalTTT_ParallelMuon" / "train_gpt.py",
    ),
    SnapshotPair(
        "random_map_adapter_snapshot",
        ROOT / "experiments" / "train_gpt_random_map_adapter.py",
        ROOT / "records" / "track_non_record_16mb" / "2026-03-28_RandomMapAdapters_Stack" / "train_gpt.py",
    ),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_snapshots(pairs: list[SnapshotPair] | tuple[SnapshotPair, ...] = DEFAULT_PAIRS) -> dict[str, object]:
    refreshed_pairs: list[dict[str, str]] = []
    for pair in pairs:
        pair.snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pair.source, pair.snapshot)
        refreshed_pairs.append(
            {
                "id": pair.id,
                "source": pair.source.as_posix(),
                "snapshot": pair.snapshot.as_posix(),
            }
        )
    return {"refreshed": refreshed_pairs}


def run_check(
    pairs: list[SnapshotPair] | tuple[SnapshotPair, ...] = DEFAULT_PAIRS,
    *,
    refresh: bool = False,
) -> dict[str, object]:
    refreshed: list[dict[str, str]] = []
    if refresh:
        refreshed = refresh_snapshots(pairs)["refreshed"]
    encoded_pairs: list[dict[str, object]] = []
    all_match = True
    for pair in pairs:
        source_sha = sha256_file(pair.source)
        snapshot_sha = sha256_file(pair.snapshot)
        match = source_sha == snapshot_sha
        all_match = all_match and match
        encoded_pairs.append(
            {
                "id": pair.id,
                "source": pair.source.as_posix(),
                "snapshot": pair.snapshot.as_posix(),
                "source_sha256": source_sha,
                "snapshot_sha256": snapshot_sha,
                "match": match,
            }
        )
    return {"all_match": all_match, "pairs": encoded_pairs, "refreshed": refreshed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or refresh experiment snapshot parity.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Copy canonical experiment sources into their committed snapshot locations before checking parity.",
    )
    args = parser.parse_args()
    payload = run_check(refresh=args.refresh)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
