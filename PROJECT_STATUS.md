# Project Status

This file is the fork-specific status surface for this repository. It describes what this project is now, what is complete, what remains blocked, and which parts are original fork work versus inherited packaging.

## What This Fork Is

This repository started from OpenAI's `parameter-golf` challenge code and now serves four roles at once:

1. baseline challenge entrypoints for PyTorch and MLX
2. a promoted integrated submission package under `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/`
3. a non-record random-map adapter experiment package under `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/`
4. an audit and contract layer that mechanically checks evidence, launchability, provenance, and fixed-path packaging

The repo is strongest as a submission/evidence fork. It is still not a broad experiment platform with fully modular training code.

## What Is Inherited vs Original

### Inherited / promoted packaging

- `experiments/train_gpt_stack.py`
- `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/`

That promoted package preserves an inherited 3-seed result and packages it with launchability and provenance checks. It is valuable evidence, but it is not a fresh fork-owned research win.

### Original fork work

- `experiments/train_gpt_random_map_adapter.py`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/`
- audit helpers under `experiments/`
- contract tests under `tests/`
- environment/snapshot verification utilities

The original contribution is narrow by design: a random-map adapter seam on top of the promoted non-TTT stack, plus the verification machinery around it.

## Complete Surfaces

### Local verification contract

The repo now has one canonical local verification path:

1. install `requirements-dev.txt` into the project venv
2. run `python experiments/check_local_environment.py`
3. run `python experiments/check_snapshot_parity.py`
4. run `python -m pytest -q`

The intended interpreter is the project venv interpreter, not a global `pytest` shim.

### Canonical source and snapshot discipline

- `experiments/*.py` is the editable source of truth for fork-owned experiment scripts
- `records/**/train_gpt.py` remains a committed snapshot artifact for review and provenance
- `python experiments/check_snapshot_parity.py --refresh` refreshes the snapshot copies from the canonical experiment sources before checking parity

### Learned-gate adapter extension

`experiments/train_gpt_random_map_adapter.py` now supports two new env vars:

- `RANDOM_MAP_ADAPTER_GATE_ENABLED`
- `RANDOM_MAP_ADAPTER_GATE_INIT`

When enabled, each adapted Q/V projection gets a learned scalar gate on top of the existing frozen random-map adapter path. When disabled, the script preserves the earlier ungated behavior.

## Current Evidence State

### Promoted integrated stack

The promoted S03 legal-TTT package is complete as an evidence package. The current committed record surface includes the reviewed script, README, submission metadata, and 3 seed logs.

### Random-map adapter package

The non-record package is complete as a fixed-path experiment package and now contains a real Linux/CUDA rerun of the learned-gate variant:

- `baseline_no_adapter.log`
- `random_map_adapter.log`
- `runpod_capability_check_2026-04-05.txt`

The remote A/B result closes the branch as a negative result:

- baseline `val_bpb`: `2.2096`
- learned-gate adapter `val_bpb`: `2.2804`
- signed delta: `+0.0708`
- baseline total bytes: `7,335,405`
- learned-gate total bytes: `7,277,705`

The adapter made the artifact smaller, but it made the metric materially worse. That means the branch is now implemented, remotely verified, and archived rather than still open.

## What Is Still Blocked

No major closeout blocker remains in the fork itself.

As of 2026-04-06, the closeout state is:

- the local code and docs are complete
- the RunPod control path was proven
- the learned-gate rerun was executed on Linux/CUDA
- the fixed logs were copied back into the repo
- the branch outcome is now explicit: negative result, not promotion candidate

## Recommended Entry Points

- new participant baseline: `train_gpt.py`
- Apple Silicon local iteration: `train_gpt_mlx.py`
- promoted integrated record audit: `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/`
- non-record adapter experiment: `experiments/train_gpt_random_map_adapter.py`
- local environment check: `experiments/check_local_environment.py`
- snapshot parity check: `experiments/check_snapshot_parity.py`

## Final Repo Identity

This fork is finished as a public-ready research/evidence snapshot when these statements remain true:

- local verification runs from the project venv with `python -m pytest -q`
- record snapshots stay byte-identical with their canonical experiment sources
- the promoted legal-TTT package remains intact
- the random-map adapter package is explicitly labeled as a remotely verified negative-result branch rather than a promotion candidate

That is the intended final interpretation of this repository.
