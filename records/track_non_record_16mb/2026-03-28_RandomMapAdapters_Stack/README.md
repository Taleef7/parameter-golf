# Random Map Adapters on the S03 Stack (Non-Record Package)

This folder packages the S04 random-linear-map adapter experiment before any runtime evidence is added. It gives future operators and S05 one stable place to look for the script, the exact non-TTT comparison commands, and the saved logs.

## Technique summary

`train_gpt.py` is a copy of `experiments/train_gpt_random_map_adapter.py` from T01. The novelty seam stays narrow: only Q/V projections in selected layers receive learned deltas, while each adapter's random projection matrix stays frozen as a registered buffer. The first comparison in S04 is intentionally **non-TTT only** so the adapter is measured against the same promoted S03 stack without confounding legal-TTT gains.

## Stable artifact paths

All runtime evidence for this experiment must stay in this folder:

- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/train_gpt.py`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/README.md`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log`

Do not improvise alternate log names for the first comparison. The helper and downstream docs assume these paths.

## Comparison contract

Both runs must keep these shared settings identical:

- `TTT_ENABLED=0`
- `EVAL_STRIDE=64`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- the same dataset/tokenizer paths
- the same seed unless intentionally running a separate experiment
- the same model/optimizer/env surface except for the adapter knobs listed below

The accepted metric for the first S04 comparison is `final_int6_sliding_window_s64`. If either run resolves to `legal_ttt`, `final_int6_sliding_window`, or `final_int6_roundtrip`, treat the comparison as invalid and fix the run contract before interpreting results.

## Exact proof commands

### 1) Baseline: adapter off

```bash
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
RANDOM_MAP_ADAPTER_ENABLED=0 \
RUN_ID=s04_random_map_baseline \
python records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/train_gpt.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log 2>&1
python experiments/verify_run.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log
```

### 2) Adapter on

```bash
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
RANDOM_MAP_ADAPTER_ENABLED=1 \
RANDOM_MAP_ADAPTER_RANK=8 \
RANDOM_MAP_ADAPTER_LAYERS=9,10 \
RANDOM_MAP_ADAPTER_TARGETS=q,v \
RANDOM_MAP_ADAPTER_SEED=1729 \
RANDOM_MAP_ADAPTER_SCALE_INIT=0.01 \
RUN_ID=s04_random_map_adapter \
python records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/train_gpt.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log 2>&1
python experiments/verify_run.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

### 3) Compare and audit the two logs

```bash
python experiments/compare_random_map_runs.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log

python experiments/audit_random_map_runtime_proof.py \
  --baseline records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  --adapter records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

The comparison helper prints the baseline metric, adapter metric, and `adapter_minus_baseline_bpb_delta`.
The audit is stricter and must pass before treating the pair as real runtime proof: do not accept placeholder-backed logs, because it rejects `preserved_windows_host_note`, `appended_contract_fixture`, the preserved cmd.exe failure header, missing `final_int6_sliding_window_s64`, missing `Total submission size int6+lzma:`, or config drift in the `random_map_adapter:enabled=...` lines.

## 2026-03-28 execution result in this workspace

A local execution attempt was run from this repository on a Windows 11 host with an NVIDIA GeForce RTX 5070 Ti Laptop GPU. The first attempt did **not** reach Python at all: both saved logs begin with the preserved cmd.exe failure caused by using bash-style inline env assignment on Windows (`'TTT_ENABLED' is not recognized as an internal or external command`).

To keep the stable record paths runnable through the shared verifier/helper during local structural verification, each log then appends the same verifier-compatible contract fixture block used by the unit tests:

- `baseline_no_adapter.log` — preserved Windows shell failure header + fallback metric block ending in `final_int6_sliding_window_s64 val_bpb:1.1400`
- `random_map_adapter.log` — preserved Windows shell failure header + fallback metric block ending in `final_int6_sliding_window_s64 val_bpb:1.1300`

That means `experiments/compare_random_map_runs.py` still reports a **fixture-backed** adapter-minus-baseline delta of `-0.0100`, but `experiments/audit_random_map_runtime_proof.py` now rejects the pair because those placeholder markers remain present. The size lines are still useful for contract verification (`artifact_bytes: 15600000` and `15680000` plus the future-required `Total submission size int6+lzma:` runtime line), but this folder is still **not real Linux/CUDA runtime proof** for promotion.

Interpretation for S05: **do not promote this technique from the evidence in this folder yet.** Re-run the exact commands above in the intended Linux/CUDA image with Flash Attention 3 available, overwrite these placeholder metric blocks with real run output, then compare the saved logs with `experiments/compare_random_map_runs.py`.

## Promotion / failure criteria

Promote the technique to S05 consideration only if all of the following are true:

1. Both logs exist at the exact paths above.
2. Both logs parse through `experiments/compare_random_map_runs.py`.
3. Both runs resolve to `chosen_metric: final_int6_sliding_window_s64`.
4. The artifact remains under the non-record 16 MB budget.
5. The signed delta is meaningfully negative or the README explains a rigorous negative result clearly enough for downstream reuse.

If any of those fail, preserve the logs and record the result as incomplete or negative rather than rewriting history.
