# Random Map Adapters on the S03 Stack (Non-Record Package)

This folder packages the S04 random-linear-map adapter experiment on top of the promoted S03 non-TTT stack. It is a fixed-path, mechanically auditable A/B comparison surface for the adapter-on vs adapter-off question.
The final fork-owned extension adds an optional learned multiplicative gate to the adapter path while keeping the random projection matrix frozen.

## Technique summary

`train_gpt.py` is a copy of `experiments/train_gpt_random_map_adapter.py`. The novelty seam stays narrow: only Q/V projections in selected layers receive learned deltas, while each adapter's random projection matrix stays frozen as a registered buffer.
The fork-specific closeout extension is `RANDOM_MAP_ADAPTER_GATE_ENABLED=1`, which multiplies the learned adapter output by a small learned scalar gate per adapted projection. When the gate is disabled, behavior matches the earlier ungated adapter path.

## Stable artifact paths

All evidence for this comparison stays in this folder:

- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/train_gpt.py`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/README.md`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log`
- `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log`

Do not improvise alternate log names. The verifier, comparison helper, and audit all assume these fixed paths.

## Capability/auth gate before reruns

Before overwriting `baseline_no_adapter.log` or `random_map_adapter.log`, prove a real remote Linux/CUDA control path from this workspace:

1. `runpodctl` is installed locally and `RUNPOD_API_KEY` is set, **or** an authenticated `ssh <remote-host>` / `scp <remote-host>` route already reaches the target checkout.
2. The remote host can run `python -c "import flash_attn_interface"` successfully.
3. The dataset root, tokenizer path, and `python experiments/verify_run.py <log>` all work from that same remote checkout.

Do not rerun or overwrite the fixed logs until one of those control paths is proven from this workspace.
If any prerequisite fails, preserve that failure separately instead of inserting `preserved_windows_host_note` or `appended_contract_fixture` markers into the fixed evidence logs.
The current local control-path result is recorded separately in `runpod_capability_check_2026-04-05.txt`.

## Comparison contract

Both runs keep these shared settings identical:

- `TTT_ENABLED=0`
- `EVAL_STRIDE=64`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- identical dataset/tokenizer paths, seed, and remaining stack knobs

The accepted metric contract for this comparison is the stride-64 non-TTT sliding-window surface:

- `final_int6_sliding_window ... stride:64` when `EVAL_STRIDE=64`, or
- `final_int6_sliding_window_s64 ... stride:64` when the script emits a supplemental 64-stride pass

Each log must also contain `Total submission size int6+lzma:`.
The final closeout rerun target also enables:

- `RANDOM_MAP_ADAPTER_GATE_ENABLED=1`
- `RANDOM_MAP_ADAPTER_GATE_INIT=1.0`

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
RANDOM_MAP_ADAPTER_GATE_ENABLED=1 \
RANDOM_MAP_ADAPTER_GATE_INIT=1.0 \
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

The audit rejects placeholder markers such as `preserved_windows_host_note`, `appended_contract_fixture`, and the preserved cmd.exe failure header.

## Current fixed-path evidence summary

The saved evidence pair now comes from the real Linux/CUDA learned-gate rerun executed on the official RunPod Parameter Golf template. The verifier/comparison/audit stack accepts the pair through the stride-64 non-TTT contract:

- `baseline_no_adapter.log` -> `chosen_metric: final_int6_sliding_window`, `val_bpb: 2.2096`, `Total submission size int6+lzma: 7335405 bytes`
- `random_map_adapter.log` -> `chosen_metric: final_int6_sliding_window`, `val_bpb: 2.2804`, `Total submission size int6+lzma: 7277705 bytes`
- `adapter_minus_baseline_bpb_delta: +0.0708`

Interpretation: the learned-gate adapter is a **negative result** on the verified remote rerun. It reduces artifact size by 57,700 bytes, but it worsens `val_bpb` by 0.0708 relative to the baseline under the same non-TTT stride-64 contract. This remains a **non-record** package, so it does not count toward the S06 real-ablation ledger and should not be confused with the promoted legal-TTT submission evidence.

## Keep / drop decision

**Decision: archive as a reusable non-record negative-result package; do not promote.**

Why:

1. The verified learned-gate rerun is worse than baseline by `+0.0708` bpb.
2. The adapter does reduce artifact size from `7,335,405` to `7,277,705` bytes, but that size win is not enough to justify the loss increase.
3. The experiment is still useful evidence because it closes the branch with a real remote negative result instead of leaving it ambiguous.

If a future rerun or architecture change materially improves this branch under the same fixed-path contract, update these same files in place and rerun the verifier/comparison/audit commands above.
