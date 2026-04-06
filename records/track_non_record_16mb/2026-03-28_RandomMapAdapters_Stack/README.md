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
3. The dataset root, tokenizer path, and `python experiments/verify_run.py --help` all work from that same remote checkout.

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

The accepted metric for this comparison is `final_int6_sliding_window_s64`. Each log must also contain `Total submission size int6+lzma:`.
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

The saved evidence pair currently audits cleanly through the local verifier/comparison/audit stack.
Those logs predate the learned-gate extension, so they remain the last audited ungated pair until a Linux/CUDA rerun proves the new gate-enabled variant:

- `baseline_no_adapter.log` -> `chosen_metric: final_int6_sliding_window_s64`, `val_bpb: 1.1400`, `Total submission size int6+lzma: 15600000 bytes`
- `random_map_adapter.log` -> `chosen_metric: final_int6_sliding_window_s64`, `val_bpb: 1.1300`, `Total submission size int6+lzma: 15680000 bytes`
- `adapter_minus_baseline_bpb_delta: -0.0100`

Interpretation: the ungated adapter improves `final_int6_sliding_window_s64` by 0.0100 bpb, but it also increases total artifact size by 80,000 bytes. This remains a **non-record** package, so it does not count toward the S06 real-ablation ledger and should not be confused with the promoted legal-TTT submission evidence.
The learned-gate variant is implemented in `train_gpt.py`, but it still needs a clean Linux/CUDA rerun before these fixed logs can be replaced.

## Keep / drop decision

**Decision: drop from the promoted submission path for now; keep as a reusable non-record experiment package.**

Why:

1. The measured delta is modest (`-0.0100` bpb) relative to the already-promoted legal-TTT stack.
2. The adapter increases artifact size from 15,600,000 to 15,680,000 bytes.
3. The experiment is useful evidence for future reuse, but it is not part of the audited promoted submission package.

If a future rerun on the intended Linux/CUDA host produces a materially better delta under the same fixed-path contract, update these same files in place and rerun the verifier/comparison/audit commands above.
