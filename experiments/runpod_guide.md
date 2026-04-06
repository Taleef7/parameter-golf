# RunPod Execution Guide — S03 Integrated Stack

**Platform:** RunPod  
**Pod type:** 1× H100 SXM5 80 GB  
**Docker image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`  
**Disk:** 50 GB  
**Target artifact:** `experiments/train_gpt_stack.py`

> **Linux/CUDA only.** `train_gpt_stack.py` imports Flash Attention 3 and requires CUDA.

## 1. Pod setup

After SSH-ing into the pod:

```bash
cd /workspace
git clone <your-repo-url> parameter-golf
cd parameter-golf
pip install zstandard
ls /workspace/data/datasets/fineweb10B_sp1024/
mkdir -p logs
```

If the dataset is stored elsewhere, either set `DATA_PATH` explicitly or create a symlink to `/workspace/data/datasets/fineweb10B_sp1024`.

## 2. Canonical S03 run contract

The integrated script already defaults to the promoted S03 architecture. For an operator, the important knobs are the runtime cap, eval stride, and whether legal TTT is enabled.

### Preferred proof run (`legal_ttt`)

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
python experiments/train_gpt_stack.py \
  > logs/stack_legal_ttt.txt 2>&1
python experiments/verify_run.py logs/stack_legal_ttt.txt
```

Expected verifier behavior:

- prints `chosen_metric: legal_ttt`
- prints `val_bpb: <value>`

### Non-TTT fallback proof (stride-64 sliding-window contract)

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
EVAL_STRIDE=64 \
TTT_ENABLED=0 \
python experiments/train_gpt_stack.py \
  > logs/stack_no_ttt.txt 2>&1
python experiments/verify_run.py logs/stack_no_ttt.txt
```

Expected verifier behavior:

- prints `chosen_metric: final_int6_sliding_window` when `EVAL_STRIDE=64`
- or prints `chosen_metric: final_int6_sliding_window_s64` when the script emits a supplemental 64-stride pass
- prints `val_bpb: <value>`

## 3. Real env vars still wired in the script

Use only env vars that exist in `experiments/train_gpt_stack.py`. The commonly touched ones for S03 are:

### Data and reproducibility

- `DATA_PATH`
- `TOKENIZER_PATH`
- `RUN_ID`
- `SEED`

### Runtime and evaluation

- `MAX_WALLCLOCK_SECONDS`
- `ITERATIONS`
- `TRAIN_BATCH_TOKENS`
- `TRAIN_SEQ_LEN`
- `EVAL_SEQ_LEN`
- `VAL_BATCH_SIZE`
- `VAL_LOSS_EVERY`
- `TRAIN_LOG_EVERY`
- `EVAL_STRIDE`
- `WARMUP_STEPS`
- `WARMDOWN_ITERS`

### Architecture / stack selection

- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `BIGRAM_VOCAB_SIZE`
- `BIGRAM_DIM`
- `XSA_LAST_N`
- `ROPE_DIMS`
- `LN_SCALE`
- `VE_ENABLED`
- `VE_DIM`
- `VE_LAYERS`
- `TIE_EMBEDDINGS`

### Optimization / averaging

- `MATRIX_LR`
- `SCALAR_LR`
- `TIED_EMBED_LR`
- `HEAD_LR`
- `MUON_MOMENTUM`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `MUON_BACKEND_STEPS`
- `MUON_WD`
- `ADAM_WD`
- `BETA1`
- `BETA2`
- `ADAM_EPS`
- `GRAD_CLIP_NORM`
- `SWA_ENABLED`
- `SWA_EVERY`
- `LAWA_ENABLED`
- `LAWA_K`
- `LAWA_FREQ`

### TTT

- `TTT_ENABLED`
- `TTT_LR`
- `TTT_EPOCHS`
- `TTT_CHUNK_TOKENS`
- `TTT_FREEZE_BLOCKS`
- `TTT_MOMENTUM`
- `TTT_BATCH_SEQS`
- `TTT_GRAD_CLIP`

## 4. What the verifier accepts

`experiments/verify_run.py` ignores stale aliases and accepts only these metrics, in order:

1. `legal_ttt`
2. `final_int6_sliding_window_s64`
3. `final_int6_sliding_window`
4. `final_int6_roundtrip`

If none of those appear, the verifier exits non-zero and prints a useful error naming the accepted metrics.

## 5. S04 random-map-adapter runbook (Linux/CUDA, non-TTT only)

Use this when executing the fork-specific S04 closeout comparison. The baseline and adapter runs must share the same non-TTT runtime contract so the only intentional delta is the adapter configuration.
The final fork-owned extension is a learned multiplicative gate on top of the frozen random-map adapter. Leave the gate disabled for the baseline and enable it for the adapter rerun.

### Capability gate before any rerun

Prove the remote execution path before you touch `baseline_no_adapter.log` or `random_map_adapter.log`.

1. Prove the control path from this workspace:
   - **RunPod CLI path:** `runpodctl` is installed locally and `RUNPOD_API_KEY` is present.
   - **SSH path:** an authenticated `ssh <remote-host>` / `scp <remote-host>` route already reaches the Linux/CUDA workspace that will run the job.
2. Prove the remote runtime prerequisites on the Linux/CUDA host before starting either run:
   - `python -c "import flash_attn_interface"` succeeds.
   - the dataset root and tokenizer path you will use are readable.
   - `python -c "from experiments import verify_run; print('verify_run:ok')"` works from the remote repo checkout so the same verifier is available there.

Do not rerun or overwrite the fixed logs until one of those control paths is proven from this workspace.
If any prerequisite check fails, stop at that failure boundary, preserve the failing command output separately, and leave the fixed evidence logs untouched rather than appending placeholder fixtures.

### Shared settings that must stay identical

- `TTT_ENABLED=0`
- `EVAL_STRIDE=64`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- same dataset/tokenizer paths, seed, and remaining stack knobs

Closeout-specific adapter settings:

- `RANDOM_MAP_ADAPTER_GATE_ENABLED=1`
- `RANDOM_MAP_ADAPTER_GATE_INIT=1.0`

### Baseline command

```bash
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
RANDOM_MAP_ADAPTER_ENABLED=0 \
RUN_ID=s04_random_map_baseline \
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log 2>&1
python experiments/verify_run.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log
```

### Adapter command

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
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log 2>&1
python experiments/verify_run.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

### Remote command form

Run both jobs from the same remote repo checkout and write directly to the fixed artifact paths there.
The command shape must stay the same even if the host name differs:

```bash
ssh <remote-host> 'cd /workspace/parameter-golf && \
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
RANDOM_MAP_ADAPTER_ENABLED=0 \
RUN_ID=s04_random_map_baseline \
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log 2>&1'

ssh <remote-host> 'cd /workspace/parameter-golf && \
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
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log 2>&1'
```

After the remote runs finish, copy the artifacts back onto the same local fixed paths before running the local verifier/audit commands:

```bash
scp <remote-host>:/workspace/parameter-golf/records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log
scp <remote-host>:/workspace/parameter-golf/records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

Only after both copy-back steps succeed should you run `python experiments/verify_run.py`, `python experiments/compare_random_map_runs.py`, and `python experiments/audit_random_map_runtime_proof.py` locally against the fixed evidence pair.

### Comparison command

```bash
python experiments/compare_random_map_runs.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log

python experiments/audit_random_map_runtime_proof.py \
  --baseline records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  --adapter records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

Expected comparison behavior:

- both logs satisfy the stride-64 non-TTT contract:
  - `chosen_metric: final_int6_sliding_window` with `stride:64`, or
  - `chosen_metric: final_int6_sliding_window_s64`
- the helper prints both chosen metrics
- the helper prints `adapter_minus_baseline_bpb_delta: +/-0.xxxx`
- do not accept placeholder-backed proof: the audit rejects `preserved_windows_host_note`, `appended_contract_fixture`, and the preserved cmd.exe failure header
- the audit also requires the baseline/adaptor `random_map_adapter:enabled=...` config lines plus a stride-64 runtime line and `Total submission size int6+lzma:` in each saved log

The committed fixed-path logs now come from the RunPod learned-gate rerun. They prove the Linux/CUDA path and show that the learned-gate variant underperformed the baseline, so do not treat this branch as a promoted candidate.

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError: zstandard` | Optional compression dependency missing | `pip install zstandard` |
| `FileNotFoundError` for dataset shards | `DATA_PATH` does not point at FineWeb shards | Fix `DATA_PATH` or create the expected symlink |
| `RuntimeError: CUDA is required` | Run attempted outside a CUDA pod | Move to RunPod/Linux with GPU |
| `Error: no accepted metric found in log` | Run crashed early or log only contains stale aliases | Inspect the log tail and confirm one of the accepted metrics was emitted |
| `compare_random_map_runs.py` rejects `legal_ttt` | The run was not kept on the first S04 non-TTT contract | Re-run with `TTT_ENABLED=0` and confirm stride-64 eval completed |
| Verifier chooses `final_int6_roundtrip` | Sliding-window stride-64 eval did not complete | Check `EVAL_STRIDE`, inspect the log tail, and do not trust the comparison until both logs satisfy the stride-64 contract |
| Verifier chooses `final_int6_sliding_window` | This is acceptable only if the corresponding runtime line includes `stride:64` | Inspect the log tail and only accept the comparison if the stride-64 line is present |
