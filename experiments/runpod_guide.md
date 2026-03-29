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

### Non-TTT fallback proof (`final_int6_sliding_window_s64`)

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

- prints `chosen_metric: final_int6_sliding_window_s64`
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

Use this when executing the first S04 comparison. The baseline and adapter runs must share the same non-TTT runtime contract so the only intentional delta is the adapter configuration.

### Shared settings that must stay identical

- `TTT_ENABLED=0`
- `EVAL_STRIDE=64`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- same dataset/tokenizer paths, seed, and remaining stack knobs

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
RUN_ID=s04_random_map_adapter \
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log 2>&1
python experiments/verify_run.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

### Comparison command

```bash
python experiments/compare_random_map_runs.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

Expected comparison behavior:

- both logs resolve to `chosen_metric: final_int6_sliding_window_s64`
- the helper prints both chosen metrics
- the helper prints `adapter_minus_baseline_bpb_delta: +/-0.xxxx`

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError: zstandard` | Optional compression dependency missing | `pip install zstandard` |
| `FileNotFoundError` for dataset shards | `DATA_PATH` does not point at FineWeb shards | Fix `DATA_PATH` or create the expected symlink |
| `RuntimeError: CUDA is required` | Run attempted outside a CUDA pod | Move to RunPod/Linux with GPU |
| `Error: no accepted metric found in log` | Run crashed early or log only contains stale aliases | Inspect the log tail and confirm one of the accepted metrics was emitted |
| `compare_random_map_runs.py` rejects `legal_ttt` | The run was not kept on the first S04 non-TTT contract | Re-run with `TTT_ENABLED=0` and confirm stride-64 eval completed |
| Verifier chooses `final_int6_roundtrip` or `final_int6_sliding_window` | Sliding-window stride-64 eval did not complete | Check `EVAL_STRIDE`, inspect the log tail, and do not trust the comparison until both logs choose `final_int6_sliding_window_s64` |
