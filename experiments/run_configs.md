# Integrated Stack Run Configs (S03)

This file documents the real env-var surface for the integrated stack in `experiments/train_gpt_stack.py`.
It intentionally omits stale S02 toggles that are no longer wired in the script.

## Canonical success metrics

Use `python experiments/verify_run.py <log>` to extract the accepted end-of-run metric.
The verifier prefers metrics in this order:

1. `legal_ttt`
2. `final_int6_sliding_window_s64`
3. `final_int6_sliding_window`
4. `final_int6_roundtrip`

Interpretation:

- `TTT_ENABLED=1` -> `legal_ttt` is the canonical success metric.
- `TTT_ENABLED=0` and `EVAL_STRIDE=64` -> the script emits `final_int6_sliding_window ... stride:64`.
- `TTT_ENABLED=0` and `EVAL_STRIDE!=64` with the supplemental 64-stride pass enabled -> the script also emits `final_int6_sliding_window_s64`.
- If stride-64 is unavailable, the verifier falls back to `final_int6_sliding_window`, then `final_int6_roundtrip`.

## Promoted S03 stack

The integrated stack already defaults to the intended S03 architecture in `train_gpt_stack.py`, so the main run contract is mostly about making the key knobs explicit.

### Canonical integrated run (TTT enabled)

```bash
TTT_ENABLED=1 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
python experiments/train_gpt_stack.py
```

### Canonical integrated run (TTT disabled fallback)

```bash
TTT_ENABLED=0 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
python experiments/train_gpt_stack.py
```

## Real env vars that matter for S03

### Runtime / data

| Env var | Default | Why you might change it |
|---|---:|---|
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Point the script at the dataset root. |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | Override the SentencePiece model path. |
| `RUN_ID` | random UUID | Control the emitted log filename under `logs/`. |
| `SEED` | `1337` | Reproduce or vary the run seed. |
| `MAX_WALLCLOCK_SECONDS` | `600.0` | Cap training wallclock before early stop. |

### Training / evaluation controls

| Env var | Default | Why you might change it |
|---|---:|---|
| `ITERATIONS` | `20000` | Shorten or extend training. |
| `TRAIN_BATCH_TOKENS` | `786432` | Adjust global train batch size. |
| `TRAIN_SEQ_LEN` | `2048` | Set training context length. |
| `EVAL_SEQ_LEN` | `2048` | Override eval context length. |
| `VAL_BATCH_SIZE` | `524288` | Adjust validation batch size. |
| `VAL_LOSS_EVERY` | `4000` | Change validation cadence. |
| `TRAIN_LOG_EVERY` | `500` | Change train logging cadence. |
| `EVAL_STRIDE` | `64` | Controls stride-64 sliding-window eval. With `EVAL_STRIDE=64` the script logs `final_int6_sliding_window ... stride:64`; otherwise it may also emit `final_int6_sliding_window_s64` for the supplemental 64-stride pass. |
| `WARMUP_STEPS` | `20` | Number of optimizer warmup steps. |
| `WARMDOWN_ITERS` | `3500` | Length of wallclock-aware warmdown. |

### Integrated architecture knobs

| Env var | Default | S03 setting / role |
|---|---:|---|
| `NUM_LAYERS` | `11` | Integrated stack depth. |
| `MODEL_DIM` | `512` | Model width. |
| `NUM_HEADS` | `8` | Attention heads. |
| `NUM_KV_HEADS` | `4` | GQA KV heads. |
| `MLP_MULT` | `3.0` | 3x MLP expansion. |
| `BIGRAM_VOCAB_SIZE` | `2048` | Bigram-hash vocabulary. |
| `BIGRAM_DIM` | `128` | Bigram embedding width. |
| `XSA_LAST_N` | `4` | Enables XSA on the last 4 blocks. |
| `ROPE_DIMS` | `16` | Partial RoPE width. |
| `LN_SCALE` | `1` | Per-layer `1/sqrt(layer+1)` scaling enabled. |
| `VE_ENABLED` | `1` | Value embedding is on by default. |
| `VE_DIM` | `128` | Value embedding width. |
| `VE_LAYERS` | `9,10` | Value embedding target layers. |
| `TIE_EMBEDDINGS` | `1` | Weight tying is on by default. |

### Optimizer / averaging knobs

| Env var | Default | S03 setting / role |
|---|---:|---|
| `MATRIX_LR` | `0.025` | Parallel Muon bank learning rate. |
| `SCALAR_LR` | `0.025` | Scalar/control-tensor AdamW learning rate. |
| `TIED_EMBED_LR` | `0.035` | Token-embedding LR when embeddings are tied. |
| `HEAD_LR` | `0.008` | LM head LR when embeddings are untied. |
| `MUON_MOMENTUM` | `0.99` | Integrated Muon momentum target. |
| `MUON_MOMENTUM_WARMUP_START` | `0.92` | Starting momentum during warmup. |
| `MUON_MOMENTUM_WARMUP_STEPS` | `1500` | Momentum warmup length. |
| `MUON_BACKEND_STEPS` | `5` | Newton-Schulz backend iterations. |
| `MUON_WD` | `0.04` | Matrix weight decay. |
| `ADAM_WD` | `0.04` | AdamW weight decay. |
| `BETA1` | `0.9` | Adam beta1. |
| `BETA2` | `0.95` | Adam beta2. |
| `ADAM_EPS` | `1e-8` | Adam epsilon. |
| `GRAD_CLIP_NORM` | `0.3` | Global training grad clip. |
| `SWA_ENABLED` | `1` | Tight SWA enabled by default. |
| `SWA_EVERY` | `50` | SWA cadence. |
| `LAWA_ENABLED` | `0` | Optional LAWA alternative, off by default. |
| `LAWA_K` | `10` | LAWA queue length. |
| `LAWA_FREQ` | `100` | LAWA snapshot cadence. |

### TTT knobs

| Env var | Default | S03 setting / role |
|---|---:|---|
| `TTT_ENABLED` | `0` | Turn on legal score-first TTT. |
| `TTT_LR` | `0.002` | TTT SGD learning rate. |
| `TTT_EPOCHS` | `3` | TTT epochs per chunk. |
| `TTT_CHUNK_TOKENS` | `32768` | Non-overlapping chunk size. |
| `TTT_FREEZE_BLOCKS` | `2` | Freeze the first N blocks during TTT. |
| `TTT_MOMENTUM` | `0.9` | TTT SGD momentum. |
| `TTT_BATCH_SEQS` | `32` | Per-rank TTT microbatch in sequences. |
| `TTT_GRAD_CLIP` | `1.0` | TTT gradient clipping. |

## S04 random-map-adapter comparison protocol (non-TTT)

Use this section for the fork-specific adapter-vs-baseline closeout on top of the promoted S03 stack. The goal is one reproducible non-TTT A/B pair, not a free-form sweep.
The current fork extension adds an optional learned multiplicative gate on top of the frozen random-map adapter path. That gate stays off by default so the existing ungated behavior remains available.

### Capability/auth gate before reruns

Before overwriting `records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log` or `.../random_map_adapter.log`, prove all of the following:

1. A control path exists from this workspace:
   - `runpodctl` is installed locally and `RUNPOD_API_KEY` is set, **or**
   - an authenticated `ssh <remote-host>` / `scp <remote-host>` path already reaches the target Linux/CUDA repo checkout.
2. The remote Linux/CUDA host satisfies the runtime contract:
   - `python -c "import flash_attn_interface"` succeeds remotely.
   - the dataset root and tokenizer path are readable remotely.
   - the remote checkout can import `experiments.verify_run`.

Do not rerun or overwrite the fixed logs until one of those control paths is proven from this workspace.
If any prerequisite check fails, preserve that failure output separately and treat the rerun as blocked instead of appending placeholder content to the fixed evidence logs.

### Shared contract for both runs

Keep these settings identical between baseline and adapter runs:

- `TTT_ENABLED=0`
- `EVAL_STRIDE=64`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- identical dataset/tokenizer paths, seed, and remaining stack knobs

Both logs must satisfy the stride-64 non-TTT contract:

- `chosen_metric: final_int6_sliding_window_s64`, or
- `chosen_metric: final_int6_sliding_window` with a runtime line containing `stride:64`

If either log resolves to `legal_ttt` or any fallback without the stride-64 signal, do not compare them.
The closeout rerun target uses `RANDOM_MAP_ADAPTER_GATE_ENABLED=1` with `RANDOM_MAP_ADAPTER_GATE_INIT=1.0`. If those gate vars are absent or disabled, you are reproducing the older ungated adapter path instead of the final fork-owned extension.

### Baseline run (adapter off)

```bash
TTT_ENABLED=0 \
EVAL_STRIDE=64 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
RANDOM_MAP_ADAPTER_ENABLED=0 \
RUN_ID=s04_random_map_baseline \
python experiments/train_gpt_random_map_adapter.py \
  > records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log 2>&1
```

### Adapter run (adapter on)

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
```

### Remote execution / copy-back form

Run both commands from the same remote repo checkout and write directly to the fixed artifact paths there:

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

scp <remote-host>:/workspace/parameter-golf/records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log
scp <remote-host>:/workspace/parameter-golf/records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

Only after the copy-back succeeds should you run the local verifier/comparison/audit commands against the fixed evidence pair.

### Verification / comparison

```bash
python experiments/compare_random_map_runs.py \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log

python experiments/audit_random_map_runtime_proof.py \
  --baseline records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/baseline_no_adapter.log \
  --adapter records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/random_map_adapter.log
```

The comparison helper reuses `experiments/verify_run.py`, fails if either log is not on the non-TTT stride-64 contract, and prints the signed `adapter_minus_baseline_bpb_delta`.

The audit helper is stricter: do not accept placeholder-backed proof. It rejects `preserved_windows_host_note`, `appended_contract_fixture`, and the preserved cmd.exe failure header; it also requires each saved log to contain the expected `random_map_adapter:enabled=...` config line, a stride-64 runtime line (`final_int6_sliding_window ... stride:64` or `final_int6_sliding_window_s64 ... stride:64`), and a `Total submission size int6+lzma:` line.

The current committed fixed logs now come from the learned-gate closeout rerun on RunPod. They prove the remote path and show that the learned-gate variant is a negative result relative to the baseline under this contract.

## Notes for future updates

- If `train_gpt_stack.py` or `train_gpt_random_map_adapter.py` adds or removes env vars, update this file and `experiments/runpod_guide.md` in the same change.
- Keep docs aligned with the verifier's metric precedence so operator guidance matches the actual success extraction logic.
