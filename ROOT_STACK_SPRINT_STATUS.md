# Root Stack Sprint Status

Date: 2026-04-07

Branch: `feature/root-stack-sprint`

Code checkpoint: `47f538f` (`feat: add root-safe stack sprint knobs`)

## Goal

Test Claude's recommendation in the safest form for this fork:

- keep the root `train_gpt.py` readable and baseline-oriented
- add only root-safe proven knobs first
- run a low-budget 1xH100 sweep
- package only if we find a clean, legal, reproducible win

## Implemented Root Knobs

The branch adds these optional root-script features:

- `MLP_NEGATIVE_SLOPE`
- `EMA_ENABLED`
- `EMA_DECAY`
- `EVAL_STRIDE`

Behavior:

- `MLP_NEGATIVE_SLOPE=0.0` preserves the baseline ReLU-squared MLP.
- `MLP_NEGATIVE_SLOPE=0.5` uses LeakyReLU(0.5)-squared.
- `EMA_ENABLED=1` maintains parameter EMA and applies it before final export/eval.
- `EVAL_STRIDE=64` adds an extra `final_int8_zlib_sliding_window[_exact]` final metric without renaming the existing roundtrip metrics.

Local focused verification:

```text
python -m pytest -q tests/test_root_stack_contract.py
7 passed
```

Additional checks:

```text
python -m py_compile train_gpt.py experiments/audit_ablation_evidence.py tests/test_root_stack_contract.py
pytest tests/test_record_stack_contract.py -k repository_root_baseline_remains_distinct_newcomer_entrypoint
1 passed
```

The full test suite was not rerun cleanly in this sandbox because old pytest temp directories in the worktree are owned by the sandbox user and produce `PermissionError` warnings. The focused tests for this change passed.

## RunPod Environment

Template:

- official RunPod `Parameter Golf`
- image: `runpod/parameter-golf:latest`
- GPU: `1x NVIDIA H100 80GB HBM3`

Common run contract:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
VAL_LOSS_EVERY=0
MAX_WALLCLOCK_SECONDS=600
SEED=42
```

Dataset:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

All pods created for this sprint were stopped/deleted after log recovery.

Final RunPod state after cleanup:

```text
clientBalance: 0.8895682767
currentSpendPerHr: 0
```

## Results

Lower `val_bpb` is better.

| Run | Change | Primary metric | val_bpb | Size bytes | Status |
|---|---|---:|---:|---:|---|
| `r0_control` | baseline root defaults | `final_int8_zlib_roundtrip_exact` | `1.34962573` | `13,076,892` | legal control |
| `r1_seq2048_lr` | `TRAIN_SEQ_LEN=2048`, LR retune | `final_int8_zlib_roundtrip_exact` | `1.32375678` | `11,568,165` | legal win |
| `r2_seq4096_lr` | `TRAIN_SEQ_LEN=4096`, LR retune | `final_int8_zlib_roundtrip_exact` | `1.30923066` | `11,639,971` | legal win |
| `r3_11l_mlp3x` | `NUM_LAYERS=11`, `MLP_MULT=3` | `final_int8_zlib_roundtrip_exact` | `1.29844295` | `17,077,044` | metric win, illegal size |
| `r4_11l_mlp3x_muon99_wd3000` | `r3` + Muon 0.99 warmup + `WARMDOWN_ITERS=3000` | `final_int8_zlib_roundtrip_exact` | `1.36502907` | `14,520,832` | legal regression |
| `r5_leaky_slope05` | `r2` + `MLP_NEGATIVE_SLOPE=0.5` | `final_int8_zlib_roundtrip_exact` | `1.30666091` | `11,460,359` | legal win |
| `r6_ema` | `r2` + `EMA_ENABLED=1` | `final_int8_zlib_roundtrip_exact` | `1.44569405` | `10,696,689` | legal regression |
| `r7_sliding_control` | `r2` + `EVAL_STRIDE=64` | `final_int8_zlib_sliding_window_exact` | `1.30349569` | `11,493,119` | best completed legal run |
| `r8_sliding_leaky_ema` | `r2` + LeakyReLU + EMA + stride-64 | `final_int8_zlib_roundtrip_exact` | `1.41370521` | `10,816,788` | interrupted before sliding; already worse |
| `r8b_sliding_leaky_no_ema` | `r2` + LeakyReLU + stride-64, EMA off | `final_int8_zlib_roundtrip_exact` | `1.30156080` | `11,643,685` | interrupted before sliding; promising but incomplete |

## Interpretation

Confirmed:

- The baseline can be beaten on 1xH100 with simple proven config changes.
- `TRAIN_SEQ_LEN=4096` plus lower LR is a strong legal improvement over root defaults.
- LeakyReLU(0.5)-squared is a small but real positive move under the same roundtrip metric.
- Sliding-window eval is a real metric improvement on the 4096-context control.

Rejected for this root path:

- 11 layers + 3x MLP improves BPB but exceeds the 16MB artifact cap under the root int8+zlib export.
- Muon 0.99 + long warmdown on top of the illegal 11L/3x config regresses badly in this 1xH100/10-shard setting.
- EMA is strongly harmful under the 10-minute 1xH100 contract. It also reduced compressed size, but metric quality collapsed.

Incomplete:

- `r8b_sliding_leaky_no_ema` is the best-looking next candidate, but RunPod stopped the pod before the stride-64 metric completed. Its roundtrip metric (`1.30156080`) is better than `r5` and `r2`, and the size is legal, but it is not a completed sliding-window proof.

## Current Recommendation

Do not package this as an upstream PR yet.

Reason:

- the best completed legal run is a config/systems stack (`r7`) rather than a novel clean submission package
- the likely best candidate (`r8b`) is incomplete because stride-64 eval did not finish
- there is no second-seed or 30-minute confirmation

If another `$5-7` is added, run exactly one next job:

```bash
RUN_ID=r8b_sliding_leaky_no_ema_retry \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=600 \
SEED=42 \
TRAIN_SEQ_LEN=4096 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
EVAL_STRIDE=64 \
MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Promotion gate:

- if the completed `final_int8_zlib_sliding_window_exact` beats `r7` by any amount and stays under `16,000,000` bytes, run one confirmation
- otherwise stop this root-safe path and escalate to a dedicated stack bridge script instead of adding more features to root `train_gpt.py`

## Local Log Locations

Logs are intentionally under ignored `logs/` paths for now:

```text
logs/runpod/root-stack-sprint/r0_control.log
logs/runpod/root-stack-sprint/r1_seq2048_lr.log
logs/runpod/root-stack-sprint/r2_seq4096_lr.log
logs/runpod/root-stack-sprint/r3_11l_mlp3x.log
logs/runpod/root-stack-sprint/r4_11l_mlp3x_muon99_wd3000.log
logs/runpod/root-stack-sprint/r5_leaky_slope05.log
logs/runpod/root-stack-sprint/r6_ema.log
logs/runpod/root-stack-sprint/r7_sliding_control.log
logs/runpod/root-stack-sprint/r8_sliding_leaky_ema.partial.log
logs/runpod/root-stack-sprint/r8b_sliding_leaky_no_ema.partial.log
```

If this becomes a submission package, move the selected control and candidate logs under a new `records/track_non_record_16mb/.../` folder with a README and `submission.json`.
