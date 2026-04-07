# Root-Safe LeakyReLU + Sliding Eval (1xH100 Non-Record)

This folder packages a non-record root-script experiment from `feature/root-stack-sprint`.

It is not an 8xH100 record-track submission. It is a 1xH100, 10-train-shard evidence package showing that a small root-safe stack can beat the matched root baseline while staying under the 16MB artifact cap.

## Summary

The experiment keeps the root `train_gpt.py` baseline structure and adds only two active changes for the winning path:

- longer context and lower learning rates: `TRAIN_SEQ_LEN=4096`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `TIED_EMBED_LR=0.03`
- LeakyReLU-squared MLP plus stride-64 sliding-window final evaluation: `MLP_NEGATIVE_SLOPE=0.5`, `EVAL_STRIDE=64`

EMA was tested separately and rejected for this setting. The 11-layer / 3x-MLP config improved BPB but exceeded the 16MB artifact cap under the root int8+zlib export path, so it is not used here.

## Environment

- Template: official RunPod `Parameter Golf`
- Image: `runpod/parameter-golf:latest`
- GPU: `1x NVIDIA H100 80GB HBM3`
- Branch: `feature/root-stack-sprint`
- Code checkpoint: `bc10936`
- Dataset command: `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`

Common data/tokenizer settings:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
VAL_LOSS_EVERY=0
SEED=42
```

## Evidence Files

- `train_gpt.py` - exact script snapshot used for the run package
- `control_r7_sliding_control.log` - matched 600s stride-64 control
- `r8b_sliding_leaky_no_ema_retry.log` - 600s LeakyReLU + stride-64 retry
- `r9_sliding_leaky_no_ema_1800.log` - 1800s LeakyReLU + stride-64 confirmation
- `submission.json` - non-record package metadata

## Results

Lower `val_bpb` is better.

| Run | Wallclock cap | Active change | Metric | val_bpb | Total bytes | Interpretation |
|---|---:|---|---|---:|---:|---|
| `control_r7_sliding_control` | 600s | `TRAIN_SEQ_LEN=4096`, LR retune, `EVAL_STRIDE=64` | `final_int8_zlib_sliding_window_exact` | `1.30349569` | `11,493,119` | matched completed control |
| `r8b_sliding_leaky_no_ema_retry` | 600s | control + `MLP_NEGATIVE_SLOPE=0.5` | `final_int8_zlib_sliding_window_exact` | `1.29661412` | `11,480,224` | matched 600s win |
| `r9_sliding_leaky_no_ema_1800` | 1800s | same winning config, longer run | `final_int8_zlib_sliding_window_exact` | `1.21339931` | `14,318,706` | non-record confirmation |

The matched 600s LeakyReLU + stride-64 run improves over the matched stride-64 control by:

```text
1.29661412 - 1.30349569 = -0.00688157 bpb
```

The 1800s confirmation remains under the 16MB cap and reaches:

```text
final_int8_zlib_sliding_window_exact val_loss:2.04877391 val_bpb:1.21339931
Total submission size int8+zlib: 14318706 bytes
```

## Exact 600s Winning Command

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

## Exact 1800s Confirmation Command

```bash
RUN_ID=r9_sliding_leaky_no_ema_1800 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=1800 \
SEED=42 \
TRAIN_SEQ_LEN=4096 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
EVAL_STRIDE=64 \
MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## PR Guidance

This is submission-shaped evidence, but it should be treated as a non-record package only.

Do not present the 1800s result as a record-track result. It is a useful 1xH100 sign-of-life for the root-safe stack and a basis for a future 8xH100 or dedicated-stack experiment if more budget is available.
