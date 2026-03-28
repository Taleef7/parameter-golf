[Skip to content](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) to refresh your session.Dismiss alert

{{ message }}

[openai](https://github.com/openai)/ **[parameter-golf](https://github.com/openai/parameter-golf)** Public

- [Notifications](https://github.com/login?return_to=%2Fopenai%2Fparameter-golf) You must be signed in to change notification settings
- [Fork\\
2.7k](https://github.com/login?return_to=%2Fopenai%2Fparameter-golf)
- [Star\\
4.4k](https://github.com/login?return_to=%2Fopenai%2Fparameter-golf)


## Collapse file tree

## Files

main

Search this repository(forward slash)` forward slash/`

/

# README.md

Copy path

BlameMore file actions

BlameMore file actions

## Latest commit

[![0hq](https://avatars.githubusercontent.com/u/30643741?v=4&size=40)](https://github.com/0hq)[0hq](https://github.com/openai/parameter-golf/commits?author=0hq)

[Launch snapshot](https://github.com/openai/parameter-golf/commit/a15093adad328a650d421e53c078cbd2c45beb0e)

last weekMar 18, 2026

[a15093a](https://github.com/openai/parameter-golf/commit/a15093adad328a650d421e53c078cbd2c45beb0e) · last weekMar 18, 2026

## History

[History](https://github.com/openai/parameter-golf/commits/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md)

Open commit details

[View commit history for this file.](https://github.com/openai/parameter-golf/commits/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) History

46 lines (40 loc) · 1.78 KB

/

# README.md

Top

## File metadata and controls

- Preview

- Code

- Blame


46 lines (40 loc) · 1.78 KB

[Raw](https://github.com/openai/parameter-golf/raw/refs/heads/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md)

Copy raw file

Download raw file

Edit and raw actions

This record captures the `Simple Baseline`.

Trainer changes in this snapshot:

- current repository `train_gpt.py` snapshot copied into the record folder
- published `fineweb10B_sp1024` dataset and tokenizer loaded from the new Hugging Face export
- 10-minute wallclock cap on `8xH100`
- periodic validation every `200` steps on the full `fineweb_val_*` split

Configuration:

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):

```
NCCL_IB_DISABLE=1 \
RUN_ID=hf_verify_sp1024_8gpu \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):

- Timed training stopped at `13780/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0606`, `val_bpb:1.2172`
- Post-quant roundtrip eval: `val_loss:2.0727`, `val_bpb:1.2244`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
- Train time: `600038ms` (`step_avg:43.54ms`)
- Peak memory: `10184 MiB allocated`, `10200 MiB reserved`
- Serialized model int8+zlib: `15815847 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `15863489 bytes`

Training volume:

- Global batch: `524288` tokens/step
- Total train tokens seen: `7224688640`

Included files:

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)

You can’t perform that action at this time.