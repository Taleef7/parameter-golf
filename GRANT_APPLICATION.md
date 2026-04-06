# OpenAI Compute Grant Application — Parameter Golf

## Submission Status

- **Status:** Submitted
- **Recorded for S06:** This repository now treats the compute-grant application as a submitted artifact, not a planned draft. The requested-compute narrative below is preserved as the original application text.
- **Historical framing note:** The competitive score ranges in the original application text are historical to the submission date and should not be read as current leaderboard state.

## Project Description

Parameter Golf is an open benchmarking challenge (openai/parameter-golf) that asks: how low can you push bits-per-byte on the FineWeb validation set within strict parameter budgets? I am competing in the **10-minute / 16 MB parameter** track, where the current public baseline sits at ~1.22 bpb and top entries approach ~1.16 bpb.

My goal is to make a meaningful, reproducible contribution to the public leaderboard by systematically evaluating under-explored architectural techniques and submitting the best result as a pull request.

## Technical Approach

I plan three phases:

1. **Ablation study** — evaluate 5+ techniques against the baseline: RoPE positional encodings, grouped-query attention, SwiGLU FFN, weight tying, and dynamic sparse attention. Each run is ~30 minutes on 1×H100. Estimated: 5 runs × 0.5 hr = **2.5 H100-hours**.

2. **Full validation** — combine the top 2–3 ablation winners and run 3 full validation sweeps. Each run is ~45 minutes. Estimated: 3 runs × 0.75 hr = **2.25 H100-hours**.

3. **Novel technique** — implement and evaluate an H-net tokenization layer or a state-space hybrid (Mamba-style recurrence in the MLP block), neither of which currently appears on the leaderboard. Estimated: 4 runs × 0.5 hr = **2 H100-hours**.

**Total requested: ~7 H100-hours** (budget to ~8 hr with overhead).

## Why These Choices

RoPE + GQA + SwiGLU are well-validated at scale but rarely co-optimized in the ultra-constrained (16 MB) regime. H-net tokenization is particularly interesting here because the parameter budget is so tight that spending fewer parameters on the embedding matrix could free capacity for deeper MLP layers — a trade-off untested in this benchmark.

## Expected Outcome

A pull request to the public `openai/parameter-golf` repository with:
- Reproducible training configuration
- val_bpb result competitive with or better than current top entries
- A novel technique not yet on the leaderboard, with clear documentation of what was learned

All runs will be logged and results archived for reproducibility.
