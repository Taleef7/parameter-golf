# Stack Integration + Legal TTT + Parallel Muon

**Promoted S03 record artifact:** `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py`

**Canonical metric:** `legal_ttt` when `TTT_ENABLED=1` | **Fallback metric:** `final_int6_sliding_window_s64` when `TTT_ENABLED=0` | **Artifact boundary:** `< 16 MB`

This folder is the reproducible S03 home for the integrated stack. It promotes `experiments/train_gpt_stack.py` into a concrete record artifact without replacing the repository-root `train_gpt.py` newcomer baseline.

## Integrated stack

The promoted `train_gpt.py` carries the S03 stack exactly as copied from `experiments/train_gpt_stack.py`:

- parameter banking with Parallel Muon
- LeakyReLU(0.5)^2 MLP
- BigramHash + partial RoPE + XSA last-4
- VE128 on layers 9-10
- int6 + lzma export path
- legal score-first TTT support
- sliding-window evaluation with stride-64 fallback metric

## Canonical 1×H100 run contract

Run from the repository root so the default dataset/tokenizer paths resolve as documented elsewhere in the repo.

### Preferred proof run (`TTT_ENABLED=1` → `legal_ttt`)

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
python records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py \
  > records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log 2>&1
python experiments/verify_run.py \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log
```

Expected verifier behavior:

- prints `chosen_metric: legal_ttt`
- prints `val_bpb: <value>`
- exits non-zero if the log does not contain an accepted S03 metric

### Fallback proof run (`TTT_ENABLED=0` → `final_int6_sliding_window_s64`)

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
EVAL_STRIDE=64 \
TTT_ENABLED=0 \
python records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py \
  > records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log 2>&1
python experiments/verify_run.py \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log
```

Expected verifier behavior:

- prints `chosen_metric: final_int6_sliding_window_s64`
- prints `val_bpb: <value>`
- treats `legal_ttt` as preferred if both metrics appear in the same log

## Artifact-size and proof-log expectations

The promoted script logs the compressed export size as:

- `Serialized model int6+lzma: ... bytes`
- `Total submission size int6+lzma: ... bytes`

S03 treats the run as acceptable only if the recorded total submission size stays below `16,000,000` bytes (`< 16 MB`).

The accepted proof log for this promoted artifact lives at:

- `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log`

T02 creates the folder contract and T03 is responsible for placing the accepted proof log there.

## Current execution status

The promoted artifact is byte-identical to the already-proven stack under:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

To keep S03's canonical record folder self-contained, the accepted proof log from that identical artifact was copied into:

- `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log`

Measured result from the accepted log:

- `chosen_metric: legal_ttt`
- `val_bpb: 1.1189`
- `Total submission size int6+lzma: 15990006 bytes`
- source run: 8×H100 SXM, `SEED=2025`

The blocked local Windows proof attempt is still preserved separately for diagnosis at:

- `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_blocked_local.log`

That local attempt fails during interpreter startup with `ModuleNotFoundError: No module named 'flash_attn_interface'`, so this workspace still does not satisfy the Linux/CUDA runtime dependency contract for rerunning the stack locally.

## Relationship to other entrypoints

- `train_gpt.py` at the repository root remains the readable newcomer baseline and must stay untouched.
- `experiments/train_gpt_stack.py` remains the editable source-of-truth staging script for the integrated stack.
- `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py` is the promoted reproducible record artifact for downstream S03 work.
