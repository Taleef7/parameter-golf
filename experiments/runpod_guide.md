# RunPod Execution Guide — S02 Ablation Experiments

**Platform:** RunPod  
**Pod type:** 1× H100 SXM5 80 GB  
**Docker image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`  
**Disk:** 50 GB  
**Expected runtime per run:** ~30–45 min on 1× H100 (wallclock cap 10 h max; typical full run ~2.5 h; baseline stopped at ~10 min/wallclock for demo)

> **⚠️ Linux/RunPod only.** The Flash Attention 3 import in `train_gpt_stack.py` will fail on Windows or macOS. Run all experiments on the RunPod pod, not locally.

---

## 1. Pod Setup

After SSH-ing into the pod:

```bash
# 1. Clone / upload repo
cd /workspace
git clone <your-repo-url> parameter-golf
cd parameter-golf

# 2. Install missing Python dependency
pip install zstandard

# 3. Confirm dataset path (must exist before any run)
ls /workspace/data/datasets/fineweb10B_sp1024/
# If missing, create a symlink to wherever the dataset lives:
# ln -s /path/to/dataset /workspace/data/datasets/fineweb10B_sp1024

# 4. Create log directory
mkdir -p logs
```

---

## 2. Global Env Vars (set once per session)

```bash
export MAX_WALLCLOCK_SECONDS=72000   # 20 h cap — ensures a full run completes
export SWA_EVERY=50                  # default; set explicitly to avoid surprises
```

> **Note:** `TTT_ENABLED=1` only affects eval time, not training — disable it for speed during ablations and re-enable only for the final S03 stack run.

---

## 3. Ablation Runs

Run each block in order. Each block exports env vars, launches training with stdout+stderr redirected to a log file, and then extracts the final `val_bpb`.

---

### Run 1 — FullSOTA (baseline ceiling)

```bash
python experiments/train_gpt_stack.py \
  > logs/run_fullsota.txt 2>&1
python experiments/verify_run.py logs/run_fullsota.txt
# Expected val_bpb: ~1.119
```

---

### Run 2 — WiderLonger

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  python experiments/train_gpt_stack.py \
  > logs/run_widerlonger.txt 2>&1
python experiments/verify_run.py logs/run_widerlonger.txt
# Expected val_bpb: ~1.155
```

---

### Run 3 — WiderLonger+SWA

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  python experiments/train_gpt_stack.py \
  > logs/run_widerlonger_swa.txt 2>&1
python experiments/verify_run.py logs/run_widerlonger_swa.txt
# Expected val_bpb: ~1.145
```

---

### Run 4 — WiderLonger+SWA+XSA

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  XSA_LAST_N=4 \
  python experiments/train_gpt_stack.py \
  > logs/run_widerlonger_swa_xsa.txt 2>&1
python experiments/verify_run.py logs/run_widerlonger_swa_xsa.txt
# Expected val_bpb: ~1.131
```

---

### Run 5 — +EMA+GPTQ

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  XSA_LAST_N=4 \
  EMA_ENABLED=1 GPTQ_INT6=1 \
  python experiments/train_gpt_stack.py \
  > logs/run_ema_gptq.txt 2>&1
python experiments/verify_run.py logs/run_ema_gptq.txt
# Expected val_bpb: ~1.123
```

---

### Run 6 — +LateQAT

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  XSA_LAST_N=4 \
  EMA_ENABLED=1 GPTQ_INT6=1 \
  LATE_QAT_THRESHOLD=0.15 \
  python experiments/train_gpt_stack.py \
  > logs/run_lateqat.txt 2>&1
python experiments/verify_run.py logs/run_lateqat.txt
# Expected val_bpb: ~1.125
```

---

### Run 7 — +LeakyReLU2

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  XSA_LAST_N=4 \
  EMA_ENABLED=1 GPTQ_INT6=1 \
  LATE_QAT_THRESHOLD=0.15 \
  LEAKY_SLOPE=0.5 \
  python experiments/train_gpt_stack.py \
  > logs/run_leakyrelu2.txt 2>&1
python experiments/verify_run.py logs/run_leakyrelu2.txt
# Expected val_bpb: ~1.119
```

---

### Run 8 — +TTT (full stack)

```bash
NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 MUON_WD=0.04 \
  XSA_LAST_N=4 \
  EMA_ENABLED=1 GPTQ_INT6=1 \
  LATE_QAT_THRESHOLD=0.15 \
  LEAKY_SLOPE=0.5 \
  TTT_ENABLED=1 \
  python experiments/train_gpt_stack.py \
  > logs/run_ttt.txt 2>&1
python experiments/verify_run.py logs/run_ttt.txt
# Expected val_bpb: ~1.116
```

---

## 4. Recording Results in EXPERIMENTS.md

After each run, append a row to `EXPERIMENTS.md`:

```
| N | <config name> | <key env vars> | <bpb from verify_run.py> | <notes> |
```

Example after Run 1:
```
| 2 | FullSOTA | (all defaults) | 1.119 | Full SOTA stack ceiling |
```

Row numbering: row 1 is the pre-existing baseline entry; new runs start at row 2.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: zstandard` | `pip install zstandard` |
| `FileNotFoundError: fineweb10B_sp1024` | Check dataset path; create symlink |
| `ImportError: flash_attn` | Must run on Linux/CUDA pod, not Windows |
| `verify_run.py` prints no `val_bpb` | Run likely crashed early; check log tail for error |
| Pod OOM | Reduce `TRAIN_SEQ_LEN` to 1024 or `NUM_LAYERS` to 8 |
