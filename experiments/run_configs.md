# Ablation Run Configs

All 8 ablation experiments for S02. Run each with:
```
python experiments/train_gpt_stack.py
```
with the listed env vars exported in the shell before running.

---

## 1. FullSOTA

**Config name:** FullSOTA  
**Env var overrides:** _(all defaults from train_gpt_stack.py)_  
**Expected bpb:** ~1.119  
**Description:** Run the SOTA script with no modifications — establishes the full-stack ceiling.  
**Run as:** `python experiments/train_gpt_stack.py`

---

## 2. WiderLonger

**Config name:** WiderLonger  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048`  
**Expected bpb:** ~1.155  
**Description:** Increase model depth, MLP width, and sequence length to test capacity scaling.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 python experiments/train_gpt_stack.py`

---

## 3. WiderLonger+SWA

**Config name:** WiderLonger+SWA  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04`  
**Expected bpb:** ~1.145  
**Description:** Adds Stochastic Weight Averaging and Muon weight decay on top of WiderLonger.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 python experiments/train_gpt_stack.py`

---

## 4. WiderLonger+SWA+XSA

**Config name:** WiderLonger+SWA+XSA  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4`  
**Expected bpb:** ~1.131  
**Description:** Adds Cross-Sequence Attention on the last 4 layers on top of WiderLonger+SWA.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 python experiments/train_gpt_stack.py`

---

## 5. +EMA+GPTQ

**Config name:** +EMA+GPTQ  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1`  
**Expected bpb:** ~1.123  
**Description:** Adds EMA averaging and GPTQ INT6 quantization-aware training on top of previous stack.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 python experiments/train_gpt_stack.py`

---

## 6. +LateQAT

**Config name:** +LateQAT  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15`  
**Expected bpb:** ~1.125  
**Description:** Adds late-stage quantization-aware training triggered at 15% loss threshold.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15 python experiments/train_gpt_stack.py`

---

## 7. +LeakyReLU2

**Config name:** +LeakyReLU2  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15 LEAKY_SLOPE=0.5`  
**Expected bpb:** ~1.119  
**Description:** Increases LeakyReLU slope to 0.5 for stronger negative gradient flow.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15 LEAKY_SLOPE=0.5 python experiments/train_gpt_stack.py`

---

## 8. +TTT

**Config name:** +TTT  
**Env var overrides:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15 LEAKY_SLOPE=0.5 TTT_ENABLED=1`  
**Expected bpb:** ~1.116  
**Description:** Adds Test-Time Training on top of the full stack — expected best result.  
**Run as:** `NUM_LAYERS=11 MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 SWA_ENABLED=1 MUON_WD=0.04 XSA_LAST_N=4 EMA_ENABLED=1 GPTQ_INT6=1 LATE_QAT_THRESHOLD=0.15 LEAKY_SLOPE=0.5 TTT_ENABLED=1 python experiments/train_gpt_stack.py`
