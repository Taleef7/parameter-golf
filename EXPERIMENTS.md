# Ablation Experiments — Parameter Golf M001 S02

Tracking table for all ablation runs on 1×H100. Baseline from `logs/baseline_sp1024.txt`.

| Run | Config | Key Env Vars | val_bpb | Notes |
|-----|--------|--------------|---------|-------|
| 0 | NaiveBaseline | none | 1.2244 | H100 reference from logs/baseline_sp1024.txt |
| 1 | 2026-03-28_StackIntegration_LegalTTT_ParallelMuon | `MAX_WALLCLOCK_SECONDS=600`, `ITERATIONS=9000`, `EVAL_STRIDE=64`, `TTT_ENABLED=1` | blocked | Canonical S03 record folder. Local executor preserved `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log`, which exits on a `ModuleNotFoundError` before any accepted metric or artifact-size lines are emitted; no BPB is claimed yet. |
