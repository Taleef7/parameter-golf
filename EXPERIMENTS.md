# Ablation Experiments — Parameter Golf M001 S02

Tracking table for all ablation runs on 1×H100. Baseline from `logs/baseline_sp1024.txt`.

| Run | Config | Key Env Vars | val_bpb | Notes |
|-----|--------|--------------|---------|-------|
| 0 | NaiveBaseline | none | 1.2244 | H100 reference from logs/baseline_sp1024.txt |
| 1 | 2026-03-28_StackIntegration_LegalTTT_ParallelMuon | `MAX_WALLCLOCK_SECONDS=600`, `ITERATIONS=9000`, `EVAL_STRIDE=64`, `TTT_ENABLED=1` | 1.1189 | Canonical S03 record folder. `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log` is the accepted proof log, copied from the byte-identical 2026-03-23 precursor artifact's 8×H100 `SEED=2025` run so the promoted record folder stays self-contained. Verifier output: `chosen_metric: legal_ttt`, `val_bpb: 1.1189`; artifact size line: `Total submission size int6+lzma: 15990006 bytes`. Local Windows import failure evidence remains preserved as `train_blocked_local.log`. |
