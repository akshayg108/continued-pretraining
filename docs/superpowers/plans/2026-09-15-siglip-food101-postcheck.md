# SigLIP-2 Food-101 Checkpoint Audit

**Goal:** Evaluate existing Food-101 CP checkpoints with the verified official input normalization, without updating encoder weights or changing archived results.

**Architecture:** A CPU-only planner resolves exact mainrule or recheck checkpoint/result pairs. A V100 array runs one frozen kNN/PyTorch-LP evaluation per pair. The evaluator reuses the current shared loaders and strict backbone loader. Input hashes bind each result to its checkpoint. Different training attempts remain separate.

**Constraints:** Food-101 only, SigLIP-2 B/16 MAP features, mean/std 0.5, full training split, k=20, unchanged LP recipe, no CP or FT training, no checkpoint fallback, no manuscript changes. Missing artifacts are listed rather than replaced. Complete low-scoring runs are included. All new outputs use a separate namespace.

## Implementation

- [x] Add failing tests for checkpoint discovery, provenance, source separation, frozen evaluation, and V100 launch/staging.
- [x] Implement `eval/siglip_food101_postcheck.py` with `plan` and `run` commands. Use `eval.full_ft.checkpoint.load_backbone_state` only for strict weight loading, never its training runner.
- [x] Add `run/slurm/cp-siglip/post-cp/submit_food101.sh` and `food101_v100.sh`. Preserve the existing baseline launcher.
- [x] Run the new CPU tests, baseline regression tests, shell syntax checks, and strict checkpoint-loader tests. All 127 selected tests passed. A real CPU construction with timm 1.0.25 confirmed official normalization and attention pooling.
- [x] Document cluster commands and output paths. GPU evaluation and checkpoint availability require the cluster and are not locally verified.
