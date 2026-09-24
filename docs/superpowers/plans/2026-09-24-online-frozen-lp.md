# Online Frozen Linear Probe Implementation Plan

**Goal:** Replace one-shot augmented feature-cache LP with per-epoch augmented-image LP for pre/post-CP evaluation, while preserving training checkpoints and legacy results.

**Architecture:** A dedicated frozen evaluator forwards newly cropped/flipped images through the encoder every epoch, without encoder gradients or mutable training state. Shared metadata versions the new evaluation separately from CP training. Runners isolate new outputs and reuse compatible CP checkpoints.

**Constraints:** Keep all encoder readouts, pretrained input normalization, CP training recipes, kNN, geometry, GPU assignments and concurrency unchanged. LP keeps L2 features, Linear + cross entropy, Adam lr=0.001 and classifier batch=512; run 150 actual epochs, forwarding images in chunks of 32, rather than enforcing the old cached-feature 10000-step floor. Validation/test retain deterministic preprocessing. Never delete or overwrite legacy results or checkpoints for migration.

## Tasks

- [x] Add failing CPU tests for refreshed image access, frozen weights/buffers/gradients, tail batches, deterministic test preprocessing, metadata and legacy-result separation.
- [x] Implement `linear_probe_online_evaluate` and integrate `zero_shot_eval` without LP feature caching. Return `lp` protocol metadata with metrics.
- [x] Add dedicated weak LP transforms/loaders and CLI LP settings. Export stage-specific `pre_lp` / `post_lp` metadata. Keep CP/evaluation model state boundaries intact.
- [x] Version pre-CP outputs under `outputs/precp_full/lp_online_v1`; keep existing ImageNet reference banks. Version CP evaluation outputs under each seed's `lp_online_v1` directory while reusing the seed's `cp.ckpt` after immutable training-recipe validation.
- [x] Test missing, partial, compatible completed and incompatible checkpoint/config recovery; prevent mixing old/new baselines and final evaluations.
- [x] Document defaults, GPU cost, result locations, submission sequence and checkpoint reuse. Run full local tests, syntax/lint and dry-run commands, then independent review.

## Verification Commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q continued_pretraining.py stable_cp run tests
bash -n run/slurm/submit_cp_full.sh run/slurm/cp_full.sh run/slurm/precp.sh
.venv/bin/python run/precp.py run --root /tmp/cp-lp-dry-run --task-id 4 --encoder MAE-Mean --dry-run
.venv/bin/python run/cp_full.py run --root /tmp/cp-lp-dry-run --task-id 176 --dry-run
git diff --check
```

## Verification Outcome

56 CPU tests passed using Torch 2.10.0 and the pinned SPT checkout at 9aa93f8b.
The local environment contained iCloud-offloaded dependency files; tests used
exact-version wheels from the local uv cache without changing the installed
environment. A real tiny timm ViT smoke test exercised the MAE patch-mean readout
and confirmed unchanged encoder tensors. Both runner dry-runs, Python compilation,
shell syntax, and whitespace checks passed.

Independent review found advancing DataLoader generator/worker state between LP
calls. Fresh seeded loaders now isolate each evaluation; regression tests cover
shuffled calls with zero workers and two persistent workers. Re-review confirmed
identical sampling/views/metrics and an unchanged caller generator.

No production GPU training or real cluster checkpoint fixture was available.
Checkpoint tests include a real synthetic PyTorch checkpoint and strict restore
gate tests; they do not substitute for a cluster smoke run.
