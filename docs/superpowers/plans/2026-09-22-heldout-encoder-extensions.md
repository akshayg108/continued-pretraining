# Held-Out Encoder Extensions Implementation Plan

**Goal:** Add SigLIP-2 and DINOv3-L CP on the eight held-out targets without changing the running DINOv3-B/CLIP implementation.

**Architecture:** Add `eval/heldout_extensions` and `run/slurm/heldout-extensions`. Reuse immutable data, metric, optimizer, and training helpers. Read the original manifest and verified preparation artifacts to pin the same training indices, partitions, and geometry samples. Publish new baselines, predictions, attempts, results, and reports in `heldout_extensions_1000_v1`.

**Tech Stack:** Python, pytest, timm, PyTorch/Lightning, stable-datasets, Bash, Slurm.

## Constraints

- Eight existing targets; three objectives; seeds 42, 43, 44 run serially in each CP job; exactly 1,000 distinct training images; no FT.
- SigLIP-2: `vit_base_patch16_siglip_224.v2_webli`, 768-dimensional MAP readout, mean/std 0.5, existing base-model recipes. Preparation/DIET/SimCLR use V100; LeJEPA uses A100.
- DINOv3-L: `vit_large_patch16_dinov3.lvd1689m`, 1,024-dimensional CLS readout, ImageNet mean/std. Per the approved update, match the base-model held-out recipes: LeJEPA/SimCLR batch 256 and accumulation 1; DIET batch 32. Older ViT-L scripts remain unchanged. All jobs use ordinary A100, without an 80GB constraint.
- Preserve existing 224-pixel transforms, 150 epochs, 15 frozen epochs, two trainable blocks, and kNN/LP protocols.
- Each encoder-target preparation freezes its three baselines and geometry before its three CP jobs can start. No global preparation barrier.
- Do not modify files hashed by `eval.heldout_cp.protocol.implementation_sha256` or the roundoff recovery wrapper.
- No automatic submission or cancellation. The user authorized committing and pushing the extension after local verification.

## Tasks

- [x] Protocol: first test the 48-job/144-fit grid, native normalization, encoder-specific pooling/dimensions/recipes, source-artifact pinning, and prediction immutability; then implement `protocol.py`.
- [x] Runtime: test model/config validation, exact source indices, feature dimensions/readouts, numerical roundoff, and serial fresh-process seeds; then implement `runtime.py`, `prepare.py`, and `run.py`.
- [x] Interface and collection: test CPU-only plan/dry-run, paired mean/sample SD, and incomplete-result handling; implement `__main__.py` and `collect.py`.
- [x] Slurm: test resource grouping, encoder-target dependencies, no premature release on failure, and pinned Python; implement `submit.py`, `submit.sh`, and `worker.sh`.
- [x] Document preparation, submission, monitoring, collection, resource requirements, and the distinction between native mean/std and full checkpoint-default transforms.
- [x] Run focused tests, existing held-out/native/ViT-L regression tests, shell syntax, dry-run, and `git diff --check`. Confirm the original implementation hash remains unchanged.

## Verification Commands

```bash
python3 -m pytest -q tests/test_heldout_extensions.py tests/test_heldout_extensions_runtime.py tests/test_heldout_extensions_slurm.py
python3 -m pytest -q tests
bash -n run/slurm/heldout-extensions/submit.sh
bash -n run/slurm/heldout-extensions/worker.sh
git diff --check
```

GPU execution must be verified on the cluster; local tests do not establish that every A100 memory capacity can run the fixed recipe.

## Verification Results

- Extension tests: 32 passed, including synthetic end-to-end preparation/training/collection and scheduler failure cases.
- Full `tests` directory: 510 passed, 1 skipped; one existing FT-checkpoint warning.
- Both shell entrypoints pass `bash -n`; the CLI loads without GPU dependencies.
- Original held-out implementation hash remains `3d181c4845e2d0e0278cc508e312ff12ce5f402ada8e27edb569fadcc95e3ade`.
- Independent read-only review found no actionable correctness issue.
- No cluster jobs were submitted, no original jobs were cancelled, and no GPU training was run locally.
