# Held-Out CP Implementation Plan

> **For agentic workers:** Use inline execution with test-driven development and verification before completion. Do not change existing experiments or submit cluster jobs locally.

**Goal:** Run DINOv3 and CLIP with LeJEPA, DIET, and SimCLR on eight held-out datasets, 1,000 distinct training images and seeds 42/43/44, on V100 GPUs.

**Architecture:** Register the eight stable-datasets readers through a fixed-split adapter. Separate GPU preparation (baseline evaluation and initial uniformity) from 48 grouped CP jobs. Reuse existing training and evaluation functions, with immutable manifests and per-seed provenance in a new output namespace.

**Tech Stack:** Python, stable-datasets, stable-pretraining, timm, PyTorch/Lightning, pytest, Slurm.

## Fixed Protocol

- OOD: BloodMNIST, TissueMNIST, AID, RESISC45. Fine-grained: StanfordDogs, JenaFlowers30, Flavia, IP102.
- Model-native RGB mean/std for training and both evaluations; preserve current 224-pixel augmentations and CLS readout.
- Official test splits where available. For AID/RESISC45/JenaFlowers30-all/Flavia, fixed seed-42 stratified 80/10/10 split. StanfordDogs validation uses 10% of its official training split, never its test split.
- Sample exactly 1,000 training images per seed, shared across models/methods and kNN/LP. Preserve the entire held-out test split.
- Initial geometry uses the existing 5,000-stratified/3,000-uniform protocol, restricted to the training pool, fixed before each dataset's CP.
- Training: 150 epochs, 15 frozen and 15 warmup, last two blocks, AdamW 1e-4, weight decay 0.05, 16-mixed precision. DIET batch 32; LeJEPA/SimCLR batch 256, no gradient accumulation.
- Three seeds serially in each of 48 V100 CP jobs; one array capped at 12, no FT. Each dataset's six CP jobs depend only on its own preparation element. The cluster QoS bounds overlapping preparation and CP jobs to 12 per user.
- Record raw pre/post scores, paired deltas, sample SD, actual counts, indices and model/data/code provenance. Never impute missing seeds or choose runs by score.

## Tasks

- [x] Add fixture tests for disjoint fixed splits, exact stratified sampling, and lazy indexed dataset access.
- [x] Implement the held-out data adapter and integrate with existing loaders, without changing legacy dataset behavior.
- [x] Add protocol tests for all 144 fits, normalization, recipes, validation, and immutable predictions.
- [x] Implement preparation, pre-training geometry freeze, fresh-weight CP execution, and result collection.
- [x] Add Slurm tests for V100, pinned interpreter, dependency, and array limits; implement submission scripts and instructions.
- [x] Run offline tests, existing regression tests, syntax checks, dry-run submission, and final diff review. Report unavailable GPU/download validation explicitly.

## Verification

- Full repository suite: 432 passed, 1 skipped; includes 47 held-out tests.
- Real shard-backed StableDataset fixture verifies numeric label access without image decoding.
- Concurrent prediction publication and image-content cache replacement have regression coverage.
- Ruff, shell syntax checks, and git whitespace checks passed.
- Launcher dry run from outside the repository produced 8 preparation jobs, one held 48-task V100 CP array capped at 12, all 48 dataset-local dependencies, and release after configuration.
- Independent code review findings were fixed and rechecked.
- No cluster submission, GPU training, or full dataset download was performed for this implementation.

## Dataset-Local Dependency Update

**Approved behavior:** Each dataset's six CP jobs become eligible when that dataset's preparation array element succeeds, without waiting for other datasets.

- [x] Add failing scheduler tests for all 48 task-to-preparation dependencies, held submission, release ordering, and failure without release.
- [x] Submit the existing CP array held, update each task's dependency to its own preparation element, then release only after all updates succeed. Preserve the single CP-array throttle; the cluster's nvidia MaxJobsPU=12 bounds overlapping preparation and CP jobs.
- [x] Replace global prediction publication with immutable dataset-local geometry records. Freeze each dataset before its CP starts, preserving the fixed hypothesis and recipe without requiring other datasets' results.
- [x] Update collection to validate completed datasets independently; partial results must remain readable when another dataset is unprepared.
- [x] Update README and run focused tests, full regression tests, shell checks, and a submission dry run. Do not submit or push.

**Verified:** Focused scheduler/protocol tests: 32 passed. Full suite: 432 passed, 1 skipped. Independent review: no blocking findings; all 47 held-out tests passed. Ruff, shell syntax, whitespace checks, and a launcher dry run passed. No cluster execution or push was performed.
