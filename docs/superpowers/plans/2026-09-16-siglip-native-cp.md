# SigLIP Native-Normalization CP Implementation Plan

**Goal:** Rerun all 15 SigLIP MAX targets with checkpoint-native mean/std, three CP objectives, and seeds 42/43/44, without baseline evaluation or FT.

**Approved design:** 63 jobs and 135 CP fits. OCTMNIST, PathMNIST, and Food-101 use one method/target/seed per job (27 jobs). The other 12 targets use one method/target per job with three sequential seeds (36 jobs). Preserve the existing main-rule recipes. All 2-block fits use V100; 4/6-block fits and full-backbone DIET/SimCLR use A100 without a memory constraint; full-backbone LeJEPA uses A100 80GB. Old outputs and checkpoints must never be read as recovery inputs or overwritten.

**Architecture:** Add an opt-in normalization mode to the unified training CLI, applied before any loader is built. Add an isolated CPU-plannable runner using existing manifest, locking, hashing, and atomic-publication helpers. Add Slurm wrappers for filtering datasets/objectives, staging a private cache, and dry-run verification.

## Work Items

- [x] Add failing CPU tests for normalization, exact task coverage, immutable recipes, isolated fresh starts, result validation, and Slurm submission.
- [x] Implement `--normalization-mode pretrained` in `continued_pretraining.py`; leave the default dataset normalization unchanged and record actual normalization/configuration in result JSON.
- [x] Implement `eval/siglip_native_cp.py` with isolated manifests, verified completion skipping, fresh attempt directories, post-only CP evaluation, and a result summary command.
- [x] Implement `run/slurm/cp-siglip/native-norm/{submit.sh,array.sh,README.md}` with dataset/method selection, per-task GPU resources, and safe cache staging.
- [x] Run focused and existing regression tests, shell syntax checks, a 63-job/135-command dry run, and inspect the final diff. Do not submit cluster jobs from the local workspace.

## Invariants

- Every planned fit has exactly one method, one target, and one seed.
- Official RGB mean and std are `[0.5, 0.5, 0.5]`; verify against the loaded checkpoint configuration, not the target dataset preset.
- Only mean/std changes. Input size, crop/augmentation recipe, pooling, epochs, depth, objective parameters, and precision stay unchanged.
- Skip pre-CP metrics because the native-normalization baseline audit is already available. Run the existing post-CP kNN/LP evaluation; do not enable either SFT flag.
- Completed same-protocol fits may be skipped only after checking result identity, normalization, configuration, checksum, and checkpoint. Incomplete attempts restart from public weights in a new directory.
- Preserve failed attempts; never filter completed results based on their score.
- The training configuration and actual normalization are written into result JSON and checked before publication.

## Verification Commands

```bash
python3 -m pytest tests/test_cp_normalization.py tests/test_siglip_native_cp.py tests/test_siglip_native_cp_slurm.py -q
bash -n run/slurm/cp-siglip/native-norm/submit.sh
bash -n run/slurm/cp-siglip/native-norm/array.sh
SIGLIP_NATIVE_OUTPUT_BASE=/tmp/siglip-native-check bash run/slurm/cp-siglip/native-norm/submit.sh --dry-run
```

Actual CUDA training and Slurm scheduling require the cluster. Local tests must not claim GPU verification.

## Initial Verification Results

- Focused protocol/normalization/Slurm tests: 49 passed.
- Full `tests/` suite: 321 passed, 1 skipped; one existing FT checkpoint warning.
- Both shell entrypoints passed `bash -n`; Python compilation passed.
- Full dry-run verified 63 distinct job IDs and 135 unique method/target/seed commands,
  native normalization, no FT flags, no checkpoint resume, and no training artifacts.
- Mock execution covered sequential seeds, retries, locks, completion checksums,
  result collection, Slurm submission, and private-cache cleanup.
- No cluster jobs submitted; no GPU training executed locally.

## GPU Routing Revision

- [x] Update protocol and Slurm tests for 24 V100, 30 A100, and 9 A100 80GB jobs; keep 135 fits and all recipes unchanged.
- [x] Add immutable task GPU profiles in `eval/siglip_native_cp.py`, profile-aware CUDA checks, and resource-specific submission groups.
- [x] Update `submit.sh` to submit up to three arrays and `array.sh` to validate its manifest's GPU profile. Only the 80GB group may carry `--constraint=80g`.
- [x] Keep `--concurrency` as a total cap: divide slots across active groups; if there are fewer slots than groups, chain arrays with `afterany` dependencies.
- [x] Update the README; rerun focused tests, the full suite, and shell syntax checks. No cluster submission.

Revision verification: 75 focused tests passed; full suite 347 passed, 1 skipped,
with the existing FT checkpoint warning. Both shell syntax checks passed. The
full dry-run covered 24/30/9 resource-group jobs, 135 distinct fits, and a shared
12-job concurrency cap. V100 peak memory remains untested; no GPU training or
cluster submission was performed.
