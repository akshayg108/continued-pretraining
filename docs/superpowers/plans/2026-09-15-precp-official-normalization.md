# Pre-CP Official-Normalization Audit

**Goal:** Prepare four V100 array jobs, one per encoder, for frozen MAX-budget
kNN, PyTorch LP, and pre-CP uniformity with checkpoint-native RGB mean/std.

**Scope:** Keep the existing model IDs, readouts, 224-pixel transforms, data
splits, k=20, and LP settings. Never train CP or full FT, load CP checkpoints,
modify shared dataset caches, overwrite old results, or edit the manuscript.
SigLIP covers the other fourteen targets with seeds 42/43/44. Food-101 already
has the native-normalization baseline, so compute only its geometry at seed 42.
CLIP, DINOv3 ViT-B, and MAE cover all fifteen targets with seeds 42/43/44.

**Files:**
- `eval/precp_official_norm.py`: lightweight planning, per-seed evaluation,
  bounded-memory exact uniformity, provenance, and result summaries.
- `run/slurm/pre-cp-official/submit.sh`: exactly four V100 array elements.
- `run/slurm/pre-cp-official/array.sh`: sequential evaluation and per-dataset
  private node-local staging, with failure isolation and cleanup.
- `run/slurm/pre-cp-official/README.md`: launch commands and metric semantics.
- `tests/test_precp_official_norm.py`: CPU tests and shell integration tests.

## Implementation Checklist

- [x] Add failing tests for four-encoder coverage, native normalization checks,
  MAX counts, no baseline repetition for SigLIP Food-101, and pure dry runs.
- [x] Implement the planner and isolated evaluator using the existing shared
  loaders and frozen-evaluation functions. Record exact model, transforms,
  package versions, source hashes, split fingerprints, and seed-level metrics.
- [x] Compute uniformity at t=2 over all distinct pairs of clean MAX training
  features using bounded-size GPU blocks. Also emit a separate 5000-to-3000
  subsampled estimate for comparison with the historical estimator. Do not
  imply identical sample membership across old and new geometry pipelines.
- [x] Implement four V100 jobs with three sequential seeds per target, staged
  caches, explicit failures, no overwrites, and an isolated output namespace.
- [x] Run focused and related regression tests, shell syntax checks, and all
  four dry runs. CUDA evaluation remains a cluster-side verification step.

## Verification Contract

Use `python3 -m pytest -q tests/test_precp_official_norm.py` for the new tests.
Compare blocked uniformity to a dense NumPy upper-triangle calculation on small
synthetic features. Exercise the runner with stubbed data/model boundaries while
checking dispatch to the existing evaluators. Exercise staging cleanup and
nonzero failure propagation without a GPU. Validate that dry runs load neither
CUDA nor the cluster environment and create no output directories.

The four array elements must plan 177 frozen-evaluation records and one
geometry-only record. Each successful record is written atomically without
replacement. Missing or invalid records remain visible in the summary and
never become zero-valued measurements.

## Verification Results

- New tests and the existing SigLIP Food-101 precheck/postcheck suites: 71 passed.
- All four dry runs passed without submitting jobs or creating result folders.
- Both shell scripts passed individual `bash -n` checks. Python compilation passed.
- Installed timm 1.0.25 checkpoint metadata matched all four expected mean/std pairs.
- CPU smoke tests using real timm architectures with random weights passed for
  all four readouts, with finite `(2, 768)` feature arrays and finite uniformity.
- Full shared-pipeline execution was not tested locally because Lightning is
  unavailable. No CUDA evaluation, cluster submission, commit, or push was performed.
