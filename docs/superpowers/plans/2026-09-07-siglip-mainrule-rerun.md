# SigLIP main-grid unfreezing rerun

## Goal

Run a separate, auditable MAX-budget SigLIP-2 pass with the main-grid CP
unfreezing schedule, followed by full-parameter post-CP FT. The user's request
approves this design. Do not change the old frozen two-block protocol or its
outputs. Work in the existing dirty worktree and do not commit unrelated work.

## Global constraints

- Exactly 29 grouped tasks: LeJEPA and SimCLR on galaxy10, eurosat,
  organamnist, plant_village, food101, pathmnist, octmnist; DIET on all 15.
- Each task runs seeds 42, 43, 44 sequentially, CP then full FT per seed.
- CP: 150 epochs, 15 frozen epochs, AdamW 1e-4 / 0.05, warmup 15.
- After warmup: n < 10000 -> 2 blocks; n <= 25000 -> 4; n <= 50000 -> 6;
  otherwise -1 (all parameters). Use native SigLIP-2 MAP pooling.
- LeJEPA effective batch 256, accumulation 1/2/2/4 for depths 2/4/6/all;
  SimCLR batch 256, accumulation 1; DIET batch 32, accumulation 1.
- Existing objective-specific recipes are unchanged. No pre-CP evaluation
  rerun. CP checkpoints are saved; FT checkpoints are never saved.
- New namespace: <output-base>/siglip_mainrule_v1, default base
  /scratch/gs4133/zhd/CP/outputs. No fallback to old checkpoints/results.
- Resume must verify recipe identity and completed CP checkpoint digest;
  outputs without a matching protocol receipt must not be silently accepted.
- Reuse eval/full_ft/run.py for post-CP FT. Its trainable/total parameter gate
  and no-checkpoint trainer remain mandatory.
- One submission command, array 0-28%12, A100, 96 GB RAM, 8 CPUs, 96h/task.
  Stage each dataset once to private node-local storage for the three seeds.
- No real job submission from this machine. Verify CPU tests and dry runs;
  report the lack of an actual A100/cluster training test.

## Task 1: Protocol and grouped CP/FT runner

Owner: root. Files: eval/siglip_mainrule/{__init__,protocol,run}.py,
tests/test_siglip_mainrule.py.

Use torch-free metadata from eval/full_ft/manifest.py. Freeze exactly 29
method/dataset tasks, ordered by descending sample count then method. Each
task is compatible with the full-FT manifest schema, plus cp_recipe. Manifest
contains schema_version=1, protocol, output_root, tasks. Load validates the
entire canonical manifest, not merely row counts.

Public CLI for shell task:

```
python -m eval.siglip_mainrule.protocol --output-base BASE --output MANIFEST
python -m eval.siglip_mainrule.run --manifest MANIFEST --task-id ID \
  --cache-dir CACHE [--num-workers 8] [--dry-run]
```

Python protocol interface: load_manifest(path) returns the document;
document['tasks'][i]['processed_subpath'] is the dataset staging path.
Default output base is /scratch/gs4133/zhd/CP/outputs. The runner always runs
all three seeds, on CUDA, and uses subprocesses to release CP memory before FT.
Dry run prints all six CP/FT commands without loading ML libraries or writing
training artifacts. FT subprocess uses the same manifest with
eval/full_ft/run.py and one seed at a time.

Write failing tests first for exact coverage/depth/batch recipes, isolation,
command arguments, sequential ordering, CP/FT resume and fail-closed checks.
Implement, rerun, and review.

## Task 2: Cluster submission and dataset staging

Owner: shell worker. Files exclusively:
run/slurm/cp-siglip/mainrule/{array,submit}.sh,
tests/test_siglip_mainrule_slurm.py.

Use existing run/slurm/full_ft/{array,submit}.sh as conventions. New submit
builds a unique manifest through Task 1's CLI, submits exactly one array with
29 elements, and supports --dry-run. Default concurrency is 12; optionally
allow --concurrency 1..12. No flags may alter the frozen CP recipe.
Environment variables use SIGLIP_MAINRULE_ prefix for REPO_ROOT, OUTPUT_BASE,
MANIFEST, CACHE_DIR, LOG_DIR and SKIP_ENV_SETUP. PYTHON override supported.

Array reads manifest/task identity, stages the one processed_subpath once to
a private directory with capacity checks/cleanup trap, and calls Task 1's
runner. Respect SLURM_SUBMIT_DIR for spool safety. Conda/module initialization
before strict mode, but all substantive operations under set -euo pipefail.
CUDA must be asserted for real runs. --dry-run must avoid sbatch, CUDA,
environment activation and staging writes, and show the 3-seed plan.

TDD: fixed resource/count/CLI tests, mocked sbatch, spool-safe repo lookup,
dry-run sequence, bad args, no old artifact paths. Run bash -n. Do not edit
shared helpers or old scripts. You are not alone in the codebase; do not
revert others' edits. No commit. Report tests and concerns.

## Task 3: Integration, regression and handoff

Owner: root and independent reviewer. Add eval/siglip_mainrule/README.md with
dated protocol amendment, command, outputs, recovery and interpretation scope.
Run new tests, full-FT regression, old SigLIP DIET protocol regression, syntax
and compile checks, and all 29 dry-run tasks. Review cross-process checkpoint
and full-unfreezing boundaries. Preserve old preregistration/data unchanged.

## Progress

- [x] Audited main-grid recipes, affected cells, existing full-FT implementation.
- [x] Task 1: TDD protocol and runner (26 tests passed, including CUDA guard and strict numeric schema).
- [x] Task 2: TDD scripts and shell checks (7 tests and bash -n passed).
- [x] Task 3: Integration/review/handoff.

Final verification: 99 passed / 1 skipped (Lightning unavailable), plus both
shell syntax checks and Python compilation. Native-MAP gradients and CP
freeze-boundary checks pass. Independent review found one missing direct-run
CUDA guard; a RED/GREEN regression fixed it and targeted re-review approved.
No training artifacts or historical preregistrations were changed, and no
cluster jobs were submitted. Actual A100 memory/runtime remain unverified.
