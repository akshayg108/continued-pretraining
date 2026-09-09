# Full-Parameter FT Repair and Rerun Plan

> **For agentic workers:** Use superpowers:subagent-driven-development and
> superpowers:test-driven-development. Keep edits in the shared checkout;
> do not commit, submit jobs, or revert unrelated changes.

**Goal:** Correct inherited CP freezing and provide storage-free supervised FT
reruns for the main grid and SigLIP, with three seeds per scheduler task.

**Architecture:** Reuse the existing SFT recipe and data functions. Unfreeze the
evaluation copy before optimizer construction and use a checkpoint-disabled
Lightning trainer. A standard-library manifest groups three checkpoint paths per
configuration. A standalone runner writes atomic per-seed metrics and a collector
reports available-seed means and explicit missingness. Slurm stages each target
cache on node-local storage and runs the grouped seeds serially.

**Tech Stack:** Python, PyTorch, Lightning, stable-pretraining, timm, Bash/Slurm.

## Global Constraints

- Preserve CP training weights, recipes, and all frozen kNN/LP outputs.
- Full FT updates every backbone parameter and a freshly seeded classifier.
- SFT recipe: 150 epochs, batch 32, AdamW lr 1e-4, weight decay 0.05,
  15-epoch linear warmup, cosine decay, no label smoothing, existing transforms.
- Do not save FT model or optimizer checkpoints, including automatic callbacks.
- Keep CP checkpoints. Never delete existing checkpoints or canonical results.
- Scope: DINOv3/CLIP/MAE main grid and SigLIP. No ViT-L extension in this task.
- Support all recorded budgets/objectives; the first submission selects
  LeJEPA/SimCLR/DIET, 500/MAX, with SigLIP MAX only.
- Each scheduler task owns one encoder/objective/dataset/budget/phase and runs
  seeds 42, 43, 44 sequentially. Default scheduling is post-only. Optional pre-CP
  tasks require explicit selection and are shared across objectives.
- Missing CP files are recorded and skipped. Invalid or failed runs are errors,
  not missing files. Means use successful seeds and expose their exact identities.
- A seed interrupted without a saved FT checkpoint restarts that seed; completed
  validated per-seed result JSON files are resumable.
- Default resources: one A100, eight CPUs, 96 GB RAM, 96 hours, concurrency 12.
  Cluster resource names remain overridable through sbatch options/environment.
- New metrics live under a separate full-FT result directory. Old reported
  averages are not silently mixed with new per-seed results.

## Task 1: Shared SFT Repair

Files: `stable_cp/evaluation/sft_eval.py`, `continued_pretraining.py`,
`tests/test_full_ft_sft.py`.

- [x] Reproduce the frozen-copy failure with real torch parameters.
- [x] Add failing tests for full trainability before module/optimizer creation,
  original-model immutability, deterministic head initialization, and no saves.
- [x] Enable every parameter of the copied model using
  `backbone_copy.requires_grad_(True)` before `_setup_sft_module`.
- [x] Seed before classifier creation; use `Trainer(enable_checkpointing=False)`
  and direct `trainer.fit(module, datamodule=sft_data)` to prevent Manager from
  injecting checkpoint callbacks. Preserve the module optimizer/scheduler.
- [x] Remove SFT path creation/deletion from `_run_sft_phase`; retain CP storage.
- [x] Run `python3 -m pytest tests/test_full_ft_sft.py -q`.

## Task 2: Grouped Manifest

Files: `eval/full_ft/manifest.py`, `tests/test_full_ft_manifest.py`.

Public API: `build_tasks(repo_root, *, methods=None, budgets=None, encoders=None,
phases=("post",), checkpoint_root=None) -> list[dict]`.

Each task contains `task_id`, `phase`, `scope`, `encoder`, `method`, `dataset`,
`budget`, `n_samples`, `model_id`, `pool`, `processed_subpath`, and `checkpoints`.
`checkpoints` maps string seeds to ordered candidate-path lists. `method="PRE"`
for shared pre tasks. Checkpoint candidates must be CP files, never SFT or teacher
files. Known inventory paths take precedence over exact launch-derived paths.

- [x] Test source-grid coverage, budget aliases, MAX=3334 for Aircraft, shared pre
  deduplication, exact seed grouping, SigLIP three objectives, and path relocation.
- [x] Build from `eval/outputs/cp_long_refreshed.csv` and existing CP scripts;
  enrich from `postcp_sweep_fixed.csv` and `nd12_operator.csv`.
- [x] Add SigLIP DIET MAX candidates using the existing launcher layout.
- [x] Expose CLI filters and `--output` JSON; assign contiguous task IDs only
  after selection, and preserve source provenance in the manifest document.
- [x] Run `python3 -m pytest tests/test_full_ft_manifest.py -q`.

## Task 3: Standalone Runner and Collector

Files: `eval/full_ft/run.py`, `eval/full_ft/collect.py`,
`eval/full_ft/checkpoint.py`, `tests/test_full_ft_runner.py`.

CLI: `run.py --manifest FILE --task-id N --cache-dir DIR --outdir DIR
--device cuda [--seeds 42 43 44] [--dry-run]`.

- [x] Test strict backbone extraction (including MAE wrappers and SigLIP pool),
  refusal of SFT state dictionaries, missing-seed skips, invalid-resume rejection,
  available-seed aggregation, and JSON output without model files.
- [x] Load CPU CP state, validate all target tensors and shapes, initialize the
  matching timm architecture without pretrained downloads for post tasks.
- [x] Reuse `_create_shared_eval_data` and `_create_sft_data` without evaluating
  kNN/LP. Preserve sampled indices and log their digest and actual sample count.
- [x] Run repaired `sft_evaluate`, write finite metrics and source identity
  atomically per seed. Do not treat corruption or training failures as missing.
- [x] Collector reports exact successful/missing/failed seeds, per-seed data and
  configuration means. Match pre/post by seed when calculating new deltas.
- [x] Run CPU/unit smoke tests and `--dry-run` over real manifest selections.

## Task 4: Slurm and SigLIP DIET Integration

Files: `run/slurm/full_ft/array.sh`, `run/slurm/full_ft/submit.sh`,
`run/slurm/cp-siglip/cp/diet_max_array.sh`,
`eval/F5_decision_score/SIGLIP_DIET_FT_ADDENDUM.md`, `eval/full_ft/README.md`,
appropriate shell/CLI regression tests.

- [x] Add failing dry-run tests for seed grouping, first-wave filters, resource
  overrides, and staging behavior. All new sbatch tasks group three seeds.
- [x] Stage only the selected processed dataset using TMPDIR/tmpdata/dev/shm,
  with capacity checks and private temporary directories; fall back explicitly
  to shared storage. Clean up only directories created by this task.
- [x] Submission wrapper creates immutable task manifests, logs the exact
  sbatch command, and provides a one-command 500/MAX three-objective launch.
- [x] SigLIP DIET groups three original preregistered cells per array task and
  appends full FT without changing CP hyperparameters or frozen P1. Keep old
  completed CP results resumable; record FT in the new namespace.
- [x] Document CP-first/FT-second ordering, no-save restart behavior, missing
  seeds, collection, synchronization, and full-grid versus first-wave commands.
- [x] Run `bash -n`, mocked-sbatch tests, full new unit suite, existing SigLIP
  protocol tests, and an independent final code review.

## Progress

- Context and root cause inspected; original worktree contains only unrelated
  untracked `eval/complement_search/`.
- User clarification: rerun post only, not pre. Both the manifest builder and
  submission wrapper now default to post. First wave: 315 tasks / 945 possible
  seed fits; all recorded budgets and objectives: 885 tasks / 2655 seed fits.
  Optional pre support remains explicit and is not part of this launch.
- No cluster jobs will be submitted during implementation.
- Task 1 RED: four expected failures and one pass; GREEN: all five tests pass.
- Initial combined suite: 47 passed. Independent review found Slurm spool-path,
  staging, resume, and CLIP-head defects; targeted RED/GREEN regressions repaired
  them. The final independent review reports no remaining launch-blocking finding.
- Final combined unit/CLI suite: 61 passed, one integration-module skip in the
  base environment. This includes the seven-test Slurm suite and mocked
  submission. All three shell scripts pass `bash -n`, changed Python modules
  compile, and `git diff --check` is clean. The frozen SigLIP preregistration
  re-verifies without any source or endpoint change.
- A temporary Python 3.12 environment with real Lightning 2.6.5 and the archived
  stable-pretraining implementation passed all three CPU integration tests.
  It does not modify the shared Python installation or archived source.
- Real integration exposed Lightning's implicit HPC restore: `ckpt_path=None`
  is insufficient, so FT and CP use private empty trainer directories. FT
  rejects all checkpoint saves and disables Slurm automatic requeue saves.
- CP required a compatibility repair: separate canonical save destination from
  existing resume input, temporarily disable Manager's job-ID cache, and retain
  full optimizer state. Existing CP files are never deleted to start a new run.
- Four timm architectures instantiate locally without downloads; all expose
  the expected 768-dimensional backbone. Unused CLIP native heads are excluded,
  but every feature-producing tensor, including SigLIP pooling, loads strictly.
- A100 execution, real processed datasets, cluster dependency compatibility,
  and actual Slurm scheduling remain untested locally. README requires a first
  three-seed dataset smoke before the full submission.
