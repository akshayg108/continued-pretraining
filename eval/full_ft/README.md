# Full-Parameter Fine-Tuning Reruns

This pipeline reruns supervised full-parameter fine-tuning without saving FT model or optimizer checkpoints. CP checkpoints and historical metrics remain unchanged; new seed-level results live under `outputs/full_ft_v1`.

## Submit

The default first wave selects LeJEPA, SimCLR, and DIET; budgets 500 and MAX; DINOv3, CLIP, MAE, and SigLIP; and post-CP FT only. SigLIP has MAX configurations only:

```bash
bash run/slurm/full_ft/submit.sh
```

This creates 315 grouped post-CP tasks, up to 945 seed-level FT fits. No pre-CP
baseline is scheduled. Missing CP weights and previously completed FT seeds
reduce the actual number of fits. The 96-hour limit applies to each three-seed
group, not each seed.

| First-wave scope | Grouped tasks | Seed fits before skips |
|---|---:|---:|
| Main grid, 500, three encoders and three objectives | 135 | 405 |
| Main grid, MAX, three encoders and three objectives | 135 | 405 |
| SigLIP, MAX, three objectives | 45 | 135 |
| Total | 315 | 945 |

SigLIP DIET CP-first plus full-FT uses the frozen 15-dataset driver directly:

```bash
sbatch run/slurm/cp-siglip/cp/diet_max_array.sh
```

The full grid, including MAE-CP and all recorded sample budgets, is:

```bash
bash run/slurm/full_ft/submit.sh --all
```

The full selection has 885 post-CP tasks, up to 2655 seed fits. Pre-CP FT remains
available only through an explicit `--phases pre` or `--phases pre post` override;
it is not part of the requested rerun. Selections larger than `FULL_FT_ARRAY_LIMIT`
(default 1000) are split into immutable, reindexed manifests and submitted
sequentially with `afterany` dependencies. Each invocation keeps at most 12 tasks
live. Do not simultaneously launch independent 12-way waves if your total
allocation is 12 GPUs.

Inspect without submitting with `--dry-run`. Resource defaults are one A100, eight CPUs, 96 GB, 96 hours, and array concurrency 12. Override them with `FULL_FT_GRES`, `FULL_FT_CPUS`, `FULL_FT_MEM`, `FULL_FT_TIME`, and `FULL_FT_CONCURRENCY`, or pass alternate manifest-builder filters after `--`.

Cluster defaults are `/scratch/gs4133/zhd/CP/data` for data, `/scratch/gs4133/zhd/CP/outputs/ckpts` for CP checkpoints, `/scratch/gs4133/zhd/CP/outputs/full_ft_v1` for results, and `/scratch/gs4133/zhd/CP/outputs/slurm-log` for scheduler logs.

Each immutable manifest task groups seeds 42, 43, and 44. The array driver stages only that task's processed dataset into a private job/task directory under `$TMPDIR`, `/tmpdata`, or `/dev/shm` after a capacity check. It explicitly falls back to shared storage and removes only the directory it created.

Missing CP checkpoint candidates are recorded as missing. Invalid checkpoints and failed training remain errors. Completed validated seed JSON files resume; an interrupted seed restarts from the beginning because FT checkpoints are intentionally disabled.

## Collect

After synchronizing all seed JSON files into one result directory:

```bash
python eval/full_ft/collect.py --outdir /scratch/gs4133/zhd/CP/outputs/full_ft_v1 --output eval/outputs/full_ft_v1
```

The collector writes `per_seed.csv`, `summary.csv`, `paired_deltas.csv`, and `availability.csv`. Available-seed means identify the exact successful seeds; pre/post deltas are paired by seed.

Before launching, synchronize the minimal code and metadata set: `eval/full_ft/`, `stable_cp/evaluation/sft_eval.py`, `continued_pretraining.py`, `eval/outputs/cp_long_refreshed.csv`, `eval/outputs/postcp_sweep_fixed.csv`, `eval/outputs/nd12_operator.csv`, and the relevant Slurm scripts. Code, environment, and processed caches must remain fixed during a run. Resume is based on validated result JSON files, not later code updates.

## Synchronization and First Run

From the local repository, replace `<cluster>` with your SSH host. This transfers
source and small metadata only, not CP weights, feature dumps, or FT results:

```bash
rsync -avR --exclude='__pycache__' --exclude='*.pyc' \
  continued_pretraining.py stable_cp/evaluation/sft_eval.py \
  stable_cp/evaluation/zero_shot_eval.py \
  stable_cp/data/datasets.py stable_cp/data/loaders.py \
  eval/full_ft/ run/slurm/full_ft/ \
  run/slurm/cp-siglip/cp/diet_max_array.sh \
  eval/F5_decision_score/siglip_diet_protocol.py \
  eval/F5_decision_score/SIGLIP_DIET_PREREG.md \
  eval/F5_decision_score/SIGLIP_DIET_FT_ADDENDUM.md \
  eval/outputs/cp_long_refreshed.csv eval/outputs/postcp_sweep_fixed.csv \
  eval/outputs/nd12_operator.csv eval/outputs/preregister_siglip.csv \
  eval/outputs/c2_siglip_score.csv eval/outputs/siglip_diet_preregister.csv \
  results.xlsx \
  <cluster>:/scratch/gs4133/zhd/CP/continued-pretraining/
```

On the cluster, use the existing `env` environment. Validate the frozen SigLIP
inputs before running the first three-seed dataset task:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining
mkdir -p /scratch/gs4133/zhd/CP/outputs/slurm-log
python eval/F5_decision_score/siglip_diet_protocol.py --verify-prereg
sbatch --array=0 run/slurm/cp-siglip/cp/diet_max_array.sh
```

After that task succeeds, submit the remaining SigLIP DIET datasets:

```bash
sbatch --array=1-14%12 run/slurm/cp-siglip/cp/diet_max_array.sh
```

After the SigLIP wave finishes, submit the first-wave full-FT reruns:

```bash
bash run/slurm/full_ft/submit.sh --dry-run
bash run/slurm/full_ft/submit.sh
```

The same output namespace is used by both entry points. Already completed,
validated SigLIP DIET FT seeds are skipped in the later wave. SigLIP DIET accounts
for 15 of the 315 tasks (45 seed fits), not 45 additional FT fits. Its CP training
and frozen evaluations are extra work and are excluded from the FT count.
Without compatible seed-level pre FT results, the collector reports post levels
and does not manufacture paired deltas from old aggregate baselines.

Check logs for `sft_trainable_params == sft_total_params` in the result JSON,
successful seeds, and explicit missing checkpoint notices. Summaries are under
the collector's output directory; source seed JSON files remain under
`/scratch/gs4133/zhd/CP/outputs/full_ft_v1`.

All seeds in a task share its 96-hour walltime. SigLIP DIET tasks additionally
include CP training and frozen evaluation, so a large dataset can require a
resubmission. CP resumes its checkpoint; an incomplete FT seed restarts without
checkpoint recovery, while completed FT seeds are skipped. No completion-time
estimate is implied by the walltime setting.

## Conditional Runtime Budget

For all 945 first-wave fits, the unchanged recipe has
`sum(3 * 150 * floor(n_samples / 32)) = 68,951,250` optimizer steps:
911,250 at 500 samples and 68,040,000 at MAX. No representative A100 timing is
available locally. With 12 continuously occupied GPUs, training-only duration
would be:

| Assumed average seconds per optimizer step | Ideal training-only days |
|---|---:|
| 0.10 | 6.65 |
| 0.20 | 13.30 |
| 0.30 | 19.95 |

These are scenarios, not measured performance or completion commitments.
Validation, final testing, data staging, queue delays, stragglers, and restarts
add time; missing weights and completed fits remove work. The separate SigLIP
DIET CP stage is not included. The largest OctMNIST task has 1,370,700 training
steps across its three seeds: at 0.25 seconds per step, training alone takes
95.19 hours, so validation would risk exceeding the 96-hour cap. Measure an
early epoch on that dataset before relying on a full-wave deadline.

## Verification

The unit/CLI regression suite passes 61 tests (the real integration module is
skipped in the base environment). It includes exact manifest counts, missing seeds,
paired aggregation, CP preservation, and a mocked `sbatch` test of the
1000+110-task full-grid split and its dependency. A separate real Lightning plus
archived stable-pretraining CPU integration suite passed three tests: full FT
updates previously frozen layers without saving (ordinary and simulated Slurm
environments), and CP saves/resumes its canonical checkpoint and optimizer state.

These drivers have not been exercised on an A100, on real processed datasets, or
submitted through Slurm in this workspace. The cluster smoke above is still
required to check the installed dependency versions and dataset caches.

## Checkpoint Safety

FT unfreezes the copied backbone before creating its optimizer, seeds the new
classifier, disables automatic Slurm requeue saves, and uses an isolated empty
trainer directory. Any attempted FT checkpoint write is an error. Completed FT
JSON files are the only resume mechanism for FT.

CP still saves checkpoints. Its save destination and load path are now separate:
new CP runs load nothing, existing CP runs require `--resume`, and a retry never
deletes an existing checkpoint. The SigLIP DIET launcher passes `--resume`.
Job-level Manager cache redirection is disabled temporarily for CP so grouped
seeds cannot inherit another seed's restart checkpoint. The canonical CP filename
is retained and optimizer state is restored when resuming.

Keep the processed datasets unchanged throughout a wave. Result provenance
records code/environment, CP file SHA256, and train-index/test-label digests;
it does not hash every image in the processed caches. CLIP's unused native head
is discarded before standalone FT; all feature tensors, including SigLIP MAP
pooling, must match the CP checkpoint exactly.
