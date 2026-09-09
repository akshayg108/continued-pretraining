# SigLIP-2 main-grid unfreezing rerun

## Protocol amendment: 2026-09-07

This is a separate `siglip_mainrule_v1` MAX-budget pass, not a continuation of
the old fixed-two-block SigLIP experiments. The old scripts, DIET preregistration,
checkpoints and results are retained unchanged. This pass changes CP depth to
the main-grid budget rule and evaluates each new checkpoint with full FT.
Any later analysis must identify this protocol change explicitly; the old DIET
preregistration does not by itself preregister this revised recipe. No new
scientific endpoint is introduced here.

## Workload

| Objective | Datasets | Grouped Slurm tasks | CP fits | Post-CP FT fits |
|---|---|---:|---:|---:|
| LeJEPA | Seven affected MAX datasets below | 7 | 21 | 21 |
| SimCLR | Seven affected MAX datasets below | 7 | 21 | 21 |
| DIET-CP | All 15 datasets | 15 | 45 | 45 |
| Total | | 29 | 87 | 87 |

One task runs `CP(42) -> FT(42) -> CP(43) -> FT(43) -> CP(44) -> FT(44)`.
Each CP process also reports its final frozen kNN and LP. No pre-CP evaluation
or pre-CP FT is rerun. There are 29 array elements, not 29 individual model fits.
The seven affected datasets use:

| Dataset | MAX training images | CP blocks trained after epoch 15 |
|---|---:|---|
| Galaxy10 | 14,188 | Last 4 |
| EuroSAT | 16,200 | Last 4 |
| OrganAMNIST | 34,561 | Last 6 |
| PlantVillage | 43,596 | Last 6 |
| Food-101 | 75,750 | All parameters (`-1`) |
| PathMNIST | 89,996 | All parameters (`-1`) |
| OCTMNIST | 97,477 | All parameters (`-1`) |

The other eight DIET tasks use the last 2 blocks: BreastMNIST, DermaMNIST,
DTD, FGVC-Aircraft, Cars196, CUB200, Flowers102, and OxfordPet. The other
eight LeJEPA/SimCLR datasets are not rerun because their two-block CP depth
already matches the main-grid rule. Their old FT jobs are not included here.

The model is `vit_base_patch16_siglip_224.v2_webli`, with native `map` pooling.
All CP runs use 150 epochs, 15 encoder-frozen epochs, AdamW (lr 1e-4, weight
decay 0.05), 15 warmup epochs and cosine decay. The rule is 2 blocks below
10,000 images, 4 through 25,000, 6 through 50,000, and all parameters above
50,000. `-1` also unfreezes patch embedding, normalization and attention pooling;
it is not interchangeable with a count of all Transformer blocks.

LeJEPA retains 8 views, projection 128, hidden width 2048, lambda 0.02 and
effective batch 256. Its accumulation is 1/2/2/4 for 2/4/6/all-block CP.
SimCLR retains batch 256 without accumulation, temperature 0.5, projection 128
and hidden width 2048. DIET retains batch 32, label smoothing 0.3, and disabled
mixup/cutmix (probability 0, both alpha parameters 1). No automatic batch-size
reduction changes the contrastive recipe after an out-of-memory failure.

## Submit all tasks

Synchronize the code first. From the repository on the cluster:

```bash
bash run/slurm/cp-siglip/mainrule/submit.sh
```

This submits one `0-28%12` array: one A100, 8 CPUs, 96 GB host RAM and 96 hours
per element. Tasks are ordered by decreasing dataset size. Slurm can choose its
own scheduling order. The 96 hours cover all three CP/FT seed pairs together;
this is a time limit, not a measured runtime guarantee.

To inspect without submitting, use `--dry-run`. It writes a unique manifest,
prints the array command and the first task's six commands, but does not stage
data, load CUDA, or write training outputs. For a local preview:

```bash
SIGLIP_MAINRULE_OUTPUT_BASE=/tmp/siglip-mainrule-preview \
  bash run/slurm/cp-siglip/mainrule/submit.sh --dry-run
```

Optional `--concurrency N` accepts 1 through 12. Paths can be configured through
`SIGLIP_MAINRULE_REPO_ROOT`, `SIGLIP_MAINRULE_OUTPUT_BASE`,
`SIGLIP_MAINRULE_CACHE_DIR`, `SIGLIP_MAINRULE_LOG_DIR`, and
`SIGLIP_MAINRULE_MANIFEST`. The latter must name a new manifest file.
`PYTHON` selects the Python executable. Default shared cache:
`/scratch/gs4133/zhd/CP/data`.

The array activates the existing `env` conda environment. It copies only its
dataset's processed cache to a private node-local directory, checks free space,
uses that copy for all three seeds and removes it on exit. It fails if no
local location has sufficient capacity. Node-local storage is not GPU VRAM.

## Outputs and recovery

Default output root: `/scratch/gs4133/zhd/CP/outputs/siglip_mainrule_v1/`.

- `checkpoints/<method>/<dataset>/cp/`: new CP checkpoints, retained for reuse.
- `cp_results/<method>/<dataset>/seed<seed>.json`: final kNN/LP metrics.
- `provenance/<method>/<dataset>/seed<seed>.json`: recipe, code/software identity
  and completed CP checkpoint/result checksums.
- `full_ft/post/SigLIP/<method>/<dataset>/MAX_n<n>/seed<seed>.json`: FT metrics,
  checkpoint hash, split hashes and trainable/total parameter counts.

FT starts from the just-produced CP weights in a separate process. It resets
every backbone parameter to trainable before constructing the optimizer and
uses the existing `full_ft_v1` recipe: 150 epochs, batch 32, AdamW 1e-4/0.05.
There is no frozen-encoder stage in FT. Every result must pass the equality
`sft_trainable_params == sft_total_params`. FT writes no checkpoints.

Re-running the submit command uses new manifests but the same output namespace.
Matching completed CP is skipped; completed FT is validated by the full-FT
runner and skipped. Partial CP can resume only with a matching protocol receipt.
An interrupted FT starts that seed's FT again, since there is no FT checkpoint.
Changed recipes/software/code, unbound old weights and changed completed CP
artifacts cause a loud error, not a silent skip. A failed seed is reported, the
remaining seeds are attempted, and the task ultimately exits nonzero.

For only a failed array element, retain its printed manifest path and run:

```bash
sbatch --chdir="$PWD" --array=TASK_ID \
  --export=ALL,SIGLIP_MAINRULE_MANIFEST=/absolute/path/to/selection.json,SIGLIP_MAINRULE_REPO_ROOT="$PWD" \
  run/slurm/cp-siglip/mainrule/array.sh
```

If files must be synchronized selectively, the new package and scripts depend
on `continued_pretraining.py`, `stable_cp/` and `eval/full_ft/` from this same
working tree. Sync those dependencies too; copying only the new shell scripts
would omit the full-FT fix. Do not reuse the old `diet_max_array.sh` launcher
for this pass. Do not merge these outputs into the old fixed-two-block result
directory or present both protocols as the same experiment.

## Local verification

```bash
python3 -m pytest tests eval/F5_decision_score/test_siglip_diet_extension.py -q
bash -n run/slurm/cp-siglip/mainrule/array.sh
bash -n run/slurm/cp-siglip/mainrule/submit.sh
```

The tests exercise the complete 29-task grid, all 174 commands, CP receipts and
resume rejection, three-seed ordering, full-FT result gates, mocked submission,
dataset staging/cleanup and the existing full-unfreezing implementation.
They do not establish A100 peak memory use, wall time or cluster environment
compatibility. The first real task log should be checked for successful CP
unfreezing at epoch 15 and the full-FT trainable-parameter audit.

Local verification on 2026-09-07: 99 tests passed; one Lightning-dependent
integration module was skipped because Lightning is not installed locally.
Both shell scripts passed `bash -n`. Independent review approved the pass
after a regression-tested CUDA preflight was added to the Python runner.
