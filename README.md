# Continued Pretraining

One workflow: dataset preparation, pre-CP evaluation, continued pretraining,
and post-CP evaluation. CP methods are `lejepa`, `simclr`, `diet`, and `mae`.
Frozen evaluation uses weighted cosine kNN and a PyTorch linear probe.
Optional supervised fine-tuning (FT) is retained.

## Installation

Use Python 3.10 or newer and a CUDA-compatible PyTorch installation for GPU jobs.

```bash
python -m pip install -e .
```

This installs the lab's `galilai-group/stable-pretraining`, pinned to tested
commit `9aa93f8b`, and `stable-datasets` from `pyproject.toml`. The former personal
SPT fork is no longer required. Model weights are downloaded by TIMM; some
checkpoints require a Hugging Face account with access to the model.

For local SPT development, the nested checkout can be installed explicitly:

```bash
python -m pip install -e ./stable-pretraining
```

## Run

From the repository root:

```bash
python continued_pretraining.py \
    --dataset bloodmnist \
    --backbone vit_base_patch16_dinov3.lvd1689m \
    --cp-method diet --n-samples 1000 --seed 42 \
    --epochs 150 --freeze-epochs 15 --warmup-epochs 15 \
    --num-trained-blocks 2 --batch-size 32 --accumulate-grad-batches 1 \
    --mixup-cutmix-prob 0.0 \
    --cache-dir /path/to/data --checkpoint-dir /path/to/checkpoints \
    --results-json /path/to/results/diet-bloodmnist-seed42.json
```

Use `WANDB_MODE=offline` or `WANDB_MODE=disabled` when online W&B logging is
unavailable. `--help` lists datasets and method-specific options.

- `--no-cp`: run pre-CP evaluation only.
- `--full-train`: use every image in the training split instead of `--n-samples`.
- `--skip-baseline` / `--skip-final-eval`: skip pre-CP / post-CP frozen evaluation.
- `--pre-cp-sft` / `--post-cp-sft`: additionally fine-tune a copy of the encoder
  before / after CP; the encoder used by CP is not modified by FT.
- `--no-cp --pre-cp-sft`: evaluate supervised FT without CP.
- `--resume`: explicitly resume an existing CP checkpoint, including optimizer
  state. Existing checkpoints are otherwise never overwritten.

Each invocation runs one encoder, dataset, method, and seed. Results include
pre/post kNN and LP scores, normalization, and the training configuration.
Checkpoints are stored under `CHECKPOINT_DIR/METHOD/`. Use separate directories
when changing the training recipe.

Start new runs in a new checkpoint directory after upgrading from the old SPT
fork. Old MAE decoder checkpoint keys differ. `--resume` is tested for checkpoints
created with the pinned version, not for migration of old training state.

## Preprocessing And Evaluation

All training and evaluation transforms read mean/std from the loaded encoder's
`pretrained_cfg`. Dataset-specific normalization and random initialization are
not options on this branch. The existing 224-pixel resize/crop and augmentation
recipes are unchanged; official normalization does not mean replacing the whole
training transform with the model's inference transform.

| Encoder | Example TIMM checkpoint | Default pooling |
| --- | --- | --- |
| DINOv3-B | `vit_base_patch16_dinov3.lvd1689m` | CLS |
| DINOv3-L | `vit_large_patch16_dinov3.lvd1689m` | CLS |
| CLIP | `vit_base_patch16_clip_224.openai` | CLS |
| SigLIP-2 | `vit_base_patch16_siglip_224.v2_webli` | MAP |
| MAE | `vit_base_patch16_224.mae` | Patch mean |

`--pool-strategy` can override the default. CP, kNN, LP, and optional FT share
the sampled training indices. kNN uses clean training features; LP uses the
existing augmented training features. Frozen pre/post evaluation uses the same test
split and L2-normalized features. Dataset readers and split rules live in
`stable_cp/data/`.

`--n-samples` specifies the exact number of distinct training images. Sampling
follows class proportions while ensuring at least one image per class, so the
budget must be at least the class count. For the nominal 100-image setting,
pass the class count instead when it exceeds 100.

SPT's `OnlineKNN` and `OnlineProbe` monitor training; they do not replace frozen
pre/post evaluation. The reported kNN keeps sklearn inverse-cosine-distance
weighting, and LP keeps the existing normalized-feature Adam protocol.

SPT now handles gradient accumulation inside `Module` via `Manager`; CP forwards
return unscaled losses. Logged losses are therefore unscaled, and `global_step`
counts main optimizer updates rather than also counting online-probe updates.
Upstream also fixes distributed SimCLR to use cross-rank negatives; multi-GPU
results are not numerically equivalent to runs with the old fork.

Physical batch size and gradient accumulation remain explicit; they are not
automatically changed based on GPU type. Existing recipe examples:

| Method / encoder | Batch | Accumulation |
| --- | ---: | ---: |
| DIET | 32 | 1 |
| DINOv3-L LeJEPA / SimCLR | 128 | 2 |
| Other SimCLR / MAE-CP | 256 | 1 |
| Other LeJEPA, last 2 blocks | 256 | 1 |
| Other LeJEPA, last 4 or 6 blocks | 128 | 2 |
| Other LeJEPA, all blocks | 64 | 4 |

## Slurm

Run from the repository root after creating the log directory:

```bash
mkdir -p logs
export PYTHON=/path/to/environment/bin/python3
sbatch --partition=nvidia --account=civil --qos=nvidia \
    --gres=gpu:v100:1 --cpus-per-task=8 --mem=96G --time=96:00:00 \
    --output=logs/cp-%j.out --error=logs/cp-%j.err \
    run/slurm/run.sh \
    --dataset bloodmnist --backbone vit_base_patch16_dinov3.lvd1689m \
    --cp-method diet --n-samples 1000 --seed 42 \
    --epochs 150 --freeze-epochs 15 --warmup-epochs 15 \
    --num-trained-blocks 2 --batch-size 32 --accumulate-grad-batches 1 \
    --mixup-cutmix-prob 0.0 \
    --cache-dir /path/to/data --checkpoint-dir /path/to/checkpoints \
    --results-json /path/to/results/diet-bloodmnist-seed42.json
```

Choose GPU resources with `sbatch`; use A100 for the existing LeJEPA and
DINOv3-L jobs. `CP_REPO_ROOT` overrides the default repository location
(`SLURM_SUBMIT_DIR`). The launcher uses the supplied interpreter directly,
without changing Conda environments or creating experiment manifests.

## Full Pre-CP Grid

The grid evaluates five encoders on the original 15 datasets and eight held-out
datasets, with seeds 42, 43, and 44. kNN and LP both use the full training split;
validation and test images are not added to it. Existing dataset splits, pooling,
and evaluation recipes are retained, with each encoder's pretrained mean/std.
There is no CP, FT, checkpoint saving, or online W&B logging in this grid.
Stanford Dogs keeps its existing validation holdout from the official training
split; custom splits are unchanged. Galaxy10's split varies with the seed.

The 17 Slurm tasks contain one dataset each, except task 0, which serializes all
seven MedMNIST datasets. Every task runs all five encoders and three seeds in
separate processes. In total there are 345 evaluations. The default resources
are one V100, eight CPUs, 96 GB RAM, and 96 hours, with at most 12 tasks running.
The extraction batch is 32 with two loader workers; these are frozen evaluations,
not the memory-intensive CP configurations above.
Task 0 has 105 evaluations rather than 15, so it will usually finish later.

Start from the new project directory on the cluster:

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
cd "$CP_ROOT"
git clone --branch organized --single-branch \
    git@github.com:akshayg108/continued-pretraining.git
cd continued-pretraining
module load miniconda/3-4.11.0
bash run/setup_env.sh "$CP_ROOT"
source run/precp_env.sh
```

The setup script creates `CP_ROOT/env` with Python 3.11, CUDA 12.8 PyTorch 2.10.0
and torchvision 0.25.0, and the pinned lab `stable-pretraining` and user
`stable-datasets` revisions. It does not modify the old environment. PyTorch
is pinned to retain V100 support; do not upgrade it independently. A dependency
snapshot is saved in `outputs/environment/pip-freeze.txt`.

For gated DINOv3 weights, authorize your Hugging Face account for the checkpoints
and log in to the new cache, or supply `HF_TOKEN` through your environment:

```bash
"$CP_ROOT/env/bin/hf" auth login
"$CP_PYTHON" run/precp.py list
"$CP_PYTHON" run/precp.py run --task-id 1 --dry-run
bash run/slurm/submit_precp.sh
squeue -u "$USER"
```

Dataset archives and processed data go to `data/stable_datasets`; model caches
go to `data/huggingface` and `data/torch`. Download, package, and temporary caches
also live under `data`. Results and per-evaluation logs are stored in
`outputs/precp_full/{results,logs}/ENCODER/DATASET/seedSEED.{json,log}`; Slurm logs
are in `outputs/slurm-log/precp-JOB_TASK.{out,err}`.

Completed JSON results are skipped on resubmission. A failed evaluation is logged
and the job continues through the other combinations, then exits nonzero. Retry
the affected task IDs, or submit the full grid again. Slurm resource overrides
are accepted by the submission script. Invalid or mismatched existing JSON files
stop the command for inspection rather than being silently overwritten:

```bash
bash run/slurm/submit_precp.sh --array=0,16%2 --gres=gpu:a100:1
"$CP_PYTHON" run/precp.py report
```

The report writes `outputs/precp_full/results.csv` and `summary.csv`, including
actual training/test sizes, per-seed scores, means and sample standard deviations.
Missing seeds stay missing rather than contributing zero to a mean. Re-running
a failed combination restarts that evaluation; already completed combinations
are not repeated.

## Layout

```text
continued_pretraining.py   CLI and pre/CP/post orchestration
stable_cp/data/           Dataset registry, splits, transforms, loaders
stable_cp/methods/        LeJEPA, SimCLR, DIET, MAE
stable_cp/evaluation/     Frozen kNN/LP and optional FT
stable_cp/callbacks/      Unfreezing and online validation
run/slurm/run.sh          Generic single-job launcher
run/setup_env.sh          Independent cluster environment installation
run/precp.py              Full pre-CP grid and CSV report
run/slurm/submit_precp.sh  Submit the 17-task pre-CP array
```

Historical experiment grids, geometry/causal analyses, rerun tools, and their
results remain in the `MAE` branch, not in this core branch.
