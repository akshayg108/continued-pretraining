# Continued Pretraining

One workflow: dataset preparation, pre-CP evaluation, continued pretraining,
and post-CP evaluation. CP methods are `lejepa`, `simclr`, `diet`, and `mae`.
Frozen evaluation uses weighted cosine kNN and a PyTorch linear probe.
Optional supervised fine-tuning (FT) is retained.

The separate [source-coverage experiment](run/SOURCE_PRETRAIN.md) trains paired
ViT-B models from scratch on ImageNet alone or ImageNet plus three target domains.

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
not options in the CP entry point. The existing 224-pixel resize/crop and augmentation
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

All GPU launchers stage image data on node-local storage before computation.
Missing downloads and Arrow caches are first built under the shared `--cache-dir`
inside the same job. Only the current dataset's processed splits are copied;
archives remain shared. `CP_NODE_TMPDIR` can select the local disk; otherwise the
launcher uses `SLURM_TMPDIR`, writable `/tmpdata`, or `/tmp`, in that order.
Insufficient local space stops the job rather than falling back to shared reads.
Results, checkpoints, and logs keep their original shared paths. The private
local copy is removed on normal exit and catchable termination.

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
separate processes. In total there are 345 evaluations. One additional reference
job runs first; the 17 evaluation tasks depend on its successful completion.
Each dataset is staged once for all encoders and seeds. Task 0 stages and releases
one MedMNIST dataset at a time. The reference job stages only its selected 5000
ImageNet images. There are no separate CPU preparation jobs.
The default resources
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

Copy the existing Hugging Face ImageNet validation cache before submission;
the preparation job reads local images and does not download ImageNet:

```bash
mkdir -p "$CP_ROOT/data/imagenet_val"
rsync -ah --info=progress2 \
    /scratch/gs4133/zhd/CP/data/imagenet_val/ "$CP_ROOT/data/imagenet_val/"
```

For gated DINOv3 weights, authorize your Hugging Face account for the checkpoints
and log in to the new cache, or supply `HF_TOKEN` through your environment:

```bash
"$CP_ROOT/env/bin/hf" auth login
"$CP_PYTHON" run/precp_reference.py --dry-run
"$CP_PYTHON" run/precp.py list
"$CP_PYTHON" run/precp.py run --task-id 1 --dry-run
bash run/slurm/submit_precp.sh
squeue -u "$USER"
```

Persistent dataset archives and processed data go to `data/stable_datasets`; model caches
go to `data/huggingface` and `data/torch`. Download, package, and temporary caches
also live under `data`. Results and per-evaluation logs are stored in
`outputs/precp_full/{results,logs}/ENCODER/DATASET/seedSEED.{json,log}`; Slurm logs
are in `outputs/slurm-log/precp-JOB_TASK.{out,err}`.
Image reads during evaluation use the node-local processed cache. First-time
download/cache construction can still leave the GPU idle; if the cluster cancels
the job, resubmit it to reuse completed shared caches and results.

Geometry reuses clean training features already extracted for kNN. Every metric
uses the same label-independent, sorted seed-42 sample of `min(n_train, 5000)`
images; there is no further target subsampling. The reference job selects 5000
ImageNet validation images once using the same sampling seed and extracts each
encoder's reference with its own mean/std and pooling.

| Descriptor | Definition on L2-normalized feature rows |
| --- | --- |
| Uniformity | Log mean `exp(-2 * squared_distance)` over distinct target pairs |
| Mean cosine | Mean cosine similarity over distinct target pairs |
| RankMe | Exponential singular-value entropy of the uncentered target matrix |
| MMD | Biased squared RBF-MMD, including diagonal pairs |
| Overlap | Mean ImageNet fraction among 50 cosine neighbors, excluding self |

MMD bandwidth uses the reciprocal median squared distance over all pooled target
and reference rows. Overlap queries every selected target against the same target
bank plus the reference bank. Thus neither bandwidth nor overlap uses an extra
smaller subset. These statistics use float64 arithmetic. The protocol is
`precp_geometry_5000_v1`; it differs from historical 3000-target angular metrics
and 2000-query overlap, so those old values must not be mixed into this grid.

Reference features are saved to `outputs/precp_full/reference/ENCODER.npz`.
Selected raw target features and training-row indices are saved to
`outputs/precp_full/features/ENCODER/DATASET/seedSEED.npz`. No full-training-set
feature archive is written. JSON records include sample counts, MMD bandwidth,
and reference paths. For unchanged train splits, deterministic geometry can be
identical across evaluation seeds; LP still follows its original seeded protocol.

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
actual training/test/geometry sizes, per-seed scores and all five geometry
descriptors, means and sample standard deviations.
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
run/data_cache.py         Shared cache preparation and node-local staging
run/precp_reference.py    ImageNet reference feature extraction
run/slurm/submit_precp.sh  Submit reference preparation and the 17-task array
```

Historical experiment grids, geometry/causal analyses, rerun tools, and their
results remain in the `MAE` branch, not in this core branch.
