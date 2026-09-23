# Continued Pretraining

One workflow: dataset preparation, pre-CP evaluation, continued pretraining,
and post-CP evaluation. CP methods are `lejepa`, `simclr`, `diet`, and `mae`.
Frozen evaluation uses weighted cosine kNN and a PyTorch linear probe.
Optional supervised fine-tuning (FT) is retained.

## Installation

Use Python 3.10 or newer and a CUDA-compatible PyTorch installation for GPU jobs.

```bash
python -m pip install -e '.[dev]'
```

This installs `stable-pretraining` and `stable-datasets` from the repositories
listed in `pyproject.toml`. Model weights are downloaded by TIMM; some checkpoints
require a Hugging Face account with access to the model.

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

## Layout And Tests

```text
continued_pretraining.py   CLI and pre/CP/post orchestration
stable_cp/data/           Dataset registry, splits, transforms, loaders
stable_cp/methods/        LeJEPA, SimCLR, DIET, MAE
stable_cp/evaluation/     Frozen kNN/LP and optional FT
stable_cp/callbacks/      Unfreezing and online validation
run/slurm/run.sh          Generic single-job launcher
tests/                   Core regression and CPU smoke tests
```

```bash
python -m pytest -q tests
```

Historical experiment grids, geometry/causal analyses, rerun tools, and their
results remain in the `MAE` branch, not in this core branch.
