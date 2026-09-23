# Source Coverage Experiment

Train a random-initialized ViT-B/16 (`vit_base_patch16_224`, all layers trainable)
under two source conditions, paired at seed 42:

- A, `imagenet`: all 1,281,167 ImageNet-1K training images in 1,000 classes.
- B, `mixed`: the same ImageNet training data plus the complete OCTMNIST,
  PathMNIST, and Galaxy10 training splits.

Both conditions concatenate and shuffle naturally without replacement per cycle;
there is no target oversampling or domain balancing. Target train/test partitioning
uses seed 42 independently of the training seed. No test images enter training,
and labels are not used by the LeJEPA loss. ImageNet labels train a detached
online linear probe only. At fixed total steps, B has lower ImageNet
exposure than A; this is a source-mixture comparison, not an isolated causal test
of target coverage at matched ImageNet exposure.

## Fixed Recipe

The physical batch is 256 images with accumulation 1 on one H200. Each image has
two 224-pixel global crops and eight 96-pixel local crops. LeJEPA uses the
all-view mean as the invariance center, lambda 0.05, and per-view SIGReg with
1,024 random slices and Epps-Pulley quadrature at 17 points up to `t_max=3`.
The projector is 768-2048-2048-128; its output width is separate from the
1,024 random SIGReg directions. Training uses BF16 mixed precision,
float32 loss calculations, and activation checkpointing.

AdamW uses peak learning rate `5e-4`, weight decay `0.05`, and betas `(0.9, 0.999)`.
The budget is 500,400 optimizer steps, with 25,020 warmup steps followed by cosine
decay to `5e-7`. This equals `100 * floor(1,281,167 / 256)` updates, not 100 mixed
dataset epochs. The final run records actual per-domain image exposures.
Memory fit and completion within the requested wall time must be checked on the
cluster; neither is guaranteed by the H200 request.
Normalization is the standard ImageNet mean/std in both conditions; there are
no downloaded encoder weights in this from-scratch experiment.

## Update The Server Checkout

Use the `organized` branch in the separate `CP_new` workspace. The commands
below run on the cluster, not the local workstation.

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
cd "$CP_ROOT/continued-pretraining"
git switch organized
git pull --ff-only origin organized
```

## Environment And ImageNet

Use the existing checkout and `CP_new/env`; run setup only if the environment is
missing. These commands do not modify the old CP environment.

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
cd "$CP_ROOT/continued-pretraining"
if [[ ! -x "$CP_ROOT/env/bin/python3" ]]; then
    module load miniconda/3-4.11.0
    bash run/setup_env.sh "$CP_ROOT"
fi
source run/precp_env.sh
export IMAGENET_SOURCE=hf
export IMAGENET_TRAIN_DIR="$CP_ROOT/data/imagenet/train"
export IMAGENET_VAL_DIR="$CP_ROOT/data/imagenet_val"
unset IMAGENET_TRAIN_ARCHIVE
"$CP_PYTHON" -c 'import stable_pretraining; print(stable_pretraining.__file__)'
"$CP_PYTHON" run/source_pretrain.py --help
```

The default source is the gated Hugging Face dataset
[`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k), pinned at
revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`. Use an account that has
accepted the dataset terms and has access. Authenticate **after** sourcing
`run/precp_env.sh`, which places the Hugging Face cache and saved login under
`CP_ROOT/data/huggingface`. Enter the token interactively; do not put it in scripts.

```bash
"$CP_PYTHON" -c 'from huggingface_hub import login; login()'
df -h "$CP_ROOT/data"
```

The preparation job downloads the training split, saves a memory-mapped Hugging
Face Arrow dataset at `IMAGENET_TRAIN_DIR`, and checks the exact training
image/class counts. This directory does not need to exist before submission.
Training reads the saved local dataset, not an online stream. An existing
`IMAGENET_VAL_DIR` cache is reused; otherwise the preparation job downloads the
validation split too. It does not download the unlabeled ImageNet test split.
Split-only downloads disable Hugging Face's whole-repository metadata checks;
each downloaded split is checked against its exact row count and all 1,000 labels.

Plan for approximately **500 GiB of free shared disk** for downloaded parquet
files, the Arrow build cache, the saved training dataset, target caches and
checkpoints. This is a planning estimate, not an exact size requirement. Each
concurrent training task also needs room for a complete node-local input copy.

The validation cache is used only for probe validation, never as pretraining
data. It must be a Hugging Face `save_to_disk` dataset with `image` and `label`
columns, or a DatasetDict with a `validation` split. WNID label names are remapped
to training class indices; otherwise labels must be the official ImageNet-1K IDs
in lexicographic WNID order. Optionally reuse the previous validation cache before
submission, but do not create an empty destination unless copying into it:

```bash
mkdir -p "$IMAGENET_VAL_DIR"
rsync -a --info=progress2 \
    /scratch/gs4133/zhd/CP/data/imagenet_val/ "$IMAGENET_VAL_DIR/"
```

Preparation also downloads or reuses the three target caches under
`CP_ROOT/data/stable_datasets`, and prepares five validation images per class
using seed 42 under
`data/source_imagenet_validation_5000`, preserving images, labels and indices.

Every training job stages the full ImageNet training dataset and all three target
caches on node-local storage, including condition A for held-out evaluation.
The prepared 5,000-image validation subset is staged locally as well.
Staging uses `CP_NODE_TMPDIR`, then `SLURM_TMPDIR`, then writable `/tmpdata` or `/tmp`.
Do not point scratch at the shared source directory. Local copies are temporary;
checkpoints and outputs remain under `CP_ROOT/outputs`.

### Optional Local ImageNet Source

To use an existing complete ImageFolder instead, set `IMAGENET_SOURCE=local`,
point `IMAGENET_TRAIN_DIR` at its 1,000 WNID class directories, and unset
`IMAGENET_TRAIN_ARCHIVE`. The original nested training archive is also accepted:

```bash
export IMAGENET_SOURCE=local
export IMAGENET_TRAIN_DIR="$CP_ROOT/data/imagenet/train"
export IMAGENET_TRAIN_ARCHIVE="$CP_ROOT/data/ILSVRC2012_img_train.tar"
```

In archive mode the preparation job extracts the archive and reuses completed
class directories after interruption. Local mode requires an existing validation
cache at `IMAGENET_VAL_DIR`; it does not download ImageNet. Do not rename a
parquet file or another archive format to `ILSVRC2012_img_train.tar`.

## Submit And Resume

First submit a separate ten-step smoke run to check real H200 memory fit and
the online probes before committing to the full budget:

```bash
SOURCE_STEPS=10 SOURCE_TASKS='0-1%2' \
SOURCE_OUTPUT_DIR="$CP_ROOT/outputs/source_coverage_smoke" \
    bash run/slurm/submit_source_pretrain.sh
squeue -u "$USER"
```

After both tasks have completed, check the smoke outputs. This reports
`Complete runs: 2/2` and both online metric files when successful:

```bash
"$CP_PYTHON" run/source_pretrain.py report --root "$CP_ROOT" \
    --output-dir "$CP_ROOT/outputs/source_coverage_smoke"
for CONDITION in imagenet mixed; do
    "$CP_PYTHON" -m json.tool \
        "$CP_ROOT/outputs/source_coverage_smoke/$CONDITION/seed42/online_metrics.json"
done
```

Then submit the full-budget pair. Smoke checkpoints are not reused:

```bash
SOURCE_STEPS=500400 SOURCE_TASKS='0-1%2' \
SOURCE_OUTPUT_DIR="$CP_ROOT/outputs/source_coverage_v1" \
    bash run/slurm/submit_source_pretrain.sh
```

Submission creates one V100 preparation job and an `afterok` training array
`0-1%2`. Tasks 0/1 run A/B at the same seed 42, for two training runs total.
Each training task requests one H200. Both stages request 16 CPUs, 128 GB RAM,
and 96 hours. Preparation uses an explicit `--gres=gpu:v100:1` submission override;
when submitting `source_pretrain.sh prepare` directly, include that flag too.
Defaults are partition `nvidia`, account `civil`, and QoS `nvidia`; standard `SBATCH_*`
environment variables such as `SBATCH_PARTITION`, `SBATCH_ACCOUNT`, and
`SBATCH_QOS` can override them. Logs are `outputs/slurm-log/source-*`.
Use `SOURCE_TASKS='0-1%1'` to allow only one H200 training task at a time.
If preparation fails, its dependent array cannot start. Fix the preparation
error, cancel that still-pending array, and rerun the submission script; completed
shared datasets and download caches are reused. A partial Arrow export may need
to be rebuilt, but already downloaded Hugging Face files are cached.

Short runs also evaluate initial/final geometry and stage the full datasets.
Their warmup is shortened automatically; their outputs are not full-budget
results. Do not reuse their output directory for the full run.

Training always passes `--resume`. After preparation has succeeded, resubmit only
interrupted task IDs without rerunning preparation or cancelling other jobs:

```bash
export SOURCE_STEPS=500400
export SOURCE_OUTPUT_DIR="$CP_ROOT/outputs/source_coverage_v1"
sbatch --chdir="$CP_REPO_ROOT" --export=ALL --array='0-1%2' \
    --output="$CP_ROOT/outputs/slurm-log/source-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/source-%A_%a.err" \
    run/slurm/source_pretrain.sh train
```

Use the same `SOURCE_STEPS`, `SOURCE_OUTPUT_DIR`, and ImageNet path when resuming.
The latest saved checkpoint restores model, optimizer, scheduler, progress, and
image counts, online probe, its optimizer, and kNN queues. The shuffled index
stream resumes at the next update; stochastic
augmentations after a restart need not be bitwise identical. Completed training
is not rerun, and a changed experiment configuration is rejected.

## Results

The default root is `outputs/source_coverage_v1`. Each
`CONDITION/seedSEED/` directory contains:

- `config.json`: complete recipe and source sizes.
- `initial/geometry.json` and `final/geometry.json`: fixed held-out target metrics.
- `initial/DATASET.npz` and `final/DATASET.npz`: raw final-normalized CLS features,
  selected test indices, and metadata.
- `checkpoints/last.ckpt` and periodic checkpoints: resumable training state,
  saved every 1,000 updates; `checkpoints/final.ckpt` marks finished training.
- `encoder.pt`: final encoder state and recipe.
- `training.json`: completed update count and actual per-domain image counts.
- `logs/version_*/metrics.csv`: LeJEPA losses and online validation curves.
- `online_metrics.json`: final checkpoint's online kNN/LP validation metrics.

Geometry uses up to 5,000 fixed seed-42 test images per target, independently of
training seeds. Clean images are resized to 224 by 224 and ImageNet-normalized;
metrics use L2-normalized CLS features without the projector. Uniformity is
`U = log(mean(exp(-2 * squared_distance)))` over distinct pairs. Smaller, more
negative U means more spread-out features. Mean pairwise cosine and uncentered
RankMe accompany U; these descriptors are not downstream accuracy measures.

```bash
"$CP_PYTHON" run/source_pretrain.py report --root "$CP_ROOT"
```

The report writes `geometry_results.csv` and `paired_geometry.csv`, including
matched-seed `mixed - imagenet` uniformity differences. Pass `--output-dir` to
report a nondefault output root. Only completed runs with final geometry are
included.

Inspect Slurm history, including finished tasks, using the IDs printed at submission:

```bash
sacct -X -j YOUR_PREP_JOB,YOUR_TRAIN_ARRAY \
    --format=JobID%24,State%24,ExitCode,Elapsed
```

## Online Monitoring

Native `stable_pretraining.callbacks.OnlineKNN` and `OnlineProbe` evaluate
ImageNet top-1/top-5 accuracy every 5,000 updates, and once at the final checkpoint.
Short runs use their total step count as the validation interval.
Both use L2-normalized backbone CLS features, not projector outputs.
The LP uses first-global-crop training features, AdamW at `1e-3`, weight decay
`1e-6`, and a constant learning rate; its gradients do not reach the encoder.
Target-domain labels are ignored by the LP and excluded from the kNN queue.
The kNN uses the native cosine-distance weighting, `k=20`, temperature `0.07`,
and the most recent 20,000 ImageNet training features.

Validation uses the same fixed clean images in A and B. Metrics are logged as
`eval/imagenet_knn_top1`, `eval/imagenet_knn_top5`,
`eval/imagenet_lp_top1_epoch`, and `eval/imagenet_lp_top5_epoch`.
Accuracy values are fractions, not percentages. Online scores use a changing
encoder and historical training features; they are monitoring signals, not the
offline frozen-feature kNN/LP protocol. Mixed training provides fewer ImageNet
probe updates/examples at a fixed total budget. No early stopping or checkpoint
selection uses these scores or the target uniformity outcome.
