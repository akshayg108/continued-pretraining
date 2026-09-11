# DINOv3 ViT-L: complete the remaining eight datasets

This entry point runs **CP plus pre/post kNN and linear probing only**.
It never requests pre-CP or post-CP fine-tuning. The existing seven-dataset
panel and its checkpoints are not changed or reused as initial weights.
Each new CP fit starts from the public DINOv3 ViT-L pretrained weights.

## Submit

Run from the repository on the cluster after syncing these new files:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining

# Preview a selection. Nothing is submitted or trained.
bash run/slurm/cp-L/completion/submit.sh \
  --datasets breastmnist flowers102 oxford_pet --dry-run

# Submit the same selection: 9 tasks, 27 CP fits.
bash run/slurm/cp-L/completion/submit.sh \
  --datasets breastmnist flowers102 oxford_pet

# Submit all eight missing datasets: 24 tasks, 72 CP fits.
bash run/slurm/cp-L/completion/submit.sh

# Optionally lower concurrency (default 12, allowed 1..12).
bash run/slurm/cp-L/completion/submit.sh --datasets breastmnist --concurrency 1
```

Every selected dataset runs **LeJEPA, SimCLR, and DIET-CP**. For N datasets,
there are 3N Slurm array tasks and 9N CP fits. Each task runs seeds 42, 43,
and 44 sequentially. `--concurrency 1` serializes tasks, not seeds or epochs.
There is no method, seed, epoch, or batch override in this fixed protocol.

Dry run creates a selection manifest and prints the complete task list plus
the three seed commands for the first task. It does not call `sbatch`, stage
data, import torch, or create training artifacts. Tasks are ordered by
decreasing training-set size. This puts long tasks early in the array but
does not guarantee Slurm's scheduling order.

## Dataset selection and unfreezing

| `--datasets` key | MAX images | Trainable Transformer blocks after epoch 15 |
| --- | ---: | ---: |
| `breastmnist` | 546 | Last 2 |
| `octmnist` | 97,477 | All encoder parameters |
| `organamnist` | 34,561 | Last 6 |
| `pathmnist` | 89,996 | All encoder parameters |
| `plant_village` | 43,596 | Last 6 |
| `food101` | 75,750 | All encoder parameters |
| `flowers102` | 1,020 | Last 2 |
| `oxford_pet` | 3,680 | Last 2 |

The size rule is `<10,000 -> 2`, `10,000..25,000 -> 4`,
`25,001..50,000 -> 6`, and `>50,000 -> all`. The existing completed targets
(`galaxy10`, `dermamnist`, `eurosat`, `fgvc_aircraft`, `cars196`, `cub200`,
and `dtd`) are deliberately outside this completion selection.

## Recipe and resources

- Backbone: `vit_large_patch16_dinov3.lvd1689m`, native CLS pooling.
- CP: 150 epochs, encoder frozen for 15, warmup 15, AdamW LR `1e-4`, WD `0.05`.
- LeJEPA: batch 128, accumulation 2, eight views, projector 128/2048, lambda 0.02.
- SimCLR: batch 128, accumulation 2, temperature 0.5, projector 128/2048.
- DIET-CP: batch 32, accumulation 1, label smoothing 0.3, mixup/cutmix probability 0.
- kNN uses k=20. Each invocation computes its own pre/post frozen metrics,
  including macro-F1 and accuracy. Pretrained baselines are not shared across
  objective tasks. These evaluations do not fine-tune the encoder.
- One A100 80GB, 8 CPUs, 96GB host RAM, and a **96-hour limit for all three
  seeds combined**. Up to 12 tasks from this array can run concurrently.

The batches preserve the existing ViT-L recipes. In particular, SimCLR's
negative pool is based on a 128-image forward batch, not a single 256-image
batch. Accumulating two gradients does not change that distinction.

Both the submission command and array script request `--gres=gpu:a100:1`
and `--constraint=80g`, using the same constraint as the existing ViT-L scripts.
The runner checks that exactly one visible GPU is an A100 with at least
75 GiB of driver-reported memory (a full 80GB card reports about 79 GiB).
It rejects 40GB cards, smaller MIG slices, and other GPU models.

The processed dataset must already exist under the shared cache. Each task
copies just that dataset to a private directory on `$TMPDIR`, `/tmpdata`, or
`/dev/shm`, checks space with 5GB headroom, and removes only its private copy
on exit. Missing caches or insufficient local space fail rather than falling
back to random reads on shared scratch.

**Runtime boundary:** local tests do not establish GPU memory use or duration.
The newly added full-encoder runs are more demanding than the old last-two-
or last-four-block runs, especially eight-view LeJEPA. A small-dataset smoke
does not validate full-depth memory use. Check a full-depth job after its
15 frozen epochs before assuming all jobs fit. There is no automatic
microbatch reduction, since it would change the protocol.

## Results and safe restarts

The default output namespace is separate from the old `ckpts/cp-L` directory:

```text
/scratch/gs4133/zhd/CP/outputs/
  vitl_completion_v1/
    checkpoints/<method>/<dataset>/cp/<dataset>_<model>_n<N>_s<seed>.ckpt
    cp_results/<method>/<dataset>/seed<seed>.json
    provenance/<method>/<dataset>/seed<seed>.json
  vitl_completion_manifests/selection-<timestamp>-<pid>.json
  slurm-log/vitl-completion-<array>_<task>.out
  slurm-log/vitl-completion-<array>_<task>.err
```

Methods in paths are `LeJEPA`, `SimCLR`, and `DIET`. Checkpoint saves are
CP checkpoints only. No FT weights or FT metrics are produced.

Resubmit the same dataset selection to continue an interrupted task. Valid
completed seeds are skipped after checking result and checkpoint hashes.
Incomplete, receipt-bound checkpoints resume CP optimizer/training state.
A failed seed is reported and the other seeds are still attempted. The task
exits nonzero if any seed fails. A later selection may contain fewer or more
datasets without changing existing experiment identities or output paths.

Unknown artifacts, changed completed files, changed recipes, or changed
recorded software/training code stop a restart rather than silently mixing
results. Do not edit the training code mid-run. Do not delete or import old
checkpoints into this namespace. A per-seed lock prevents concurrent writes
from accidental duplicate submissions.

Optional path overrides are `VITL_COMPLETION_OUTPUT_BASE`,
`VITL_COMPLETION_CACHE_DIR`, `VITL_COMPLETION_LOG_DIR`, and
`VITL_COMPLETION_REPO_ROOT`. `VITL_COMPLETION_MANIFEST` can specify a new
manifest filename, which must not already exist. For local dry runs, set
`VITL_COMPLETION_OUTPUT_BASE` to a writable local temporary directory.

## Local checks

```bash
python3 -m pytest -q tests/test_vitl_completion.py tests/test_vitl_completion_slurm.py
python3 -m pytest -q tests
bash -n run/slurm/cp-L/completion/submit.sh run/slurm/cp-L/completion/array.sh
```

The tests simulate GPU properties, Slurm submission, training, interruption,
hash validation, and node-local staging. They do not submit real jobs.
