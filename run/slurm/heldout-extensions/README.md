# Held-out targets: SigLIP-2 and DINOv3-L

This independent suite adds two encoders to the existing eight-target,
1,000-image experiment. It does not change the running DINOv3-B/CLIP suite
or its frozen implementation hash.

## Fixed Configuration

| Setting | SigLIP-2 (`SigLIP` in artifacts) | DINOv3-L (`DINOv3L` in artifacts) |
|---|---|---|
| timm checkpoint | `vit_base_patch16_siglip_224.v2_webli` | `vit_large_patch16_dinov3.lvd1689m` |
| Features | Native MAP readout, 768 dimensions | CLS readout, 1,024 dimensions |
| RGB mean | `(0.5, 0.5, 0.5)` | `(0.485, 0.456, 0.406)` |
| RGB std | `(0.5, 0.5, 0.5)` | `(0.229, 0.224, 0.225)` |
| Preparation GPU | V100 | A100 |
| LeJEPA GPU | A100 | A100 |
| DIET / SimCLR GPU | V100 | A100 |
| LeJEPA / SimCLR batch, accumulation | 256, 1 | 256, 1 |
| DIET batch, accumulation | 32, 1 | 32, 1 |

Mean/std are read from the loaded checkpoint configuration and checked against
the values above. See the [SigLIP-2 checkpoint configuration](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli/blob/main/config.json)
and [DINOv3-L checkpoint configuration](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/blob/main/config.json).
The suite retains the original 224-pixel resize/crop/augmentation protocol;
it does not adopt every checkpoint-default transform. In particular, DINOv3-L
continues to use the paper's CLS readout and 224-pixel inputs.

All recipes use 150 CP epochs, 15 frozen epochs, then the last two blocks;
learning rate `1e-4`, weight decay `0.05`, and seeds `42,43,44`.
LeJEPA has eight views, SimCLR two, and DIET one. Both extension encoders use
the original held-out base-model recipes, including batch 256 without
gradient accumulation for LeJEPA/SimCLR. This changes only the new ViT-L
held-out suite; the older ViT-L batch-128, accumulation-2 scripts are unchanged.
No FT is run. Each seed starts from public pretrained weights, never from a
previous seed or an incomplete checkpoint.

A100 jobs use `--gres=gpu:a100:1`, without an 80GB constraint. No automatic
batch-size or precision fallback is applied. Local tests do not measure GPU
memory usage; a 40GB allocation still requires an actual training check,
including the transition to two trainable blocks after the 15 frozen epochs.

## Data and Preparation

The source is the already prepared `heldout_cp_1000_v1` manifest. All eight
original preparations must be complete; CP completion is not required.
The planner checks original artifact hashes and requires DINOv3-B and CLIP
to have identical training indices, data fingerprints, and geometry indices.
No original artifact or prediction record is replaced.

New preparation jobs reuse these exact samples and dataset partitions:

- CP training, clean kNN reference, and augmented LP training use the same
  1,000 distinct training images, paired before/after CP and across encoders.
- Geometry uses the original at-most-3,000 training-pool indices, with fresh
  features from the new encoder.
- Test sets and labels are unchanged. The experiment does not subsample the
  test set to 1,000 images.
- New pre-CP kNN/LP scores and uniformity are computed for each encoder.
  No scores from a different encoder are reused.
- Scores slightly outside `[0,1]` by at most `1e-6` are clipped, with raw
  values recorded under `evaluation_numerics`. Other invalid scores fail.

There are **16 preparation jobs**, one per encoder-target pair, followed by
**48 CP jobs / 144 fits**. Each CP job runs one encoder, one objective, one
dataset, and three serial seeds in separate Python processes. A preparation
freezes geometry and all three baselines before its three CP jobs start.
Dependencies are encoder-target-local; no global preparation barrier is added.

## Submit

After synchronizing the code to the server:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining
export HELDOUT_PYTHON=/home/gs4133/.conda/envs/env/bin/python3
SOURCE=/scratch/gs4133/zhd/CP/outputs/heldout_cp_manifests/heldout-cp-20260922T045909Z-2968777.json

bash run/slurm/heldout-extensions/submit.sh \
    --source-manifest "$SOURCE" --concurrency 12 --dry-run

bash run/slurm/heldout-extensions/submit.sh \
    --source-manifest "$SOURCE" --concurrency 12
```

The dry run verifies source preparations and writes a new manifest, but makes
no scheduler calls. The actual submission creates a separate manifest and
prints its path. Four arrays are submitted: V100/A100 preparation arrays and
V100/A100 CP arrays. CP arrays stay held until every dependency has been set.
On a dependency configuration failure, do not release the arrays manually
without repairing all dependencies.

Each array uses `%12`; the cluster's `nvidia` QoS enforces the shared 12-job
user limit, including existing jobs. The script does not divide this into
smaller per-GPU quotas and does not cancel existing work. `--concurrency`
sets each array's throttle, not a new global limit below the QoS limit.

Optional environment overrides: `HELDOUT_REPO_ROOT`, `HELDOUT_PYTHON`,
`HELDOUT_OUTPUT_BASE`, `HELDOUT_CACHE_DIR`. Use an absolute Python executable;
no `conda run`, environment activation, or module reload is performed.
Do not resubmit the whole suite while its jobs are still active.

## Array Mapping

| Target | SigLIP prep | SigLIP CP | DINOv3-L prep | DINOv3-L CP |
|---|---:|---|---:|---|
| BloodMNIST | 0 | 0, 1, 2 | 8 | 24, 25, 26 |
| TissueMNIST | 1 | 3, 4, 5 | 9 | 27, 28, 29 |
| AID | 2 | 6, 7, 8 | 10 | 30, 31, 32 |
| RESISC45 | 3 | 9, 10, 11 | 11 | 33, 34, 35 |
| StanfordDogs | 4 | 12, 13, 14 | 12 | 36, 37, 38 |
| JenaFlowers30 | 5 | 15, 16, 17 | 13 | 39, 40, 41 |
| Flavia | 6 | 18, 19, 20 | 14 | 42, 43, 44 |
| IP102 | 7 | 21, 22, 23 | 15 | 45, 46, 47 |

Every CP triple is **LeJEPA, DIET, SimCLR**, in that order. Array IDs only
have meaning together with their Slurm parent job ID and extension manifest.

## Outputs and Collection

- Manifests and submission receipts: `outputs/heldout_extension_manifests/`.
- New results: `outputs/heldout_extensions_1000_v1/`.
- Preparation records: `pre/{encoder}/{dataset}/seed{seed}.json`.
- Geometry: `geometry/{encoder}/{dataset}.json`.
- Frozen initial predictions: `predictions/{encoder}/{dataset}.json`.
- CP results: `cp_results/{encoder}/{method}/{dataset}/seed{seed}.json`.
- Attempt logs/checkpoints: `attempts/{encoder}/{method}/{dataset}/seed{seed}/{attempt_id}/`.
- Slurm logs: `outputs/slurm-log/heldout-extensions/heldout-ext-{prep,cp}-%A_%a.{out,err}`.

Set `MANIFEST` to the path printed by the actual submission, then run:

```bash
PY=/home/gs4133/.conda/envs/env/bin/python3
REPORTS=/scratch/gs4133/zhd/CP/outputs/heldout_extensions_1000_v1/reports
"$PY" -m eval.heldout_extensions collect \
    --manifest "$MANIFEST" --outdir "$REPORTS"
cat "$REPORTS/status.json"
cat "$REPORTS/summary.csv"
cat "$REPORTS/correlations.csv"
```

`seed_results.csv` contains verified pre/post/delta F1 and accuracy per seed.
`summary.csv` reports paired F1 means and sample SD using only verified runs.
Missing results remain missing, not zero. `correlations.csv` reports each
objective and the mean of the three objectives only when all eight targets
have all required seeds. Original results remain in their separate namespace.

## Local Tests

```bash
python3 -m pytest -q tests/test_heldout_extensions.py \
    tests/test_heldout_extensions_runtime.py tests/test_heldout_extensions_slurm.py
bash -n run/slurm/heldout-extensions/submit.sh
bash -n run/slurm/heldout-extensions/worker.sh
```

Tests use synthetic preparation artifacts and mock GPU/training boundaries.
They exercise both readouts, both feature dimensions, all six encoder-method
setups, numerical roundoff, resource grouping, dependency failures, fresh
seed processes, verified skips, and paired result collection. They do not
download checkpoints or run real GPU training.
