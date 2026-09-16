# Four-encoder native-normalization audit

This submits exactly four array elements, each using one V100. Datasets and
seeds run sequentially inside each element. No CP or full fine-tuning is run.

| Array element | Encoder | Coverage |
| --- | --- | --- |
| 0 | SigLIP-2 | 14 targets x seeds 42/43/44, plus Food-101 geometry at seed 42 |
| 1 | CLIP | All 15 targets x seeds 42/43/44 |
| 2 | DINOv3 ViT-B | All 15 targets x seeds 42/43/44 |
| 3 | MAE | All 15 targets x seeds 42/43/44 |

Food-101's SigLIP-2 native-normalization kNN/LP baseline has already been run.
It is not recomputed or silently imported into this audit. Its geometry-only
record is explicitly identified in the summary.

## Launch

Synchronize the new evaluator and this launcher directory to the cluster first.
From the cluster repository:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining
bash run/slurm/pre-cp-official/submit.sh --dry-run
bash run/slurm/pre-cp-official/submit.sh
```

The dry run does not submit jobs, load models, or write output directories.
The actual submission reserves `gpu:v100:1`, 8 CPUs, 96 GB host RAM, and 96 hours
per array element. It activates the existing `env` conda environment.

Results are written beneath
`/scratch/gs4133/zhd/CP/outputs/precp_official_norm_v1/<array-job-id>/<encoder>/<dataset>/`.
Each seed produces `seed42.json`, `seed43.json`, or `seed44.json`.
SigLIP Food-101 produces `geometry_seed42.json` instead.
Existing result files are never overwritten. Each dataset is copied into a
private node-local cache that is removed when the next dataset starts. Shared
processed caches and old experiment outputs are not modified.

Inspect a completed or partial run:

```bash
python3 -m eval.precp_official_norm summarize \
  --outdir /scratch/gs4133/zhd/CP/outputs/precp_official_norm_v1/ARRAY_JOB_ID
```

The CSV includes each seed, both F1 and accuracy, and both uniformity estimates.
Missing or invalid results print `CHECK`, and an incomplete summary exits 1.
The final validation count goes to stderr so redirecting stdout gives a clean CSV.

## Controlled changes

Mean/std are read from each loaded model's `pretrained_cfg` and checked against
the expected checkpoint values. The override is applied to all evaluation
transforms, including augmented LP training images and clean kNN/test images.

All other choices stay with the existing CP pipeline:

- Exact existing timm model IDs and pretrained weights, not CP checkpoints.
- 224-pixel images, original splits, and the complete MAX training set.
- CLS readout for DINOv3 and CLIP, MAP for SigLIP-2, mean patch tokens for MAE.
- FP32 feature extraction, batch 64, kNN k=20.
- Existing augmented LP training features and clean test features.
- PyTorch LP: Adam, learning rate 0.001, batch 512, at least 150 epochs and
  10,000 steps. Only the linear classifier is optimized.

This is an **official mean/std audit**, not a change to every aspect of the
published models' inference pipelines. In particular, CLIP retains
`vit_base_patch16_clip_224.openai` and its existing activation implementation.
It is not silently switched to the QuickGELU model variant or projected CLIP
embedding. Native resize/crop recipes are not substituted for the existing
224-pixel transforms.

## Uniformity

`uniformity_t2` is `log(mean(exp(-2 * ||z_i-z_j||^2)))` after L2 normalization,
using **every distinct pair in the clean MAX training split**. Self-pairs are
excluded. Row blocks bound GPU memory without approximating the pair average.
No test images or ImageNet reference set are used.

`uniformity_t2_subset` uses the historical sampling rule: select up to 5000
stratified training examples with random state 42, then up to 3000 examples
without replacement with random state 42. This secondary estimate makes the
estimator change visible. It does not assert identical sample membership to
the old geometry script, which used a separate loader. Only this secondary
subsample selection uses class labels. The full-pair descriptor does not.

The old geometry script used ImageNet mean/std for every encoder. Its DINOv3
and MAE normalization was already native, unlike some dataset-specific baseline
transforms. This audit places geometry and frozen evaluation under the same
native mean/std and MAX split.

Each JSON records normalization, transforms, model weights hash, code hashes,
package versions, seed, index/label hashes, split fingerprint, and GPU. A failure
does not create a successful result and does not prevent other seeds/datasets
from being attempted. Any failure makes the array element exit nonzero.

## Local checks

```bash
python3 -m pytest -q tests/test_precp_official_norm.py
bash -n run/slurm/pre-cp-official/submit.sh
bash -n run/slurm/pre-cp-official/array.sh
```

CPU tests do not substitute for a cluster-side CUDA evaluation.
