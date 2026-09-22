# Held-out continued pre-training suite

This suite evaluates DINOv3 and CLIP with LeJEPA, DIET, and SimCLR on four
out-of-distribution datasets (BloodMNIST, TissueMNIST, AID, RESISC45) and four
fine-grained datasets (Stanford Dogs, Jena Flowers 30, Flavia, IP102). Each fit
uses 1,000 training images, the last two encoder blocks, official model mean and
standard deviation for both encoders, and seeds 42, 43, and 44.

Install the pinned `stable-datasets` revision into the same interpreter used by
Slurm; do not use `conda run`:

```bash
/home/gs4133/.conda/envs/env/bin/python3 -m pip install \
  'git+https://github.com/haodongzhang0118/stable-datasets.git@cc01e36e7f5b2eac04684852002e69bc8ebe4541'
```

From the cluster repository, validate the plan and then submit:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining
bash run/slurm/heldout-cp/submit.sh --concurrency 12 --dry-run
bash run/slurm/heldout-cp/submit.sh --concurrency 12
```

The first array contains eight preparation jobs that compute pre-CP metrics and
initial uniformity. Each dataset's six CP jobs depend only on that dataset's
successful preparation, not on the entire preparation array. For example, AID
preparation is array element 2; CP elements 6, 7, 8, 30, 31, and 32 use
`afterok:PREP_JOB_2`. They become eligible as soon as AID preparation succeeds,
subject to resource availability and queue limits. A different dataset's failed
preparation does not block AID.

The launcher submits the 48-job CP array held, assigns every task's dependency,
and releases it only after all dependency updates succeed. If an update fails,
the CP array remains held and the script prints its job ID. Do not manually
release it until every dependency is configured. Each CP element runs its three
seeds serially and does no full fine-tuning. Both arrays use one V100 per job,
eight CPUs, 96 GB RAM, and a 96-hour limit. `--concurrency` caps each array;
all CP tasks share one throttle rather than splitting capacity between datasets.
Preparation and CP may overlap. Their combined running count is also subject
to the cluster's per-user nvidia QoS limit, previously confirmed as 12.

The hypothesis, dataset list, and recipe are fixed in the submission manifest.
Each preparation job freezes both encoders' initial scores and baseline hashes
in `predictions/DATASET.json` before that dataset's CP begins. The complete
cross-dataset ordering follows from these fixed scores once all preparations
finish; this is not a claim that all eight scores were available before any CP.

The launcher writes a new manifest under
`$HELDOUT_OUTPUT_BASE/heldout_cp_manifests` for every invocation. Override
cluster locations with `HELDOUT_REPO_ROOT`, `HELDOUT_CACHE_DIR`,
`HELDOUT_OUTPUT_BASE`, and `HELDOUT_LOG_DIR`. Override the interpreter only with
an absolute executable path in `HELDOUT_PYTHON`.

Collect the report with the manifest printed by the launcher. This also works
while some datasets are still being prepared or trained:

```bash
/home/gs4133/.conda/envs/env/bin/python3 -m eval.heldout_cp collect \
  --manifest /scratch/gs4133/zhd/CP/outputs/heldout_cp_manifests/MANIFEST.json \
  --outdir /scratch/gs4133/zhd/CP/outputs/heldout_cp_report
```

Official test splits are used when available. AID, RESISC45, Jena Flowers
30-all, and Flavia use a fixed stratified 80/10/10 split with seed 42. Stanford
Dogs uses 10% of its official training split for validation. Only model-native
mean and standard deviation are substituted; this is not a claim that the full
official resize and crop transform is reproduced.

CP, the clean kNN reference set, and the augmented linear-probe training set
share the same 1,000 distinct image indices for each seed, across encoders and
objectives. The entire held-out test partition is evaluated. Every fit starts
from public pretrained weights in a separate process and checkpoint directory.
Geometry uses at most 3,000 images from the training pool, never validation or
test data. Reports include per-seed pre/post scores, paired differences and
sample standard deviations. Correlations are calculated across eight dataset
means only after all three seeds are verified for every target in the comparison.

Results are stored in `$HELDOUT_OUTPUT_BASE/heldout_cp_1000_v1`. Resubmission
skips verified complete runs; an incomplete seed restarts from public weights
in a new attempt directory. Do not modify code or update dependencies while
the arrays are queued or running: manifests lock the implementation and
preparation records lock dependency versions and data identities.

Run offline checks without downloading datasets or allocating a GPU:

```bash
python3 -m pytest tests/test_heldout_data.py tests/test_heldout_cp.py \
  tests/test_heldout_runtime.py tests/test_heldout_cp_slurm.py -q
```
