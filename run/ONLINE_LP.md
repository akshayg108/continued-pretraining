# LP-Only Reruns

Use `run/lp_only.py` for the existing experiments. It trains only a new linear
classifier. It never starts or resumes CP, runs FT, extracts kNN banks, computes
geometry, or stages ImageNet reference images. Existing CP weights, result JSONs,
kNN scores, and geometry files are left untouched.

## LP Settings

- 150 complete passes through the training split, including the final short batch.
- Batch 512, with the entire batch passed through the frozen encoder by default.
- Optional `--forward-batch-size 128` splits only encoder forwards. It does not
  change the classifier batch or its optimizer update count. No automatic OOM retry.
- Frozen encoder in evaluation mode; only the linear classifier receives gradients.
- Fresh random resized crop and horizontal flip whenever an image is read.
- Deterministic test resize and encoder-specific pretrained mean/std.
- Existing readout, L2 feature normalization, Adam LR 0.001, zero weight decay.
- No minimum-step floor, scheduler change, or test-set hyperparameter selection.

150 epochs is the agreed project budget, not the DINOv3 official recipe.
[DINOv3's public LP defaults](https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/linear.py)
are 10 nominal epochs of 1,250 iterations each, or 12,500 optimizer updates, with
batch 128 per GPU. Those nominal epochs are not full passes over each target dataset.
Full-batch 512 GPU memory use has not been measured locally.
The Slurm time limit remains 96 hours. A large pre-CP dataset job contains
15 fresh-view LP fits and may exceed that limit; runtime has not been benchmarked
on the cluster. Completed seed results are retained and skipped on resubmission.

## Task Layout

- Pre-CP: 23 dataset tasks, each with the five encoders and seeds 42, 43, 44.
  The seven MedMNIST datasets have separate jobs.
- Canonical encoder readouts: DINOv3 CLS, CLIP CLS, SigLIP-2 MAP, MAE-Mean,
  and DINOv3-L CLS. MAE uses `LayerNorm(mean(raw patch tokens))`.
- Post-CP: original CP task IDs, one encoder x dataset x CP method per job.
  Each job processes the completed checkpoints among seeds 42, 43, 44.
- The post array includes only tasks with at least one completed checkpoint.
  Missing and incomplete checkpoints are skipped, never trained.
- Every job copies its dataset's processed cache to node-local storage once.
- Pre-CP uses V100. Post-CP retains the original V100/A100 task routing.
  Arrays run sequentially with `%10`, so these LP jobs total at most ten running
  tasks. This cap does not include unrelated source-pretraining jobs.

## Delete Interrupted Checkpoints

Run from the updated repository on the cluster, after the canceled jobs have
finished exiting. This command inspects only the ten canceled tasks listed below
and deletes only their incomplete `cp.ckpt` files. Finished seeds, logs, settings,
and evaluation results are preserved. No server files have been deleted locally.

```bash
(
    set -euo pipefail
    export CP_ROOT=/scratch/gs4133/zhd/CP_new
    source run/precp_env.sh
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export LD_LIBRARY_PATH="$CP_ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    active=$(squeue -h -u "$USER" -o '%F')
    if printf '%s\n' "$active" | grep -qx 18138000; then
        printf 'Array 18138000 still has active or completing tasks. Wait before deleting.\n' >&2
        exit 1
    fi

    "$CP_PYTHON" - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, "run")
from cp_full import TASKS, SEEDS, seed_dir
from lp_only import checkpoint_status

root = Path(os.environ["CP_ROOT"])
for task_id in (57, 58, 59, 61, 62, 63, 65, 66, 67, 69):
    task = TASKS[task_id]
    for seed in SEEDS:
        path = seed_dir(root, task, seed) / "cp.ckpt"
        status = checkpoint_status(path, task)
        if status == "incomplete":
            path.unlink()
            print(f"DELETED incomplete: {path}")
        else:
            print(f"KEEP {status}: {path}")
PY
)
```

A job canceled during its previous LP may already have a completely trained CP
checkpoint. Such a checkpoint is retained and is eligible for the LP-only rerun.

## Submit

Use an updated checkout that is not referenced by another active job. Do not
resubmit the old CP training launcher for this recovery.

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
source run/precp_env.sh
bash run/slurm/submit_lp_only.sh
squeue -u "$USER"
```

The launcher uses 150 epochs and batch 512 by default. It submits pre-CP first,
then the completed V100 and A100 post-CP tasks, skipping empty post arrays.
Post arrays use `afterany`: they need checkpoint weights, not newly computed
pre-CP scores, and therefore do not become permanently blocked by a failed
pre-CP LP. Wait for both phases before comparing scores.

For an explicit forward chunk on a subsequent submission:

```bash
bash run/slurm/submit_lp_only.sh --forward-batch-size 128
```

Do not submit both examples at once. A chunked rerun uses the same classifier
batch but may have floating-point differences; use the same settings pre/post.

## Outputs

Pre-CP LP-only JSONs and adjacent logs:
`outputs/precp_full/lp_online_v1/lp_results/ENCODER/DATASET/seedSEED.{json,log}`.

Post-CP LP-only JSON:
`outputs/results/DATASET/ENCODER/METHOD/Full/SEED/lp_online_v1/lp.json`.
These are LP-only records, not replacement full-evaluation records. The old
full JSONs retain their cached-view LP values alongside kNN and geometry.

Inspect the job output for `RUN`, `DONE`, and `SKIP` entries and the per-seed
LP logs for training progress. Rerunning the same LP command skips matching
completed LP-only results; it never resumes a partly trained linear head.
