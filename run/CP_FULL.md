# Full-Training CP Grid

This grid uses all training images from the 11 targets whose training split has
fewer than 10,000 images: breastmnist, dermamnist, dtd, fgvc_aircraft, cars196,
cub200, flowers102, oxford_pet, aid, jena_flowers30, and flavia. The threshold is
on the training split, not the sum of all splits. Seeds are 42, 43, and 44.

There are 176 tasks: 99 on V100 and 77 on A100, each running three seeds
sequentially (528 fits). No preparation job or preparation dependency is added;
submission first checks the existing pre-CP results and reference banks.
Each task stages target and reference images on node-local storage.
The four encoders are DINOv3-B, CLIP, SigLIP-2, and DINOv3-L; each uses LeJEPA,
SimCLR, DIET, and MAE objectives. All LeJEPA and all DINOv3-L tasks use A100;
the other tasks use V100.

## Training And Evaluation

- Use public pretrained weights and full training splits, with no supervised FT.
- Train for 150 epochs: freeze the backbone for the first 15, then train its last
  two blocks. Warmup is 15 epochs; all other settings come from the runner's fixed
  recipe. Full training data does not mean full-parameter backbone training.
- DIET uses batch 32 with accumulation 1. LeJEPA and SimCLR use 256 with
  accumulation 1, except on DINOv3-L, which uses 128 with accumulation 2.
- LeJEPA uses lambda 0.05 and 1,024 random SigReg projection directions.
  Its learned MLP projector still outputs 128 dimensions (hidden size 2,048).
- MAE uses batch 256 with accumulation 1 on all four encoders. This choice
  for MAE-CP on DINOv3-L has not yet been
  validated for GPU memory usage; GPU training is required to confirm it.
- Reuse matching pre-CP baselines; evaluate post-CP kNN, LP, and geometry.
  Pretrained normalization and encoder readout must match the baseline.
  Frozen feature extraction uses batch 32 and two workers, independently of CP
  training batches. LP optimization keeps the existing evaluation recipe.
  Merge the matching seed's pre-CP scores and geometry into its result JSON,
  recording post-minus-pre deltas without rerunning the baseline.
- All five geometry metrics (`uniformity_t2`, `mean_pairwise_cos`,
  `rankme_l2_uncentered`, `mmd_rbf`, and `neighbor_overlap_k50`) use the same
  seed-42 target subset of `min(n_train, 5000)` images. Re-encode the same 5,000
  ImageNet reference images with each CP seed's current post-CP encoder.
- Keep checkpoints and result/configuration metadata for each seed. Completed
  results are skipped only when the runner validates them; keep partially
  completed artifacts for resumption instead of deleting them.

## Isolated Cluster Deployment

Do not pull into a directory still used by running or queued source/pre-CP jobs.
In particular, an old pre-CP driver launches new Python child processes between
evaluations, so a live pull can mix revisions even when its current process has
already imported Python modules. Leave the existing original directory and the
separate MAE worktree unchanged.

After the CP changes are committed and made available on remote `organized`,
create a new detached worktree at that exact revision. These commands do not
publish the local changes or imply they have already been pushed:

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
ORIGINAL="$CP_ROOT/continued-pretraining"
git -C "$ORIGINAL" fetch origin organized
REV=$(git -C "$ORIGINAL" rev-parse origin/organized)
CP_WORKTREE="$CP_ROOT/continued-pretraining-cp-full-${REV:0:12}"
git -C "$ORIGINAL" worktree add --detach "$CP_WORKTREE" "$REV"
cd "$CP_WORKTREE"
source run/precp_env.sh

"$CP_PYTHON" run/cp_full.py check --root "$CP_ROOT"
"$CP_PYTHON" run/cp_full.py list
bash run/slurm/submit_cp_full.sh
```

Reuse the existing environment; do not reinstall or upgrade it while other jobs
are using it. Fetching and adding a worktree do not update the original working
tree. A pull there is safe only when no live or queued job can read any file it
would change; waiting for all original-directory jobs to finish is the simplest
way to establish that condition.

Both arrays request account `civil`, partition/QoS `nvidia`, one GPU, eight CPUs,
96 GB host memory, and 96 hours per task. A100 requests use `gpu:a100:1` with no
GPU-memory-size constraint. The separate concurrency limits default to 12:

```bash
CP_V100_CONCURRENCY=12 CP_A100_CONCURRENCY=12 \
    bash run/slurm/submit_cp_full.sh
```

Additional arguments to the submission script are passed to both `sbatch`
calls, for example `--time=48:00:00`. Array task IDs, GPU type, working directory,
and log paths remain controlled by the script. The two limits are independent;
cluster QoS limits may reduce their combined running count. Submission is not
transactional: if the second submission fails, the first array remains queued
and its job ID has already been printed. Do not blindly resubmit both arrays.

## Outputs

Each seed writes to:

```text
outputs/results/<dataset>/<encoder>/<method>/Full/<42|43|44>/
    cp.ckpt
    result.json
    config.json
    post_reference.npz
    post_features.npz
    run.log
```

The output encoder folder is `SigLiP-2` to match the existing result folders;
the display name and baseline lookup remain `SigLIP-2`. Existing pre-CP and
source-pretraining output namespaces are untouched.

```bash
"$CP_PYTHON" run/cp_full.py report --root "$CP_ROOT"
```
