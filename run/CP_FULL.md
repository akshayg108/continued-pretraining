# Full-Training CP Grid

This grid uses all training images from the 11 targets whose training split has
fewer than 10,000 images: breastmnist, dermamnist, dtd, fgvc_aircraft, cars196,
cub200, flowers102, oxford_pet, aid, jena_flowers30, and flavia. The threshold is
on the training split, not the sum of all splits. Seeds are 42, 43, and 44.

There are 220 tasks: 132 on V100 and 88 on A100, each running three seeds
sequentially (660 fits). No preparation job or preparation dependency is added;
submission first checks the existing pre-CP results and reference banks.
It also imports the training module, SQLite, and the SPT registry with one CPU
thread before submitting, leaving compute-node thread settings unchanged.
The import check and CP tasks prefer `$CP_ROOT/env/lib` for runtime libraries;
this avoids loading an older system C++ runtime for ICU/SQLite. The override
is local to the CP process tree, not the shared environment or `sbatch`.
Each task stages target and reference images on node-local storage.
The five encoders are DINOv3-B, CLIP, SigLIP-2, DINOv3-L, and MAE; each uses
LeJEPA, SimCLR, DIET, and MAE objectives. All LeJEPA and all DINOv3-L tasks use A100;
the other tasks use V100.
The V100 array runs first with a `%10` limit. The A100 array has the same limit
and starts only after every V100 task has ended, so at most 10 CP jobs run at once.
When existing `cp-full` jobs are found, the new V100 array also waits for all of
them to end. Submit batches sequentially from one terminal; the queue snapshot
does not provide a lock against simultaneous independent submissions.

## Training And Evaluation

- Use public pretrained weights and full training splits, with no supervised FT.
- Train for 150 epochs: freeze the backbone for the first 15, then train its last
  two blocks. Warmup is 15 epochs; all other settings come from the runner's fixed
  recipe. Full training data does not mean full-parameter backbone training.
- DIET uses batch 32 with accumulation 1. LeJEPA and SimCLR use 256 with
  accumulation 1, except on DINOv3-L, which uses 128 with accumulation 2.
- LeJEPA uses lambda 0.05 and 1,024 random SigReg projection directions.
  Its learned MLP projector still outputs 128 dimensions (hidden size 2,048).
- MAE-CP uses batch 256 with accumulation 1 on all five encoders. This choice
  for MAE-CP on DINOv3-L has not yet been
  validated for GPU memory usage; GPU training is required to confirm it.
- Reuse matching pre-CP baselines; evaluate post-CP kNN, LP, and geometry.
  Pretrained normalization and encoder readout must match the baseline.
  Frozen feature extraction uses batch 32 and two workers, independently of CP
  training batches. LP optimization keeps the existing evaluation recipe.
  Merge the matching seed's pre-CP scores and geometry into its result JSON,
  recording post-minus-pre deltas without rerunning the baseline.
- The MAE encoder uses `vit_base_patch16_224.mae` with
  `LayerNorm(mean(raw patch tokens))`, using the pretrained final LayerNorm.
  Its baseline and pre-CP reference come only from `MAE-Mean`, never old `MAE`
  or `MAE-CLS` results. Baseline, reference, and post-CP metadata must record
  `mae_patch_mean_pretrained_ln_v1`. MAE-CP online full-image validation uses
  that same readout; masked training and reconstruction remain unchanged.
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

Both arrays request account `civil`, partition/QoS `nvidia`, one node, one GPU, eight CPUs,
96 GB host memory, and 96 hours per task. A100 requests use `gpu:a100:1` with no
GPU-memory-size constraint. Set the concurrency limit for each sequential array with:

```bash
CP_CONCURRENCY=10 bash run/slurm/submit_cp_full.sh
```

The full grid makes two ordinary `sbatch` submissions: 132 V100 tasks, then 88 A100
tasks with `--dependency=afterany:<V100-array-ID>`. A100 waits for the entire V100
array to finish, regardless of success or failure. Seeing `Dependency` for A100
while V100 is active is expected. There is no hold, GPU-request update, or release
step. See the [Slurm array documentation](https://slurm.schedmd.com/job_array.html).
Cluster QoS limits and GPU availability can reduce the actual running count.
The first array additionally uses `afterany` for all current user's existing
`cp-full` array parents. This avoids adding another ten jobs alongside an existing
batch. Unrelated source-pretraining and pre-CP jobs are not included in this cap.
`CP_CONCURRENCY` accepts integers from 1 through 10.

Additional arguments are passed to `sbatch`, for example `--time=48:00:00`.
Array task IDs, GPU type, one-node allocation, job name, dependencies, working
directory, and log paths remain controlled by the script. Dependency arguments
and `SBATCH_DEPENDENCY` are rejected rather than overriding that ordering. Logs use
`cp-full-<v100|a100>-<array>_<task>.out` and `.err`.
The old `CP_V100_CONCURRENCY` and `CP_A100_CONCURRENCY` variables are no longer
used. Do not submit another copy of the same encoder selection while it is active.
If the A100 submission fails, the V100 array remains submitted and its ID has
already been printed. Do not blindly resubmit both arrays.

## Add Only The MAE Encoder

The original four-encoder task IDs 0-175 and their recipes are unchanged. MAE
is appended as IDs 176-219: 44 jobs, 132 fits, with 33 V100 jobs followed by 11
A100 jobs. Use a new pinned worktree as above so running or queued jobs continue
reading their original code. Then submit only MAE:

```bash
CP_ENCODERS=MAE CP_CONCURRENCY=10 bash run/slurm/submit_cp_full.sh
```

The preflight checks only MAE's 33 full-training baselines and its matching
reference bank. The launcher automatically waits for existing `cp-full` jobs;
there is no need to cancel or resubmit the other encoders. Data staging, three
sequential seeds, checkpoint resumption, and post-CP evaluation remain unchanged.

`CP_ENCODERS` can contain space-separated encoder keys. Leaving it unset selects
all five encoders. The two-array launcher requires a selection with tasks on
both GPU types to submit both arrays; an empty GPU group is simply skipped.
Inspect or report a selection without submitting anything:

```bash
"$CP_PYTHON" run/cp_full.py list --encoder MAE
"$CP_PYTHON" run/cp_full.py check --root "$CP_ROOT" --encoder MAE
"$CP_PYTHON" run/cp_full.py report --root "$CP_ROOT" --encoder MAE
```

MAE-only reports are `outputs/results/cp_full_results.MAE.csv` and
`cp_full_summary.MAE.csv`, with a completion denominator of 132. They do not
overwrite the combined five-encoder reports, whose denominator is 660.

## Four-Block Full-Training Cohort

Set `CP_GROUP=four-block` to select only the datasets whose **training split**
falls in the original four-block range, `10000 <= n_train <= 25000`:

| Dataset | Training images |
| --- | ---: |
| BloodMNIST | 11,959 |
| Galaxy10 | 14,188 |
| EuroSAT | 16,200 |
| Stanford Dogs | 10,800 |

This adds 80 jobs (five encoders x four datasets x four CP methods), each running
seeds 42/43/44 sequentially, for 240 fits. The new IDs are 220-299; IDs 0-219 and
their two-block recipes remain unchanged. Without `CP_GROUP`, submission still
selects only the original small cohort.

The backbone stays frozen for 15 epochs, then only its **last four Transformer
blocks** train for the remainder of the 150 epochs. This does not include
OCTMNIST, PathMNIST, Food101, or TissueMNIST, whose full splits require full
unfreezing under the original policy, nor the six-block targets.

All other recipe settings above remain unchanged: the existing repeat sampler,
CP batch/accumulation sizes, lambda 0.05, SigReg 1,024 directions, pretrained
normalization, and encoder readouts. MAE uses the new patch-mean readout and its
`MAE-Mean` baseline. Post-CP runs kNN, the original **cached-feature LP**, and all
five geometry metrics, with no FT. Each CP encoder re-encodes the same 5,000
ImageNet reference images; target and reference data are staged node-locally.

The launcher submits two ordinary arrays:

- 64 A100 jobs for DINOv3-B, CLIP, SigLiP-2, and MAE, across all CP methods.
- 16 A100 80GB jobs for DINOv3-L, across all CP methods.

The 80GB request follows the old `MAE` branch's
`run/slurm/cp-L/completion/submit.sh`: `--gres=gpu:a100:1 --constraint=80g`
and `--exclude=cn253,cn259`. Host memory remains 96 GB; `--mem` is not GPU VRAM.
Both groups use eight CPUs, one GPU, and a 96-hour limit. Actual four-block GPU
memory use has not been tested locally.

As with the small cohort, both arrays use `%10`, run sequentially through
`afterany`, and first wait for any existing `cp-full` arrays. Thus this submission
does not add another ten concurrent CP jobs alongside an active batch. A pending
`Dependency` state is expected while that earlier work is running.

In a new pinned worktree (see deployment instructions above):

```bash
unset CP_ENCODERS
CP_GROUP=four-block CP_CONCURRENCY=10 bash run/slurm/submit_cp_full.sh
```

The preflight checks 60 matching pre-CP baselines (five encoders x four datasets x
three seeds) and the reference banks before submitting. Output paths still use
`outputs/results/<dataset>/<encoder>/<method>/Full/<seed>/`. Slurm logs use
`cp-four-block-<a100|a100-80gb>-<array>_<task>.out` and `.err`.
To inspect or report without submitting:

```bash
"$CP_PYTHON" run/cp_full.py list --group four-block
"$CP_PYTHON" run/cp_full.py check --group four-block --root "$CP_ROOT"
"$CP_PYTHON" run/cp_full.py report --group four-block --root "$CP_ROOT"
```

Reports are `outputs/results/cp_full_results.four-block.csv` and
`cp_full_summary.four-block.csv`, with a denominator of 240. They do not replace
the small-cohort reports. `--encoder MAE` restricts any of these commands to MAE;
the corresponding launcher filter remains `CP_ENCODERS=MAE`.

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
the display name and baseline lookup remain `SigLIP-2`. The MAE output folder
is `MAE`, while its baseline lookup is `MAE-Mean`. Existing pre-CP and
source-pretraining output namespaces are untouched.

```bash
"$CP_PYTHON" run/cp_full.py report --root "$CP_ROOT"
```
