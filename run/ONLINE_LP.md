# Fresh-View Frozen LP Migration

Both pre-CP and post-CP now use `frozen_online_lp_v1`. Training rereads every
image each epoch with random resized crop and horizontal flip. Only the linear
head learns; encoder parameters, buffers, and train/eval flags are restored on
return, including failure. Test preprocessing is deterministic. CP training,
source-pretraining, training-time SPT probes, kNN, and geometry are unchanged.

## Fixed Evaluation Budget

| Setting | Value |
| --- | --- |
| Epochs | 150 complete passes, no minimum step override |
| Classifier batch | 512, final partial batch retained |
| Encoder forward chunk | At most 32 images, no gradients |
| Optimizer | Adam, LR 0.001, weight decay 0 |
| Features | Existing readout followed by L2 normalization |
| Training view | RGB, random resized crop (scale 0.08-1.0), horizontal flip (p=0.5) |
| Test view | RGB, deterministic square resize, encoder-specific normalization |

The CP batch and accumulation settings do not control the LP batch. LP does not
use the CP balanced-repeat sampler: it shuffles and visits every training image
once per epoch, retaining the final short batch. The MAE encoder still uses
`LayerNorm(mean(raw patch tokens))` with its pretrained LayerNorm.

Each LP evaluation creates independent seeded loaders and workers. Running a
baseline first does not advance the post-CP probe's sampling stream; a post-only
rerun with the same seed and encoder uses the same LP sampling protocol.

Fresh-view LP repeatedly runs the encoder and is much slower than optimizing a
cached feature bank. A 150-epoch run is not comparable to the old once-augmented
LP even when the classifier optimizer is unchanged. Keep the two protocols
separate in tables; new pre/post differences must use new baselines.

## Output Isolation And Checkpoints

- New pre-CP JSON, logs, target features, and reports live under
  `outputs/precp_full/lp_online_v1/`. Original `precp_full/results` stays intact.
- The existing `outputs/precp_full/reference/` banks are reused because the
  pretrained encoder, deterministic inputs, and readout have not changed.
- CP weights remain at `outputs/results/<dataset>/<encoder>/<method>/Full/<seed>/cp.ckpt`.
- New post-CP JSON, logs, geometry artifacts, and evaluation configuration live
  in that seed's `lp_online_v1/` subdirectory. Original seed-level files remain.
- Combined CP reports live in `outputs/results/lp_online_v1/`.

The runner validates the original CP configuration before reusing its checkpoint.
Only the baseline hash is allowed to differ, since the LP baseline changed; all
training identity and recipe fields must match. Completed CP restores its weights
for evaluation; incomplete CP resumes from the last saved checkpoint. A run with
no checkpoint starts CP normally. A missing or incompatible configuration is an
error, not permission to overwrite old weights. Grid completion requires matching
new `pre_lp` and `post_lp` metadata and the current baseline hash.

## Cluster Sequence

Use a new pinned worktree for the published revision. Do not pull into a code
directory still referenced by running or queued source-pretraining jobs. Reuse
the existing environment and completed dataset caches.

First run the canonical five-encoder pre-CP grid (345 evaluations). Existing
matching reference banks are needed; no new ImageNet download is required:

```bash
export CP_ROOT=/scratch/gs4133/zhd/CP_new
cd /path/to/new/pinned/worktree
source run/precp_env.sh
LOG="$CP_ROOT/outputs/slurm-log"
mkdir -p "$LOG"
sbatch --chdir="$PWD" --export=ALL --array=0-16%10 \
    --job-name=precp-lp-v1 \
    --output="$LOG/precp-lp-v1-%A_%a.out" \
    --error="$LOG/precp-lp-v1-%A_%a.err" \
    run/slurm/precp.sh --encoder DINOv3 CLIP SigLIP-2 MAE-Mean DINOv3-L
```

Every job stages data on the compute node. No old pre-CP result can satisfy the
new completion check. This reevaluates kNN/geometry as well as LP, without changing
their recipes. Wait for the required new baselines to finish, then:

```bash
"$CP_PYTHON" run/precp.py report --root "$CP_ROOT" \
    --encoder DINOv3 CLIP SigLIP-2 MAE-Mean DINOv3-L
"$CP_PYTHON" run/cp_full.py check --root "$CP_ROOT"
CP_CONCURRENCY=10 bash run/slurm/submit_cp_full.sh
"$CP_PYTHON" run/cp_full.py report --root "$CP_ROOT"
```

The CP launcher retains the ordinary V100 `%10` array followed by A100 `%10`.
It validates the 165 baselines needed by the 11-target, five-encoder CP grid and
uses existing compatible CP checkpoints. Do not submit duplicate grids while an
earlier copy is active. Source-pretraining's pending array and its OnlineProbe
are independent of this migration; these commands do not cancel or resubmit it.
