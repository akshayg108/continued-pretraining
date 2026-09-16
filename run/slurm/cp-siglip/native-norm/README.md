# SigLIP native-normalization CP rerun

Run all 15 MAX targets with `vit_base_patch16_siglip_224.v2_webli`, MAP pooling,
DIET / LeJEPA / SimCLR, and seeds 42, 43, 44. CP training and post-CP evaluation
both use the loaded backbone's native RGB mean/std (`[0.5, 0.5, 0.5]` for both).
Only mean/std changes: augmentations, image size, pooling, precision, optimizer,
and the existing main-rule recipes remain unchanged. This is not an adoption of
the checkpoint's entire inference resize/crop pipeline.

No baseline evaluation, FT, or SFT is run. Post-CP kNN and PyTorch LP macro-F1
and accuracy are collected; the unified evaluator also retains its existing
sklearn probe and k-means diagnostics. Compare against the separate native
pre-CP audit, never against the legacy dataset-normalized baseline.

## Jobs and resources

| Targets | Jobs per target | Seeds per job | Total jobs |
| --- | ---: | --- | ---: |
| OCTMNIST, PathMNIST, Food-101 | 9 (method x seed) | One | 27 |
| Other 12 targets | 3 (one per method) | 42, 43, 44 sequentially | 36 |

Total: **63 jobs, 135 CP fits**. Every job requests one GPU, 8 CPUs,
96 GB host RAM, and 96 hours in `nvidia` / `civil`.
The submission wrapper creates one array per GPU profile:

| Trainable depth | DIET / SimCLR | LeJEPA |
| --- | --- | --- |
| Last 2 blocks | V100 | V100 |
| Last 4 or 6 blocks | A100 | A100 |
| All blocks | A100 | A100 80GB |

The full grid has **24 V100 jobs, 30 ordinary A100 jobs, and 9 A100 80GB jobs**.
Ordinary V100/A100 requests have no VRAM constraint (V100 16/32GB and A100
40/80GB are eligible). Only full-backbone LeJEPA adds `--constraint=80g`.
The runner checks the allocated GPU against the task's recorded profile.
Batch sizes and accumulation are unchanged; a GPU-type check does not establish
that every recipe fits in the smaller memory variant. OOM failures are reported,
not silently retried with a different batch size or GPU.

Default total concurrency is 12, split into fixed per-array caps (4/4/4 for the
full grid). Filtered selections redistribute these slots across nonempty groups.
The caps are not automatically increased when another group finishes. If the
requested concurrency is smaller than the number of GPU groups, arrays are
chained with `afterany` dependencies so failures do not block later groups.
Jobs copy only their own target cache to private node-local storage and remove
only that private copy on exit.

Training depth follows the existing main rule: 2 blocks below 10,000 images,
4 through 25,000, 6 through 50,000, and all blocks above 50,000.
All fits use 150 epochs, 15 frozen epochs, 15 warmup epochs, learning rate
`1e-4`, weight decay `0.05`, and kNN `k=20`.

| Method | Batch size x accumulation | Other parameters |
| --- | --- | --- |
| LeJEPA | 256 x 1 (2 blocks); 128 x 2 (4/6 blocks); 64 x 4 (all) | 8 views, projector 128, hidden 2048, lambda 0.02 |
| SimCLR | 256 x 1 | projector 128, hidden 2048, temperature 0.5 |
| DIET | 32 x 1 | smoothing 0.3, mixup/cutmix alpha 1, mixup/cutmix probability 0 |

Gradient accumulation does not pool the samples used to compute each minibatch
loss. These are the existing recipes, not newly equalized batch-level objectives.

## Submit

Sync the changed code to the cluster before creating a manifest. From the repo:

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining
bash run/slurm/cp-siglip/native-norm/submit.sh --dry-run
bash run/slurm/cp-siglip/native-norm/submit.sh --concurrency 12
```

Dry-run creates a manifest and prints all selected commands, but does not load
CUDA, copy datasets, create training artifacts, or call `sbatch`.
Use `submit.sh`, not a direct `sbatch array.sh`: GPU resources are assigned by
the wrapper, not fixed in the shared array script. Existing queued jobs are not
changed by editing these files; the routing applies to new submissions.
To select targets and/or methods (do not submit these again alongside the full arrays):

```bash
bash run/slurm/cp-siglip/native-norm/submit.sh \
    --datasets octmnist pathmnist food101 --methods LeJEPA --concurrency 9
bash run/slurm/cp-siglip/native-norm/submit.sh \
    --datasets dtd cars196 cub200 --methods DIET SimCLR
```

Supported optional environment variables: `SIGLIP_NATIVE_OUTPUT_BASE`,
`SIGLIP_NATIVE_CACHE_DIR`, `SIGLIP_NATIVE_LOG_DIR`, `SIGLIP_NATIVE_REPO_ROOT`,
`SIGLIP_NATIVE_CONCURRENCY`, and `SIGLIP_NATIVE_MANIFEST` (a new path, never
overwritten). Defaults use `/scratch/gs4133/zhd/CP/{outputs,data}`.

## Outputs and retries

The isolated protocol root is
`/scratch/gs4133/zhd/CP/outputs/siglip_native_cp_v1/`:

- `cp_results/<method>/<dataset>/seed<seed>.json`: validated post-CP metrics,
  actual normalization, and training arguments.
- `provenance/<method>/<dataset>/seed<seed>.json`: status, command, recipe,
  code fingerprint, software versions, GPU profile, actual GPU, checkpoint path,
  and checksums.
- `attempts/<method>/<dataset>/seed<seed>/<unique-id>/`: each fresh attempt's
  result, checkpoint, and provenance. Failed attempts are retained.

Manifests live in the sibling `siglip_native_cp_manifests/` directory.
Slurm logs are `slurm-log/siglip-native-<array-job-id>_<task-id>.{out,err}`.
The task-to-target/method/seed/GPU mapping is printed when submitting and stored
in the manifest. Task IDs are global within a manifest and stay unchanged when
split across the resource arrays; collect results across all three array IDs
with the same command below.

Every new attempt starts from public pretrained weights. There is no checkpoint
resume flag. A failed seed does not prevent later seeds in the same job from
running; any seed failure makes the job exit nonzero. Re-running the submit
command skips only checksum-verified completed native-protocol fits, starts
unfinished seeds in new attempt directories, and never reads legacy outputs as
completion evidence. Low but finite scores are retained, not filtered out.

Receipts lock each fit against concurrent jobs. A malformed or incompatible
existing result causes an error, not an overwrite. Do not change training/runner
code while this protocol is running: manifests and receipts are code-bound.
A crash between result publication and its completion receipt requires manual
inspection instead of silently repeating or overwriting the fit.

## Collect post-CP results

```bash
python3 -m eval.siglip_native_cp collect
```

This prints CSV with all 135 per-seed records, missing/failed/check statuses,
then available-seed means and sample SDs per method/target. Completion counts
go to stderr so redirected CSV stays valid. `VERIFIED` means provenance and
fields/checksums passed, not that training converged or performed well.
Incomplete groups are labeled `n=k/3`; failed or missing seeds are never zero-filled.
The same `--datasets` and `--methods` filters are supported.

## Local verification

```bash
python3 -m pytest tests/test_cp_normalization.py tests/test_siglip_native_cp.py tests/test_siglip_native_cp_slurm.py -q
bash -n run/slurm/cp-siglip/native-norm/submit.sh
bash -n run/slurm/cp-siglip/native-norm/array.sh
```

Tests cover planning, commands, normalization, isolation, retries, provenance,
mock submission, and private cache staging. They do not replace a cluster GPU
run and cannot predict numerical convergence or wall time.
