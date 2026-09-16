# Food-101 frozen checkpoint audit

Evaluate saved SigLIP-2 Food-101 CP weights with official image normalization
(`mean=std=[0.5, 0.5, 0.5]`). This changes evaluation preprocessing only. It does
not resume CP, run full FT, retrain the encoder, or overwrite old artifacts.
The LP classifier is trained on frozen features using the baseline audit recipe.

## Original mainrule checkpoints

```bash
bash run/slurm/cp-siglip/post-cp/submit_food101.sh --dry-run
bash run/slurm/cp-siglip/post-cp/submit_food101.sh
```

Defaults: DIET, LeJEPA, SimCLR; seeds 42, 43, 44; one V100 per checkpoint;
up to nine jobs concurrently. The planner checks the original result identity,
mainrule completion receipt and checkpoint/result SHA256 hashes. Missing files
are printed and not submitted. Invalid existing records stop planning.
Low scores are not exclusion criteria. No weights are downloaded as a fallback.

Select a smaller scope with `--methods LeJEPA --seeds 42` or limit concurrent
jobs with `--concurrency 3`.

## Separately evaluate recheck checkpoints

```bash
bash run/slurm/cp-siglip/post-cp/submit_food101.sh --source recheck --recheck-job-id 17950124 --dry-run
bash run/slurm/cp-siglip/post-cp/submit_food101.sh --source recheck --recheck-job-id 17950124
```

These are LeJEPA seeds 43, 44, 45, 46 from a separate training attempt. They are
never substituted for mainrule seeds or automatically pooled with them. Recheck
runs do not have the original runner's completion receipt; the planner validates
their result identity and hashes the exact saved checkpoint/result pair instead.

## Outputs

`/scratch/gs4133/zhd/CP/outputs/siglip_food101_postcheck_official_norm_v1/<array_job_id>/<input_source>/<method>/food101/seed<seed>.json`

Each output records the checkpoint source and hashes, original post-CP metrics,
new frozen metrics, official evaluation normalization, model-loading audit,
split hash, transforms, software versions and code hashes. Training preprocessing
has not been corrected by this evaluation. A corrected CP-training experiment is
a separate comparison.

Cluster defaults can be overridden through `SIGLIP_POST_OUTPUT_BASE`,
`SIGLIP_POST_CACHE_DIR`, `SIGLIP_POST_REPO_ROOT` and `PYTHON`. Dry-run creates
only an audit manifest and log directory. It imports no CUDA training stack and
does not submit jobs. Manifest checksums involve reading the checkpoint files.
