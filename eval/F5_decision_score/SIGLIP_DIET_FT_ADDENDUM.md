# SigLIP DIET Full-FT Addendum

The frozen P1 preregistration, source hashes, CP recipe, and kNN/LP endpoints are unchanged. Full-parameter FT is a separate follow-up stored only under `outputs/full_ft_v1`; it is not an endpoint in the frozen P1 decision.

`diet_max_array.sh` now groups one dataset per scheduler task and processes seeds 42, 43, and 44 serially after staging that dataset once. For each seed, the existing CP result and checkpoint are validated or produced first. Only after successful CP validation does the standalone full-FT runner execute for that seed. A valid CP result remains resumable even if FT later times out or fails.

FT writes atomic per-seed metrics but no model or optimizer checkpoint. Interrupted FT therefore restarts that seed, while already validated FT JSON files resume. Missing CP files are reported as missing by the general runner; corrupt files or training failures are errors.

Dry-run inspection does not submit or train:

```bash
SLURM_ARRAY_TASK_ID=0 bash run/slurm/cp-siglip/cp/diet_max_array.sh --dry-run
```
