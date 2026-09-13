# SigLIP-2 LeJEPA CP-only recovery

This launcher covers only LeJEPA on OCTMNIST, PathMNIST, and Food-101 at MAX.
Each Slurm array element runs **one seed**, without pre-CP evaluation or FT.
The candidate panel has nine seeds. The launcher submits only seeds without a
verified completed CP result in either the original mainrule pass or this pass.
Missing FT never causes a completed CP seed to be retrained.

## Training and recovery

The original mainrule CP recipe is unchanged: SigLIP-2 ViT-B/16, native MAP
pooling, 150 epochs, 15 frozen epochs, all blocks subsequently trainable,
batch 64 with four accumulation steps, eight views, AdamW 1e-4/0.05.
The original training, mainrule runner, and protocol files are not modified.
Their implementation hash remains compatible with existing result receipts.

Every unfinished seed starts from public pretrained weights. There is no
`--resume` and no reuse of partially trained weights or optimizer state.
Every launch gets a unique attempt directory. An interrupted CP-only attempt
also restarts from public weights when submitted again. Previous attempts,
original checkpoints, original results, and FT files are retained.

One seed uses one A100 80GB, eight CPUs, 96 GB host RAM, and a 96-hour limit.
The script stages only that dataset into private node-local storage and removes
the private copy on exit. The 96 hours cover one CP fit and its final kNN/LP
evaluation, not three seeds or FT. Runtime still requires cluster measurement.

## Submit

First synchronize `eval/siglip_mainrule/cp_only.py` and this directory to the
cluster repository. The existing mainrule code and dependencies must remain
available. Do not use the old `mainrule/submit.sh` for this recovery.

Stop the corresponding original jobs before submitting replacements. For the
jobs reported on September 13, 2026, the targeted cancellation command is:

```bash
scancel 17882696_1 17882696_4 17882696_7
squeue -u "$USER"
```

This does not cancel the ViT-L array or other SigLIP elements. It does not delete
completed CP metrics. Wait until these original elements have exited. The
runner also shares the original per-seed file lock and rejects a concurrent
original process on the same seed.

From the cluster repository:

```bash
bash run/slurm/cp-siglip/cp-only/submit.sh --dry-run
bash run/slurm/cp-siglip/cp-only/submit.sh
```

The dry run checks result identities and hashes, writes a selection manifest,
and prints CP commands. It neither submits jobs nor loads CUDA or stages data.
The real submission rescans completion, so a newly finished seed is skipped.
The default concurrency is 12. `--concurrency N` accepts 1 through 12.

For the reported 83/87 mainrule snapshot, the remaining tasks are:

| New array ID | Dataset | Seed |
|---:|---|---:|
| 2 | OCTMNIST | 44 |
| 5 | PathMNIST | 44 |
| 7 | Food-101 | 43 |
| 8 | Food-101 | 44 |

These are **four jobs**, not three dataset-level jobs. Selection follows actual
files, not this snapshot. All nine candidates are selected if none is complete.
If all nine have verified CP metrics, no array is submitted. Invalid completed
results or provenance cause an error rather than a silent skip or overwrite.

## Outputs

The default base is `/scratch/gs4133/zhd/CP/outputs`.

- Original completed CP: `siglip_mainrule_v1/cp_results/LeJEPA/<dataset>/seedN.json`.
- New completed CP: `siglip_cp_only_v1/cp_results/LeJEPA/<dataset>/seedN.json`.
- New receipts: `siglip_cp_only_v1/provenance/LeJEPA/<dataset>/seedN.json`.
- Attempt files: `siglip_cp_only_v1/attempts/LeJEPA/<dataset>/seedN/<unique-id>/`.
- Logs: `slurm-log/siglip-cp-only-<job>_<array-id>.out` and `.err`.

The original-only status table will not include new results. Collection must
read both namespaces. `eval.siglip_mainrule.cp_only.completed_result` implements
this lookup with original completed results preferred. Running the dry-run
command again prints the resolved file for every finished seed. New receipts
record the original training-code hash, recovery-runner hash, recipe, software,
initialization, attempt, checkpoint, and result checksums.

Optional environment variables: `SIGLIP_CP_ONLY_OUTPUT_BASE`,
`SIGLIP_CP_ONLY_REPO_ROOT`, `SIGLIP_CP_ONLY_CACHE_DIR`, `SIGLIP_CP_ONLY_LOG_DIR`,
`SIGLIP_CP_ONLY_MANIFEST` (a new filename), `SIGLIP_CP_ONLY_CONCURRENCY`, and
`PYTHON`. These affect paths or scheduling, not the CP recipe.

## Local verification

```bash
python3 -m pytest tests/test_siglip_cp_only.py tests/test_siglip_cp_only_slurm.py -q
bash -n run/slurm/cp-siglip/cp-only/submit.sh
bash -n run/slurm/cp-siglip/cp-only/array.sh
```

Tests cover selection, one-seed jobs, no FT or resume, old-artifact preservation,
fresh attempts after failure, completed-result checksums, lock conflicts,
mocked submission, and private staging. They do not run GPU training or cancel
cluster jobs.
