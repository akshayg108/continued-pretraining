# ViT-L eight-dataset completion

## Approved scope

Complete the eight datasets absent from the existing DINOv3 ViT-L CP panel:
BreastMNIST, OCTMNIST, OrganAMNIST, PathMNIST, PlantVillage, Food-101,
Flowers-102, and Oxford-IIIT Pet. Run LeJEPA, SimCLR, and DIET-CP at MAX,
with seeds 42, 43, and 44 sequentially within each dataset/objective task.
Do not run fine-tuning or modify the existing seven-dataset results.

Keep the existing ViT-L recipe: LeJEPA/SimCLR batch 128 with accumulation 2,
DIET batch 32, 150 epochs, and the main-grid size-dependent unfreezing rule.
Accumulation does not increase SimCLR's 128-image contrastive batch.
Compute pre/post kNN and linear-probe metrics in each CP invocation.

## Implementation

- [x] Add failing contract tests for the grid, dataset selection, recipes,
  GPU restriction, no-FT commands, receipts, and Slurm behavior.
- [x] Add an isolated manifest/runner with selection-independent output
  identity, checkpoint resume, and checksummed completion receipts.
- [x] Add a dataset-selectable submission script and node-local staging array.
  Require A100 plus the cluster's 80g constraint, then verify the actual GPU.
- [x] Verify focused and related regression tests, shell syntax, and dry runs.
- [x] Document commands, output paths, counts, and runtime limitations.

## Validation boundary

Local tests exercise orchestration with simulated training and GPU properties.
Actual A100 memory use and training time require a cluster run. The 96-hour
limit covers all three seeds in a task. No automatic microbatch changes or
fallback to another GPU are allowed.

## Verification results

- The initial test run failed because the new runner package did not exist.
- Focused contracts: 58 passed.
- Repository `tests/`: 140 passed, 1 skipped because Lightning is unavailable
  locally, plus the existing expected warning about disabled FT checkpointing.
- Both shell scripts pass `bash -n`.
- A three-dataset dry run generated 9 tasks, 27 CP fits, and 0 FT fits with
  `gpu:a100:1` plus `constraint=80g`. Tests also cover the full 24-task selection.
- Existing tracked files are unchanged. No cluster jobs were submitted and
  no GPU training or peak-memory measurement was performed.
