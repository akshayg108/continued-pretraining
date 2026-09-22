# Held-Out CP Run Log

## 2026-09-22: Preparation 18066506, CP 18066507

Source: the user's `squeue -u $USER` snapshot from `login3`. This records the
reported state, not a live query or a confirmation of successful completion.

- Preparation array: `18066506`, tasks `0-7`, all `RUNNING` at elapsed `0:53`.
- CP array: `18066507`, tasks `0-47`, all `PENDING` with reason `Dependency`.
- User and partition: `gs4133`, `nvidia`.
- The user confirmed that the updated stable-datasets import check passed
  before submission.
- Submission manifest, recovered from the preparation log:
  `/scratch/gs4133/zhd/CP/outputs/heldout_cp_manifests/heldout-cp-20260922T045909Z-2968777.json`.

The dataset and CP-task mapping below follows the current launcher and manifest.
Each dataset has six CP tasks, with seeds 42, 43, and 44 run serially per task.
Within each three-task group, the objective order is LeJEPA, DIET, SimCLR.

| Dataset | Preparation task | Reported node | DINOv3 CP task indices | CLIP CP task indices |
| --- | --- | --- | --- | --- |
| BloodMNIST | `18066506_0` | `dn003` | 0, 1, 2 | 24, 25, 26 |
| TissueMNIST | `18066506_1` | `dn003` | 3, 4, 5 | 27, 28, 29 |
| AID | `18066506_2` | `dn004` | 6, 7, 8 | 30, 31, 32 |
| RESISC45 | `18066506_3` | `dn004` | 9, 10, 11 | 33, 34, 35 |
| Stanford Dogs | `18066506_4` | `dn005` | 12, 13, 14 | 36, 37, 38 |
| Jena Flowers 30 | `18066506_5` | `dn007` | 15, 16, 17 | 39, 40, 41 |
| Flavia | `18066506_6` | `dn007` | 18, 19, 20 | 42, 43, 44 |
| IP102 | `18066506_7` | `dn014` | 21, 22, 23 | 45, 46, 47 |

Configured dependency: each CP task uses `afterok` for its dataset's preparation
element only. For example, `18066507_6`, `_7`, `_8`, `_30`, `_31`, and `_32`
depend on `18066506_2`. The supplied queue snapshot shows pending dependencies
but does not display their individual dependency expressions.

### Default Server Locations

- Repository: `/scratch/gs4133/zhd/CP/continued-pretraining`
- Results: `/scratch/gs4133/zhd/CP/outputs/heldout_cp_1000_v1`
- Manifest directory: `/scratch/gs4133/zhd/CP/outputs/heldout_cp_manifests`
- Preparation logs: `/scratch/gs4133/zhd/CP/outputs/slurm-log/heldout-cp/heldout-prep-18066506_TASK.out` and `.err`
- CP logs: `/scratch/gs4133/zhd/CP/outputs/slurm-log/heldout-cp/heldout-cp-18066507_TASK.out` and `.err`
- Raw downloads: `/scratch/gs4133/zhd/CP/data/stable_datasets/downloads`
- Processed data: `/scratch/gs4133/zhd/CP/data/stable_datasets/processed`

These paths assume the launcher's default environment variables; no overrides
were reported. Replace `TASK` in log paths with the array task index.

### Subsequent Preparation Status and Blocked Tasks

Source: the user's later `sacct`, expanded `squeue`, and preparation logs.

| Preparation task | Dataset | State | Exit code | Elapsed |
| --- | --- | --- | --- | --- |
| `18066506_0` | BloodMNIST | COMPLETED | 0:0 | 00:24:29 |
| `18066506_1` | TissueMNIST | COMPLETED | 0:0 | 01:14:30 |
| `18066506_2` | AID | COMPLETED | 0:0 | 00:39:35 |
| `18066506_3` | RESISC45 | COMPLETED | 0:0 | 00:20:15 |
| `18066506_4` | Stanford Dogs | COMPLETED | 0:0 | 00:45:25 |
| `18066506_5` | Jena Flowers 30 | FAILED | 1:0 | 00:23:21 |
| `18066506_6` | Flavia | COMPLETED | 0:0 | 00:26:36 |
| `18066506_7` | IP102 | CANCELLED by 0 | 0:0 | 03:01:39 |

- Jena blocked CP indices: `15,16,17,39,40,41`. DINOv3 seed 42 completed
  feature extraction and LP optimization, then `check_metrics` rejected
  `pre_knn_f1`. The supplied log does not show its value or type, so the precise
  numeric failure remains unconfirmed. No baseline JSON was published for that
  failed evaluation because validation precedes writing.
- IP102 blocked CP indices: `21,22,23,45,46,47`. The download reached 100%.
  Completed caches contain 45,095 training and 7,508 validation samples. There
  is no completed test-cache message in the supplied tail. Slurm reported
  cancellation on `dn014` at `2026-09-22T12:00:49` (cluster log timezone not
  specified); no Python exception or cancellation cause was supplied.
- All 12 CP tasks show `DependencyNeverSatisfied` against their respective
  preparation element in that snapshot.

### IP102 Retry and Jena Numerical Diagnostic

- The user submitted IP102 preparation retry `18073093_7` with the original
  manifest and rebound CP indices `21,22,23,45,46,47` of array `18066507` to
  `afterok:18073093_7`. Submission is confirmed; completion is not yet reported.
- A synthetic perfect-classification check on the server returned
  `1.0000001192092896` for 30-class macro-F1 with PyTorch `2.10.0+cu128` and
  TorchMetrics `1.9.0`. This exceeds one by one float32 epsilon, so the strict
  metric-range check rejects a numerically valid perfect score. The actual
  failed Jena score was not included in the supplied log.
- The additive `eval.heldout_metric_roundoff` entrypoint handles roundoff within
  `1e-6` of the score bounds during pre/post evaluation. It retains the original
  strict validator and manifest, clips only the out-of-range values, and stores
  raw scores plus the wrapper hash in `evaluation_numerics`. Other invalid
  scores still fail. No Jena retry job has been submitted by this assistant.
