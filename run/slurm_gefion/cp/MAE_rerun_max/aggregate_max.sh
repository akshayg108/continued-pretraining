#!/bin/bash
#SBATCH --job-name=mae-rerun-max-agg
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/mae-rerun-max-agg-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/max_manifest.csv"
JOB_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --job-file) JOB_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Start Time: $(date)"
echo "Manifest: ${MANIFEST}"
echo "Job File: ${JOB_FILE:-N/A}"
echo "=========================================="

cd /dcai/projects/iu_0092/projects/cp/continued-pretraining
source .venv/bin/activate
PYTHON_BIN="$(pwd)/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found at ${PYTHON_BIN}" >&2
    exit 1
fi

OUTPUT_ROOT="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/MAE_rerun_max"
COMBINED_DIR="${OUTPUT_ROOT}/combined"
mkdir -p "${COMBINED_DIR}" "/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
if [ -n "${JOB_FILE}" ] && [ -f "${JOB_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${JOB_FILE}"
fi

STATUS_CSV="${STATUS_CSV:-${COMBINED_DIR}/mae_rerun_max_status_${ARRAY_JOB_ID}.csv}"
COMBINED_CSV="${COMBINED_CSV:-${COMBINED_DIR}/mae_rerun_max_combined_results_${ARRAY_JOB_ID}.csv}"

echo "Status CSV: ${STATUS_CSV}"
echo "Combined CSV: ${COMBINED_CSV}"

"${PYTHON_BIN}" - "${MANIFEST}" "${STATUS_CSV}" "${COMBINED_CSV}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
status_csv = Path(sys.argv[2])
combined_csv = Path(sys.argv[3])
seeds = [42, 43, 44]
output_root = Path("/dcai/projects/iu_0092/projects/cp/outputs/logs")

header = [
    "backbone",
    "dataset",
    "n_samples",
    "model_size",
    "run",
    "pre_knn_f1",
    "pre_knn_f1_std",
    "pre_linear_f1",
    "pre_linear_f1_std",
    "post_knn_f1",
    "post_knn_f1_std",
    "post_linear_f1",
    "post_linear_f1_std",
    "post_sft_f1",
    "post_sft_f1_std",
]

combined_header = ["run_id", *header]
status_header = [
    "run_id",
    "backbone",
    "dataset",
    "n_samples",
    "num_trained_blocks",
    "expected_seeds",
    "completed_seeds",
    "missing_seeds",
    "csv_path",
    "complete",
]


def log_dir_for(row):
    rel = Path("cp/MAE_rerun_max/pretrained") / row["display_name"] / row["backbone_tag"] / "all"
    return output_root / rel


def result_path(row, seed):
    return log_dir_for(row) / f"{row['backbone_tag']}_{row['dataset']}_n{row['n_samples']}_seed{seed}.json"


def csv_path_for(row):
    return log_dir_for(row) / f"{row['backbone_tag']}_mae_cp_rerun_max_results.csv"


def fmt(value):
    return f"{value:.6f}" if value is not None else ""


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


with manifest_path.open(newline="") as f:
    manifest_rows = list(csv.DictReader(f))

grouped_rows = defaultdict(list)
status_rows = []
combined_rows = []

for row in manifest_rows:
    metrics = {
        "pre_knn_f1": [],
        "pre_linear_f1": [],
        "post_knn_f1": [],
        "post_linear_f1": [],
        "post_sft_f1": [],
    }
    missing = []
    per_seed_rows = []

    for seed in seeds:
        path = result_path(row, seed)
        if not path.exists():
            missing.append(str(seed))
            continue

        with path.open() as f:
            data = json.load(f)

        csv_row = [
            row["backbone_tag"],
            row["display_name"],
            row["n_samples"],
            row["model_size"],
            seed,
            fmt(data.get("pre_knn_f1")),
            "",
            fmt(data.get("pre_linear_f1")),
            "",
            fmt(data.get("post_knn_f1")),
            "",
            fmt(data.get("post_linear_f1")),
            "",
            fmt(data.get("post_sft_f1")),
            "",
        ]
        per_seed_rows.append(csv_row)
        combined_rows.append([row["run_id"], *csv_row])

        for key in metrics:
            value = data.get(key)
            if value is not None:
                metrics[key].append(value)

    if per_seed_rows:
        pk_m, pk_s = mean_std(metrics["pre_knn_f1"])
        pl_m, pl_s = mean_std(metrics["pre_linear_f1"])
        ok_m, ok_s = mean_std(metrics["post_knn_f1"])
        ol_m, ol_s = mean_std(metrics["post_linear_f1"])
        sf_m, sf_s = mean_std(metrics["post_sft_f1"])
        average_row = [
            row["backbone_tag"],
            row["display_name"],
            row["n_samples"],
            row["model_size"],
            "average",
            pk_m,
            pk_s,
            pl_m,
            pl_s,
            ok_m,
            ok_s,
            ol_m,
            ol_s,
            sf_m,
            sf_s,
        ]
        per_seed_rows.append(average_row)
        combined_rows.append([row["run_id"], *average_row])

    grouped_rows[csv_path_for(row)].extend(per_seed_rows)
    status_rows.append(
        [
            row["run_id"],
            row["backbone_tag"],
            row["display_name"],
            row["n_samples"],
            row["num_trained_blocks"],
            ";".join(str(seed) for seed in seeds),
            3 - len(missing),
            ";".join(missing),
            csv_path_for(row),
            "yes" if not missing else "no",
        ]
    )

for path, rows in grouped_rows.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")

status_csv.parent.mkdir(parents=True, exist_ok=True)
with status_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(status_header)
    writer.writerows(status_rows)

combined_csv.parent.mkdir(parents=True, exist_ok=True)
with combined_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(combined_header)
    writer.writerows(combined_rows)

complete = sum(1 for row in status_rows if row[-1] == "yes")
print(f"Manifest rows complete: {complete}/{len(status_rows)}")
print(f"Status report: {status_csv}")
print(f"Combined results: {combined_csv}")
PY

echo ""
echo "=========================================="
echo "MAE rerun MAX aggregation completed"
echo "End Time: $(date)"
echo "=========================================="
