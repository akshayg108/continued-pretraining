#!/bin/bash
#SBATCH --job-name=c-oct-max-agg
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/lejepa-octmnist-clip-max-agg-%j.out

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

cd /dcai/projects/iu_0092/projects/cp/continued-pretraining
echo "Working directory: $(pwd)"
source .venv/bin/activate
PYTHON_BIN="$(pwd)/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found at ${PYTHON_BIN}"
    exit 1
fi

LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA/pretrained/OctMNIST/CLIP/all"
SLURM_LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"
mkdir -p "${LOG_DIR}" "${SLURM_LOG_DIR}"

DATASET="octmnist"
DISPLAY_NAME="OctMNIST"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="CLIP"

CSV_FILE="${LOG_DIR}/${BACKBONE_TAG}_lejepa_cp_results.csv"
echo "Writing aggregated CSV to: ${CSV_FILE}"

"${PYTHON_BIN}" <<PYEOF
import csv
import json
import os
import statistics

log_dir = "${LOG_DIR}"
backbone_tag = "${BACKBONE_TAG}"
dataset = "${DATASET}"
display_name = "${DISPLAY_NAME}"
model_size = "${MODEL_SIZE}"
csv_file = "${CSV_FILE}"
seeds = [42, 43, 44]
n_samples_values = [97477]

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


def fmt(value):
    return f"{value:.6f}" if value is not None else ""


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for n_samples in n_samples_values:
        metrics = {
            "pre_knn_f1": [],
            "pre_linear_f1": [],
            "post_knn_f1": [],
            "post_linear_f1": [],
            "post_sft_f1": [],
        }

        completed_runs = 0
        for seed in seeds:
            results_file = os.path.join(
                log_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}.json"
            )
            if not os.path.exists(results_file):
                print(f"Warning: {results_file} not found, skipping seed {seed}")
                continue

            with open(results_file) as rf:
                data = json.load(rf)

            completed_runs += 1
            writer.writerow(
                [
                    backbone_tag,
                    display_name,
                    n_samples,
                    model_size,
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
            )

            for key in metrics:
                value = data.get(key)
                if value is not None:
                    metrics[key].append(value)

        if completed_runs == 0:
            print(f"[{backbone_tag}] {display_name} n={n_samples}: no results available")
            continue

        pk_m, pk_s = mean_std(metrics["pre_knn_f1"])
        pl_m, pl_s = mean_std(metrics["pre_linear_f1"])
        ok_m, ok_s = mean_std(metrics["post_knn_f1"])
        ol_m, ol_s = mean_std(metrics["post_linear_f1"])
        sf_m, sf_s = mean_std(metrics["post_sft_f1"])

        writer.writerow(
            [
                backbone_tag,
                display_name,
                n_samples,
                model_size,
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
        )

        print(
            f"[{backbone_tag}] {display_name} n={n_samples}: "
            f"pre_knn={pk_m}+-{pk_s} "
            f"pre_lp={pl_m}+-{pl_s} "
            f"post_knn={ok_m}+-{ok_s} "
            f"post_lp={ol_m}+-{ol_s} "
            f"post_sft={sf_m}+-{sf_s}"
        )
PYEOF

echo ""
echo "=========================================="
echo "Aggregation completed"
echo "  CSV file: ${CSV_FILE}"
echo "  End Time: $(date)"
echo "=========================================="
