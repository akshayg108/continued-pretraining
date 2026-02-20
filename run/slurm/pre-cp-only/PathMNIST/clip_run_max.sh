#!/bin/bash
#SBATCH --job-name=pre-path-clip-max
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/pre-cp-pathmnist-clip-max-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/pre-cp-pathmnist-clip-max-%j.err

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"

echo "Working directory: $(pwd)"
echo "=========================================="
nvidia-smi

# ============================================================
# Paths
# ============================================================
DATA_DIR="/scratch/gs4133/zhd/CP/data"
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts/pre-cp-only/PathMNIST/CLIP/max"
LOG_DIR="/scratch/gs4133/zhd/CP/outputs/logs/pre-cp-only/PathMNIST/CLIP/max"
SLURM_LOG_DIR="/scratch/gs4133/zhd/CP/outputs/slurm-log"
mkdir -p ${DATA_DIR} ${CKPT_DIR} ${LOG_DIR} ${SLURM_LOG_DIR}

# ============================================================
# Fixed parameters
# ============================================================
DATASET="pathmnist"
DISPLAY_NAME="PathMNIST"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="CLIP"
BACKBONE_TIMM="vit_base_patch16_clip_224.openai"

BATCH_SIZE=32
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

NSAMPLES=(89996)

# ============================================================
# Run a single experiment
# ============================================================
run_single() {
    local n_samples=$1
    local seed=$2

    local results_file="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${seed}.json"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] Pre-CP ${BACKBONE_TAG} | ${DATASET} | n=${n_samples} | seed=${seed}"
    echo "  Start: $(date)"
    echo "=========================================="

    python -u continued_pretraining.py \
        --cp-method simclr \
        --no-cp \
        --pre-cp-sft \
        --dataset ${DATASET} \
        --backbone ${BACKBONE_TIMM} \
        --n-samples ${n_samples} \
        --batch-size ${BATCH_SIZE} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --pool-strategy cls \
        --checkpoint-dir ${CKPT_DIR} \
        --cache-dir ${DATA_DIR} \
        --project pre-cp-clip-${DATASET} \
        --run-name "${BACKBONE_TAG}_${DATASET}_n${n_samples}_s${seed}" \
        --seed ${seed} \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed}"
    fi

    return $exit_code
}

# ============================================================
# Aggregate results across seeds
# ============================================================
aggregate_results() {
    local n_samples=$1
    local csv_file=$2

    python3 << PYEOF
import json, os, statistics

log_dir = "${LOG_DIR}"
backbone_tag = "${BACKBONE_TAG}"
dataset = "${DATASET}"
n_samples = "${n_samples}"
display_name = "${DISPLAY_NAME}"
model_size = "${MODEL_SIZE}"
csv_file = "${csv_file}"
seeds = [42, 43, 44]

knn_f1s = []
linear_f1s = []
sft_f1s = []

for i, seed in enumerate(seeds):
    results_file = os.path.join(log_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}.json")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping seed {seed}")
        continue

    with open(results_file) as f:
        data = json.load(f)

    knn_f1 = data.get("pre_knn_f1")
    linear_f1 = data.get("pre_linear_f1")
    sft_f1 = data.get("pre_sft_f1")

    if knn_f1 is not None:
        knn_f1s.append(knn_f1)
    if linear_f1 is not None:
        linear_f1s.append(linear_f1)
    if sft_f1 is not None:
        sft_f1s.append(sft_f1)

    with open(csv_file, "a") as f:
        knn_str = f"{knn_f1:.6f}" if knn_f1 is not None else ""
        lin_str = f"{linear_f1:.6f}" if linear_f1 is not None else ""
        sft_str = f"{sft_f1:.6f}" if sft_f1 is not None else ""
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},{i},{knn_str},,{lin_str},,{sft_str},\n")

if len(knn_f1s) > 0 or len(linear_f1s) > 0 or len(sft_f1s) > 0:
    def mean_std(vals):
        if len(vals) == 0:
            return "", ""
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.6f}", f"{s:.6f}"

    knn_mean, knn_std = mean_std(knn_f1s)
    lin_mean, lin_std = mean_std(linear_f1s)
    sft_mean, sft_std = mean_std(sft_f1s)

    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},average,{knn_mean},{knn_std},{lin_mean},{lin_std},{sft_mean},{sft_std}\n")

    print(f"  [{backbone_tag}] {display_name} n={n_samples}: knn_f1={knn_mean}+-{knn_std} linear_f1={lin_mean}+-{lin_std} sft_f1={sft_mean}+-{sft_std}")
else:
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: no results available")

PYEOF
}

# ============================================================
# Main loop
# ============================================================
echo ""
echo "=========================================="
echo "Starting Pre-CP-Only: ${DISPLAY_NAME} (CLIP, n=max 89996)"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="
echo ""

CSV_FILE="${LOG_DIR}/${BACKBONE_TAG}_pre_cp_results.csv"
if [ ! -f "${CSV_FILE}" ]; then
    echo "backbone,dataset,n_samples,model_size,run,knn_f1,knn_f1_std,linear_f1,linear_f1_std,sft_f1,sft_f1_std" > ${CSV_FILE}
fi
echo "CSV file: ${CSV_FILE}"

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for n_samples in "${NSAMPLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Experiment: ${BACKBONE_TAG} | ${DISPLAY_NAME} | n_samples=${n_samples}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        run_single ${n_samples} ${seed}
        if [ $? -eq 0 ]; then
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done

    echo "--- Aggregating results for n=${n_samples} ---"
    aggregate_results ${n_samples} ${CSV_FILE}
done

echo ""
echo "=========================================="
echo "All Pre-CP ${DISPLAY_NAME} ${BACKBONE_TAG} experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/"
echo "  End Time: $(date)"
echo "=========================================="
