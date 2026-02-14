#!/bin/bash
#SBATCH --job-name=sft-baseline-rand-pre
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/logs/sft-baseline-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/logs/sft-baseline-%j.err

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
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts/sft_rand/"
LOG_DIR="/scratch/gs4133/zhd/CP/outputs/logs/sft_rand/"
RESULTS_DIR="${LOG_DIR}/baseline_sft_lejepa_rand_pre"
mkdir -p ${DATA_DIR} ${CKPT_DIR} ${LOG_DIR} ${RESULTS_DIR}

# Output CSV
CSV_FILE="${RESULTS_DIR}/baseline_lejepa_rand_pre_results.csv"

# ============================================================
# Fixed parameters
# ============================================================
BACKBONE="vit_base_patch16_224"
MODEL_SIZE="ViT-B"
METHOD="sft"
EPOCHS=150
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=0
NUM_TRAINED_BLOCKS=-1
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

# ============================================================
# Experiment list: (dataset, n_samples)
# From result.csv Baselines group (LeJEPA-CP FROM SCRATCH)
# ============================================================
EXPERIMENTS=(
    # DermaMNIST
    # "dermamnist 100"
    # "dermamnist 500"
    # "dermamnist 1000"
    # "dermamnist 7007"

    # BreastMNIST
    # "breastmnist 100"
    # "breastmnist 500"

    # OCTMNIST
    # "octmnist 100"
    # "octmnist 500"
    # "octmnist 1000"
    # "octmnist 10000"
    "octmnist 97477"
    "octmnist 25000"

    # OrganAMNIST
    # "organamnist 100"
    # "organamnist 500"
    # "organamnist 1000"
    # "organamnist 10000"
    # "organamnist 25000"
    # "organamnist 34561"

    # PathMNIST
    # "pathmnist 100"
    # "pathmnist 500"
    # "pathmnist 1000"
    # "pathmnist 10000"
    # "pathmnist 25000"
    # "pathmnist 89996"

    # Galaxy10
    "galaxy10 100"
    "galaxy10 500"
    "galaxy10 1000"
    "galaxy10 10000"
    "galaxy10 14188"

    # FGVC_Aircraft
    "fgvc_aircraft 100"
    "fgvc_aircraft 500"
    "fgvc_aircraft 1000"
    "fgvc_aircraft 3400"

    # Food101
    # "food101 100"
    # "food101 500"
    # "food101 1000"
    # "food101 10000"
    # "food101 25000"
    # "food101 75750"
)

# ============================================================
# CSV column name mapping (dataset code -> display name)
# ============================================================
get_display_name() {
    case "$1" in
        dermamnist)  echo "DermaMNIST" ;;
        breastmnist) echo "BreastMNIST" ;;
        octmnist)    echo "OCTMNIST" ;;
        organamnist) echo "OrganAMNIST" ;;
        pathmnist)   echo "PathMNIST" ;;
        galaxy10)       echo "Galaxy10" ;;
        food101)        echo "Food101" ;;
        fgvc_aircraft)  echo "FGVC_Aircraft" ;;
        *)              echo "$1" ;;
    esac
}

# ============================================================
# Write CSV header
# ============================================================
echo "dataset,n_samples,model_size,run,knn_f1,knn_f1_std,linear_f1,linear_f1_std,finetune_f1,finetune_f1_std" > ${CSV_FILE}
echo "CSV file: ${CSV_FILE}"

# ============================================================
# Run a single experiment
# ============================================================
run_single() {
    local dataset=$1
    local n_samples=$2
    local seed=$3
    local results_file="${RESULTS_DIR}/${dataset}_n${n_samples}_seed${seed}.json"

    # Skip if already completed
    if [ -f "$results_file" ]; then
        echo "[SKIP] ${dataset} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] ${dataset} | n=${n_samples} | seed=${seed}"
    echo "  Start: $(date)"
    echo "=========================================="

    # Warmup epochs = 10% of total epochs
    local warmup_epochs=$((EPOCHS / 10))

    python -u continued_pretraining.py \
        --cp-method ${METHOD} \
        --random-init \
        --dataset ${dataset} \
        --backbone ${BACKBONE} \
        --n-samples ${n_samples} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --freeze-epochs ${FREEZE_EPOCHS} \
        --num-trained-blocks ${NUM_TRAINED_BLOCKS} \
        --warmup-epochs ${warmup_epochs} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --checkpoint-dir ${CKPT_DIR} \
        --cache-dir ${DATA_DIR} \
        --project sft-rand-${dataset} \
        --seed ${seed} \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${dataset} n=${n_samples} seed=${seed}"
    fi

    return $exit_code
}

# ============================================================
# Aggregate results for one (dataset, n_samples) across seeds
# ============================================================
aggregate_results() {
    local dataset=$1
    local n_samples=$2
    local display_name=$(get_display_name ${dataset})

    python3 << PYEOF
import json, os, statistics

results_dir = "${RESULTS_DIR}"
dataset = "${dataset}"
n_samples = "${n_samples}"
display_name = "${display_name}"
model_size = "${MODEL_SIZE}"
csv_file = "${CSV_FILE}"
seeds = [42, 43, 44]

knn_f1s = []
linear_f1s = []
finetune_f1s = []

for i, seed in enumerate(seeds):
    results_file = os.path.join(results_dir, f"{dataset}_n{n_samples}_seed{seed}.json")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping seed {seed}")
        continue

    with open(results_file) as f:
        data = json.load(f)

    knn_f1 = data.get("pre_knn_f1")
    linear_f1 = data.get("pre_linear_f1")
    finetune_f1 = data.get("finetune_f1")

    if knn_f1 is not None:
        knn_f1s.append(knn_f1)
    if linear_f1 is not None:
        linear_f1s.append(linear_f1)
    if finetune_f1 is not None:
        finetune_f1s.append(finetune_f1)

    # Write individual run row
    with open(csv_file, "a") as f:
        knn_str = f"{knn_f1:.6f}" if knn_f1 is not None else ""
        lin_str = f"{linear_f1:.6f}" if linear_f1 is not None else ""
        ft_str = f"{finetune_f1:.6f}" if finetune_f1 is not None else ""
        f.write(f"{display_name},{n_samples},{model_size},{i},{knn_str},,{lin_str},,{ft_str},\n")

# Write average row
if len(knn_f1s) > 0 or len(linear_f1s) > 0 or len(finetune_f1s) > 0:
    def mean_std(vals):
        if len(vals) == 0:
            return "", ""
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.6f}", f"{s:.6f}"

    knn_mean, knn_std = mean_std(knn_f1s)
    lin_mean, lin_std = mean_std(linear_f1s)
    ft_mean, ft_std = mean_std(finetune_f1s)

    with open(csv_file, "a") as f:
        f.write(f"{display_name},{n_samples},{model_size},average,{knn_mean},{knn_std},{lin_mean},{lin_std},{ft_mean},{ft_std}\n")

    print(f"  {display_name} n={n_samples}: knn_f1={knn_mean}±{knn_std} linear_f1={lin_mean}±{lin_std} finetune_f1={ft_mean}±{ft_std}")
else:
    print(f"  {display_name} n={n_samples}: no results available")

PYEOF
}

# ============================================================
# Main loop
# ============================================================
echo ""
echo "=========================================="
echo "Starting SFT Baseline Experiments (Randomly Initialized ViT, Pre-CP)"
echo "Total experiments: ${#EXPERIMENTS[@]} configs x ${#SEEDS[@]} seeds = $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} )) runs"
echo "=========================================="
echo ""

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r dataset n_samples <<< "$exp"
    display_name=$(get_display_name ${dataset})

    echo ""
    echo "============================================================"
    echo "Experiment: ${display_name} | n_samples=${n_samples}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        run_single ${dataset} ${n_samples} ${seed}
        if [ $? -eq 0 ]; then
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done

    # Aggregate after all seeds for this experiment
    echo "--- Aggregating results for ${display_name} n=${n_samples} ---"
    aggregate_results ${dataset} ${n_samples}
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results CSV: ${CSV_FILE}"
echo "  Individual JSONs: ${RESULTS_DIR}/"
echo "  End Time: $(date)"
echo "=========================================="
