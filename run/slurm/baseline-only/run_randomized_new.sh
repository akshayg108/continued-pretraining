#!/bin/bash
#SBATCH --job-name=baseline-randomized-new
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-randomized-new-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-randomized-new-%j.err

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
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts/baseline-only"
LOG_DIR="/scratch/gs4133/zhd/CP/outputs/logs/baseline-only"
SLURM_LOG_DIR="/scratch/gs4133/zhd/CP/outputs/slurm-log"
mkdir -p ${DATA_DIR} ${CKPT_DIR} ${LOG_DIR} ${SLURM_LOG_DIR}

# ============================================================
# Fixed parameters
# ============================================================
MODEL_SIZE="ViT-B"
BATCH_SIZE=32
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

# ============================================================
# Backbone definition (randomly initialized)
# ============================================================
BACKBONE_TAG="SCRATCH"
BACKBONE_TIMM="vit_base_patch16_224"

# ============================================================
# Experiment list (from results.csv Baselines group)
# ============================================================
EXPERIMENTS=(
    # Cars196 (MAX=8144, 196 classes)
    "cars196 196"
    "cars196 500"
    "cars196 1000"
    "cars196 8144"

    # CUB200 (MAX=5994, 200 classes)
    "cub200 200"
    "cub200 500"
    "cub200 1000"
    "cub200 5994"

    # Flowers102 (MAX=1020, 102 classes)
    "flowers102 102"
    "flowers102 500"
    "flowers102 1000"
    "flowers102 1020"

    # OxfordPet (MAX=3680, 37 classes)
    "oxford_pet 100"
    "oxford_pet 500"
    "oxford_pet 1000"
    "oxford_pet 3680"

    # DTD (MAX=1880, 47 classes)
    "dtd 100"
    "dtd 500"
    "dtd 1000"
    "dtd 1880"

    # EuroSAT (MAX=16200, 10 classes)
    "eurosat 100"
    "eurosat 500"
    "eurosat 1000"
    "eurosat 10000"
    "eurosat 16200"

    # PlantVillage (MAX=43596, 38 classes)
    "plant_village 100"
    "plant_village 500"
    "plant_village 1000"
    "plant_village 10000"
    "plant_village 25000"
    "plant_village 43596"
)

# ============================================================
# CSV column name mapping
# ============================================================
get_display_name() {
    case "$1" in
        dermamnist)     echo "DermaMNIST" ;;
        breastmnist)    echo "BreastMNIST" ;;
        octmnist)       echo "OCTMNIST" ;;
        organamnist)    echo "OrganAMNIST" ;;
        pathmnist)      echo "PathMNIST" ;;
        galaxy10)       echo "Galaxy10" ;;
        food101)        echo "Food101" ;;
        fgvc_aircraft)  echo "FGVC_Aircraft" ;;
        cars196)        echo "Cars196" ;;
        cub200)         echo "CUB200" ;;
        flowers102)     echo "Flowers102" ;;
        oxford_pet)     echo "OxfordPet" ;;
        dtd)            echo "DTD" ;;
        eurosat)        echo "EuroSAT" ;;
        plant_village)  echo "PlantVillage" ;;
        *)              echo "$1" ;;
    esac
}

# ============================================================
# Run a single experiment (baseline only: KNN + Linear Probe)
# ============================================================
run_single() {
    local dataset=$1
    local n_samples=$2
    local seed=$3

    local dataset_results_dir="${LOG_DIR}/${dataset}"
    mkdir -p "${dataset_results_dir}"

    local results_file="${dataset_results_dir}/${BACKBONE_TAG}_${dataset}_n${n_samples}_seed${seed}.json"

    local dataset_ckpt_dir="${CKPT_DIR}/${dataset}"
    mkdir -p "${dataset_ckpt_dir}"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${dataset} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] ${BACKBONE_TAG} | ${dataset} | n=${n_samples} | seed=${seed}"
    echo "  Start: $(date)"
    echo "=========================================="

    python -u continued_pretraining.py \
        --cp-method diet \
        --no-cp \
        --random-init \
        --dataset ${dataset} \
        --backbone ${BACKBONE_TIMM} \
        --n-samples ${n_samples} \
        --batch-size ${BATCH_SIZE} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --checkpoint-dir ${dataset_ckpt_dir} \
        --cache-dir ${DATA_DIR} \
        --project baseline-randomized-new \
        --run-name "${BACKBONE_TAG}_${dataset}_n${n_samples}_s${seed}" \
        --seed ${seed} \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${BACKBONE_TAG} | ${dataset} n=${n_samples} seed=${seed}"
    fi

    return $exit_code
}

# ============================================================
# Aggregate results across seeds
# ============================================================
aggregate_results() {
    local dataset=$1
    local n_samples=$2
    local display_name=$(get_display_name ${dataset})
    local csv_file=$3

    local dataset_results_dir="${LOG_DIR}/${dataset}"

    python3 << PYEOF
import json, os, statistics

results_dir = "${dataset_results_dir}"
backbone_tag = "${BACKBONE_TAG}"
dataset = "${dataset}"
n_samples = "${n_samples}"
display_name = "${display_name}"
model_size = "${MODEL_SIZE}"
csv_file = "${csv_file}"
seeds = [42, 43, 44]

knn_f1s = []
linear_f1s = []

for i, seed in enumerate(seeds):
    results_file = os.path.join(results_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}.json")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping seed {seed}")
        continue

    with open(results_file) as f:
        data = json.load(f)

    knn_f1 = data.get("pre_knn_f1")
    linear_f1 = data.get("pre_linear_f1")

    if knn_f1 is not None:
        knn_f1s.append(knn_f1)
    if linear_f1 is not None:
        linear_f1s.append(linear_f1)

    with open(csv_file, "a") as f:
        knn_str = f"{knn_f1:.6f}" if knn_f1 is not None else ""
        lin_str = f"{linear_f1:.6f}" if linear_f1 is not None else ""
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},{i},{knn_str},,{lin_str},\n")

if len(knn_f1s) > 0 or len(linear_f1s) > 0:
    def mean_std(vals):
        if len(vals) == 0:
            return "", ""
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.6f}", f"{s:.6f}"

    knn_mean, knn_std = mean_std(knn_f1s)
    lin_mean, lin_std = mean_std(linear_f1s)

    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},average,{knn_mean},{knn_std},{lin_mean},{lin_std}\n")

    print(f"  [{backbone_tag}] {display_name} n={n_samples}: knn_f1={knn_mean}+-{knn_std} linear_f1={lin_mean}+-{lin_std}")
else:
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: no results available")

PYEOF
}

# ============================================================
# Main loop
# ============================================================
echo ""
echo "=========================================="
echo "Starting Baseline-Only Evaluation (KNN + Linear Probe, Randomly Initialized)"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="
echo ""

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r dataset n_samples <<< "$exp"
    display_name=$(get_display_name ${dataset})

    dataset_log_dir="${LOG_DIR}/${dataset}"
    mkdir -p "${dataset_log_dir}"
    CSV_FILE="${dataset_log_dir}/${BACKBONE_TAG}_baseline_results.csv"
    if [ ! -f "${CSV_FILE}" ]; then
        echo "backbone,dataset,n_samples,model_size,run,knn_f1,knn_f1_std,linear_f1,linear_f1_std" > ${CSV_FILE}
    fi
    echo "CSV file: ${CSV_FILE}"

    echo ""
    echo "============================================================"
    echo "Experiment: ${BACKBONE_TAG} | ${display_name} | n_samples=${n_samples}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        run_single ${dataset} ${n_samples} ${seed}
        if [ $? -eq 0 ]; then
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done

    echo "--- Aggregating results for ${BACKBONE_TAG} | ${display_name} n=${n_samples} ---"
    aggregate_results ${dataset} ${n_samples} ${CSV_FILE}
done

echo ""
echo "=========================================="
echo "All baseline experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/{dataset}/"
echo "  End Time: $(date)"
echo "=========================================="
