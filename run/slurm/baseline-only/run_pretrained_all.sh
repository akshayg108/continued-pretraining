#!/bin/bash
#SBATCH --job-name=baseline-pretrained
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-pretrained-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-pretrained-%j.err

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
# Backbone definitions
# ============================================================
BACKBONE_TAGS=("DINOv3" "MAE" "CLIP")
BACKBONE_TIMM_NAMES=(
    "vit_base_patch16_dinov3.lvd1689m"
    "vit_base_patch16_224.mae"
    "vit_base_patch16_clip_224.openai"
)

# ============================================================
# Experiment lists per backbone
# Same datasets/n_samples as DIET pre-cp-only
# ============================================================

DINOV3_EXPERIMENTS=(
    # DermaMNIST (MAX=7007)
    "dermamnist 100"
    "dermamnist 500"
    "dermamnist 1000"
    "dermamnist 7007"

    # BreastMNIST (MAX=546)
    "breastmnist 100"
    "breastmnist 500"
    "breastmnist 546"

    # OCTMNIST (MAX=97477)
    "octmnist 100"
    "octmnist 500"
    "octmnist 1000"
    "octmnist 10000"
    "octmnist 25000"
    "octmnist 97477"

    # OrganAMNIST (MAX=34561)
    "organamnist 100"
    "organamnist 500"
    "organamnist 1000"
    "organamnist 10000"
    "organamnist 25000"
    "organamnist 34561"

    # PathMNIST (MAX=89996)
    "pathmnist 100"
    "pathmnist 500"
    "pathmnist 1000"
    "pathmnist 10000"
    "pathmnist 25000"
    "pathmnist 89996"

    # Galaxy10 (MAX=14188)
    "galaxy10 100"
    "galaxy10 500"
    "galaxy10 1000"
    "galaxy10 10000"
    "galaxy10 14188"

    # Food101 (MAX=75750)
    "food101 101"
    "food101 500"
    "food101 1000"
    "food101 10000"
    "food101 25000"
    "food101 75750"

    # FGVC_Aircraft (MAX=3400)
    "fgvc_aircraft 100"
    "fgvc_aircraft 500"
    "fgvc_aircraft 1000"
    "fgvc_aircraft 3400"
)

MAE_EXPERIMENTS=(
    "dermamnist 1000"
    "dermamnist 7007"
    "breastmnist 100"
    "breastmnist 546"
    "octmnist 1000"
    "octmnist 97477"
    "organamnist 1000"
    "organamnist 34561"
    "pathmnist 1000"
    "pathmnist 89996"
    "galaxy10 1000"
    "galaxy10 14188"
    "food101 1000"
    "food101 75750"
    "fgvc_aircraft 1000"
    "fgvc_aircraft 3400"
)

CLIP_EXPERIMENTS=(
    "dermamnist 1000"
    "dermamnist 7007"
    "breastmnist 100"
    "breastmnist 546"
    "octmnist 1000"
    "octmnist 97477"
    "organamnist 1000"
    "organamnist 34561"
    "pathmnist 1000"
    "pathmnist 89996"
    "galaxy10 1000"
    "galaxy10 14188"
    "food101 1000"
    "food101 75750"
    "fgvc_aircraft 1000"
    "fgvc_aircraft 3400"
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
        *)              echo "$1" ;;
    esac
}

# ============================================================
# Run a single experiment (baseline only: KNN + Linear Probe)
# ============================================================
run_single() {
    local backbone_tag=$1
    local backbone_timm=$2
    local dataset=$3
    local n_samples=$4
    local seed=$5
    local pool_strategy=$6

    local dataset_results_dir="${LOG_DIR}/${dataset}"
    mkdir -p "${dataset_results_dir}"

    local results_file="${dataset_results_dir}/${backbone_tag}_${dataset}_n${n_samples}_seed${seed}.json"

    local dataset_ckpt_dir="${CKPT_DIR}/${dataset}"
    mkdir -p "${dataset_ckpt_dir}"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${backbone_tag} | ${dataset} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] ${backbone_tag} | ${dataset} | n=${n_samples} | seed=${seed}"
    echo "  Start: $(date)"
    echo "=========================================="

    local backbone_tag_lower=$(echo "${backbone_tag}" | tr '[:upper:]' '[:lower:]')

    python -u continued_pretraining.py \
        --cp-method diet \
        --no-cp \
        --dataset ${dataset} \
        --backbone ${backbone_timm} \
        --n-samples ${n_samples} \
        --batch-size ${BATCH_SIZE} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --pool-strategy ${pool_strategy} \
        --checkpoint-dir ${dataset_ckpt_dir} \
        --cache-dir ${DATA_DIR} \
        --project baseline-pretrained \
        --run-name "${backbone_tag}_${dataset}_n${n_samples}_s${seed}" \
        --seed ${seed} \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${backbone_tag} | ${dataset} n=${n_samples} seed=${seed}"
    fi

    return $exit_code
}

# ============================================================
# Aggregate results across seeds
# ============================================================
aggregate_results() {
    local backbone_tag=$1
    local dataset=$2
    local n_samples=$3
    local display_name=$(get_display_name ${dataset})
    local csv_file=$4

    local dataset_results_dir="${LOG_DIR}/${dataset}"

    python3 << PYEOF
import json, os, statistics

results_dir = "${dataset_results_dir}"
backbone_tag = "${backbone_tag}"
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
echo "Starting Baseline-Only Evaluation (KNN + Linear Probe, Pretrained Backbones)"
echo "Backbones: ${BACKBONE_TAGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="
echo ""

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for idx in "${!BACKBONE_TAGS[@]}"; do
    BACKBONE_TAG="${BACKBONE_TAGS[$idx]}"
    BACKBONE_TIMM="${BACKBONE_TIMM_NAMES[$idx]}"

    echo ""
    echo "############################################################"
    echo "# Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
    echo "############################################################"
    echo ""

    case "$BACKBONE_TAG" in
        DINOv3) CURRENT_EXPERIMENTS=("${DINOV3_EXPERIMENTS[@]}"); POOL_STRATEGY="cls" ;;
        MAE)    CURRENT_EXPERIMENTS=("${MAE_EXPERIMENTS[@]}"); POOL_STRATEGY="mean" ;;
        CLIP)   CURRENT_EXPERIMENTS=("${CLIP_EXPERIMENTS[@]}"); POOL_STRATEGY="cls" ;;
    esac

    for exp in "${CURRENT_EXPERIMENTS[@]}"; do
        read -r dataset n_samples <<< "$exp"
        display_name=$(get_display_name ${dataset})

        # Per-dataset CSV
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
            run_single ${BACKBONE_TAG} ${BACKBONE_TIMM} ${dataset} ${n_samples} ${seed} ${POOL_STRATEGY}
            if [ $? -eq 0 ]; then
                TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
            else
                TOTAL_FAIL=$((TOTAL_FAIL + 1))
            fi
        done

        echo "--- Aggregating results for ${BACKBONE_TAG} | ${display_name} n=${n_samples} ---"
        aggregate_results ${BACKBONE_TAG} ${dataset} ${n_samples} ${CSV_FILE}
    done
done

echo ""
echo "=========================================="
echo "All baseline experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/{dataset}/"
echo "  End Time: $(date)"
echo "=========================================="
