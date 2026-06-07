#!/bin/bash
#SBATCH --job-name=baseline-mae-clip-new
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-mae-clip-new-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/baseline-mae-clip-new-%j.err

# ============================================================
# Baseline-only delta runner for MAE + CLIP.
#
# Runs the (dataset, n_samples) tuples that were ADDED to MAE/CLIP when their
# experiment lists were expanded to mirror DINOv3 (small + MAX → full list).
# All other paths/conventions mirror run_pretrained.sh, so skip-detection
# against the existing JSONs in baseline-only/<dataset>/ still works.
# ============================================================

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
# Paths (identical to run_pretrained.sh so JSONs/CSVs land in the same place)
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
# Backbones — MAE and CLIP only
# ============================================================
BACKBONE_TAGS=("MAE" "CLIP")
BACKBONE_TIMM_NAMES=(
    "vit_base_patch16_224.mae"
    "vit_base_patch16_clip_224.openai"
)

# ============================================================
# Delta experiments — the (dataset, n_samples) pairs that DINOv3 has
# but MAE/CLIP did not have in the previous run_pretrained.sh.
#
# MAE/CLIP previously only had: small (100/102/196/200/1000) + MAX per dataset.
# The full list adds intermediate sizes (100/500, 10000, 25000, etc.) to match DINOv3.
#
# 40 entries per backbone × 2 backbones × 3 seeds = 240 runs total.
# ============================================================
DELTA_EXPERIMENTS=(
    # DermaMNIST — added: 100, 500
    "dermamnist 100"
    "dermamnist 500"

    # BreastMNIST — added: 500
    "breastmnist 500"

    # OCTMNIST — added: 100, 500, 10000, 25000
    "octmnist 100"
    "octmnist 500"
    "octmnist 10000"
    "octmnist 25000"

    # OrganAMNIST — added: 100, 500, 10000, 25000
    "organamnist 100"
    "organamnist 500"
    "organamnist 10000"
    "organamnist 25000"

    # PathMNIST — added: 100, 500, 10000, 25000
    "pathmnist 100"
    "pathmnist 500"
    "pathmnist 10000"
    "pathmnist 25000"

    # Galaxy10 — added: 100, 500, 10000
    "galaxy10 100"
    "galaxy10 500"
    "galaxy10 10000"

    # Food101 — added: 101, 500, 10000, 25000
    "food101 101"
    "food101 500"
    "food101 10000"
    "food101 25000"

    # FGVC_Aircraft — added: 100, 500
    "fgvc_aircraft 100"
    "fgvc_aircraft 500"

    # Cars196 — added: 196, 500
    "cars196 196"
    "cars196 500"

    # CUB200 — added: 200, 500
    "cub200 200"
    "cub200 500"

    # Flowers102 — added: 500
    "flowers102 500"

    # OxfordPet — added: 100, 500
    "oxford_pet 100"
    "oxford_pet 500"

    # DTD — added: 100, 500
    "dtd 100"
    "dtd 500"

    # EuroSAT — added: 100, 500, 10000
    "eurosat 100"
    "eurosat 500"
    "eurosat 10000"

    # PlantVillage — added: 100, 500, 10000, 25000
    "plant_village 100"
    "plant_village 500"
    "plant_village 10000"
    "plant_village 25000"
)

# ============================================================
# CSV column name mapping (mirrors run_pretrained.sh)
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
# Identical to run_pretrained.sh::run_single
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
# Aggregate results across seeds (identical to run_pretrained.sh)
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
echo "Starting Baseline (MAE + CLIP delta — newly-added sizes only)"
echo "Backbones: ${BACKBONE_TAGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Delta size: ${#DELTA_EXPERIMENTS[@]} experiments per backbone"
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
        MAE)  POOL_STRATEGY="mean" ;;
        CLIP) POOL_STRATEGY="cls"  ;;
    esac

    for exp in "${DELTA_EXPERIMENTS[@]}"; do
        read -r dataset n_samples <<< "$exp"
        display_name=$(get_display_name ${dataset})

        # Per-dataset CSV (same as run_pretrained.sh — new rows append to existing CSV)
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
echo "All MAE + CLIP delta baseline experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/{dataset}/"
echo "  End Time: $(date)"
echo "=========================================="
