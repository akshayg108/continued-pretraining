#!/bin/bash
#SBATCH --job-name=p-path
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/precp-pathmnist-small-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/precp-pathmnist-small-%j.err

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
# Parse optional arguments (e.g., sbatch run.sh --seed 42)
# ============================================================
OVERRIDE_SEED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) OVERRIDE_SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ============================================================
# Paths
# ============================================================
DATA_DIR="/scratch/gs4133/zhd/CP/data"
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip/pre-cp/PathMNIST/SigLIP/small"
LOG_DIR="/scratch/gs4133/zhd/CP/outputs/logs/cp-siglip/pre-cp/PathMNIST/SigLIP/small"
SLURM_LOG_DIR="/scratch/gs4133/zhd/CP/outputs/slurm-log"
mkdir -p ${DATA_DIR} ${CKPT_DIR} ${LOG_DIR} ${SLURM_LOG_DIR}

# ============================================================
# Fixed parameters
# ============================================================
DATASET="pathmnist"
DISPLAY_NAME="PathMNIST"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="SigLIP"
BACKBONE_TIMM="vit_base_patch16_siglip_224.v2_webli"

EPOCHS=150
EFFECTIVE_BATCH=256
ACCUMULATE_GRAD_BATCHES=1
if [ $((EFFECTIVE_BATCH % ACCUMULATE_GRAD_BATCHES)) -ne 0 ]; then
    echo "[ERROR] EFFECTIVE_BATCH=${EFFECTIVE_BATCH} is not divisible by ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}" >&2
    exit 1
fi
BATCH_SIZE=$((EFFECTIVE_BATCH / ACCUMULATE_GRAD_BATCHES))
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
NUM_TRAINED_BLOCKS=2
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)
if [ -n "$OVERRIDE_SEED" ]; then SEEDS=($OVERRIDE_SEED); fi

# LeJEPA hyperparameters
LAMB=0.02
N_VIEWS=8
PROJ_DIM=128
HIDDEN_DIM=2048

# n_samples for small runs
NSAMPLES=(89996)

# ============================================================
# Run a single experiment
# ============================================================
run_single() {
    local n_samples=$1
    local seed=$2

    local dataset_log_dir="${LOG_DIR}"
    mkdir -p "${dataset_log_dir}"

    local results_file="${dataset_log_dir}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${seed}_pre.json"

    local dataset_ckpt_dir="${CKPT_DIR}"
    mkdir -p "${dataset_ckpt_dir}"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] Pre-CP eval ${BACKBONE_TAG} | ${DATASET} | n=${n_samples} | seed=${seed}"
    echo "  freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
    echo "  Start: $(date)"
    echo "=========================================="

    python -u continued_pretraining.py \
        --cp-method lejepa \
        --pre-cp-sft \
        --dataset ${DATASET} \
        --backbone ${BACKBONE_TIMM} \
        --n-samples ${n_samples} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --freeze-epochs ${FREEZE_EPOCHS} \
        --num-trained-blocks ${NUM_TRAINED_BLOCKS} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --lamb ${LAMB} \
        --n-views ${N_VIEWS} \
        --proj-dim ${PROJ_DIM} \
        --hidden-dim ${HIDDEN_DIM} \
        --pool-strategy map \
        --accumulate-grad-batches ${ACCUMULATE_GRAD_BATCHES} \
        --checkpoint-dir ${dataset_ckpt_dir} \
        --cache-dir ${DATA_DIR} \
        --project precp-siglip-${DATASET} \
        --run-name "${BACKBONE_TAG}_${DATASET}_n${n_samples}_blk${NUM_TRAINED_BLOCKS}_s${seed}" \
        --seed ${seed} \
        --no-cp \
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
METRICS = ["pre_knn_f1", "pre_linear_f1", "pre_sft_f1"]
acc = {m: [] for m in METRICS}

def fmt(v):
    return f"{v:.6f}" if v is not None else ""

for i, seed in enumerate(seeds):
    rf = os.path.join(log_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}_pre.json")
    if not os.path.exists(rf):
        print(f"  Warning: {rf} not found, skipping seed {seed}")
        continue
    data = json.load(open(rf))
    for m in METRICS:
        v = data.get(m)
        if v is not None:
            acc[m].append(v)
    row = f"{backbone_tag},{display_name},{n_samples},{model_size},{i}"
    for m in METRICS:
        row += f",{fmt(data.get(m))},"
    with open(csv_file, "a") as f:
        f.write(row + "\n")

def mean_std(v):
    if not v:
        return "", ""
    return f"{statistics.mean(v):.6f}", (f"{statistics.stdev(v):.6f}" if len(v) > 1 else "0.000000")

if any(acc[m] for m in METRICS):
    cells = [backbone_tag, display_name, str(n_samples), model_size, "average"]
    for m in METRICS:
        mm, ss = mean_std(acc[m]); cells += [mm, ss]
    with open(csv_file, "a") as f:
        f.write(",".join(cells) + "\n")
    summ = "  ".join(f"{m}={mean_std(acc[m])[0]}+-{mean_std(acc[m])[1]}({len(acc[m])}/3)" for m in METRICS)
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: {summ}")
else:
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: no results available")
PYEOF
}

# ============================================================
# Main loop
# ============================================================
echo ""
echo "=========================================="
echo "Starting Pre-CP eval: ${DISPLAY_NAME} (small: n=100,500,1000)"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="
echo ""

CSV_FILE="${LOG_DIR}/${BACKBONE_TAG}_lejepa_cp_results.csv"
if [ ! -f "${CSV_FILE}" ]; then
    echo "backbone,dataset,n_samples,model_size,run,pre_knn_f1,pre_knn_f1_std,pre_linear_f1,pre_linear_f1_std,pre_sft_f1,pre_sft_f1_std" > ${CSV_FILE}
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
echo "All Pre-CP eval ${DISPLAY_NAME} small experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/"
echo "  End Time: $(date)"
echo "=========================================="
