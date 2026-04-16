#!/bin/bash
#SBATCH --job-name=d-oct-max
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-2
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/lejepa-octmnist-max-%A_%a.out

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
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

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
"${PYTHON_BIN}" -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"

echo "=========================================="
nvidia-smi

OVERRIDE_SEED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) OVERRIDE_SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

DATA_DIR="/dcai/projects/iu_0092/projects/cp/data"
CKPT_DIR="/dcai/projects/iu_0092/projects/cp/outputs/ckpts/cp/LeJEPA/pretrained/OctMNIST/DINOv3/all"
LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA/pretrained/OctMNIST/DINOv3/all"
SLURM_LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"

DATASET="octmnist"
DISPLAY_NAME="OctMNIST"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="DINOv3"
BACKBONE_TIMM="vit_base_patch16_dinov3.lvd1689m"

EPOCHS=150
BATCH_SIZE=256
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
NUM_TRAINED_BLOCKS=-1
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

LAMB=0.02
N_VIEWS=8
PROJ_DIM=128
HIDDEN_DIM=2048

NSAMPLES=(97477)

if [ -n "$OVERRIDE_SEED" ]; then
    SEED="$OVERRIDE_SEED"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
else
    SEED="${SEEDS[0]}"
fi

if [ -z "${SEED}" ]; then
    echo "No seed resolved for task id '${SLURM_ARRAY_TASK_ID:-unset}'."
    exit 1
fi

echo "=========================================="
echo "Starting LeJEPA-CP: ${DISPLAY_NAME} (MAX: 97477)"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS} (all blocks)"
echo "Resolved seed: ${SEED}"
echo "=========================================="

run_single() {
    local n_samples=$1
    local results_file="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${SEED}.json"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${SEED} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] LeJEPA-CP ${BACKBONE_TAG} | ${DATASET} | n=${n_samples} | seed=${SEED}"
    echo "  freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS} (all blocks)"
    echo "  Start: $(date)"
    echo "=========================================="

    "${PYTHON_BIN}" -u continued_pretraining.py \
        --cp-method lejepa \
        --post-cp-sft \
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
        --pool-strategy cls \
        --checkpoint-dir ${CKPT_DIR} \
        --cache-dir ${DATA_DIR} \
        --project lejepa-cp-dinov3-octmnist \
        --run-name "${BACKBONE_TAG}_${DATASET}_n${n_samples}_blkALL_s${SEED}" \
        --seed ${SEED} \
        --skip-baseline \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${SEED}"
    fi

    return $exit_code
}

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for n_samples in "${NSAMPLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Experiment: ${BACKBONE_TAG} | ${DISPLAY_NAME} | n_samples=${n_samples} | seed=${SEED}"
    echo "============================================================"

    run_single "${n_samples}"
    if [ $? -eq 0 ]; then
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "LeJEPA-CP ${DISPLAY_NAME} array task completed!"
echo "  Seed: ${SEED}"
echo "  Successful runs: ${TOTAL_SUCCESS}"
echo "  Failed runs: ${TOTAL_FAIL}"
echo "  Results: ${LOG_DIR}/"
echo "  End Time: $(date)"
echo "=========================================="

if [ ${TOTAL_FAIL} -ne 0 ]; then
    exit 1
fi
