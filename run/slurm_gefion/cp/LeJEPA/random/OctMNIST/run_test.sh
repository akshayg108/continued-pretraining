#!/bin/bash
#SBATCH --job-name=l-oct-test
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/lejepa-rand-octmnist-test-%j.out

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

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
"${PYTHON_BIN}" -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"

echo "=========================================="
nvidia-smi

# ============================================================
# Paths
# ============================================================
DATA_DIR="/dcai/projects/iu_0092/projects/cp/data"
CKPT_DIR="/dcai/projects/iu_0092/projects/cp/outputs/ckpts/cp/LeJEPA/random/OctMNIST_test"
LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/logs/cp/LeJEPA/random/OctMNIST_test"
SLURM_LOG_DIR="/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"

# ============================================================
# Smoke-test parameters
# ============================================================
DATASET="octmnist"
BACKBONE_TAG="SCRATCH_TEST"
BACKBONE_TIMM="vit_base_patch16_224"

EPOCHS=1
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=0
NUM_TRAINED_BLOCKS=-1
KNN_K=5
NUM_WORKERS=4
SEED=42
N_SAMPLES=512

LAMB=0.02
N_VIEWS=4
PROJ_DIM=128
HIDDEN_DIM=512

RESULTS_FILE="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_seed${SEED}.json"

echo "=========================================="
echo "Starting LeJEPA smoke test"
echo "Dataset: ${DATASET}"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "n_samples=${N_SAMPLES} epochs=${EPOCHS} seed=${SEED}"
echo "Results file: ${RESULTS_FILE}"
echo "=========================================="

"${PYTHON_BIN}" -u continued_pretraining.py \
    --cp-method lejepa \
    --random-init \
    --post-cp-sft \
    --dataset ${DATASET} \
    --backbone ${BACKBONE_TIMM} \
    --n-samples ${N_SAMPLES} \
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
    --project lejepa-cp-rand-octmnist-test \
    --run-name "${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_s${SEED}" \
    --seed ${SEED} \
    --skip-baseline \
    --results-json ${RESULTS_FILE} 2>&1

EXIT_CODE=$?

echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
