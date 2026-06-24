#!/bin/bash
#SBATCH --job-name=lejepa-rem
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --array=0-0
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/lejepa-remaining-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/remaining_manifest.csv"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
OVERRIDE_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --task-index) TASK_INDEX="$2"; shift 2 ;;
        --seed) OVERRIDE_SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Start Time: $(date)"
echo "Manifest: ${MANIFEST}"
echo "Task index: ${TASK_INDEX}"
echo "=========================================="

cd /dcai/projects/iu_0092/projects/cp/continued-pretraining
echo "Working directory: $(pwd)"
source .venv/bin/activate
PYTHON_BIN="$(pwd)/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found at ${PYTHON_BIN}" >&2
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

eval "$("${PYTHON_BIN}" - "${MANIFEST}" "${TASK_INDEX}" "${OVERRIDE_SEED}" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
task_index = int(sys.argv[2])
override_seed = sys.argv[3]
seeds = [42, 43, 44]

with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit(f"Manifest has no rows: {manifest}")

row_index = task_index // len(seeds)
seed_index = task_index % len(seeds)

if row_index >= len(rows):
    raise SystemExit(
        f"Task index {task_index} maps to row {row_index}, "
        f"but manifest only has {len(rows)} rows"
    )

row = rows[row_index]
seed = int(override_seed) if override_seed else seeds[seed_index]
output_root = "/dcai/projects/iu_0092/projects/cp/outputs"

if row["group"] == "random":
    rel_dir = f"cp/LeJEPA/random/{row['display_name']}"
else:
    rel_dir = f"cp/LeJEPA/pretrained/{row['display_name']}/{row['backbone_tag']}"

values = {
    **row,
    "seed": seed,
    "row_index": row_index,
    "seed_index": seed_index,
    "ckpt_dir": f"{output_root}/ckpts/{rel_dir}",
    "log_dir": f"{output_root}/logs/{rel_dir}",
    "data_dir": "/dcai/projects/iu_0092/projects/cp/data",
    "slurm_log_dir": f"{output_root}/slurm-logs",
}

for key, value in values.items():
    print(f"{key.upper()}={shlex.quote(str(value))}")
PY
)"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"

EPOCHS=150
BATCH_SIZE=32
ACCUMULATE_GRAD_BATCHES=8
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
KNN_K=20
NUM_WORKERS=8
LAMB=0.02
N_VIEWS=8
PROJ_DIM=128
HIDDEN_DIM=2048

BLOCK_LABEL="${NUM_TRAINED_BLOCKS}"
if [ "${NUM_TRAINED_BLOCKS}" = "-1" ]; then
    BLOCK_LABEL="ALL"
fi

RESULTS_FILE="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_seed${SEED}.json"

echo "=========================================="
echo "Starting LeJEPA-CP remaining run"
echo "Run ID: ${RUN_ID}"
echo "Cookbook row: ${COOKBOOK_ROW}"
echo "Dataset: ${DISPLAY_NAME} (${DATASET})"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "n_samples=${N_SAMPLES} seed=${SEED}"
echo "num_trained_blocks=${NUM_TRAINED_BLOCKS} batch_size=${BATCH_SIZE} accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
echo "Results file: ${RESULTS_FILE}"
echo "=========================================="

if [ -f "${RESULTS_FILE}" ]; then
    echo "[SKIP] ${RESULTS_FILE} already exists"
    exit 0
fi

RANDOM_ARGS=()
if [ "${RANDOM_INIT}" = "yes" ]; then
    RANDOM_ARGS+=(--random-init)
fi

"${PYTHON_BIN}" -u continued_pretraining.py \
    --cp-method lejepa \
    "${RANDOM_ARGS[@]}" \
    --post-cp-sft \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE_TIMM}" \
    --n-samples "${N_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --freeze-epochs "${FREEZE_EPOCHS}" \
    --num-trained-blocks "${NUM_TRAINED_BLOCKS}" \
    --knn-k "${KNN_K}" \
    --num-workers "${NUM_WORKERS}" \
    --lamb "${LAMB}" \
    --n-views "${N_VIEWS}" \
    --proj-dim "${PROJ_DIM}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --pool-strategy "${POOL_STRATEGY}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --cache-dir "${DATA_DIR}" \
    --project "${PROJECT}" \
    --run-name "${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_blk${BLOCK_LABEL}_s${SEED}" \
    --seed "${SEED}" \
    --skip-baseline \
    --results-json "${RESULTS_FILE}" 2>&1

echo ""
echo "=========================================="
echo "LeJEPA-CP remaining run completed"
echo "Run ID: ${RUN_ID}"
echo "End Time: $(date)"
echo "=========================================="
