#!/bin/bash
#SBATCH --job-name=mae-max-eval
#SBATCH --account=iu_0092
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-0
#SBATCH --output=/dcai/projects/iu_0092/projects/cp/outputs/slurm-logs/mae-max-eval-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/max_manifest.csv"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
OVERRIDE_SEED=""
INCLUDE_POOL_STRATEGY="cls"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --task-index) TASK_INDEX="$2"; shift 2 ;;
        --seed) OVERRIDE_SEED="$2"; shift 2 ;;
        --include-pool-strategy) INCLUDE_POOL_STRATEGY="$2"; shift 2 ;;
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
echo "Include pool strategy: ${INCLUDE_POOL_STRATEGY}"
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

eval "$("${PYTHON_BIN}" - "${MANIFEST}" "${TASK_INDEX}" "${OVERRIDE_SEED}" "${INCLUDE_POOL_STRATEGY}" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
task_index = int(sys.argv[2])
override_seed = sys.argv[3]
include_pool_strategy = sys.argv[4]
seeds = [42, 43, 44]

with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

if include_pool_strategy != "all":
    rows = [row for row in rows if row["pool_strategy"] == include_pool_strategy]

if not rows:
    raise SystemExit(
        f"Manifest has no rows matching pool_strategy={include_pool_strategy}: {manifest}"
    )

row_index = task_index // len(seeds)
seed_index = task_index % len(seeds)

if row_index >= len(rows):
    raise SystemExit(
        f"Task index {task_index} maps to row {row_index}, "
        f"but selected manifest only has {len(rows)} rows"
    )

row = rows[row_index]
seed = int(override_seed) if override_seed else seeds[seed_index]
output_root = "/dcai/projects/iu_0092/projects/cp/outputs"
source_rel = f"cp/MAE/pretrained/{row['display_name']}/{row['backbone_tag']}/all"
eval_rel = f"cp/MAE/pretrained/{row['display_name']}/{row['backbone_tag']}/all_eval_fix"
ckpt_name = (
    f"{row['dataset']}_{row['backbone_timm'].replace('/', '_')}"
    f"_n{row['n_samples']}_s{seed}.ckpt"
)

values = {
    **row,
    "seed": seed,
    "row_index": row_index,
    "seed_index": seed_index,
    "source_ckpt": f"{output_root}/ckpts/{source_rel}/cp/{ckpt_name}",
    "ckpt_dir": f"{output_root}/ckpts/{eval_rel}",
    "old_results_file": (
        f"{output_root}/logs/{source_rel}/"
        f"{row['backbone_tag']}_{row['dataset']}_n{row['n_samples']}_seed{seed}.json"
    ),
    "results_file": (
        f"{output_root}/logs/{eval_rel}/"
        f"{row['backbone_tag']}_{row['dataset']}_n{row['n_samples']}_seed{seed}.json"
    ),
    "data_dir": "/dcai/projects/iu_0092/projects/cp/data",
    "slurm_log_dir": f"{output_root}/slurm-logs",
}

for key, value in values.items():
    print(f"{key.upper()}={shlex.quote(str(value))}")
PY
)"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "$(dirname "${RESULTS_FILE}")" "${SLURM_LOG_DIR}"

EPOCHS=150
BATCH_SIZE=64
ACCUMULATE_GRAD_BATCHES=4
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
KNN_K=20
NUM_WORKERS=8
DECODER_DIM=512
DECODER_DEPTH=4
MASK_RATIO=0.75

BLOCK_LABEL="${NUM_TRAINED_BLOCKS}"
if [ "${NUM_TRAINED_BLOCKS}" = "-1" ]; then
    BLOCK_LABEL="ALL"
fi

echo "=========================================="
echo "Starting MAE-CP post eval-only run"
echo "Run ID: ${RUN_ID}"
echo "Dataset: ${DISPLAY_NAME} (${DATASET})"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "n_samples=${N_SAMPLES} seed=${SEED}"
echo "pool_strategy=${POOL_STRATEGY}"
echo "Source checkpoint: ${SOURCE_CKPT}"
echo "Old results file: ${OLD_RESULTS_FILE}"
echo "Corrected results file: ${RESULTS_FILE}"
echo "Eval checkpoint dir: ${CKPT_DIR}"
echo "=========================================="

if [ ! -f "${SOURCE_CKPT}" ]; then
    echo "Source CP checkpoint not found: ${SOURCE_CKPT}" >&2
    exit 1
fi

if [ ! -f "${OLD_RESULTS_FILE}" ]; then
    echo "Old results file not found: ${OLD_RESULTS_FILE}" >&2
    exit 1
fi

if [ -f "${RESULTS_FILE}" ]; then
    echo "[SKIP] ${RESULTS_FILE} already exists"
    exit 0
fi

"${PYTHON_BIN}" -u continued_pretraining.py \
    --cp-method mae \
    --post-cp-sft \
    --eval-only-cp-checkpoint "${SOURCE_CKPT}" \
    --merge-results-json "${OLD_RESULTS_FILE}" \
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
    --decoder-dim "${DECODER_DIM}" \
    --decoder-depth "${DECODER_DEPTH}" \
    --mask-ratio "${MASK_RATIO}" \
    --pool-strategy "${POOL_STRATEGY}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --cache-dir "${DATA_DIR}" \
    --project "${PROJECT}-eval-fix" \
    --run-name "${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_blk${BLOCK_LABEL}_s${SEED}_eval_fix" \
    --seed "${SEED}" \
    --skip-baseline \
    --results-json "${RESULTS_FILE}" 2>&1

echo ""
echo "=========================================="
echo "MAE-CP post eval-only run completed"
echo "Run ID: ${RUN_ID}"
echo "End Time: $(date)"
echo "=========================================="
