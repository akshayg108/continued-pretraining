#!/bin/bash
#SBATCH --job-name=sg-diet
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --array=0-44%12
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/sg-diet-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/sg-diet-%A_%a.err

# ============================================================
# SigLIP x DIET-CP held-out extension (plan 2026-09-04, Task 2).
# One array task per (dataset, seed) cell: task_id = 3*dataset_index + seed_index,
# 45 cells, 12 running at a time. The manifest and recipe live ONLY in
# eval/F5_decision_score/siglip_diet_protocol.py (frozen; see SIGLIP_DIET_PREREG.md).
# Recipe = SigLIP grid: 2 trained blocks, MAP pooling, DIET main-grid hparams,
# frozen post kNN/LP only (no --post-cp-sft, no baseline: pre values are frozen).
#
#   sbatch run/slurm/cp-siglip/cp/diet_max_array.sh
#   sbatch --array=7,8 run/slurm/cp-siglip/cp/diet_max_array.sh   # rerun cells
#   SLURM_ARRAY_TASK_ID=0 bash run/slurm/cp-siglip/cp/diet_max_array.sh --dry-run
#
# Resume-safe: a cell is skipped only if its results JSON parses, names exactly
# this cell, has finite post_knn_f1/post_linear_f1, and its checkpoint is readable.
# ============================================================

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "[ERROR] unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROTOCOL="${REPO_ROOT}/eval/F5_decision_score/siglip_diet_protocol.py"
OUT_ROOT="${SIGLIP_DIET_OUT_ROOT:-/scratch/gs4133/zhd/CP/outputs}"
DATA_ROOT="${SIGLIP_DIET_DATA_DIR:-/scratch/gs4133/zhd/CP/data}"
PY="${PYTHON:-python3}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "${TASK_ID}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is unset (submit as an array job or export it for --dry-run)" >&2
    exit 2
fi

# ------------------------------------------------------------ environment (real run only;
# the protocol module needs numpy/pandas/scipy, which live in the conda env)
if [ "${DRY_RUN}" -eq 0 ]; then
    echo "=========================================="
    echo "SLURM Job ID: ${SLURM_ARRAY_JOB_ID:-?}_${TASK_ID}"
    echo "Job Name: ${SLURM_JOB_NAME:-?}   Node: ${SLURM_NODELIST:-?}"
    echo "Start Time: $(date)"
    echo "=========================================="
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
    set -euo pipefail
    PY=python
    cd "${REPO_ROOT}"
    export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export PYTHONFAULTHANDLER=1
    export WANDB_CONSOLE="wrap"
    echo "Python: $(which python)   Working directory: $(pwd)"
fi

# ------------------------------------------------------------ resolve the cell
CELL_ENV="$("${PY}" "${PROTOCOL}" --cell "${TASK_ID}" --format env)" || {
    echo "[ERROR] task id ${TASK_ID} is not a cell of the frozen 0..44 grid" >&2; exit 2; }
eval "${CELL_ENV}"

MODEL_ID="vit_base_patch16_siglip_224.v2_webli"
BACKBONE_TAG="SigLIP"
EPOCHS=150
BATCH_SIZE=32
ACCUMULATE_GRAD_BATCHES=1
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
NUM_TRAINED_BLOCKS=2
KNN_K=20
NUM_WORKERS=8
LABEL_SMOOTHING=0.3
MIXUP_ALPHA=1.0
CUTMIX_ALPHA=1.0
MIXUP_CUTMIX_PROB=0.0
MIXUP_CUTMIX_SWITCH_PROB=0.5

CKPT_DIR="${OUT_ROOT}/ckpts/cp-siglip/cp/DIET/${DISPLAY_NAME}/${BACKBONE_TAG}"
LOG_DIR="${OUT_ROOT}/logs/cp-siglip/cp/DIET/${DISPLAY_NAME}/${BACKBONE_TAG}"
SLURM_LOG_DIR="${OUT_ROOT}/slurm-log"
RESULTS_JSON="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_seed${SEED}.json"
CKPT_PATH="${CKPT_DIR}/cp/${DATASET}_${MODEL_ID}_n${N_SAMPLES}_s${SEED}.ckpt"
RUN_NAME="${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_blk${NUM_TRAINED_BLOCKS}_s${SEED}"

build_command() {
    # $1 = cache dir actually used for training (node-local after staging)
    printf '%s' "python -u continued_pretraining.py \
--cp-method diet \
--dataset ${DATASET} \
--backbone ${MODEL_ID} \
--n-samples ${N_SAMPLES} \
--epochs ${EPOCHS} \
--batch-size ${BATCH_SIZE} \
--lr ${LR} \
--weight-decay ${WEIGHT_DECAY} \
--freeze-epochs ${FREEZE_EPOCHS} \
--num-trained-blocks ${NUM_TRAINED_BLOCKS} \
--knn-k ${KNN_K} \
--num-workers ${NUM_WORKERS} \
--label-smoothing ${LABEL_SMOOTHING} \
--mixup-alpha ${MIXUP_ALPHA} \
--cutmix-alpha ${CUTMIX_ALPHA} \
--mixup-cutmix-prob ${MIXUP_CUTMIX_PROB} \
--mixup-cutmix-switch-prob ${MIXUP_CUTMIX_SWITCH_PROB} \
--pool-strategy map \
--accumulate-grad-batches ${ACCUMULATE_GRAD_BATCHES} \
--checkpoint-dir ${CKPT_DIR} \
--cache-dir $1 \
--project diet-cp-siglip-${DATASET} \
--run-name ${RUN_NAME} \
--seed ${SEED} \
--skip-baseline \
--results-json ${RESULTS_JSON}"
}

cell_complete() {
    "${PY}" "${PROTOCOL}" --check-result "${RESULTS_JSON}" --cell "${TASK_ID}" --ckpt "${CKPT_PATH}" >/dev/null 2>&1
}

# ------------------------------------------------------------ dry run (no module / conda / torch)
if [ "${DRY_RUN}" -eq 1 ]; then
    "${PY}" "${PROTOCOL}" --verify-prereg >/dev/null || { echo "[ERROR] preregistration check failed" >&2; exit 3; }
    if cell_complete; then SKIP=yes; else SKIP=no; fi
    echo "DRY-RUN task_id=${TASK_ID}"
    echo "DRY-RUN dataset=${DATASET}"
    echo "DRY-RUN display=${DISPLAY_NAME}"
    echo "DRY-RUN type=${DATASET_TYPE}"
    echo "DRY-RUN seed=${SEED}"
    echo "DRY-RUN n_samples=${N_SAMPLES}"
    echo "DRY-RUN processed_subpath=${PROCESSED_SUBPATH}"
    echo "DRY-RUN results_json=${RESULTS_JSON}"
    echo "DRY-RUN ckpt_path=${CKPT_PATH}"
    echo "DRY-RUN skip=${SKIP}"
    echo "DRY-RUN command=$(build_command "${DATA_ROOT}")"
    exit 0
fi

# ------------------------------------------------------------ real run
echo "Cell: ${DATASET} (${DISPLAY_NAME}, ${DATASET_TYPE}) n=${N_SAMPLES} seed=${SEED}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"

if cell_complete; then
    echo "[SKIP] ${RUN_NAME}: verified complete result + readable checkpoint already present"
    exit 0
fi

# ------------------------------------------------------------ preflight (fail closed)
"${PY}" "${PROTOCOL}" --verify-prereg
"${PY}" -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok); sys.exit(0 if ok else 1)"
"${PY}" -c "import timm; timm.create_model('${MODEL_ID}', pretrained=False, num_classes=0); print('timm model ok: ${MODEL_ID}')"
SRC_PROC="${DATA_ROOT}/stable_datasets/processed/${PROCESSED_SUBPATH}"
[ -d "${SRC_PROC}" ] || { echo "[ERROR] processed dataset cache missing: ${SRC_PROC}" >&2; exit 4; }
touch "${LOG_DIR}/.write_probe_${TASK_ID}" && rm -f "${LOG_DIR}/.write_probe_${TASK_ID}"
touch "${CKPT_DIR}/.write_probe_${TASK_ID}" && rm -f "${CKPT_DIR}/.write_probe_${TASK_ID}"
nvidia-smi || true

# ------------------------------------------------------------ stage the processed cache to node-local storage
DATA_DIR="${DATA_ROOT}"
NEED_KB=$(( $(du -sk "${SRC_PROC}" | awk '{print $1}') + 5*1024*1024 ))
STAGE_ROOT=""
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    avail_kb=$(df -Pk "${root}" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "${avail_kb}" ] && [ "${avail_kb}" -ge "${NEED_KB}" ]; then STAGE_ROOT="${root}"; break; fi
done
if [ -n "${STAGE_ROOT}" ]; then
    LOCAL_CACHE="${STAGE_ROOT}/cpdata-${SLURM_JOB_ID:-$$}"
    DEST_PROC="${LOCAL_CACHE}/stable_datasets/processed/${PROCESSED_SUBPATH}"
    echo "===== staging ${SRC_PROC} -> ${DEST_PROC} ====="
    mkdir -p "$(dirname "${DEST_PROC}")"
    rsync -a "${SRC_PROC}/" "${DEST_PROC}/"
    trap 'rm -rf "${LOCAL_CACHE}"' EXIT
    DATA_DIR="${LOCAL_CACHE}"
else
    echo "WARN: no node-local root with >= $((NEED_KB/1024/1024))G free; reading from ${DATA_ROOT}."
fi

# ------------------------------------------------------------ train + frozen post-CP kNN/LP
CMD="$(build_command "${DATA_DIR}")"
echo "[RUN] ${RUN_NAME}"
echo "${CMD}"
set +e
eval "${CMD}"
EXIT_CODE=$?
set -e
echo "  Exit Code: ${EXIT_CODE}   End: $(date)"
[ "${EXIT_CODE}" -eq 0 ] || { echo "[FAIL] ${RUN_NAME}" >&2; exit "${EXIT_CODE}"; }

if cell_complete; then
    echo "[DONE] ${RUN_NAME}: result verified -> ${RESULTS_JSON}"
else
    echo "[FAIL] ${RUN_NAME}: training exited 0 but the result/checkpoint did not verify" >&2
    exit 5
fi
