#!/bin/bash
#SBATCH --job-name=vitl-lejepa-h200
#SBATCH --array=0-1%2
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-lejepa-h200-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-lejepa-h200-%A_%a.err

# Preserve the ViT-L 128x2 recipe. Each dataset job runs three independent seeds.
# Start from public weights, evaluate kNN/LP before and after CP, and omit FT.
set -eo pipefail

usage() {
    printf '%s\n' \
        'Preview: bash run/slurm/cp-L/cp/lejepa_h200.sh --dry-run' \
        'Both:    sbatch run/slurm/cp-L/cp/lejepa_h200.sh' \
        'Plant:   sbatch --array=0 run/slurm/cp-L/cp/lejepa_h200.sh' \
        'Organ:   sbatch --array=1 run/slurm/cp-L/cp/lejepa_h200.sh' \
        'Task 0: PlantVillage. Task 1: OrganAMNIST.' \
        'Each task runs seeds 42, 43, 44 sequentially on one full H200.' \
        'LeJEPA only, MAX, six trainable blocks, batch 128 x accumulation 2.' \
        'Each submission writes to vitl_lejepa_h200_v1/<array-job-id>/.'
}

DRY_RUN=0
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    "") ;;
    *) echo "unsupported argument: $1" >&2; usage >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 2; }

if [ "${DRY_RUN}" -eq 1 ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    for task_id in 0 1; do
        SLURM_ARRAY_TASK_ID="${task_id}" bash "${BASH_SOURCE[0]}" --dry-run
    done
    exit 0
fi
if [ "${DRY_RUN}" -eq 0 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo 'Use sbatch to train, or --dry-run to preview on the login node.' >&2
    exit 2
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[01]$ ]] || { echo 'task id must be 0 or 1' >&2; exit 2; }

if [ "${DRY_RUN}" -eq 0 ] && [ "${VITL_LEJEPA_H200_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -euo pipefail

if [ -n "${VITL_LEJEPA_H200_REPO_ROOT:-}" ]; then
    REPO_ROOT="${VITL_LEJEPA_H200_REPO_ROOT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gs4133/zhd/CP/continued-pretraining}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
[ -f "${REPO_ROOT}/continued_pretraining.py" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
cd "${REPO_ROOT}"
PY="${PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

TASKS=(
    'plant_village 43596 plant_village'
    'organamnist 34561 med_mnist/organamnist-size=224'
)
read -r DATASET N SUBPATH <<< "${TASKS[${TASK_ID}]}"
CACHE_DIR="${VITL_LEJEPA_H200_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
OUTPUT_BASE="${VITL_LEJEPA_H200_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-preview}}"
OUT="${OUTPUT_BASE}/vitl_lejepa_h200_v1/${RUN_ID}"
SOURCE="${CACHE_DIR}/stable_datasets/processed/${SUBPATH}"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "STAGE source=${SOURCE} private=vitl-lejepa-h200-${RUN_ID}-${TASK_ID}"
else
    [ -d "${SOURCE}" ] || { echo "processed cache missing: ${SOURCE}" >&2; exit 4; }
    "${PY}" - <<'GPU_CHECK'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible H200 GPU is required")
p = torch.cuda.get_device_properties(0)
if "H200" not in p.name.upper() or "MIG" in p.name.upper() or p.total_memory < 130 * 1024**3:
    raise SystemExit(f"A full H200 with at least 130 GiB is required, got {p}")
print(f"GPU: {p.name}, VRAM: {p.total_memory / 1024**3:.2f} GiB")
GPU_CHECK
    nvidia-smi

    LOCAL_CACHE=""
    NEED_KB=$(( $(du -sk "${SOURCE}" | awk '{print $1}') + 5 * 1024 * 1024 ))
    for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
        [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
        AVAIL_KB=$(df -Pk "${root}" | awk 'NR==2 {print $4}')
        if [ "${AVAIL_KB:-0}" -ge "${NEED_KB}" ]; then
            LOCAL_CACHE=$(mktemp -d "${root}/vitl-lejepa-h200-${RUN_ID}-${TASK_ID}.XXXXXX")
            trap 'rm -rf -- "${LOCAL_CACHE}"' EXIT
            DEST="${LOCAL_CACHE}/stable_datasets/processed/${SUBPATH}"
            mkdir -p "${DEST}"
            rsync -a "${SOURCE}/" "${DEST}/"
            break
        fi
    done
    [ -n "${LOCAL_CACHE}" ] || { echo "insufficient private storage for ${SOURCE}" >&2; exit 5; }
    echo "STAGED ${SOURCE} -> ${LOCAL_CACHE}"
    CACHE_DIR="${LOCAL_CACHE}"
fi

FAILURES=0
for SEED in 42 43 44; do
    CMD=("${PY}" "${REPO_ROOT}/continued_pretraining.py"
        --cp-method lejepa --dataset "${DATASET}"
        --backbone vit_large_patch16_dinov3.lvd1689m --pool-strategy cls
        --n-samples "${N}" --seed "${SEED}" --cache-dir "${CACHE_DIR}" --num-workers 8
        --epochs 150 --freeze-epochs 15 --warmup-epochs 15 --num-trained-blocks 6
        --lr 0.0001 --weight-decay 0.05 --knn-k 20
        --batch-size 128 --accumulate-grad-batches 2
        --proj-dim 128 --hidden-dim 2048 --n-views 8 --lamb 0.02
        --project vitl_lejepa_h200_v1
        --run-name "LeJEPA_${DATASET}_blk6_b128a2_s${SEED}_job${RUN_ID}"
        --checkpoint-dir "${OUT}/checkpoints/LeJEPA/${DATASET}/seed${SEED}"
        --results-json "${OUT}/cp_results/LeJEPA/${DATASET}/seed${SEED}.json")
    printf 'CP_ONLY task=%s dataset=%s seed=%s blocks=6 batch=128 accumulate=2 initialization=public_pretrained\n' \
        "${TASK_ID}" "${DATASET}" "${SEED}"
    printf 'COMMAND'; printf ' %q' "${CMD[@]}"; printf '\n'
    [ "${DRY_RUN}" -eq 0 ] || continue

    mkdir -p "${OUT}/cp_results/LeJEPA/${DATASET}" "${OUT}/commands/LeJEPA/${DATASET}"
    "${PY}" - "${OUT}/commands/LeJEPA/${DATASET}/seed${SEED}.json" "${CMD[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

command = sys.argv[2:]
record = dict(protocol="vitl_lejepa_h200_v1", initialization="public_pretrained",
              command=command,
              entrypoint_sha256=hashlib.sha256(Path(command[1]).read_bytes()).hexdigest())
with Path(sys.argv[1]).open("x") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
    if "${CMD[@]}"; then
        if [ -s "${OUT}/cp_results/LeJEPA/${DATASET}/seed${SEED}.json" ]; then
            echo "SUCCESS LeJEPA ${DATASET} seed=${SEED}"
        else
            echo "FAIL LeJEPA ${DATASET} seed=${SEED}: result JSON missing" >&2
            FAILURES=$((FAILURES + 1))
        fi
    else
        status=$?
        echo "FAIL LeJEPA ${DATASET} seed=${SEED}: exit=${status}" >&2
        FAILURES=$((FAILURES + 1))
    fi
done
echo "TASK ${TASK_ID} finished: dataset=${DATASET} seeds=42,43,44 failures=${FAILURES} dry_run=${DRY_RUN}"
[ "${FAILURES}" -eq 0 ] || exit 1
