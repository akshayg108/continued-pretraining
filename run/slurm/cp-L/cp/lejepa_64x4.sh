#!/bin/bash
#SBATCH --job-name=vitl-lejepa-64x4
#SBATCH --array=0-5%6
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn253,cn259
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-lejepa-64x4-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-lejepa-64x4-%A_%a.err

# Isolated microbatch trial. Keep the original 128x2 results and protocol intact.
# Each task starts from public weights and runs one CP seed, with no FT.
set -eo pipefail

usage() {
    printf '%s\n' \
        'Preview: bash run/slurm/cp-L/cp/lejepa_64x4.sh --dry-run' \
        'Pilot:   sbatch --array=0,3 run/slurm/cp-L/cp/lejepa_64x4.sh' \
        'All:     sbatch run/slurm/cp-L/cp/lejepa_64x4.sh' \
        'Tasks 0/1/2: PlantVillage seeds 42/43/44.' \
        'Tasks 3/4/5: OrganAMNIST seeds 42/43/44.' \
        'LeJEPA only, MAX, six trainable blocks, batch 64 x accumulation 4.' \
        'Each submission writes to vitl_lejepa_64x4_v1/<array-job-id>/.'
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
    for task_id in 0 1 2 3 4 5; do
        SLURM_ARRAY_TASK_ID="${task_id}" bash "${BASH_SOURCE[0]}" --dry-run
    done
    exit 0
fi
if [ "${DRY_RUN}" -eq 0 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo 'Use sbatch to train, or --dry-run to preview on the login node.' >&2
    exit 2
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[0-5]$ ]] || { echo 'task id must be 0..5' >&2; exit 2; }

if [ "${DRY_RUN}" -eq 0 ] && [ "${VITL_LEJEPA_64X4_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -euo pipefail

if [ -n "${VITL_LEJEPA_64X4_REPO_ROOT:-}" ]; then
    REPO_ROOT="${VITL_LEJEPA_64X4_REPO_ROOT}"
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
    'plant_village 42 43596 plant_village'
    'plant_village 43 43596 plant_village'
    'plant_village 44 43596 plant_village'
    'organamnist 42 34561 med_mnist/organamnist-size=224'
    'organamnist 43 34561 med_mnist/organamnist-size=224'
    'organamnist 44 34561 med_mnist/organamnist-size=224'
)
read -r DATASET SEED N SUBPATH <<< "${TASKS[${TASK_ID}]}"
CACHE_DIR="${VITL_LEJEPA_64X4_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
OUTPUT_BASE="${VITL_LEJEPA_64X4_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-preview}}"
OUT="${OUTPUT_BASE}/vitl_lejepa_64x4_v1/${RUN_ID}"
SOURCE="${CACHE_DIR}/stable_datasets/processed/${SUBPATH}"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "STAGE source=${SOURCE} private=vitl-lejepa-64x4-${RUN_ID}-${TASK_ID}"
else
    [ -d "${SOURCE}" ] || { echo "processed cache missing: ${SOURCE}" >&2; exit 4; }
    "${PY}" -c 'from eval.vitl_completion.run import require_a100_80gb; require_a100_80gb()'
    nvidia-smi

    LOCAL_CACHE=""
    NEED_KB=$(( $(du -sk "${SOURCE}" | awk '{print $1}') + 5 * 1024 * 1024 ))
    for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
        [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
        AVAIL_KB=$(df -Pk "${root}" | awk 'NR==2 {print $4}')
        if [ "${AVAIL_KB:-0}" -ge "${NEED_KB}" ]; then
            LOCAL_CACHE=$(mktemp -d "${root}/vitl-lejepa-64x4-${RUN_ID}-${TASK_ID}.XXXXXX")
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

CMD=("${PY}" "${REPO_ROOT}/continued_pretraining.py"
    --cp-method lejepa --dataset "${DATASET}"
    --backbone vit_large_patch16_dinov3.lvd1689m --pool-strategy cls
    --n-samples "${N}" --seed "${SEED}" --cache-dir "${CACHE_DIR}" --num-workers 8
    --epochs 150 --freeze-epochs 15 --warmup-epochs 15 --num-trained-blocks 6
    --lr 0.0001 --weight-decay 0.05 --knn-k 20
    --batch-size 64 --accumulate-grad-batches 4
    --proj-dim 128 --hidden-dim 2048 --n-views 8 --lamb 0.02
    --project vitl_lejepa_64x4_v1
    --run-name "LeJEPA_${DATASET}_blk6_b64a4_s${SEED}_job${RUN_ID}"
    --checkpoint-dir "${OUT}/checkpoints/LeJEPA/${DATASET}/seed${SEED}"
    --results-json "${OUT}/cp_results/LeJEPA/${DATASET}/seed${SEED}.json")
printf 'CP_ONLY task=%s dataset=%s seed=%s blocks=6 batch=64 accumulate=4 initialization=public_pretrained\n' \
    "${TASK_ID}" "${DATASET}" "${SEED}"
printf 'COMMAND'; printf ' %q' "${CMD[@]}"; printf '\n'
[ "${DRY_RUN}" -eq 0 ] || exit 0

mkdir -p "${OUT}/cp_results/LeJEPA/${DATASET}" "${OUT}/commands/LeJEPA/${DATASET}"
"${PY}" - "${OUT}/commands/LeJEPA/${DATASET}/seed${SEED}.json" "${CMD[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

command = sys.argv[2:]
record = dict(protocol="vitl_lejepa_64x4_v1", initialization="public_pretrained",
              command=command,
              entrypoint_sha256=hashlib.sha256(Path(command[1]).read_bytes()).hexdigest())
with Path(sys.argv[1]).open("x") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
"${CMD[@]}"
