#!/bin/bash
#SBATCH --job-name=siglip-mainrule
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-mainrule-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-mainrule-%A_%a.err

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "too many arguments" >&2; exit 2; }

if [ "${DRY_RUN}" -eq 0 ] && [ "${SIGLIP_MAINRULE_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi

set -euo pipefail

if [ -n "${SIGLIP_MAINRULE_REPO_ROOT:-}" ]; then
    REPO_ROOT="${SIGLIP_MAINRULE_REPO_ROOT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gs4133/zhd/CP/continued-pretraining}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
[ -d "${REPO_ROOT}/eval/siglip_mainrule" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }

PY="${PYTHON:-python3}"
MANIFEST="${SIGLIP_MAINRULE_MANIFEST:?SIGLIP_MAINRULE_MANIFEST is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CACHE_DIR="${SIGLIP_MAINRULE_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

SUBPATH=$("${PY}" - "${MANIFEST}" "${TASK_ID}" <<'PY'
import sys
from eval.siglip_mainrule.protocol import load_manifest

tasks = load_manifest(sys.argv[1])["tasks"]
task_id = int(sys.argv[2])
if not 0 <= task_id < len(tasks):
    raise ValueError(f"task id {task_id} out of range")
print(tasks[task_id]["processed_subpath"])
PY
)
SOURCE="${CACHE_DIR}/stable_datasets/processed/${SUBPATH}"
CMD=("${PY}" -m eval.siglip_mainrule.run --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${CACHE_DIR}" --num-workers 8)

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "STAGE source=${SOURCE} private=siglip-mainrule-${SLURM_JOB_ID:-local}-${TASK_ID}"
    printf 'RUN'; printf ' %q' "${CMD[@]}"; printf ' --dry-run\n'
    "${CMD[@]}" --dry-run
    exit 0
fi

[ -d "${SOURCE}" ] || { echo "processed cache missing: ${SOURCE}" >&2; exit 4; }
"${PY}" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'
nvidia-smi

LOCAL_CACHE=""
NEED_KB=$(( $(du -sk "${SOURCE}" | awk '{print $1}') + 5 * 1024 * 1024 ))
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    AVAIL_KB=$(df -Pk "${root}" | awk 'NR==2 {print $4}')
    if [ "${AVAIL_KB:-0}" -ge "${NEED_KB}" ]; then
        LOCAL_CACHE=$(mktemp -d "${root}/siglip-mainrule-${SLURM_JOB_ID:-local}-${TASK_ID}.XXXXXX")
        trap 'rm -rf -- "${LOCAL_CACHE}"' EXIT
        DEST="${LOCAL_CACHE}/stable_datasets/processed/${SUBPATH}"
        mkdir -p "$(dirname "${DEST}")"
        rsync -a "${SOURCE}/" "${DEST}/"
        break
    fi
done
[ -n "${LOCAL_CACHE}" ] || { echo "insufficient private storage for ${SOURCE}" >&2; exit 5; }

CMD=("${PY}" -m eval.siglip_mainrule.run --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${LOCAL_CACHE}" --num-workers 8)
"${CMD[@]}"
