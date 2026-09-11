#!/bin/bash
#SBATCH --job-name=vitl-completion
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-completion-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/vitl-completion-%A_%a.err

set -eo pipefail
DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 2; }

if [ "${DRY_RUN}" -eq 0 ] && [ "${VITL_COMPLETION_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -euo pipefail

if [ -n "${VITL_COMPLETION_REPO_ROOT:-}" ]; then
    REPO_ROOT="${VITL_COMPLETION_REPO_ROOT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gs4133/zhd/CP/continued-pretraining}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
[ -d "${REPO_ROOT}/eval/vitl_completion" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
PY="${PYTHON:-python3}"
MANIFEST="${VITL_COMPLETION_MANIFEST:?VITL_COMPLETION_MANIFEST is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CACHE_DIR="${VITL_COMPLETION_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

SUBPATH=$("${PY}" - "${MANIFEST}" "${TASK_ID}" <<'PY'
import sys
from eval.vitl_completion.protocol import load_manifest
tasks = load_manifest(sys.argv[1])["tasks"]
index = int(sys.argv[2])
if not 0 <= index < len(tasks):
    raise ValueError(f"task id {index} out of range")
print(tasks[index]["processed_subpath"])
PY
)
SOURCE="${CACHE_DIR}/stable_datasets/processed/${SUBPATH}"
CMD=("${PY}" -m eval.vitl_completion.run --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${CACHE_DIR}" --num-workers 8)
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "STAGE source=${SOURCE} private=vitl-completion-${SLURM_JOB_ID:-local}-${TASK_ID}"
    printf 'RUN'; printf ' %q' "${CMD[@]}"; printf ' --dry-run\n'
    "${CMD[@]}" --dry-run
    exit 0
fi

[ -d "${SOURCE}" ] || { echo "processed cache missing: ${SOURCE}" >&2; exit 4; }
"${PY}" -c 'from eval.vitl_completion.run import require_a100_80gb; require_a100_80gb()'
nvidia-smi

LOCAL_CACHE=""
NEED_KB=$(( $(du -sk "${SOURCE}" | awk '{print $1}') + 5 * 1024 * 1024 ))
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    AVAIL_KB=$(df -Pk "${root}" | awk 'NR==2 {print $4}')
    if [ "${AVAIL_KB:-0}" -ge "${NEED_KB}" ]; then
        LOCAL_CACHE=$(mktemp -d "${root}/vitl-completion-${SLURM_JOB_ID:-local}-${TASK_ID}.XXXXXX")
        trap 'rm -rf -- "${LOCAL_CACHE}"' EXIT
        DEST="${LOCAL_CACHE}/stable_datasets/processed/${SUBPATH}"
        mkdir -p "$(dirname "${DEST}")"
        rsync -a "${SOURCE}/" "${DEST}/"
        break
    fi
done
[ -n "${LOCAL_CACHE}" ] || { echo "insufficient private storage for ${SOURCE}" >&2; exit 5; }
echo "STAGED ${SOURCE} -> ${LOCAL_CACHE}"
CMD=("${PY}" -m eval.vitl_completion.run --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${LOCAL_CACHE}" --num-workers 8)
"${CMD[@]}"
