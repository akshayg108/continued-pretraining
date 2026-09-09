#!/bin/bash
#SBATCH --job-name=full-ft
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/full-ft-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/full-ft-%A_%a.err

DRY_RUN=0
case "${1:-}" in "") ;; --dry-run) DRY_RUN=1 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac
[ "$#" -le 1 ] || { echo "too many arguments" >&2; exit 2; }
if [ "${DRY_RUN}" -eq 0 ] && [ "${FULL_FT_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -euo pipefail
if [ -n "${FULL_FT_REPO_ROOT:-}" ]; then
    REPO_ROOT="${FULL_FT_REPO_ROOT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gs4133/zhd/CP/continued-pretraining}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
[ -f "${REPO_ROOT}/continued_pretraining.py" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
PY="${PYTHON:-python3}"
MANIFEST="${FULL_FT_MANIFEST:?FULL_FT_MANIFEST is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
OUTDIR="${FULL_FT_OUTDIR:-/scratch/gs4133/zhd/CP/outputs/full_ft_v1}"
DATA_ROOT="${FULL_FT_DATA_ROOT:-/scratch/gs4133/zhd/CP/data}"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

SUBPATH=$("${PY}" - "${MANIFEST}" "${TASK_ID}" <<'PY'
import sys
from eval.full_ft.run import load_tasks
tasks = load_tasks(sys.argv[1])
i = int(sys.argv[2])
if not 0 <= i < len(tasks): raise ValueError(f"task id {i} out of range")
task = tasks[i]
print(task["processed_subpath"])
PY
)
SRC="${DATA_ROOT}/stable_datasets/processed/${SUBPATH}"
CMD=("${PY}" "${REPO_ROOT}/eval/full_ft/run.py" --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${DATA_ROOT}" --outdir "${OUTDIR}" --device cuda --seeds 42 43 44)
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "STAGE source=${SRC} private=full-ft-${SLURM_JOB_ID:-local}-${TASK_ID}"
    printf 'RUN'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi
[ -d "${SRC}" ] || { echo "processed cache missing: ${SRC}" >&2; exit 4; }
"${PY}" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'
nvidia-smi
DATA_DIR="${DATA_ROOT}"
LOCAL_CACHE=""
NEED_KB=$(( $(du -sk "${SRC}" | awk '{print $1}') + 5*1024*1024 ))
for root in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "${root}" ] && [ -d "${root}" ] && [ -w "${root}" ] || continue
    AVAIL_KB=$(df -Pk "${root}" | awk 'NR==2{print $4}')
    if [ "${AVAIL_KB:-0}" -ge "${NEED_KB}" ]; then
        LOCAL_CACHE=$(mktemp -d "${root}/full-ft-${SLURM_JOB_ID:-local}-${TASK_ID}.XXXXXX")
        trap 'rm -rf -- "${LOCAL_CACHE}"' EXIT
        DEST="${LOCAL_CACHE}/stable_datasets/processed/${SUBPATH}"
        mkdir -p "$(dirname "${DEST}")"
        rsync -a "${SRC}/" "${DEST}/"
        DATA_DIR="${LOCAL_CACHE}"
        break
    fi
done
[ "${DATA_DIR}" != "${DATA_ROOT}" ] || echo "WARN: insufficient local space; using shared cache ${SRC}"
CMD=("${PY}" "${REPO_ROOT}/eval/full_ft/run.py" --manifest "${MANIFEST}" --task-id "${TASK_ID}" --cache-dir "${DATA_DIR}" --outdir "${OUTDIR}" --device cuda --seeds 42 43 44)
"${CMD[@]}"
