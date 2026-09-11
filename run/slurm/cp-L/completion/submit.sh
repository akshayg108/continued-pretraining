#!/bin/bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash run/slurm/cp-L/completion/submit.sh [--datasets KEY ...] [--concurrency 1..12] [--dry-run]' \
        'Default: all eight missing datasets, LeJEPA/SimCLR/DIET-CP, MAX, seeds 42/43/44, no FT.' \
        'Datasets: breastmnist octmnist organamnist pathmnist plant_village food101 flowers102 oxford_pet' \
        'Each task runs three seeds sequentially on one A100 80GB. The 96h limit is per task.' \
        '--dry-run creates a manifest and previews commands without submitting or training.'
}

DRY_RUN=0
CONCURRENCY="${VITL_COMPLETION_CONCURRENCY:-12}"
DATASETS=()
SELECTED=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --concurrency)
            [ "$#" -ge 2 ] || { echo '--concurrency requires a value' >&2; exit 2; }
            CONCURRENCY="$2"; shift 2 ;;
        --datasets)
            [ "${SELECTED}" -eq 0 ] || { echo '--datasets may be specified only once' >&2; exit 2; }
            SELECTED=1; shift
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                DATASETS+=("$1"); shift
            done
            [ "${#DATASETS[@]}" -gt 0 ] || { echo '--datasets requires at least one key' >&2; exit 2; }
            ;;
        *) echo "unsupported argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && [ "${CONCURRENCY}" -ge 1 ] && [ "${CONCURRENCY}" -le 12 ] || {
    echo 'concurrency must be 1..12' >&2; exit 2
}

REPO_ROOT="${VITL_COMPLETION_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
[ -d "${REPO_ROOT}/eval/vitl_completion" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
PY="${PYTHON:-python3}"
OUTPUT_BASE="${VITL_COMPLETION_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MANIFEST="${VITL_COMPLETION_MANIFEST:-${OUTPUT_BASE}/vitl_completion_manifests/selection-${STAMP}.json}"
LOG_DIR="${VITL_COMPLETION_LOG_DIR:-${OUTPUT_BASE}/slurm-log}"
CACHE_DIR="${VITL_COMPLETION_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
mkdir -p "$(dirname "${MANIFEST}")" "${LOG_DIR}"
MANIFEST="$(cd "$(dirname "${MANIFEST}")" && pwd -P)/$(basename "${MANIFEST}")"
LOG_DIR="$(cd "${LOG_DIR}" && pwd -P)"

BUILD=("${PY}" -m eval.vitl_completion.protocol --output-base "${OUTPUT_BASE}" --output "${MANIFEST}")
if [ "${SELECTED}" -eq 1 ]; then BUILD+=(--datasets "${DATASETS[@]}"); fi
printf 'MANIFEST'; printf ' %q' "${BUILD[@]}"; printf '\n'
"${BUILD[@]}"
COUNT=$("${PY}" - "${MANIFEST}" <<'PY'
import sys
from eval.vitl_completion.protocol import load_manifest
print(len(load_manifest(sys.argv[1])["tasks"]))
PY
)

SBATCH=(sbatch --parsable --chdir="${REPO_ROOT}" --array="0-$((COUNT - 1))%${CONCURRENCY}"
    --partition=nvidia --account=civil --gres=gpu:a100:1 --constraint=80g
    --exclude=cn253,cn259 --cpus-per-task=8 --mem=96G --time=96:00:00
    --output="${LOG_DIR}/vitl-completion-%A_%a.out"
    --error="${LOG_DIR}/vitl-completion-%A_%a.err"
    --export="ALL,VITL_COMPLETION_MANIFEST=${MANIFEST},VITL_COMPLETION_REPO_ROOT=${REPO_ROOT},VITL_COMPLETION_CACHE_DIR=${CACHE_DIR}"
    "${REPO_ROOT}/run/slurm/cp-L/completion/array.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'

if [ "${DRY_RUN}" -eq 1 ]; then
    echo 'PREVIEW first selected task, all three seeds (task list above covers the entire selection).'
    VITL_COMPLETION_MANIFEST="${MANIFEST}" VITL_COMPLETION_REPO_ROOT="${REPO_ROOT}" \
    VITL_COMPLETION_CACHE_DIR="${CACHE_DIR}" SLURM_ARRAY_TASK_ID=0 \
        bash "${REPO_ROOT}/run/slurm/cp-L/completion/array.sh" --dry-run
else
    "${SBATCH[@]}"
fi
