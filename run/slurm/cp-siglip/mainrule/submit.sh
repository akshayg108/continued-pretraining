#!/bin/bash
set -euo pipefail

if [ -n "${SIGLIP_MAINRULE_REPO_ROOT:-}" ]; then
    REPO_ROOT="${SIGLIP_MAINRULE_REPO_ROOT}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
[ -d "${REPO_ROOT}/eval/siglip_mainrule" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }

DRY_RUN=0
CONCURRENCY="${SIGLIP_MAINRULE_CONCURRENCY:-12}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --concurrency)
            [ "$#" -ge 2 ] || { echo "--concurrency requires a value" >&2; exit 2; }
            CONCURRENCY="$2"; shift 2 ;;
        *) echo "unsupported argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && [ "${CONCURRENCY}" -ge 1 ] && [ "${CONCURRENCY}" -le 12 ] || {
    echo "concurrency must be 1..12" >&2
    exit 2
}

PY="${PYTHON:-python3}"
OUTPUT_BASE="${SIGLIP_MAINRULE_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MANIFEST="${SIGLIP_MAINRULE_MANIFEST:-${OUTPUT_BASE}/siglip_mainrule_manifests/selection-${STAMP}.json}"
LOG_DIR="${SIGLIP_MAINRULE_LOG_DIR:-${OUTPUT_BASE}/slurm-log}"
CACHE_DIR="${SIGLIP_MAINRULE_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
mkdir -p "$(dirname "${MANIFEST}")" "${LOG_DIR}"
MANIFEST="$(cd "$(dirname "${MANIFEST}")" && pwd -P)/$(basename "${MANIFEST}")"
LOG_DIR="$(cd "${LOG_DIR}" && pwd -P)"

BUILD=("${PY}" -m eval.siglip_mainrule.protocol --output-base "${OUTPUT_BASE}" --output "${MANIFEST}")
printf 'MANIFEST'; printf ' %q' "${BUILD[@]}"; printf '\n'
(
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
    "${BUILD[@]}"
)

SBATCH=(sbatch --parsable --chdir="${REPO_ROOT}" --array="0-28%${CONCURRENCY}"
    --partition=nvidia --account=civil --gres=gpu:a100:1 --cpus-per-task=8
    --mem=96G --time=96:00:00
    --output="${LOG_DIR}/siglip-mainrule-%A_%a.out"
    --error="${LOG_DIR}/siglip-mainrule-%A_%a.err"
    --export="ALL,SIGLIP_MAINRULE_MANIFEST=${MANIFEST},SIGLIP_MAINRULE_REPO_ROOT=${REPO_ROOT},SIGLIP_MAINRULE_OUTPUT_BASE=${OUTPUT_BASE},SIGLIP_MAINRULE_CACHE_DIR=${CACHE_DIR},SIGLIP_MAINRULE_LOG_DIR=${LOG_DIR}"
    "${REPO_ROOT}/run/slurm/cp-siglip/mainrule/array.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'

if [ "${DRY_RUN}" -eq 1 ]; then
    SIGLIP_MAINRULE_MANIFEST="${MANIFEST}" \
    SIGLIP_MAINRULE_REPO_ROOT="${REPO_ROOT}" \
    SIGLIP_MAINRULE_CACHE_DIR="${CACHE_DIR}" \
    SLURM_ARRAY_TASK_ID=0 \
        bash "${REPO_ROOT}/run/slurm/cp-siglip/mainrule/array.sh" --dry-run
else
    "${SBATCH[@]}"
fi
