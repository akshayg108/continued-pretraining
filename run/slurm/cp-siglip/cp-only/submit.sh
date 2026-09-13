#!/bin/bash
set -euo pipefail

REPO_ROOT="${SIGLIP_CP_ONLY_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
[ -f "${REPO_ROOT}/eval/siglip_mainrule/cp_only.py" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
DRY_RUN=0
CONCURRENCY="${SIGLIP_CP_ONLY_CONCURRENCY:-12}"
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
    echo "concurrency must be 1..12" >&2; exit 2;
}

PY="${PYTHON:-python3}"
OUTPUT_BASE="${SIGLIP_CP_ONLY_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MANIFEST="${SIGLIP_CP_ONLY_MANIFEST:-${OUTPUT_BASE}/siglip_cp_only_manifests/selection-${STAMP}.json}"
LOG_DIR="${SIGLIP_CP_ONLY_LOG_DIR:-${OUTPUT_BASE}/slurm-log}"
CACHE_DIR="${SIGLIP_CP_ONLY_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
mkdir -p "$(dirname "${MANIFEST}")" "${LOG_DIR}"
MANIFEST="$(cd "$(dirname "${MANIFEST}")" && pwd -P)/$(basename "${MANIFEST}")"
LOG_DIR="$(cd "${LOG_DIR}" && pwd -P)"
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"

"${PY}" -m eval.siglip_mainrule.cp_only plan --output-base "${OUTPUT_BASE}" --manifest "${MANIFEST}"
TASK_IDS=$("${PY}" - "${MANIFEST}" <<'PY'
import sys
from eval.siglip_mainrule.cp_only import load_manifest
print(",".join(map(str, load_manifest(sys.argv[1])["selected_task_ids"])))
PY
)
if [ -z "${TASK_IDS}" ]; then
    echo "Nothing to submit: all nine CP seeds have verified results."
    exit 0
fi

SBATCH=(sbatch --parsable --chdir="${REPO_ROOT}" --array="${TASK_IDS}%${CONCURRENCY}"
    --partition=nvidia --account=civil --gres=gpu:a100:1 --constraint=80g
    --cpus-per-task=8 --mem=96G --time=96:00:00
    --output="${LOG_DIR}/siglip-cp-only-%A_%a.out"
    --error="${LOG_DIR}/siglip-cp-only-%A_%a.err"
    --export="ALL,SIGLIP_CP_ONLY_MANIFEST=${MANIFEST},SIGLIP_CP_ONLY_REPO_ROOT=${REPO_ROOT},SIGLIP_CP_ONLY_CACHE_DIR=${CACHE_DIR}"
    "${REPO_ROOT}/run/slurm/cp-siglip/cp-only/array.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
if [ "${DRY_RUN}" -eq 1 ]; then
    IFS=',' read -r -a SELECTED <<< "${TASK_IDS}"
    for task_id in "${SELECTED[@]}"; do
        SIGLIP_CP_ONLY_MANIFEST="${MANIFEST}" \
        SIGLIP_CP_ONLY_REPO_ROOT="${REPO_ROOT}" \
        SIGLIP_CP_ONLY_CACHE_DIR="${CACHE_DIR}" \
        SLURM_ARRAY_TASK_ID="${task_id}" \
            bash "${REPO_ROOT}/run/slurm/cp-siglip/cp-only/array.sh" --dry-run
    done
else
    "${SBATCH[@]}"
fi
