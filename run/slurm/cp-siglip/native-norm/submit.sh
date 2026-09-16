#!/bin/bash
set -euo pipefail

REPO_ROOT="${SIGLIP_NATIVE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
[ -f "${REPO_ROOT}/eval/siglip_native_cp.py" ] || { echo "invalid repo root: ${REPO_ROOT}" >&2; exit 2; }
DRY_RUN=0
CONCURRENCY="${SIGLIP_NATIVE_CONCURRENCY:-12}"
PLAN_ARGS=(plan)
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --concurrency)
            [ "$#" -ge 2 ] || { echo "--concurrency requires a value" >&2; exit 2; }
            CONCURRENCY="$2"; shift 2 ;;
        --datasets|--methods)
            OPTION="$1"
            PLAN_ARGS+=("$1")
            shift
            COUNT=0
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                PLAN_ARGS+=("$1")
                COUNT=$((COUNT + 1))
                shift
            done
            [ "${COUNT}" -gt 0 ] || { echo "${OPTION} requires at least one value" >&2; exit 2; }
            ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash submit.sh [--dry-run] [--concurrency 1..12]' \
                '                      [--datasets DATASET ...] [--methods DIET LeJEPA SimCLR]' \
                'Default: all 15 MAX datasets, three methods, seeds 42/43/44, 63 jobs.' \
                'Large targets: one seed per job. Others: three sequential seeds per job.' \
                'GPU routing: 2 blocks -> V100; 4/6 blocks -> A100;' \
                '             full DIET/SimCLR -> A100; full LeJEPA -> A100 80GB.' \
                '--concurrency is a total cap shared by the resource arrays.' \
                'Native SigLIP mean/std, fresh public weights, post-CP evaluation only, no FT.'
            exit 0 ;;
        *) echo "unsupported argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && [ "${CONCURRENCY}" -ge 1 ] && [ "${CONCURRENCY}" -le 12 ] || {
    echo "concurrency must be 1..12" >&2; exit 2;
}

PY="${PYTHON:-python3}"
OUTPUT_BASE="${SIGLIP_NATIVE_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MANIFEST="${SIGLIP_NATIVE_MANIFEST:-${OUTPUT_BASE}/siglip_native_cp_manifests/selection-${STAMP}.json}"
LOG_DIR="${SIGLIP_NATIVE_LOG_DIR:-${OUTPUT_BASE}/slurm-log}"
CACHE_DIR="${SIGLIP_NATIVE_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
mkdir -p "$(dirname "${MANIFEST}")" "${LOG_DIR}"
MANIFEST="$(cd "$(dirname "${MANIFEST}")" && pwd -P)/$(basename "${MANIFEST}")"
LOG_DIR="$(cd "${LOG_DIR}" && pwd -P)"
for VALUE in "${MANIFEST}" "${REPO_ROOT}" "${CACHE_DIR}"; do
    [[ "${VALUE}" != *,* ]] || { echo "paths cannot contain commas (Slurm --export)" >&2; exit 2; }
done
export PYTHONPATH="${REPO_ROOT}:$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"

"${PY}" -m eval.siglip_native_cp "${PLAN_ARGS[@]}" --output-base "${OUTPUT_BASE}" --manifest "${MANIFEST}"
GROUP_ROWS=$("${PY}" - "${MANIFEST}" "${CONCURRENCY}" <<'PY'
import sys
from eval.siglip_native_cp import GPU_RESOURCES, load_manifest, submission_groups
for group in submission_groups(load_manifest(sys.argv[1]), int(sys.argv[2])):
    profile = group["gpu_profile"]
    resources = GPU_RESOURCES[profile]
    print("\t".join([profile, resources["gres"], resources["constraint"] or "-",
                     ",".join(map(str, group["task_ids"])), str(group["concurrency"]),
                     str(int(group["serial"]))]))
PY
)
if [ -z "${GROUP_ROWS}" ]; then
    echo "Nothing to submit: all selected native CP seeds have verified results."
    exit 0
fi

PREVIOUS_JOB=""
while IFS=$'\t' read -r PROFILE GRES CONSTRAINT TASK_IDS LIMIT SERIAL; do
    SBATCH=(sbatch --parsable --chdir="${REPO_ROOT}" --array="${TASK_IDS}%${LIMIT}"
        --job-name="siglip-native-${PROFILE}"
        --partition=nvidia --account=civil --nodes=1 --ntasks-per-node=1
        --gres="${GRES}" --cpus-per-task=8 --mem=96G --time=96:00:00
        --output="${LOG_DIR}/siglip-native-%A_%a.out"
        --error="${LOG_DIR}/siglip-native-%A_%a.err"
        --export="ALL,SIGLIP_NATIVE_MANIFEST=${MANIFEST},SIGLIP_NATIVE_REPO_ROOT=${REPO_ROOT},SIGLIP_NATIVE_CACHE_DIR=${CACHE_DIR}")
    if [ "${CONSTRAINT}" != - ]; then
        SBATCH+=(--constraint="${CONSTRAINT}")
    fi
    if [ "${SERIAL}" -eq 1 ] && [ -n "${PREVIOUS_JOB}" ]; then
        SBATCH+=(--dependency="afterany:${PREVIOUS_JOB}")
    fi
    SBATCH+=("${REPO_ROOT}/run/slurm/cp-siglip/native-norm/array.sh")
    printf 'GPU_GROUP profile=%s concurrency=%s tasks=%s\n' "${PROFILE}" "${LIMIT}" "${TASK_IDS}"
    printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
    if [ "${DRY_RUN}" -eq 1 ]; then
        IFS=',' read -r -a SELECTED <<< "${TASK_IDS}"
        for task_id in "${SELECTED[@]}"; do
            SIGLIP_NATIVE_MANIFEST="${MANIFEST}" \
            SIGLIP_NATIVE_REPO_ROOT="${REPO_ROOT}" \
            SIGLIP_NATIVE_CACHE_DIR="${CACHE_DIR}" \
            SLURM_ARRAY_TASK_ID="${task_id}" \
                bash "${REPO_ROOT}/run/slurm/cp-siglip/native-norm/array.sh" --dry-run
        done
        PREVIOUS_JOB="DRY_RUN_${PROFILE}"
    else
        SUBMITTED=$("${SBATCH[@]}")
        printf 'SUBMITTED profile=%s job=%s\n' "${PROFILE}" "${SUBMITTED}"
        PREVIOUS_JOB="${SUBMITTED%%;*}"
        [[ "${PREVIOUS_JOB}" =~ ^[0-9]+$ ]] || { echo "unrecognized sbatch response: ${SUBMITTED}" >&2; exit 3; }
    fi
done <<< "${GROUP_ROWS}"
