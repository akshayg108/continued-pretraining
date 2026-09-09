#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-python3}"
DRY_RUN=0
MANIFEST_DIR="${FULL_FT_MANIFEST_DIR:-${REPO_ROOT}/eval/outputs/full_ft_manifests}"
METHODS=(LeJEPA SimCLR DIET)
BUDGETS=(500 MAX)
ENCODERS=(DINOv3 CLIP MAE SigLIP)
PHASES=(post)
CHECKPOINT_ROOT=""
ALL_GRID=0
METHODS_SET=0
BUDGETS_SET=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --methods) METHODS_SET=1; METHODS=(); shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
        --budgets) BUDGETS_SET=1; BUDGETS=(); shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do BUDGETS+=("$1"); shift; done ;;
        --encoders) ENCODERS=(); shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do ENCODERS+=("$1"); shift; done ;;
        --phases) PHASES=(); shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do PHASES+=("$1"); shift; done ;;
        --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
        --all) ALL_GRID=1; shift ;;
        --output|--repo-root) echo "$1 is managed by submit.sh" >&2; exit 2 ;;
        --) shift ;;
        *) echo "unsupported argument: $1" >&2; exit 2 ;;
    esac
done

CONCURRENCY="${FULL_FT_CONCURRENCY:-12}"
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && [ "${CONCURRENCY}" -ge 1 ] && [ "${CONCURRENCY}" -le 12 ] || { echo "FULL_FT_CONCURRENCY must be 1..12" >&2; exit 2; }
TIME="${FULL_FT_TIME:-96:00:00}"
if [[ "${TIME}" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
    SECONDS_REQUESTED=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
else
    echo "FULL_FT_TIME must use HH:MM:SS" >&2
    exit 2
fi
[ "${SECONDS_REQUESTED}" -gt 0 ] && [ "${SECONDS_REQUESTED}" -le 345600 ] || {
    echo "FULL_FT_TIME must be positive and at most 96:00:00" >&2
    exit 2
}
LIMIT="${FULL_FT_ARRAY_LIMIT:-1000}"
[[ "${LIMIT}" =~ ^[0-9]+$ ]] && [ "${LIMIT}" -ge 1 ] || { echo "FULL_FT_ARRAY_LIMIT must be positive" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${MANIFEST_DIR}"
MANIFEST_DIR="$(cd "${MANIFEST_DIR}" && pwd -P)"
MANIFEST="${MANIFEST_DIR}/selection-${STAMP}.json"
if [ "${DRY_RUN}" -eq 1 ]; then
    LOG_DIR="${FULL_FT_LOG_DIR:-${MANIFEST_DIR}/slurm-log}"
else
    LOG_DIR="${FULL_FT_LOG_DIR:-/scratch/gs4133/zhd/CP/outputs/slurm-log}"
fi
mkdir -p "${LOG_DIR}"
LOG_DIR="$(cd "${LOG_DIR}" && pwd -P)"
BUILD=("${PY}" "${REPO_ROOT}/eval/full_ft/manifest.py" --repo-root "${REPO_ROOT}")
if [ "${ALL_GRID}" -eq 1 ]; then
    [ "${METHODS_SET}" -eq 1 ] || METHODS=(LeJEPA SimCLR DIET MAE)
    [ "${BUDGETS_SET}" -eq 1 ] || BUDGETS=(100 500 1000 10000 25000 MAX)
fi
[ "${#METHODS[@]}" -gt 0 ] || { echo "empty --methods selection" >&2; exit 2; }
[ "${#BUDGETS[@]}" -gt 0 ] || { echo "empty --budgets selection" >&2; exit 2; }
[ "${#ENCODERS[@]}" -gt 0 ] || { echo "empty --encoders selection" >&2; exit 2; }
[ "${#PHASES[@]}" -gt 0 ] || { echo "empty --phases selection" >&2; exit 2; }
BUILD+=(--methods "${METHODS[@]}" --budgets "${BUDGETS[@]}" --encoders "${ENCODERS[@]}" --phases "${PHASES[@]}")
[ -z "${CHECKPOINT_ROOT}" ] || BUILD+=(--checkpoint-root "${CHECKPOINT_ROOT}")
BUILD+=(--output "${MANIFEST}")
printf 'MANIFEST'; printf ' %q' "${BUILD[@]}"; printf '\n'; "${BUILD[@]}"

CHUNKS=$("${PY}" - "${MANIFEST}" "${LIMIT}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); limit=int(sys.argv[2]); doc=json.loads(p.read_text()); tasks=doc["tasks"]
if not tasks: raise SystemExit("empty task selection")
for n,start in enumerate(range(0,len(tasks),limit)):
    chunk=tasks[start:start+limit]
    for i,t in enumerate(chunk): t["task_id"]=i
    out=p.with_name(f"{p.stem}.chunk{n:03d}.json")
    out.write_text(json.dumps({**doc,"tasks":chunk},indent=2)+"\n")
    print(f"{out}\t{len(chunk)}")
PY
)

PREVIOUS=""
while IFS=$'\t' read -r CHUNK COUNT; do
    SBATCH=(sbatch --parsable --chdir="${REPO_ROOT}" --array="0-$((COUNT-1))%${CONCURRENCY}" --partition="${FULL_FT_PARTITION:-nvidia}" --account="${FULL_FT_ACCOUNT:-civil}" --gres="${FULL_FT_GRES:-gpu:a100:1}" --cpus-per-task="${FULL_FT_CPUS:-8}" --mem="${FULL_FT_MEM:-96G}" --time="${TIME}" --output="${LOG_DIR}/full-ft-%A_%a.out" --error="${LOG_DIR}/full-ft-%A_%a.err" --export="ALL,FULL_FT_MANIFEST=${CHUNK},FULL_FT_REPO_ROOT=${REPO_ROOT}" )
    [ -z "${PREVIOUS}" ] || SBATCH+=(--dependency="afterany:${PREVIOUS}")
    SBATCH+=("${REPO_ROOT}/run/slurm/full_ft/array.sh")
    printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
    if [ "${DRY_RUN}" -eq 0 ]; then PREVIOUS=$("${SBATCH[@]}"); fi
done <<< "${CHUNKS}"
