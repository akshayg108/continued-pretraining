#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash submit.sh [--concurrency 1..12] [--dry-run]" >&2
    exit 2
}

CONCURRENCY=12
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --concurrency)
            [ "$#" -ge 2 ] || usage
            CONCURRENCY=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *) usage ;;
    esac
done
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || usage
[ "$CONCURRENCY" -ge 1 ] && [ "$CONCURRENCY" -le 12 ] || usage

REPO_ROOT="${HELDOUT_REPO_ROOT:-/scratch/gs4133/zhd/CP/continued-pretraining}"
CACHE_DIR="${HELDOUT_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
OUTPUT_BASE="${HELDOUT_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
LOG_DIR="${HELDOUT_LOG_DIR:-$OUTPUT_BASE/slurm-log/heldout-cp}"
PY="${HELDOUT_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
case "$PY" in /*) ;; *) echo "HELDOUT_PYTHON must be an absolute executable path" >&2; exit 2 ;; esac
[ -x "$PY" ] || { echo "HELDOUT_PYTHON must be an absolute executable path: $PY" >&2; exit 2; }
[ -d "$REPO_ROOT" ] || { echo "Missing repository: $REPO_ROOT" >&2; exit 2; }
if [ "$DRY_RUN" -eq 0 ]; then
    command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 2; }
    command -v scontrol >/dev/null || { echo "Missing scontrol" >&2; exit 2; }
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST_DIR="$OUTPUT_BASE/heldout_cp_manifests"
mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST="$MANIFEST_DIR/heldout-cp-${STAMP}-$$.json"
[ ! -e "$MANIFEST" ] || { echo "Refusing to replace manifest: $MANIFEST" >&2; exit 2; }

"$PY" -m eval.heldout_cp plan --output-base "$OUTPUT_BASE" --manifest "$MANIFEST"
[ -f "$MANIFEST" ] || { echo "Planner did not create manifest: $MANIFEST" >&2; exit 1; }
chmod a-w "$MANIFEST"

COMMON=(--partition=nvidia --qos=nvidia --account=civil --nodes=1
    --ntasks-per-node=1 --gres=gpu:v100:1 --cpus-per-task=8 --mem=96G
    --time=96:00:00 --chdir="$REPO_ROOT")
EXPORTS="ALL,HELDOUT_PYTHON=$PY,HELDOUT_REPO_ROOT=$REPO_ROOT,HELDOUT_CACHE_DIR=$CACHE_DIR,HELDOUT_OUTPUT_BASE=$OUTPUT_BASE,HELDOUT_LOG_DIR=$LOG_DIR,HELDOUT_MANIFEST=$MANIFEST"
PREP=(sbatch --parsable --job-name=heldout-prep --array="0-7%$CONCURRENCY"
    "${COMMON[@]}" --output="$LOG_DIR/heldout-prep-%A_%a.out"
    --error="$LOG_DIR/heldout-prep-%A_%a.err" --export="$EXPORTS"
    "$REPO_ROOT/run/slurm/heldout-cp/prepare.sh")

printf 'SUBMIT'; printf ' %q' "${PREP[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 1 ]; then
    PREP_JOB=DRY_RUN_PREP
else
    PREP_RESULT=$("${PREP[@]}")
    [[ "$PREP_RESULT" =~ ^[0-9]+(\;[^[:space:];]+)?$ ]] || {
        echo "Invalid preparation job id: $PREP_RESULT" >&2
        exit 1
    }
    PREP_JOB=${PREP_RESULT%%;*}
    printf 'Submitted preparation job %s\n' "$PREP_JOB"
fi

CP=(sbatch --parsable --job-name=heldout-cp --array="0-47%$CONCURRENCY"
    --hold "${COMMON[@]}"
    --output="$LOG_DIR/heldout-cp-%A_%a.out" --error="$LOG_DIR/heldout-cp-%A_%a.err"
    --export="$EXPORTS" "$REPO_ROOT/run/slurm/heldout-cp/array.sh")
printf 'SUBMIT'; printf ' %q' "${CP[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 0 ]; then
    CP_RESULT=$("${CP[@]}")
    [[ "$CP_RESULT" =~ ^[0-9]+(\;[^[:space:];]+)?$ ]] || {
        echo "Invalid CP job id: $CP_RESULT; CP array was submitted held" >&2
        exit 1
    }
    CP_JOB=${CP_RESULT%%;*}
    printf 'Submitted held CP job %s\n' "$CP_JOB"
else
    CP_JOB=DRY_RUN_CP
fi

# Manifest order: encoder (2), dataset (8), objective (3).
# Keep one CP array so ready datasets share the same throttle.
for ((TASK_ID=0; TASK_ID<48; TASK_ID++)); do
    DATASET_ID=$(( (TASK_ID % 24) / 3 ))
    UPDATE=(scontrol update "JobId=${CP_JOB}_${TASK_ID}"
        "Dependency=afterok:${PREP_JOB}_${DATASET_ID}")
    printf 'CONTROL'; printf ' %q' "${UPDATE[@]}"; printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        if ! "${UPDATE[@]}"; then
            echo "Dependency update failed for ${CP_JOB}_${TASK_ID}; CP array $CP_JOB remains held. Do not release it until all dependencies are configured." >&2
            exit 1
        fi
    fi
done
RELEASE=(scontrol release "$CP_JOB")
printf 'CONTROL'; printf ' %q' "${RELEASE[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 0 ]; then
    "${RELEASE[@]}"
    printf 'Submitted preparation=%s cp=%s manifest=%s; dataset-local dependencies configured\n' "$PREP_JOB" "$CP_JOB" "$MANIFEST"
else
    printf 'Dry run only; manifest=%s\n' "$MANIFEST"
fi
