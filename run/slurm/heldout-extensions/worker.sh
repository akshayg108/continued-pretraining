#!/bin/bash
set -euo pipefail

STAGE="${1:-}"
case "$STAGE" in
    prepare) ID_FLAG=--preparation-id ;;
    run) ID_FLAG=--task-id ;;
    *) echo "Usage: worker.sh {prepare|run} [--dry-run]" >&2; exit 2 ;;
esac
shift
[ "$#" -le 1 ] && { [ "$#" -eq 0 ] || [ "$1" = --dry-run ]; } || {
    echo "Usage: worker.sh {prepare|run} [--dry-run]" >&2; exit 2;
}
PY="${HELDOUT_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
case "$PY" in /*) ;; *) echo "HELDOUT_PYTHON must be an absolute executable path" >&2; exit 2 ;; esac
[ -x "$PY" ] || { echo "HELDOUT_PYTHON must be an absolute executable path: $PY" >&2; exit 2; }
REPO_ROOT="${HELDOUT_REPO_ROOT:-/scratch/gs4133/zhd/CP/continued-pretraining}"
MANIFEST="${HELDOUT_EXT_MANIFEST:?Missing HELDOUT_EXT_MANIFEST}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"
CACHE_DIR="${HELDOUT_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
printf 'python=%s stage=%s task_id=%s manifest=%s\n' "$PY" "$STAGE" "$TASK_ID" "$MANIFEST"
exec "$PY" -u -m eval.heldout_extensions "$STAGE" --manifest "$MANIFEST" \
    "$ID_FLAG" "$TASK_ID" --cache-dir "$CACHE_DIR" --num-workers "$WORKERS" "$@"
