#!/bin/bash
set -euo pipefail

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: bash array.sh [--dry-run]" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Usage: bash array.sh [--dry-run]" >&2; exit 2; }

PY="${HELDOUT_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
case "$PY" in /*) ;; *) echo "HELDOUT_PYTHON must be an absolute executable path" >&2; exit 2 ;; esac
[ -x "$PY" ] || { echo "HELDOUT_PYTHON must be an absolute executable path: $PY" >&2; exit 2; }
REPO_ROOT="${HELDOUT_REPO_ROOT:-/scratch/gs4133/zhd/CP/continued-pretraining}"
CACHE_DIR="${HELDOUT_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
MANIFEST="${HELDOUT_MANIFEST:?Missing HELDOUT_MANIFEST}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

export OMP_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
printf 'python=%s stage=cp task_id=%s manifest=%s\n' "$PY" "$TASK_ID" "$MANIFEST"
COMMAND=("$PY" -m eval.heldout_cp run --manifest "$MANIFEST"
    --task-id "$TASK_ID" --cache-dir "$CACHE_DIR" --num-workers "$WORKERS")
[ "$DRY_RUN" -eq 0 ] || COMMAND+=(--dry-run)
"${COMMAND[@]}"

