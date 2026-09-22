#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00

set -euo pipefail

STAGE="${1:-}"
case "$STAGE" in prepare|run) ;; *) echo "Usage: roundoff.sh prepare|run [--dry-run]" >&2; exit 2 ;; esac
[ "$#" -le 2 ] || { echo "Too many arguments" >&2; exit 2; }
DRY_RUN=0
case "${2:-}" in "") ;; --dry-run) DRY_RUN=1 ;; *) echo "Unknown argument: $2" >&2; exit 2 ;; esac

PY="${HELDOUT_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
case "$PY" in /*) ;; *) echo "HELDOUT_PYTHON must be absolute" >&2; exit 2 ;; esac
[ -x "$PY" ] || { echo "Python is not executable: $PY" >&2; exit 2; }
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
printf 'python=%s stage=%s task_id=%s manifest=%s\n' "$PY" "$STAGE" "$TASK_ID" "$MANIFEST"
COMMON=(--manifest "$MANIFEST" --cache-dir "$CACHE_DIR" --num-workers "$WORKERS")
[ "$DRY_RUN" -eq 0 ] || COMMON+=(--dry-run)
if [ "$STAGE" = prepare ]; then
    "$PY" -m eval.heldout_metric_roundoff prepare "${COMMON[@]}" --dataset-id "$TASK_ID"
else
    for SEED in 42 43 44; do
        "$PY" -m eval.heldout_metric_roundoff fit "${COMMON[@]}" --task-id "$TASK_ID" --seed "$SEED"
    done
fi
