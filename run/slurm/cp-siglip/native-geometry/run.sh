#!/bin/bash
set -eo pipefail

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: submit through submit.sh, or run.sh --dry-run" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }
if [ "$DRY_RUN" -eq 0 ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "Submit through submit.sh" >&2; exit 2; }
    if [ "${SIGLIP_GEOMETRY_SKIP_ENV_SETUP:-0}" != 1 ]; then
        module load miniconda/3-4.11.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate env
    fi
fi
set -u
REPO_ROOT="${SIGLIP_GEOMETRY_REPO_ROOT:?Missing repository path}"
OUTPUT_BASE="${SIGLIP_GEOMETRY_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${SIGLIP_GEOMETRY_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
IMAGENET_DIR="${SIGLIP_GEOMETRY_IMAGENET_DIR:-$CACHE_DIR/imagenet_val}"
OUT="$OUTPUT_BASE/siglip_native_geometry_v1/${SLURM_JOB_ID:-dry-run}"
PY="${PYTHON:-python3}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD:$(dirname "$PWD"):${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PY" -m eval.siglip_native_geometry plan
[ "$DRY_RUN" -eq 0 ] || exit 0
printf 'OUTPUT_DIRECTORY=%s\n' "$OUT"
"$PY" -m eval.siglip_native_geometry run --cache-dir "$CACHE_DIR" \
    --imagenet-dir "$IMAGENET_DIR" --outdir "$OUT" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}"
tar -czf "$OUT/figure_geometry.tar.gz" -C "$OUT" geometry.csv geometry_per_seed.csv imagenet.json results features
printf 'RESULTS=%s\nARCHIVE=%s\n' "$OUT/geometry.csv" "$OUT/figure_geometry.tar.gz"
