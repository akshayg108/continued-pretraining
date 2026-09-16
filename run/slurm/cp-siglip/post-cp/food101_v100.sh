#!/bin/bash
set -eo pipefail
DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }
if [ "$DRY_RUN" -eq 0 ] && [ "${SIGLIP_POST_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -u
REPO_ROOT="${SIGLIP_POST_REPO_ROOT:?Submit with submit_food101.sh}"
MANIFEST="${SIGLIP_POST_MANIFEST:?Missing audit manifest}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing task ID}"
OUTPUT_BASE="${SIGLIP_POST_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${SIGLIP_POST_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
PY="${PYTHON:-python3}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}"
OUT="$OUTPUT_BASE/siglip_food101_postcheck_official_norm_v1/$RUN_ID"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD:$(dirname "$PWD"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
CMD=("$PY" -m eval.siglip_food101_postcheck run --manifest "$MANIFEST" --task-id "$TASK_ID"
    --cache-dir "$CACHE_DIR" --outdir "$OUT" --num-workers 8)
if [ "$DRY_RUN" -eq 1 ]; then
    "${CMD[@]}" --dry-run
    exit 0
fi
[ -n "${SLURM_JOB_ID:-}" ] || { echo "Submit with submit_food101.sh" >&2; exit 2; }
"$PY" -c 'import torch; p = torch.cuda.get_device_properties(0); assert torch.cuda.device_count() == 1 and "V100" in p.name, p; print(p)'
SOURCE="$CACHE_DIR/stable_datasets/processed/food101"
[ -d "$SOURCE" ] || { echo "Missing processed Food-101 cache: $SOURCE" >&2; exit 4; }
SOURCE_KB=$(du -sk "$SOURCE" | awk '{print $1}')
NEED_KB=$((SOURCE_KB + 5 * 1024 * 1024))
LOCAL_CACHE=""
for ROOT in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "$ROOT" ] && [ -d "$ROOT" ] && [ -w "$ROOT" ] || continue
    FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
    if [ "${FREE_KB:-0}" -ge "$NEED_KB" ]; then
        LOCAL_CACHE=$(mktemp -d "$ROOT/siglip-post-food-${SLURM_JOB_ID}-${TASK_ID}.XXXXXX")
        break
    fi
done
[ -n "$LOCAL_CACHE" ] || { echo "Insufficient node-local storage" >&2; exit 5; }
trap 'rm -rf -- "$LOCAL_CACHE"' EXIT
DEST="$LOCAL_CACHE/stable_datasets/processed/food101"
mkdir -p "$DEST"
rsync -a "$SOURCE/" "$DEST/"
"$PY" -m eval.siglip_food101_postcheck run --manifest "$MANIFEST" --task-id "$TASK_ID" \
    --cache-dir "$LOCAL_CACHE" --outdir "$OUT" --num-workers 8
