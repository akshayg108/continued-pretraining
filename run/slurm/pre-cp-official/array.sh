#!/bin/bash
set -eo pipefail

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: bash array.sh --dry-run, or submit through submit.sh" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:-}" in
    0) ENCODER=SigLIP ;;
    1) ENCODER=CLIP ;;
    2) ENCODER=DINOv3 ;;
    3) ENCODER=MAE ;;
    *) echo "Expected array task 0, 1, 2, or 3" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" -eq 0 ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "Submit through submit.sh" >&2; exit 2; }
    if [ "${PRECP_NORM_SKIP_ENV_SETUP:-0}" != 1 ]; then
        module load miniconda/3-4.11.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate env
    fi
fi
set -u

REPO_ROOT="${PRECP_NORM_REPO_ROOT:?Missing PRECP_NORM_REPO_ROOT}"
OUTPUT_BASE="${PRECP_NORM_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${PRECP_NORM_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}"
OUT="$OUTPUT_BASE/precp_official_norm_v1/$RUN_ID"
PY="${PYTHON:-python3}"
cd "$REPO_ROOT"
export PYTHONPATH="$PWD:$(dirname "$PWD"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
printf 'JOB encoder=%s budget=MAX normalization=official gpu=v100 output=%s\n' "$ENCODER" "$OUT"
if [ "$DRY_RUN" -eq 1 ]; then
    "$PY" -m eval.precp_official_norm plan --encoder "$ENCODER"
    exit 0
fi
"$PY" -c 'from eval.precp_official_norm import check_gpu; print(check_gpu())'
PLAN=$("$PY" -m eval.precp_official_norm plan --encoder "$ENCODER" --tsv)

LOCAL_CACHE=""
CURRENT_DATASET=""
ERRORS=0
cleanup() {
    if [ -n "$LOCAL_CACHE" ]; then
        rm -rf -- "$LOCAL_CACHE"
        LOCAL_CACHE=""
    fi
}
trap cleanup EXIT

stage_dataset() {
    local SOURCE="$CACHE_DIR/stable_datasets/processed/$1"
    local SOURCE_KB NEED_KB ROOT FREE_KB DEST
    [ -d "$SOURCE" ] || { echo "Missing processed cache: $SOURCE" >&2; return 4; }
    SOURCE_KB=$(du -sk "$SOURCE" | awk '{print $1}') || return 4
    [[ "$SOURCE_KB" =~ ^[0-9]+$ ]] || return 4
    NEED_KB=$((SOURCE_KB + 5 * 1024 * 1024))
    for ROOT in "${TMPDIR:-}" /tmpdata /dev/shm; do
        [ -n "$ROOT" ] && [ -d "$ROOT" ] && [ -w "$ROOT" ] || continue
        FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}') || continue
        if [ "${FREE_KB:-0}" -ge "$NEED_KB" ]; then
            LOCAL_CACHE=$(mktemp -d "$ROOT/precp-native-${SLURM_JOB_ID}-${ENCODER}.XXXXXX") || return 5
            break
        fi
    done
    [ -n "$LOCAL_CACHE" ] || { echo "Insufficient node-local storage" >&2; return 5; }
    DEST="$LOCAL_CACHE/stable_datasets/processed/$1"
    mkdir -p "$DEST" || return 5
    rsync -a "$SOURCE/" "$DEST/" || return 5
    printf 'STAGED encoder=%s dataset=%s source=%s private=%s\n' "$ENCODER" "$CURRENT_DATASET" "$SOURCE" "$LOCAL_CACHE"
}

while IFS=$'\t' read -r DATASET SUBPATH SEED N GEOMETRY_ONLY; do
    if [ "$DATASET" != "$CURRENT_DATASET" ]; then
        cleanup
        CURRENT_DATASET="$DATASET"
        if ! stage_dataset "$SUBPATH"; then
            cleanup
            ERRORS=$((ERRORS + 1))
            echo "FAIL staging encoder=$ENCODER dataset=$DATASET" >&2
        fi
    fi
    [ -n "$LOCAL_CACHE" ] || continue
    printf 'PRE_CP encoder=%s dataset=%s seed=%s n=%s geometry_only=%s initialization=public_pretrained\n' \
        "$ENCODER" "$DATASET" "$SEED" "$N" "$GEOMETRY_ONLY"
    if "$PY" -m eval.precp_official_norm run --encoder "$ENCODER" --dataset "$DATASET" \
        --seed "$SEED" --cache-dir "$LOCAL_CACHE" --outdir "$OUT" \
        --num-workers "${SLURM_CPUS_PER_TASK:-8}"; then
        printf 'SUCCESS encoder=%s dataset=%s seed=%s\n' "$ENCODER" "$DATASET" "$SEED"
    else
        STATUS=$?
        ERRORS=$((ERRORS + 1))
        printf 'FAIL encoder=%s dataset=%s seed=%s exit=%s\n' "$ENCODER" "$DATASET" "$SEED" "$STATUS" >&2
    fi
done <<< "$PLAN"
printf 'FINISHED encoder=%s errors=%s output=%s\n' "$ENCODER" "$ERRORS" "$OUT"
[ "$ERRORS" -eq 0 ]
