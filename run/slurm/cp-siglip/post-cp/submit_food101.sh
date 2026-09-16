#!/bin/bash
set -euo pipefail

REPO_ROOT="${SIGLIP_POST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
OUTPUT_BASE="${SIGLIP_POST_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
CACHE_DIR="${SIGLIP_POST_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
PY="${PYTHON:-python3}"
DRY_RUN=0
CONCURRENCY=9
PLAN_ARGS=(--source mainrule)
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --concurrency)
            [ "$#" -ge 2 ] || { echo "--concurrency needs a value" >&2; exit 2; }
            CONCURRENCY="$2"; shift 2 ;;
        --source|--recheck-job-id)
            [ "$#" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
            PLAN_ARGS+=("$1" "$2"); shift 2 ;;
        --methods|--seeds)
            OPTION="$1"; shift
            [ "$#" -gt 0 ] && [[ "$1" != --* ]] || { echo "$OPTION needs values" >&2; exit 2; }
            PLAN_ARGS+=("$OPTION")
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do PLAN_ARGS+=("$1"); shift; done ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] && [ "$CONCURRENCY" -ge 1 ] && [ "$CONCURRENCY" -le 12 ] || {
    echo "Concurrency must be 1..12" >&2; exit 2;
}
cd "$REPO_ROOT"
export PYTHONPATH="$PWD:$(dirname "$PWD"):${PYTHONPATH:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MANIFEST="$OUTPUT_BASE/siglip_food101_postcheck_manifests/selection-$STAMP.json"
LOG_DIR="$OUTPUT_BASE/slurm-log"
mkdir -p "$(dirname "$MANIFEST")" "$LOG_DIR"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd -P)"
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd -P)/$(basename "$MANIFEST")"
LOG_DIR="$(cd "$LOG_DIR" && pwd -P)"
"$PY" -m eval.siglip_food101_postcheck plan --output-base "$OUTPUT_BASE" \
    --manifest "$MANIFEST" "${PLAN_ARGS[@]}"
COUNT=$("$PY" - "$MANIFEST" <<'PY'
import sys
from eval.siglip_food101_postcheck import load_manifest
print(len(load_manifest(sys.argv[1])["tasks"]))
PY
)
if [ "$COUNT" -eq 0 ]; then
    echo "Nothing submitted: no complete checkpoint/result pairs found."
    exit 0
fi
if [ "$COUNT" -eq 1 ]; then ARRAY=0; else ARRAY="0-$((COUNT - 1))"; fi
SBATCH=(sbatch --parsable --job-name=siglip-post-food --chdir="$REPO_ROOT"
    --array="${ARRAY}%${CONCURRENCY}" --partition=nvidia --account=civil
    --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1 --cpus-per-task=8 --mem=96G --time=24:00:00
    --output="$LOG_DIR/siglip-post-food-%A_%a.out" --error="$LOG_DIR/siglip-post-food-%A_%a.err"
    --export="ALL,SIGLIP_POST_MANIFEST=$MANIFEST,SIGLIP_POST_REPO_ROOT=$REPO_ROOT,SIGLIP_POST_OUTPUT_BASE=$OUTPUT_BASE,SIGLIP_POST_CACHE_DIR=$CACHE_DIR"
    "$REPO_ROOT/run/slurm/cp-siglip/post-cp/food101_v100.sh")
printf 'SUBMIT'; printf ' %q' "${SBATCH[@]}"; printf '\n'
if [ "$DRY_RUN" -eq 1 ]; then
    for ((id=0; id<COUNT; id++)); do
        SIGLIP_POST_MANIFEST="$MANIFEST" SIGLIP_POST_REPO_ROOT="$REPO_ROOT" \
        SIGLIP_POST_OUTPUT_BASE="$OUTPUT_BASE" SIGLIP_POST_CACHE_DIR="$CACHE_DIR" \
        SLURM_ARRAY_TASK_ID="$id" bash "${SBATCH[${#SBATCH[@]}-1]}" --dry-run
    done
else
    "${SBATCH[@]}"
fi
