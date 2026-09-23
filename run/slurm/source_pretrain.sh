#!/usr/bin/env bash
#SBATCH --job-name=source-pretrain
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
set -euo pipefail

STAGE="${1:-train}"
if (( $# > 1 )) || [[ "$STAGE" != prepare && "$STAGE" != train ]]; then
    printf 'Usage: %s [prepare|train]\n' "$0" >&2
    exit 2
fi

cd -- "${CP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source run/precp_env.sh
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SKLEARN_WORKING_MEMORY=256

ARGS=("$STAGE" --root "$CP_ROOT"
    --imagenet-dir "${IMAGENET_TRAIN_DIR:-$CP_ROOT/data/imagenet/train}"
    --imagenet-val-dir "${IMAGENET_VAL_DIR:-$CP_ROOT/data/imagenet_val}"
    --output-dir "${SOURCE_OUTPUT_DIR:-$CP_ROOT/outputs/source_coverage_v1}")
if [[ "$STAGE" == prepare ]]; then
    case "${IMAGENET_SOURCE:-hf}" in
        hf) ARGS+=(--download-imagenet) ;;
        local)
            if [[ -n "${IMAGENET_TRAIN_ARCHIVE:-}" ]]; then
                ARGS+=(--imagenet-archive "$IMAGENET_TRAIN_ARCHIVE")
            fi
            ;;
        *) printf 'IMAGENET_SOURCE must be hf or local\n' >&2; exit 2 ;;
    esac
else
    TASK="${SLURM_ARRAY_TASK_ID:-}"
    if [[ ! "$TASK" =~ ^[0-1]$ ]]; then
        printf 'Training requires SLURM_ARRAY_TASK_ID in 0..1; got: %s\n' "$TASK" >&2
        exit 2
    fi
    CONDITIONS=(imagenet mixed)
    ARGS+=(--condition "${CONDITIONS[$((TASK % 2))]}"
        --seed 42 --steps "${SOURCE_STEPS:-500400}" --resume)
fi

exec "$CP_PYTHON" -u run/source_pretrain.py "${ARGS[@]}"
