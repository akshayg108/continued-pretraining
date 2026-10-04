#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
if [[ -n "${CP_TASK_IDS:-}" ]]; then
    if [[ ! "$CP_TASK_IDS" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
        printf 'CP_TASK_IDS must contain comma-separated task IDs or ranges.\n' >&2
        exit 2
    fi
    # Explicit recovery lists were audited against completed seed results.
    unset SBATCH_DEPENDENCY SBATCH_ARRAY_INX
    job=$(sbatch --parsable --partition=nvidia --account=civil --qos=nvidia \
        --chdir="$REPO" --export=ALL --job-name=cp-full \
        --nodes=1 --ntasks=1 --gres=gpu:a100:1 --constraint=80g \
        --exclude=cn253,cn259 --cpus-per-task=24 --mem=256G --time=96:00:00 \
        --array="$CP_TASK_IDS" \
        --output="$CP_ROOT/outputs/slurm-log/cp-merged-%A_%a.out" \
        --error="$CP_ROOT/outputs/slurm-log/cp-merged-%A_%a.err" \
        "$REPO/run/slurm/cp_full.sh" --num-workers 16)
    printf 'CP merged array: %s (no array throttle; no dependency)\n' "${job%%;*}"
    exit 0
fi
CONCURRENCY="${CP_CONCURRENCY:-10}"
GROUP="${CP_GROUP:-small}"
case "$GROUP" in
    small) gpu_groups=(v100 a100); log_prefix=cp-full ;;
    four-block) gpu_groups=(a100 a100-80gb); log_prefix=cp-four-block ;;
    six-block) gpu_groups=(a100-80gb); log_prefix=cp-six-block ;;
    *) printf 'CP_GROUP must be small, four-block, or six-block.\n' >&2; exit 2 ;;
esac
if [[ ! "$CONCURRENCY" =~ ^([1-9]|10)$ ]]; then
    printf 'CP_CONCURRENCY must be an integer from 1 to 10.\n' >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --dep*|-d*)
            printf 'The launcher manages array dependencies; dependency overrides are not supported.\n' >&2
            exit 2
            ;;
    esac
done
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    printf 'Unset SBATCH_DEPENDENCY; the launcher manages array dependency ordering.\n' >&2
    exit 2
fi

encoder_args=()
if [[ -n "${CP_ENCODERS:-}" ]]; then
    read -r -a encoder_args <<< "$CP_ENCODERS"
    encoder_args=(--encoder ${encoder_args[@]+"${encoder_args[@]}"})
fi

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$CP_ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CP_PYTHON" -c 'import continued_pretraining; import sqlite3; from stable_pretraining.registry.logger import RegistryLogger; print("CP runtime imports OK")'

"$CP_PYTHON" "$REPO/run/cp_full.py" check --root "$CP_ROOT" --group "$GROUP" ${encoder_args[@]+"${encoder_args[@]}"}
"$CP_PYTHON" "$REPO/run/cp_full.py" list --group "$GROUP" ${encoder_args[@]+"${encoder_args[@]}"}
task_lists=()
for gpu in "${gpu_groups[@]}"; do
    tasks=$("$CP_PYTHON" "$REPO/run/cp_full.py" array --gpu "$gpu" --group "$GROUP" ${encoder_args[@]+"${encoder_args[@]}"})
    if [[ -n "$tasks" && ! "$tasks" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        printf 'The CP runner returned an invalid Slurm task list: %s\n' "$tasks" >&2
        exit 1
    fi
    task_lists+=("$tasks")
done

existing_jobs=""
if [[ "$GROUP" != six-block ]]; then
    existing_jobs=$(squeue -h -u "$(id -un)" -n cp-full -o '%F' | sort -u)
fi
existing_dependency=afterany
while read -r job; do
    [[ -z "$job" ]] && continue
    if [[ ! "$job" =~ ^[1-9][0-9]*$ ]]; then
        printf 'squeue returned an invalid CP array parent ID: %s\n' "$job" >&2
        exit 1
    fi
    existing_dependency+=":$job"
done <<< "$existing_jobs"

common=("$@" --chdir="$REPO" --export=ALL --nodes=1 --ntasks=1 --job-name=cp-full)
runner_args=()
if [[ "$GROUP" == six-block ]]; then
    common+=(--cpus-per-task=24 --mem=128G)
    runner_args=(--num-workers 16)
fi
for index in "${!gpu_groups[@]}"; do
    gpu="${gpu_groups[$index]}"
    tasks="${task_lists[$index]}"
    [[ -z "$tasks" ]] && continue
    submit_args=("${common[@]}" --gres="gpu:${gpu%%-*}:1")
    if [[ "$gpu" == a100-80gb ]]; then
        submit_args+=(--constraint=80g --exclude=cn253,cn259)
    fi
    if [[ "$existing_dependency" != afterany ]]; then
        submit_args+=(--dependency="$existing_dependency")
    fi
    job=$(sbatch --parsable "${submit_args[@]}" \
        --array="$tasks%$CONCURRENCY" \
        --output="$CP_ROOT/outputs/slurm-log/$log_prefix-$gpu-%A_%a.out" \
        --error="$CP_ROOT/outputs/slurm-log/$log_prefix-$gpu-%A_%a.err" \
        "$REPO/run/slurm/cp_full.sh" ${runner_args[@]+"${runner_args[@]}"})
    job="${job%%;*}"
    if [[ ! "$job" =~ ^[1-9][0-9]*$ ]]; then
        printf 'sbatch returned an invalid %s array job ID: %s\n' "$gpu" "$job" >&2
        exit 1
    fi
    printf 'CP %s array: %s (concurrency=%s; dependency=%s)\n' \
        "$gpu" "$job" "$CONCURRENCY" "$existing_dependency"
    existing_dependency="afterany:$job"
done
