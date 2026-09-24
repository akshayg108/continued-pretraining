#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/run/precp_env.sh"
CONCURRENCY="${CP_CONCURRENCY:-10}"
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

"$CP_PYTHON" "$REPO/run/cp_full.py" check --root "$CP_ROOT" ${encoder_args[@]+"${encoder_args[@]}"}
"$CP_PYTHON" "$REPO/run/cp_full.py" list ${encoder_args[@]+"${encoder_args[@]}"}
V100_TASKS=$("$CP_PYTHON" "$REPO/run/cp_full.py" array --gpu v100 ${encoder_args[@]+"${encoder_args[@]}"})
A100_TASKS=$("$CP_PYTHON" "$REPO/run/cp_full.py" array --gpu a100 ${encoder_args[@]+"${encoder_args[@]}"})
for tasks in "$V100_TASKS" "$A100_TASKS"; do
    if [[ ! "$tasks" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        printf 'The CP runner returned an invalid or empty Slurm task list: %s\n' "$tasks" >&2
        exit 1
    fi
done

existing_jobs=$(squeue -h -u "$(id -un)" -n cp-full -o '%F' | sort -u)
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
v100_args=("${common[@]}")
if [[ "$existing_dependency" != afterany ]]; then
    v100_args+=(--dependency="$existing_dependency")
fi
v100_job=$(sbatch --parsable "${v100_args[@]}" \
    --gres=gpu:v100:1 --array="$V100_TASKS%$CONCURRENCY" \
    --output="$CP_ROOT/outputs/slurm-log/cp-full-v100-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/cp-full-v100-%A_%a.err" \
    "$REPO/run/slurm/cp_full.sh")
v100_job="${v100_job%%;*}"
if [[ ! "$v100_job" =~ ^[1-9][0-9]*$ ]]; then
    printf 'sbatch returned an invalid V100 array job ID: %s\n' "$v100_job" >&2
    exit 1
fi
printf 'V100 CP array: %s (concurrency=%s)\n' "$v100_job" "$CONCURRENCY"

a100_job=$(sbatch --parsable "${common[@]}" \
    --gres=gpu:a100:1 --array="$A100_TASKS%$CONCURRENCY" \
    --dependency="afterany:$v100_job" \
    --output="$CP_ROOT/outputs/slurm-log/cp-full-a100-%A_%a.out" \
    --error="$CP_ROOT/outputs/slurm-log/cp-full-a100-%A_%a.err" \
    "$REPO/run/slurm/cp_full.sh")
printf 'A100 CP array: %s (concurrency=%s; after V100 array %s ends)\n' \
    "${a100_job%%;*}" "$CONCURRENCY" "$v100_job"
