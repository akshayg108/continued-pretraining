#!/usr/bin/env bash
#SBATCH --job-name=cp-full
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

cd -- "${CP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source run/precp_env.sh
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    printf 'Expected a numeric SLURM_ARRAY_TASK_ID.\n' >&2
    exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SKLEARN_WORKING_MEMORY=256

# Keep ICU/SQLite on the environment's C++ runtime before torch is imported.
export LD_LIBRARY_PATH="$CP_ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$CP_PYTHON" -u run/cp_full.py run --root "$CP_ROOT" \
    --task-id "$SLURM_ARRAY_TASK_ID" --num-workers 8 "$@"
