#!/usr/bin/env bash
#SBATCH --job-name=precp-full
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --array=0-16%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
set -euo pipefail

cd -- "${CP_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
source run/precp_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SKLEARN_WORKING_MEMORY=256

exec "$CP_PYTHON" -u run/precp.py run --root "$CP_ROOT" \
    --task-id "$SLURM_ARRAY_TASK_ID" "$@"
