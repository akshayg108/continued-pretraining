#!/usr/bin/env bash
#SBATCH --job-name=lp-only
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

cd -- "${CP_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
source run/precp_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$CP_ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$CP_PYTHON" -u run/lp_only.py run --root "$CP_ROOT" \
    --task-id "$SLURM_ARRAY_TASK_ID" "$@"
