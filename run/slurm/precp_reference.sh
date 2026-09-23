#!/usr/bin/env bash
#SBATCH --job-name=precp-reference
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail

cd -- "${CP_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
source run/precp_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

exec "$CP_PYTHON" -u run/precp_reference.py --root "$CP_ROOT" --stage-data "$@"
