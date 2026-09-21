#!/bin/bash
#SBATCH --job-name=siglip-descriptors
#SBATCH --partition=compute
#SBATCH --account=civil
#SBATCH --qos=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-descriptors-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-descriptors-%j.err

set -eo pipefail
module load miniconda/3-4.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env
set -u

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

SOURCE=/scratch/gs4133/zhd/CP/outputs/siglip_native_geometry_v1/18030375
OUT="/scratch/gs4133/zhd/CP/outputs/siglip_descriptor_baselines_v1/${SLURM_JOB_ID:?}"

python3 -m eval.descriptor_baselines siglip-native \
    --geometry-dir "$SOURCE" \
    --outdir "$OUT" \
    --threads "$OMP_NUM_THREADS"

printf '\nRESULT_DIRECTORY=%s\n' "$OUT"
