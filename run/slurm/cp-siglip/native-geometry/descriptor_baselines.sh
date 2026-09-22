#!/bin/bash
#SBATCH --job-name=siglip-descriptors
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-descriptors-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-descriptors-%j.err

set -euo pipefail

# Pin the interpreter; reloading conda can leave PATH pointing at base Python.
DESCRIPTOR_PYTHON="${DESCRIPTOR_PYTHON:-/home/gs4133/.conda/envs/env/bin/python3}"
if [[ "$DESCRIPTOR_PYTHON" != /* || ! -x "$DESCRIPTOR_PYTHON" ]]; then
    echo "DESCRIPTOR_PYTHON must be an executable absolute path: $DESCRIPTOR_PYTHON" >&2
    exit 2
fi

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

"$DESCRIPTOR_PYTHON" -c '
import sys
print(f"PYTHON={sys.executable}", flush=True)
print(f"VERSION={sys.version}", flush=True)
import numpy, scipy, sklearn, threadpoolctl
print(f"DEPENDENCIES_OK numpy={numpy.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}", flush=True)
'

SOURCE=/scratch/gs4133/zhd/CP/outputs/siglip_native_geometry_v1/18030375
OUT="/scratch/gs4133/zhd/CP/outputs/siglip_descriptor_baselines_v1/${SLURM_JOB_ID:?}"

"$DESCRIPTOR_PYTHON" -m eval.descriptor_baselines siglip-native \
    --geometry-dir "$SOURCE" \
    --outdir "$OUT" \
    --threads "$OMP_NUM_THREADS"

printf '\nRESULT_DIRECTORY=%s\n' "$OUT"
