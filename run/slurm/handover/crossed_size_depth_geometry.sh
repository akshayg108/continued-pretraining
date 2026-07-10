#!/usr/bin/env bash
#SBATCH --job-name=cp-xsd-geometry
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/cp-xsd-geometry-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/cp-xsd-geometry-%A_%a.err

# Run only after every crossed_size_depth_lejepa.sh task has reached a terminal
# state and checkpoint completeness has been audited. This script reads the
# isolated handover checkpoint root and writes separate geometry shards.

set -euo pipefail

module load miniconda/3-4.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi

DATASETS=(galaxy10 eurosat food101 organamnist)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}

DATA_ROOT=/scratch/gs4133/zhd/CP/data
CKPT_ROOT=/scratch/gs4133/zhd/CP/outputs/ckpts/cp_handover/crossed_size_depth
OUT_DIR=eval/outputs/handover/crossed_size_depth_geometry_shards
mkdir -p "${OUT_DIR}"

python -u eval/F2_forces/postcp_sweep.py \
    --ckpt-root "${CKPT_ROOT}" \
    --download-dir "${DATA_ROOT}/stable_datasets/downloads" \
    --processed-dir "${DATA_ROOT}/stable_datasets/processed" \
    --imagenet-dir "${DATA_ROOT}/imagenet_val" \
    --methods LeJEPA \
    --encoders DINOv3 CLIP \
    --datasets "${DATASET}" \
    --out "${OUT_DIR}/${DATASET}.csv"

echo "Wrote geometry shard: ${OUT_DIR}/${DATASET}.csv"
