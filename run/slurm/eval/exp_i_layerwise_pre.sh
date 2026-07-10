#!/bin/bash
#SBATCH --job-name=expI-layer-pre
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/expI-layer-pre-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/expI-layer-pre-%j.err

# ============================================================
# Exp I (pre-CP side) — layer-wise virtual encoders (eval/utils/layerwise_geometry.py).
# SINGLE job (no array): 4 encoders x 15 datasets x 12 blocks, per-layer geometry +
# internal kNN. No ImageNet, no checkpoints — timm pretrained weights only.
# Reads datasets straight from /scratch (60 light passes; staging all 15 would cost
# more than it saves).
#
# Output: eval/outputs/layerwise_pre.csv  — send it back as-is (no shards to concat).
# Can run CONCURRENTLY with exp_i_layerwise_post.sh and exp_j_transport.sh.
# ============================================================

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi

DATA_ROOT="/scratch/gs4133/zhd/CP/data"

python -u eval/utils/layerwise_geometry.py \
    --download-dir  "${DATA_ROOT}/stable_datasets/downloads" \
    --processed-dir "${DATA_ROOT}/stable_datasets/processed" \
    --output eval/outputs/layerwise_pre.csv \
    2>&1

echo ""
echo "Written: eval/outputs/layerwise_pre.csv"
echo "End Time: $(date)"
