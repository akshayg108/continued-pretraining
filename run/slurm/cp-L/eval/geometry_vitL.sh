#!/bin/bash
#SBATCH --job-name=L-geo
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/L-geo-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/L-geo-%j.err

# ============================================================
# cp-L R1 appetizer — ViT-L pre-CP geometry (eval/geometry_vitL.py).
# ZERO TRAINING; run this FIRST, before any cp-L training job:
# if the R1 rank-stability verdict at the end of the log FAILs (uniformity or
# overlap rank correlation vs ViT-B <= 0.8), STOP and reconsider before
# spending GPU on the 63 CP runs (eval/DESIGN_vitL_robustness.md).
#
# Single job, 15 datasets + ImageNet-val, one forward pass each (light
# sequential IO — no node staging needed, same idiom as exp_i_layerwise_pre).
# Output: eval/outputs/geometry_vitL.csv — send it back together with the
# R1 verdict lines from this log.
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

python -u eval/geometry_vitL.py \
    --imagenet-dir  "${DATA_ROOT}/imagenet_val" \
    --download-dir  "${DATA_ROOT}/stable_datasets/downloads" \
    --processed-dir "${DATA_ROOT}/stable_datasets/processed" \
    --output eval/outputs/geometry_vitL.csv \
    2>&1

echo ""
echo "Written: eval/outputs/geometry_vitL.csv (R1 verdict printed above)"
echo "End Time: $(date)"
