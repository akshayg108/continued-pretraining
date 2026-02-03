#!/bin/bash
#SBATCH --job-name=mae-cp-cifar10
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/logs/slurm-%j.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/logs/slurm-%j.err

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

echo "Python: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)" || true
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"
cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"
echo "Working directory: $(pwd)"
echo "=========================================="
nvidia-smi
DATA_DIR="/scratch/gs4133/zhd/CP/data"
CKPT_DIR="/scratch/gs4133/zhd/CP/outputs/ckpts"
LOG_DIR="/scratch/gs4133/zhd/CP/outputs/logs"
mkdir -p ${DATA_DIR}
mkdir -p ${CKPT_DIR}
mkdir -p ${LOG_DIR}
echo "Data directory: ${DATA_DIR}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="
python -u mae/mae_cp.py \
    --dataset cifar10 \
    --backbone facebook/vit-mae-base \
    --n-samples 1000 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --mask-ratio 0.75 \
    --decoder-dim 512 \
    --decoder-depth 4 \
    --num-trained-blocks 2 \
    --warmup-epochs 10 \
    --knn-k 20 \
    --num-workers 8 \
    --checkpoint-dir ${CKPT_DIR} \
    --cache-dir ${DATA_DIR} \
    --project mae-cp-cifar10 \
    --seed 42 2>&1
EXIT_CODE=$?
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then