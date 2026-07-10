#!/bin/bash
#SBATCH --job-name=nd2-sweep
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=cn253,cn259
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/nd2-sweep-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/nd2-sweep-%A_%a.err

# ============================================================
# ND2 — spectral geometry (rankme / alpha-ReQ / coherence + uniformity) over ALL
# pretrained-variant cp/ checkpoints (eval/new_direction/nd2_spectral_sweep.py).
# Same shard-parallel machinery as the F2 postcp_sweep: 16 shards, resumable per
# shard CSV. Checkpoints are read from /scratch (dominant IO, cannot be staged
# cheaply — same tradeoff as exp_i); datasets are read from /scratch too since
# every shard touches many datasets.
#
# After the whole array finishes, concat shards on the login node:
#   python -c "import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('eval/outputs/nd2_spectral_shards/*.csv'))]).to_csv('eval/outputs/nd2_spectral_sweep.csv', index=False)"
# and send me eval/outputs/nd2_spectral_sweep.csv.
# Queue-friendly tip:  sbatch --array=0-15%6 run/slurm/new_direction/nd2_spectral_sweep.sh
# ============================================================

echo "=========================================="
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"
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
DL_DIR="${DATA_ROOT}/stable_datasets/downloads"
PROC_DIR="${DATA_ROOT}/stable_datasets/processed"
CKPT_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp"
OUT_DIR="eval/outputs/nd2_spectral_shards"
mkdir -p "${OUT_DIR}"

i=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "ND2 spectral sweep: shard ${i}/16"
echo "=========================================="

python -u eval/new_direction/nd2_spectral_sweep.py \
    --ckpt-root "${CKPT_ROOT}" \
    --download-dir  "${DL_DIR}" \
    --processed-dir "${PROC_DIR}" \
    --shard "${i}/16" \
    --out "${OUT_DIR}/shard_${i}.csv" \
    2>&1

echo ""
echo "Shard written: ${OUT_DIR}/shard_${i}.csv"
echo "End Time: $(date)"
