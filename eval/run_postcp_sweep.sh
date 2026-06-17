#!/bin/bash
#SBATCH --job-name=postcp-sweep
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/postcp-sweep-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/postcp-sweep-%A_%a.err

# run this: sbatch --array=0-11 eval/run_postcp_sweep.sh

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "=========================================="

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "from stable_datasets import images; print('stable_datasets: OK')" || echo "stable_datasets: NOT importable"

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "Working directory: $(pwd)"
nvidia-smi
echo "=========================================="

# ============================================================
# Optional args:  sbatch run_postcp_sweep.sh --seed 42
#   --seed N   restrict to one seed (passes --seeds N)
# ============================================================
OVERRIDE_SEED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) OVERRIDE_SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ============================================================
# Paths
# ============================================================
CKPT_ROOT="/scratch/gs4133/zhd/CP/outputs/ckpts/cp"
DOWNLOAD_DIR="/scratch/gs4133/zhd/CP/data/stable_datasets/downloads"
PROCESSED_DIR="/scratch/gs4133/zhd/CP/data/stable_datasets/processed"
IMAGENET_DIR="/scratch/gs4133/zhd/CP/data/imagenet_val"   # set to "" to skip neighbor-overlap (~2x faster; not needed for Δcv→Δknn)
SLURM_LOG_DIR="/scratch/gs4133/zhd/CP/outputs/slurm-log"
mkdir -p "${SLURM_LOG_DIR}" eval/outputs

# ============================================================
# Sharding: with `--array=0-7` each task does its slice; without it, one job does everything.
# ============================================================
SHARD_ARG=""
OUT="eval/outputs/postcp_sweep.csv"
if [ -n "${SLURM_ARRAY_TASK_COUNT}" ]; then
    SHARD_ARG="--shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    OUT="eval/outputs/postcp_sweep_${SLURM_ARRAY_TASK_ID}.csv"
fi

# Optional seed restriction
SEED_ARG=""
if [ -n "${OVERRIDE_SEED}" ]; then SEED_ARG="--seeds ${OVERRIDE_SEED}"; fi

# Optional ImageNet overlap (neighbor_overlap; re-embeds ImageNet per ckpt, ~2x time)
IMN_ARG=""
if [ -n "${IMAGENET_DIR}" ]; then IMN_ARG="--imagenet-dir ${IMAGENET_DIR}"; fi

echo "OUT=${OUT}  SHARD_ARG='${SHARD_ARG}'  SEED_ARG='${SEED_ARG}'  IMN_ARG='${IMN_ARG}'"
echo "=========================================="

python -u eval/postcp_sweep.py \
    --ckpt-root "${CKPT_ROOT}" \
    --download-dir "${DOWNLOAD_DIR}" \
    --processed-dir "${PROCESSED_DIR}" \
    --out "${OUT}" \
    ${SHARD_ARG} ${SEED_ARG} ${IMN_ARG}

echo "=========================================="
echo "Exit Code: $?  End Time: $(date)"
echo "Output: ${OUT}"
echo "=========================================="
