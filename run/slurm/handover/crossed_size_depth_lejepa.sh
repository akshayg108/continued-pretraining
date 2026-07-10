#!/usr/bin/env bash
#SBATCH --job-name=cp-xsd-lejepa
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-215%24
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/cp-xsd-lejepa-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/cp-xsd-lejepa-%A_%a.err

# Crossed size-depth handover. See EXPERIMENT_PREREGISTRATION.md.
# Index order: seed, depth, size, backbone, dataset. Do not alter the arrays
# without updating the preregistration and the expected job count.

set -euo pipefail

module load miniconda/3-4.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

cd /scratch/gs4133/zhd/CP/continued-pretraining
export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE=wrap

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi

DATASETS=(galaxy10 eurosat food101 organamnist)
DISPLAY_NAMES=(Galaxy10 EuroSAT Food101 OrganAMNIST)
BACKBONE_TAGS=(DINOv3 CLIP)
BACKBONE_TIMMS=(vit_base_patch16_dinov3.lvd1689m vit_base_patch16_clip_224.openai)
SIZES=(1000 5000 10000)
BLOCKS=(2 4 6)
SEEDS=(42 43 44)

task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
seed_index=$((task % ${#SEEDS[@]})); task=$((task / ${#SEEDS[@]}))
block_index=$((task % ${#BLOCKS[@]})); task=$((task / ${#BLOCKS[@]}))
size_index=$((task % ${#SIZES[@]})); task=$((task / ${#SIZES[@]}))
backbone_index=$((task % ${#BACKBONE_TAGS[@]})); task=$((task / ${#BACKBONE_TAGS[@]}))
dataset_index=$task

if (( dataset_index >= ${#DATASETS[@]} )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

DATASET=${DATASETS[$dataset_index]}
DISPLAY_NAME=${DISPLAY_NAMES[$dataset_index]}
BACKBONE_TAG=${BACKBONE_TAGS[$backbone_index]}
BACKBONE_TIMM=${BACKBONE_TIMMS[$backbone_index]}
N_SAMPLES=${SIZES[$size_index]}
NUM_TRAINED_BLOCKS=${BLOCKS[$block_index]}
SEED=${SEEDS[$seed_index]}

DATA_DIR=/scratch/gs4133/zhd/CP/data
CKPT_ROOT=/scratch/gs4133/zhd/CP/outputs/ckpts/cp_handover/crossed_size_depth
LOG_ROOT=/scratch/gs4133/zhd/CP/outputs/logs/cp_handover/crossed_size_depth
CKPT_DIR="${CKPT_ROOT}/LeJEPA/pretrained/${DISPLAY_NAME}/${BACKBONE_TAG}/n${N_SAMPLES}/blk${NUM_TRAINED_BLOCKS}"
LOG_DIR="${LOG_ROOT}/LeJEPA/pretrained/${DISPLAY_NAME}/${BACKBONE_TAG}/n${N_SAMPLES}/blk${NUM_TRAINED_BLOCKS}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

RESULTS_JSON="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_blk${NUM_TRAINED_BLOCKS}_s${SEED}.json"
if [[ -f "${RESULTS_JSON}" ]]; then
    echo "[SKIP] Existing result: ${RESULTS_JSON}"
    exit 0
fi

echo "Crossed size-depth configuration"
echo "dataset=${DATASET} backbone=${BACKBONE_TAG} n=${N_SAMPLES} blocks=${NUM_TRAINED_BLOCKS} seed=${SEED}"

python -u continued_pretraining.py \
    --cp-method lejepa \
    --post-cp-sft \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE_TIMM}" \
    --n-samples "${N_SAMPLES}" \
    --epochs 150 \
    --batch-size 256 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --freeze-epochs 15 \
    --num-trained-blocks "${NUM_TRAINED_BLOCKS}" \
    --knn-k 20 \
    --num-workers 8 \
    --lamb 0.02 \
    --n-views 8 \
    --proj-dim 128 \
    --hidden-dim 2048 \
    --pool-strategy cls \
    --accumulate-grad-batches 1 \
    --checkpoint-dir "${CKPT_DIR}" \
    --cache-dir "${DATA_DIR}" \
    --project "lejepa-cp-crossed-size-depth-${DATASET}" \
    --run-name "${BACKBONE_TAG}_${DATASET}_n${N_SAMPLES}_blk${NUM_TRAINED_BLOCKS}_s${SEED}" \
    --seed "${SEED}" \
    --skip-baseline \
    --results-json "${RESULTS_JSON}"
