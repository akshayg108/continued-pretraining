#!/bin/bash
#SBATCH --job-name=d-gal-max
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/Continued-Pretraining/outputs/slurm-log/simclr-galaxy10-max-%j.out
#SBATCH --error=/scratch/gs4133/zhd/Continued-Pretraining/outputs/slurm-log/simclr-galaxy10-max-%j.err

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
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"

cd /scratch/gs4133/zhd/Continued-Pretraining/continued-pretraining
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"

echo "Working directory: $(pwd)"
echo "=========================================="
nvidia-smi

DATA_DIR="/scratch/gs4133/zhd/Continued-Pretraining/data"
CKPT_DIR="/scratch/gs4133/zhd/Continued-Pretraining/outputs/ckpts/cp/SimCLR/Galaxy10/DINOv3/all"
LOG_DIR="/scratch/gs4133/zhd/Continued-Pretraining/outputs/logs/cp/SimCLR/Galaxy10/DINOv3/all"
SLURM_LOG_DIR="/scratch/gs4133/zhd/Continued-Pretraining/outputs/slurm-log"
mkdir -p ${DATA_DIR} ${CKPT_DIR} ${LOG_DIR} ${SLURM_LOG_DIR}

DATASET="galaxy10"
DISPLAY_NAME="Galaxy10"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="DINOv3"
BACKBONE_TIMM="vit_base_patch16_dinov3.lvd1689m"

EPOCHS=150
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
NUM_TRAINED_BLOCKS=4
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

TEMPERATURE=0.5
PROJ_DIM=128
HIDDEN_DIM=2048

NSAMPLES=(14188)

run_single() {
    local n_samples=$1
    local seed=$2
    local results_file="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${seed}.json"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] SimCLR-CP ${BACKBONE_TAG} | ${DATASET} | n=${n_samples} | seed=${seed}"
    echo "  freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
    echo "  Start: $(date)"
    echo "=========================================="

    python -u continued_pretraining.py \
        --cp-method simclr \
        --post-cp-sft \
        --dataset ${DATASET} \
        --backbone ${BACKBONE_TIMM} \
        --n-samples ${n_samples} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --freeze-epochs ${FREEZE_EPOCHS} \
        --num-trained-blocks ${NUM_TRAINED_BLOCKS} \
        --knn-k ${KNN_K} \
        --num-workers ${NUM_WORKERS} \
        --temperature ${TEMPERATURE} \
        --proj-dim ${PROJ_DIM} \
        --hidden-dim ${HIDDEN_DIM} \
        --checkpoint-dir ${CKPT_DIR} \
        --cache-dir ${DATA_DIR} \
        --project simclr-cp-dinov3-${DATASET} \
        --run-name "${BACKBONE_TAG}_${DATASET}_n${n_samples}_blk${NUM_TRAINED_BLOCKS}_s${seed}" \
        --seed ${seed} \
        --results-json ${results_file} 2>&1

    local exit_code=$?
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"
    [ $exit_code -ne 0 ] && echo "[FAIL] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed}"
    return $exit_code
}

aggregate_results() {
    local n_samples=$1
    local csv_file=$2
    python3 << PYEOF
import json, os, statistics
log_dir = "${LOG_DIR}"
backbone_tag = "${BACKBONE_TAG}"
dataset = "${DATASET}"
n_samples = "${n_samples}"
display_name = "${DISPLAY_NAME}"
model_size = "${MODEL_SIZE}"
csv_file = "${csv_file}"
seeds = [42, 43, 44]
metrics = {k: [] for k in ["pre_knn_f1","pre_linear_f1","post_knn_f1","post_linear_f1","post_sft_f1"]}
for i, seed in enumerate(seeds):
    rf = os.path.join(log_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}.json")
    if not os.path.exists(rf):
        print(f"  Warning: {rf} not found, skipping seed {seed}")
        continue
    with open(rf) as f:
        data = json.load(f)
    for key in metrics:
        val = data.get(key)
        if val is not None: metrics[key].append(val)
    def fmt(v): return f"{v:.6f}" if v is not None else ""
    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},{i},"
                f"{fmt(data.get('pre_knn_f1'))},,{fmt(data.get('pre_linear_f1'))},,"
                f"{fmt(data.get('post_knn_f1'))},,{fmt(data.get('post_linear_f1'))},,"
                f"{fmt(data.get('post_sft_f1'))},\n")
def mean_std(vals):
    if not vals: return "", ""
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.6f}", f"{s:.6f}"
if any(len(v) > 0 for v in metrics.values()):
    ms = {k: mean_std(v) for k, v in metrics.items()}
    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},average,"
                f"{ms['pre_knn_f1'][0]},{ms['pre_knn_f1'][1]},"
                f"{ms['pre_linear_f1'][0]},{ms['pre_linear_f1'][1]},"
                f"{ms['post_knn_f1'][0]},{ms['post_knn_f1'][1]},"
                f"{ms['post_linear_f1'][0]},{ms['post_linear_f1'][1]},"
                f"{ms['post_sft_f1'][0]},{ms['post_sft_f1'][1]}\n")
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: aggregated")
else:
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: no results available")
PYEOF
}

echo ""
echo "=========================================="
echo "Starting SimCLR-CP: ${DISPLAY_NAME} (MAX n=14188)"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="

CSV_FILE="${LOG_DIR}/${BACKBONE_TAG}_simclr_cp_results.csv"
echo "backbone,dataset,n_samples,model_size,run,pre_knn_f1,pre_knn_f1_std,pre_linear_f1,pre_linear_f1_std,post_knn_f1,post_knn_f1_std,post_linear_f1,post_linear_f1_std,post_sft_f1,post_sft_f1_std" > ${CSV_FILE}

TOTAL_SUCCESS=0
TOTAL_FAIL=0
for n_samples in "${NSAMPLES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_single ${n_samples} ${seed}
        [ $? -eq 0 ] && TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1)) || TOTAL_FAIL=$((TOTAL_FAIL + 1))
    done
    aggregate_results ${n_samples} ${CSV_FILE}
done

echo ""
echo "=========================================="
echo "Completed! Successful: ${TOTAL_SUCCESS}  Failed: ${TOTAL_FAIL}"
echo "End Time: $(date)"
echo "=========================================="
