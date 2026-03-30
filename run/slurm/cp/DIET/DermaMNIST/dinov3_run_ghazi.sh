#!/usr/bin/env bash

echo "=========================================="
echo "Local run started: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Optional conda activation for local environments.
# Usage: CP_CONDA_ENV=myenv ./dinov3_run_ghazi.sh
if command -v conda >/dev/null 2>&1 && [ -n "${CP_CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CP_CONDA_ENV}"
fi

echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import wandb; print('wandb:', wandb.__version__)" || echo "wandb: not installed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
if [ ! -f "${REPO_ROOT}/continued_pretraining.py" ]; then
    echo "[ERROR] continued_pretraining.py not found at ${REPO_ROOT}"
    echo "[ERROR] Set REPO_ROOT manually or run from a valid checkout."
    exit 1
fi
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_CONSOLE="wrap"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "=========================================="
nvidia-smi

# ============================================================
# Paths
# ============================================================
DATA_DIR="/home/ja-am/temp"
OUTPUT_ROOT="/home/ja-am/outputs/continued-pretraining"
CKPT_DIR="${OUTPUT_ROOT}/ckpts/cp/DIET/DermaMNIST/DINOv3"
LOG_DIR="${OUTPUT_ROOT}/logs/cp/DIET/DermaMNIST/DINOv3"
RUN_LOG_DIR="${OUTPUT_ROOT}/logs/local"
WANDB_DIR="${OUTPUT_ROOT}/wandb"
WANDB_CACHE_DIR="${OUTPUT_ROOT}/wandb-cache"
WANDB_CONFIG_DIR="${OUTPUT_ROOT}/wandb-config"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${LOG_DIR}" "${RUN_LOG_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR

# ============================================================
# Fixed parameters
# ============================================================
DATASET="dermamnist"
DISPLAY_NAME="DermaMNIST"
MODEL_SIZE="ViT-B"
BACKBONE_TAG="DINOv3"
BACKBONE_TIMM="vit_base_patch16_dinov3.lvd1689m"

EPOCHS=150
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=0.05
FREEZE_EPOCHS=15
NUM_TRAINED_BLOCKS=2
KNN_K=20
NUM_WORKERS=8
SEEDS=(42 43 44)

# CP hyperparameters
TEMPERATURE=0.5
PROJ_DIM=128
HIDDEN_DIM=2048

NSAMPLES=(100 500 1000 7007)

# ============================================================
# Run a single experiment
# ============================================================
run_single() {
    local n_samples=$1
    local seed=$2

    local results_file="${LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${seed}.json"
    local run_log_file="${RUN_LOG_DIR}/${BACKBONE_TAG}_${DATASET}_n${n_samples}_seed${seed}.log"

    if [ -f "$results_file" ]; then
        echo "[SKIP] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed} (results file exists)"
        return 0
    fi

    echo "=========================================="
    echo "[RUN] DIET-CP ${BACKBONE_TAG} | ${DATASET} | n=${n_samples} | seed=${seed}"
    echo "  freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
    echo "  Start: $(date)"
    echo "=========================================="

    ${PYTHON_BIN} -u continued_pretraining.py \
        --cp-method diet \
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
        --wandb-dir ${WANDB_DIR} \
        --project diet-cp-dinov3-${DATASET} \
        --run-name "${BACKBONE_TAG}_${DATASET}_n${n_samples}_blk${NUM_TRAINED_BLOCKS}_s${seed}" \
        --seed ${seed} \
        --results-json ${results_file} 2>&1 | tee -a "${run_log_file}"

    local exit_code=${PIPESTATUS[0]}
    echo "  Exit Code: ${exit_code}"
    echo "  End: $(date)"
    echo "  Log: ${run_log_file}"

    if [ $exit_code -ne 0 ]; then
        echo "[FAIL] ${BACKBONE_TAG} | ${DATASET} n=${n_samples} seed=${seed}"
    fi

    return $exit_code
}

# ============================================================
# Aggregate results across seeds
# ============================================================
aggregate_results() {
    local n_samples=$1
    local csv_file=$2

    ${PYTHON_BIN} << PYEOF
import json, os, statistics

log_dir = "${LOG_DIR}"
backbone_tag = "${BACKBONE_TAG}"
dataset = "${DATASET}"
n_samples = "${n_samples}"
display_name = "${DISPLAY_NAME}"
model_size = "${MODEL_SIZE}"
csv_file = "${csv_file}"
seeds = [42, 43, 44]

pre_knn_f1s = []
pre_linear_f1s = []
post_knn_f1s = []
post_linear_f1s = []
post_sft_f1s = []

for i, seed in enumerate(seeds):
    results_file = os.path.join(log_dir, f"{backbone_tag}_{dataset}_n{n_samples}_seed{seed}.json")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping seed {seed}")
        continue

    with open(results_file) as f:
        data = json.load(f)

    for key, lst in [
        ("pre_knn_f1", pre_knn_f1s),
        ("pre_linear_f1", pre_linear_f1s),
        ("post_knn_f1", post_knn_f1s),
        ("post_linear_f1", post_linear_f1s),
        ("post_sft_f1", post_sft_f1s),
    ]:
        val = data.get(key)
        if val is not None:
            lst.append(val)

    def fmt(v):
        return f"{v:.6f}" if v is not None else ""

    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},{i},"
                f"{fmt(data.get('pre_knn_f1'))},,{fmt(data.get('pre_linear_f1'))},,"
                f"{fmt(data.get('post_knn_f1'))},,{fmt(data.get('post_linear_f1'))},,"
                f"{fmt(data.get('post_sft_f1'))},\n")

def mean_std(vals):
    if len(vals) == 0:
        return "", ""
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.6f}", f"{s:.6f}"

if any(len(l) > 0 for l in [pre_knn_f1s, pre_linear_f1s, post_knn_f1s, post_linear_f1s, post_sft_f1s]):
    pk_m, pk_s = mean_std(pre_knn_f1s)
    pl_m, pl_s = mean_std(pre_linear_f1s)
    ok_m, ok_s = mean_std(post_knn_f1s)
    ol_m, ol_s = mean_std(post_linear_f1s)
    sf_m, sf_s = mean_std(post_sft_f1s)

    with open(csv_file, "a") as f:
        f.write(f"{backbone_tag},{display_name},{n_samples},{model_size},average,"
                f"{pk_m},{pk_s},{pl_m},{pl_s},{ok_m},{ok_s},{ol_m},{ol_s},{sf_m},{sf_s}\n")

    print(f"  [{backbone_tag}] {display_name} n={n_samples}: "
          f"pre_knn={pk_m}+-{pk_s} pre_lp={pl_m}+-{pl_s} "
          f"post_knn={ok_m}+-{ok_s} post_lp={ol_m}+-{ol_s} post_sft={sf_m}+-{sf_s}")
else:
    print(f"  [{backbone_tag}] {display_name} n={n_samples}: no results available")

PYEOF
}

# ============================================================
# Main loop
# ============================================================
echo ""
echo "=========================================="
echo "Starting DIET-CP: ${DISPLAY_NAME}"
echo "Backbone: ${BACKBONE_TAG} (${BACKBONE_TIMM})"
echo "n_samples: ${NSAMPLES[*]}"
echo "freeze_epochs=${FREEZE_EPOCHS} num_trained_blocks=${NUM_TRAINED_BLOCKS}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="
echo ""

CSV_FILE="${LOG_DIR}/${BACKBONE_TAG}_diet_cp_results.csv"
if ! echo "backbone,dataset,n_samples,model_size,run,pre_knn_f1,pre_knn_f1_std,pre_linear_f1,pre_linear_f1_std,post_knn_f1,post_knn_f1_std,post_linear_f1,post_linear_f1_std,post_sft_f1,post_sft_f1_std" > "${CSV_FILE}"; then
    echo "[ERROR] Unable to initialize CSV file: ${CSV_FILE}"
    exit 1
fi
echo "CSV file: ${CSV_FILE}"

TOTAL_SUCCESS=0
TOTAL_FAIL=0
TOTAL_AGG_FAIL=0

for n_samples in "${NSAMPLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Experiment: ${BACKBONE_TAG} | ${DISPLAY_NAME} | n_samples=${n_samples}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        run_single ${n_samples} ${seed}
        if [ $? -eq 0 ]; then
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done

    echo "--- Aggregating results for n=${n_samples} ---"
    if ! aggregate_results ${n_samples} ${CSV_FILE}; then
        echo "[FAIL] Aggregation failed for n=${n_samples}"
        TOTAL_AGG_FAIL=$((TOTAL_AGG_FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "All DIET-CP ${DISPLAY_NAME} ${BACKBONE_TAG} experiments completed!"
echo "  Successful: ${TOTAL_SUCCESS}"
echo "  Failed: ${TOTAL_FAIL}"
echo "  Aggregation failures: ${TOTAL_AGG_FAIL}"
echo "  Results: ${LOG_DIR}/"
echo "  End Time: $(date)"
echo "=========================================="

if [ ${TOTAL_FAIL} -ne 0 ] || [ ${TOTAL_AGG_FAIL} -ne 0 ]; then
    exit 1
fi
