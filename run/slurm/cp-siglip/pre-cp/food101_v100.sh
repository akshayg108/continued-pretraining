#!/bin/bash
#SBATCH --job-name=siglip-pre-food
#SBATCH --array=0-2%3
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-pre-food-%A_%a.out
#SBATCH --error=/scratch/gs4133/zhd/CP/outputs/slurm-log/siglip-pre-food-%A_%a.err

set -eo pipefail

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SEED=42 ;;
    1) SEED=43 ;;
    2) SEED=44 ;;
    *) echo "Expected array task 0, 1, or 2" >&2; exit 2 ;;
esac

REPO_ROOT="${SIGLIP_PRE_REPO_ROOT:-/scratch/gs4133/zhd/CP/continued-pretraining}"
DATA_DIR="${SIGLIP_PRE_CACHE_DIR:-/scratch/gs4133/zhd/CP/data}"
OUTPUT_BASE="${SIGLIP_PRE_OUTPUT_BASE:-/scratch/gs4133/zhd/CP/outputs}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}"
OUT="${OUTPUT_BASE}/siglip_food101_precheck_v1/${RUN_ID}/food101"

printf 'PRE_CP_ONLY dataset=food101 seed=%s initialization=public_pretrained evaluators=knn,pytorch_lp gpu=v100 output=%s/seed%s.json\n' \
    "$SEED" "$OUT" "$SEED"

if [ "$#" -eq 1 ] && [ "$1" = --dry-run ]; then
    exit 0
elif [ "$#" -ne 0 ]; then
    echo "Usage: sbatch food101_v100.sh, or bash food101_v100.sh --dry-run" >&2
    exit 2
fi
[ -n "${SLURM_JOB_ID:-}" ] || { echo "Submit this script with sbatch" >&2; exit 2; }

if [ "${SIGLIP_PRE_SKIP_ENV_SETUP:-0}" != 1 ]; then
    module load miniconda/3-4.11.0
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate env
fi
set -u

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:$(dirname "$PWD"):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import torch; p = torch.cuda.get_device_properties(0); assert torch.cuda.device_count() == 1 and "V100" in p.name, p; print(p)'

[ ! -e "$OUT/seed${SEED}.json" ] || { echo "Refusing to overwrite an existing result" >&2; exit 3; }
SOURCE="$DATA_DIR/stable_datasets/processed/food101"
[ -d "$SOURCE" ] || { echo "Missing processed Food-101 cache: $SOURCE" >&2; exit 4; }
SOURCE_KB=$(du -sk "$SOURCE" | awk '{print $1}')
NEED_KB=$((SOURCE_KB + 5 * 1024 * 1024))
LOCAL_CACHE=""
for ROOT in "${TMPDIR:-}" /tmpdata /dev/shm; do
    [ -n "$ROOT" ] && [ -d "$ROOT" ] && [ -w "$ROOT" ] || continue
    FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
    if [ "${FREE_KB:-0}" -ge "$NEED_KB" ]; then
        LOCAL_CACHE=$(mktemp -d "$ROOT/siglip-pre-food-${SLURM_JOB_ID}-${SEED}.XXXXXX")
        break
    fi
done
[ -n "$LOCAL_CACHE" ] || { echo "Insufficient node-local storage" >&2; exit 5; }
trap 'rm -rf -- "$LOCAL_CACHE"' EXIT
DEST="$LOCAL_CACHE/stable_datasets/processed/food101"
mkdir -p "$DEST" "$OUT"
rsync -a "$SOURCE/" "$DEST/"

"$PYTHON" - "$LOCAL_CACHE" "$OUT" "$SEED" <<'PY'
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import lightning as pl
import numpy as np
import torch

from continued_pretraining import _create_shared_eval_data, get_dataset_config, load_backbone
from stable_cp.evaluation.zero_shot_eval import (
    extract_features,
    knn_evaluate,
    linear_probe_pytorch_evaluate,
)

cache_dir, output_dir, seed = sys.argv[1:]
args = SimpleNamespace(
    dataset="food101", backbone="vit_base_patch16_siglip_224.v2_webli",
    n_samples=75750, batch_size=64, num_workers=8, seed=int(seed),
    cache_dir=cache_dir, pool_strategy="map",
)
pl.seed_everything(args.seed, workers=True)
ds_cfg = get_dataset_config(args.dataset)
backbone, device = load_backbone(args, img_size=ds_cfg["input_size"], pretrained=True)
if device.type != "cuda":
    raise RuntimeError("This baseline audit requires the allocated GPU.")
backbone.requires_grad_(False)
backbone.eval()
backbone.to(device)

# Reuse the exact post-CP loaders: augmented LP train, clean kNN train and test.
eval_tf, test_loader, lp_loader, knn_loader, indices = _create_shared_eval_data(
    args, ds_cfg, Path(cache_dir)
)
if len(indices) != 75750 or len(test_loader.dataset) != 25250:
    raise ValueError("Food-101 must contain 75,750 train and 25,250 test images.")

train_features, train_labels = extract_features(
    backbone, lp_loader, device, pool_strategy="map", verbose=True
)
test_features, test_labels = extract_features(
    backbone, test_loader, device, pool_strategy="map", verbose=True
)
knn_features, knn_labels = extract_features(
    backbone, knn_loader, device, pool_strategy="map", verbose=True
)
for name, features, labels, expected in (
    ("lp_train", train_features, train_labels, 75750),
    ("knn_train", knn_features, knn_labels, 75750),
    ("test", test_features, test_labels, 25250),
):
    if features.shape != (expected, 768) or len(labels) != expected:
        raise ValueError(f"Unexpected feature or label shape for {name}.")
    if not np.isfinite(features).all() or len(np.unique(labels)) != 101:
        raise ValueError(f"Non-finite features or missing classes in {name}.")
if not np.array_equal(train_labels, knn_labels):
    raise ValueError("LP and kNN train labels must use the same ordering.")

print("Running k-NN evaluation...", flush=True)
knn = knn_evaluate(knn_features, knn_labels, test_features, test_labels, k=20)
print("Running linear probe evaluation (PyTorch)...", flush=True)
lp = linear_probe_pytorch_evaluate(
    train_features, train_labels, test_features, test_labels,
    device=device, lr=1e-3, min_epochs=150, min_steps=10000, batch_size=512,
    verbose=True,
)
metrics = dict(pre_knn_f1=float(knn["knn_f1"]), pre_knn_acc=float(knn["knn_acc"]),
               pre_linear_f1=float(lp["linear_pytorch_f1"]),
               pre_linear_acc=float(lp["linear_pytorch_acc"]))
if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
    raise ValueError("All baseline metrics must be finite fractions.")

versions = {}
for package in ("torch", "timm", "lightning", "stable-pretraining", "stable-datasets",
                "scikit-learn", "torchmetrics"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "unknown"
sources = [Path("continued_pretraining.py"), Path(__import__("stable_cp").__file__).parent]
source_files = [sources[0], *sorted(sources[1].rglob("*.py"))]
code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
result = dict(
    protocol="siglip_food101_precheck_v1", status="complete",
    dataset=args.dataset, n_samples=len(indices), n_test=len(test_labels),
    backbone=args.backbone, initialization="public_pretrained", seed=args.seed,
    no_cp=True, no_ft=True, pool_strategy=args.pool_strategy,
    feature_batch_size=args.batch_size, feature_precision="float32",
    knn_k=20, lp_method="pytorch", lp_lr=1e-3, lp_min_epochs=150,
    lp_min_steps=10000, lp_batch_size=512,
    normalization=ds_cfg["normalization"], splits=ds_cfg["splits"],
    lp_train_transform=repr(lp_loader.dataset.dataset.transform),
    eval_transform=repr(eval_tf),
    train_indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
    pretrained_config=getattr(backbone, "pretrained_cfg", {}),
    gpu=torch.cuda.get_device_name(0), versions=versions,
    git_commit=commit.stdout.strip() if commit.returncode == 0 else "unknown",
    code_sha256=code_hashes, **metrics,
)
path = Path(output_dir) / f"seed{args.seed}.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False, default=str)
print(json.dumps(dict(seed=args.seed, **metrics)), flush=True)
print(f"Results saved to {path}", flush=True)
PY
