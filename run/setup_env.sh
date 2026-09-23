#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CP_ROOT="${1:-$(dirname -- "$REPO")}"
source "$REPO/run/precp_env.sh"

if [[ ! -d "$CP_ROOT/env" ]]; then
    conda create --prefix "$CP_ROOT/env" --override-channels \
        -c conda-forge python=3.11 pip -y
fi

"$CP_PYTHON" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit("The CP environment must use Python 3.11.")
PY

"$CP_PYTHON" -m pip install --upgrade pip setuptools wheel
"$CP_PYTHON" -m pip install \
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# Keep the CUDA pair fixed while resolving all project dependencies.
"$CP_PYTHON" -m pip install -e "$REPO" \
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"
"$CP_PYTHON" -m pip check
"$CP_PYTHON" - <<'PY'
import sys
from importlib.metadata import version

import torch
import torchvision
from stable_datasets import images

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"Torchvision: {torchvision.__version__}")
for package in ("stable-cp", "stable-pretraining", "stable-datasets", "timm", "lightning"):
    print(f"{package}: {version(package)}")
for name in ("MedMNIST", "AID", "RESISC45", "StanfordDogs", "JenaFlowers30", "Flavia", "IP102"):
    getattr(images, name)
print("Dataset readers imported successfully.")
PY

mkdir -p "$CP_ROOT/outputs/environment"
"$CP_PYTHON" -m pip freeze > "$CP_ROOT/outputs/environment/pip-freeze.txt"
printf 'Environment ready: %s\n' "$CP_ROOT/env"
printf 'Package snapshot: %s\n' "$CP_ROOT/outputs/environment/pip-freeze.txt"
