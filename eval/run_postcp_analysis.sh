#!/bin/bash
# CPU-only post-CP analysis (NO GPU, NO SLURM). Run on a login node:
#   bash eval/run_postcp_analysis.sh
# Steps:
#   [1] merge the GPU sweep's shard CSVs   -> eval/outputs/postcp_sweep.csv
#   [2] Exp A: off-sphere mechanism Δcv→Δknn -> eval/outputs/postcp_offsphere.csv + console
#   [3] Exp C: growth dynamics (spread + collision vs CP size) -> eval/outputs/postcp_growth_analysis.csv + console

# activate the env if available (harmless if run outside the cluster)
module load miniconda/3-4.11.0 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate env 2>/dev/null || true

set -e
cd "$(dirname "$0")/.."          # repo root (parent of eval/)
echo "workdir: $(pwd)   python: $(which python)"

echo ""
echo "=== [1/3] merge sweep shards -> eval/outputs/postcp_sweep.csv ==="
python - <<'PY'
import pandas as pd, glob, os
fs = sorted(glob.glob("eval/outputs/postcp_sweep_[0-9]*.csv"))
if fs:
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("ckpt")
    df.to_csv("eval/outputs/postcp_sweep.csv", index=False)
    print(len(df), "rows merged from", len(fs), "shards")
elif os.path.exists("eval/outputs/postcp_sweep.csv"):
    print("no shard CSVs; using existing eval/outputs/postcp_sweep.csv")
else:
    raise SystemExit("ERROR: no shards (eval/outputs/postcp_sweep_*.csv) and no merged postcp_sweep.csv")
PY

echo ""
echo "=== [2/3] Exp A: off-sphere mechanism (Δcv -> Δknn) ==="
python eval/f2_mechanism/postcp_offsphere.py --sweep eval/outputs/postcp_sweep.csv --geometry eval/outputs/geometry_15.csv

echo ""
echo "=== [3/3] Exp C: growth dynamics (uniformity spread + overlap collision vs CP size) ==="
python eval/f3_growth/postcp_growth_analysis.py --sweep eval/outputs/postcp_sweep.csv --geometry eval/outputs/geometry_15.csv

echo ""
echo "=== done. outputs: eval/outputs/{postcp_sweep,postcp_offsphere,postcp_growth_analysis}.csv ==="
