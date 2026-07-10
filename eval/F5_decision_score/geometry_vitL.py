#!/usr/bin/env python
"""
geometry_vitL.py — R1 appetizer of the ViT-L scale-robustness check
(eval/DESIGN_vitL_robustness.md): pre-CP geometry of DINOv3 ViT-L/16 on all 15 datasets.

Zero training. One forward pass per dataset + ImageNet-val, same protocol as
geometry_metrics.py (<=5000 stratified subset, cls readout). At the end, if
eval/outputs/geometry_15.csv is present, prints the pre-registered R1 verdict:
Spearman rank stability of uniformity_t2 and neighbor_overlap_k50 between the
ViT-B DINOv3 rows and the new ViT-L rows (pass: both > 0.8).

Cluster:
  python eval/geometry_vitL.py --imagenet-dir <dir> --download-dir <raw> \
      --processed-dir <arrow> --output eval/outputs/geometry_vitL.csv
"""
import argparse
import csv
from pathlib import Path

import sys
from pathlib import Path as _P0
sys.path.insert(0, str(_P0(__file__).resolve().parent.parent))  # eval/ root for shared modules
sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / 'utils'))  # eval/utils shared modules

import numpy as np
import torch

from geometry_metrics import (TARGET_DATASETS, load_target_dataset, load_imagenet_val,
                              extract_features, l2_norm_stats, wang_isola_uniformity,
                              mmd_rbf_components, neighbor_overlap)

ROOT = Path(__file__).resolve().parent.parent
TIMM_ID = "vit_large_patch16_dinov3.lvd1689m"   # registered in stable_cp/utils/backbone.py
FIELDS = ["encoder", "dataset", "n_samples", "l2_norm_cv", "uniformity_t2",
          "neighbor_overlap_k50", "mmd_rbf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-dir", type=str, default=str(ROOT / "eval/data/imagenet_val"))
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/geometry_vitL.csv"))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import timm
    model = timm.create_model(TIMM_ID, pretrained=True, num_classes=0).eval().to(device)
    feat_in, _ = extract_features(model, load_imagenet_val(args.imagenet_dir,
                                                           args.imagenet_samples),
                                  device, "cls")

    rows = []
    for ds in args.datasets:
        print(f"--- {ds} ---")
        try:
            loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        feat, _ = extract_features(model, loader, device, "cls")
        _, _, cv = l2_norm_stats(feat)
        row = {"encoder": "DINOv3L", "dataset": ds, "n_samples": len(feat),
               "l2_norm_cv": round(cv, 5),
               "uniformity_t2": round(wang_isola_uniformity(feat, t=2.0), 5),
               "neighbor_overlap_k50": round(neighbor_overlap(feat, feat_in, k=50), 5),
               "mmd_rbf": round(mmd_rbf_components(feat, feat_in)["mmd_rbf"], 5)}
        print(f"  unif={row['uniformity_t2']} ov={row['neighbor_overlap_k50']} cv={row['l2_norm_cv']}")
        rows.append(row)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.output}")

    # ---- R1 verdict (pre-registered): rank stability vs ViT-B DINOv3 ----------------
    g15 = ROOT / "eval/outputs/geometry_15.csv"
    if g15.exists() and rows:
        from scipy.stats import spearmanr
        import pandas as pd
        b = pd.read_csv(g15)
        b = b[(b.encoder == "DINOv3") & (b.dataset != "imagenet")]
        m = b.merge(pd.DataFrame(rows), on="dataset", suffixes=("_B", "_L"))
        ru = spearmanr(m.uniformity_t2_B, m.uniformity_t2_L).correlation
        ro = spearmanr(m.neighbor_overlap_k50_B, m.neighbor_overlap_k50_L).correlation
        rm = spearmanr(m.mmd_rbf_B, m.mmd_rbf_L).correlation
        print(f"\nR1 rank stability ViT-B vs ViT-L (n={len(m)} datasets):")
        print(f"  uniformity_t2 rho = {ru:+.3f}   overlap_k50 rho = {ro:+.3f}   mmd rho = {rm:+.3f}")
        print(f"  R1 verdict: {'PASS' if (ru > 0.8 and ro > 0.8) else 'FAIL'} "
              f"(pre-registered: both uniformity and overlap > 0.8)")
    else:
        print("geometry_15.csv not found or no rows — run the R1 comparison on the login node.")


if __name__ == "__main__":
    main()
