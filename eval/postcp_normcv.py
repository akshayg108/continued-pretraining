#!/usr/bin/env python
"""
postcp_normcv.py — Exp A (F2.2): post-CP L2-norm CV per config.

Prediction: MAE-CP raises the L2-norm CV of DINOv3/CLIP (features leave the sphere);
invariance-CP (LeJEPA/SimCLR) keeps CV ~constant. This is the direct test of the
"reconstruction CP pushes a sphere encoder off the sphere" mechanism of Finding 2.

Input: a manifest CSV `--manifest` with columns: method,encoder,dataset,size,ckpt
(ckpt = path to the post-CP checkpoint). See eval/postcp_manifest_template.csv.
Pre-CP CV is read from the geometry recompute (geometry_15.csv).

Run (Colab):
  python eval/postcp_normcv.py --manifest eval/postcp_manifest.csv \
      --geometry eval/outputs/geometry_15.csv \
      --download-dir /content/drive/MyDrive/CP/data/downloads \
      --processed-dir /content/drive/MyDrive/CP/data/processed \
      --out eval/outputs/postcp_normcv.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from postcp_features import extract_postcp, l2_norm_stats

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--geometry", default=str(ROOT / "eval/outputs/geometry_15.csv"))
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/postcp_normcv.csv"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    man = pd.read_csv(args.manifest)
    geom = pd.read_csv(args.geometry)
    pre_cv = {(r.encoder, r.dataset): r.l2_norm_cv
              for r in geom[geom.dataset != "imagenet"].itertuples()}

    rows = []
    for r in man.itertuples():
        if not isinstance(r.ckpt, str) or not r.ckpt.strip():
            print(f"SKIP {r.method}+{r.encoder}+{r.dataset}+{r.size}: empty ckpt path")
            continue
        print(f"\n=== {r.method}+{r.encoder}+{r.dataset}+{r.size} ===\n  ckpt={r.ckpt}")
        feat, _ = extract_postcp(r.ckpt, r.encoder, r.dataset,
                                 args.download_dir, args.processed_dir, device)
        _, _, post_cv = l2_norm_stats(feat)
        pcv = pre_cv.get((r.encoder, r.dataset), np.nan)
        rows.append(dict(method=r.method, encoder=r.encoder, dataset=r.dataset, size=r.size,
                         pre_cv=round(float(pcv), 4), post_cv=round(post_cv, 4),
                         delta_cv=round(post_cv - float(pcv), 4)))
        print(f"  pre_cv={pcv:.4f}  post_cv={post_cv:.4f}  Δcv={post_cv-float(pcv):+.4f}")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nsaved {args.out}")
    print("Prediction P2.2: Δcv >> 0 for MAE-CP on DINOv3/CLIP; Δcv ≈ 0 for LeJEPA/SimCLR-CP.")


if __name__ == "__main__":
    main()
