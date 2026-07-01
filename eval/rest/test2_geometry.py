#!/usr/bin/env python
"""Test 2 — recompute post-CP geometry for the 60 re-trained checkpoints (+ Δ vs pre-CP).

For each checkpoint in rerun_geometry.csv: load the post-CP backbone, extract features on its
target dataset, and compute the same metrics as the production sweep:
  l2_norm_cv, uniformity_t2, and (if --imagenet-dir) neighbor_overlap_k50.
Joins pre-CP values from geometry_15.csv to report d_cv / d_unif / d_overlap.

Reuses the production loaders/metrics (postcp_features.load_cp_backbone, geometry_metrics.*), so
the numbers are directly comparable to postcp_sweep.csv. GPU.

NOTE: overlap re-embeds ImageNet with EACH post-CP encoder (expensive) — only with --imagenet-dir.

Run:
  python eval/rest/test2_geometry.py --download-dir <raw> --processed-dir <arrow> \
      [--imagenet-dir <imagenet_val>] [--geometry15 eval/outputs/geometry_15.csv] \
      [--out eval/outputs/rest_geometry.csv]
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # put eval/ on the import path

import torch
from postcp_features import load_cp_backbone
from geometry_metrics import (
    load_target_dataset, extract_features, l2_norm_stats,
    wang_isola_uniformity, neighbor_overlap, load_imagenet_val,
)

ENC = {
    "dinov3": ("DINOv3", "vit_base_patch16_dinov3.lvd1689m", "cls"),
    "clip":   ("CLIP",   "vit_base_patch16_clip_224.openai", "cls"),
    "224.mae": ("MAE",   "vit_base_patch16_224.mae",         "mean"),
}


def enc_of(fn):
    for key, v in ENC.items():
        if key in fn:
            return v
    return ("?", None, "cls")


def fmt(x):
    return "" if x is None else round(float(x), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/outputs/rerun_geometry.csv")
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--imagenet-dir", default=None)
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--geometry15", default="eval/outputs/geometry_15.csv")
    ap.add_argument("--out", default="eval/outputs/rest_geometry.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # pre-CP geometry lookup: (encoder, dataset) -> (unif, overlap, cv)
    pre = {}
    for r in csv.DictReader(open(args.geometry15)):
        ov = r.get("neighbor_overlap_k50") or ""
        pre[(r["encoder"], r["dataset"])] = (
            float(r["uniformity_t2"]) if r.get("uniformity_t2") else None,
            float(ov) if ov else None,
            float(r["l2_norm_cv"]) if r.get("l2_norm_cv") else None,
        )

    imn_loader = load_imagenet_val(args.imagenet_dir, args.imagenet_samples) if args.imagenet_dir else None

    rows = [r for r in csv.reader(open(args.csv)) if r and r[0].strip() not in ("", "ckpt")]
    FIELDS = ["method", "encoder", "dataset", "size", "seed",
              "l2_norm_cv", "uniformity_t2", "neighbor_overlap_k50",
              "d_cv", "d_unif", "d_overlap", "ckpt"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    w.writeheader()

    for i, r in enumerate(rows):
        ck = r[0].strip()
        fn = os.path.basename(ck)
        m = re.match(r"(.+?)_(vit_base_patch16_[A-Za-z0-9_.]+)_n(\d+)_s(\d+)\.ckpt$", fn)
        ds, _, n, seed = m.groups()
        tag, timm_id, pool = enc_of(fn)
        parts = ck.split("/")
        method = parts[parts.index("cp") + 1]
        try:
            model = load_cp_backbone(ck, timm_id, dev)
            feat, _ = extract_features(model, load_target_dataset(ds, args.download_dir, args.processed_dir), dev, pool)
            _, _, cv = l2_norm_stats(feat)
            unif = wang_isola_uniformity(feat, t=2.0, l2_normalize=True)
            overlap = None
            if imn_loader is not None:
                feat_imn, _ = extract_features(model, imn_loader, dev, pool)  # post-CP ImageNet feats
                overlap = neighbor_overlap(feat, feat_imn, k=50)
            pu, po, pc = pre.get((tag, ds), (None, None, None))
            row = dict(method=method, encoder=tag, dataset=ds, size=n, seed=seed,
                       l2_norm_cv=fmt(cv), uniformity_t2=fmt(unif), neighbor_overlap_k50=fmt(overlap),
                       d_cv=fmt(cv - pc) if pc is not None else "",
                       d_unif=fmt(unif - pu) if pu is not None else "",
                       d_overlap=fmt(overlap - po) if (overlap is not None and po is not None) else "",
                       ckpt=ck)
            w.writerow(row)
            fout.flush()
            print(f"[{i+1}/{len(rows)}] {method}+{tag}+{ds}+n{n}+s{seed}: "
                  f"cv={fmt(cv)} unif={fmt(unif)} d_unif={row['d_unif']} overlap={fmt(overlap)}")
        except Exception as e:
            print(f"  FAIL {fn}: {type(e).__name__}: {e}")

    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
