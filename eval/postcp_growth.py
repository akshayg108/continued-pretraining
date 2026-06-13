#!/usr/bin/env python
"""
postcp_growth.py — Exp C (F3): post-CP geometry vs CP data size.

For each (config, size) post-CP checkpoint, measure how the target point-cloud sits on the
sphere AFTER CP:
  - post_uniformity (intrinsic, no ImageNet): does the cloud SPREAD as CP data grows? (P3.1)
  - post_overlap_k50 (needs ImageNet): does the expanding cloud COLLIDE with ImageNet's region?
    (P3.2/P3.3). IMPORTANT: overlap is computed in the SAME post-CP encoder space — ImageNet is
    re-embedded through each post-CP checkpoint, since the encoder weights change during CP.

Input manifest `--manifest` columns: method,encoder,dataset,size,ckpt (size-resolved schedule
for ~3 configs, e.g. LeJEPA-CP+DINOv3+{galaxy10,octmnist,organamnist} at {100,1000,10000,MAX}).
Overlay the resulting trajectories on the existing Δ-vs-size curves (delta_structure.py).

Run (Colab):
  python eval/postcp_growth.py --manifest eval/postcp_growth_manifest.csv \
      --imagenet-dir /content/drive/MyDrive/CP/data/imagenet_val \
      --download-dir /content/drive/MyDrive/CP/data/downloads \
      --processed-dir /content/drive/MyDrive/CP/data/processed \
      --out eval/outputs/postcp_growth.csv
Use --no-overlap to skip the (expensive) per-checkpoint ImageNet pass and get uniformity only.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from postcp_features import load_cp_backbone, wang_isola_uniformity, neighbor_overlap
from geometry_metrics import ENCODERS, load_target_dataset, extract_features, load_imagenet_val

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--imagenet-dir", default=None)
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/postcp_growth.csv"))
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    do_overlap = (not args.no_overlap) and args.imagenet_dir is not None

    man = pd.read_csv(args.manifest)
    imn_loader = load_imagenet_val(args.imagenet_dir, args.imagenet_samples) if do_overlap else None

    rows = []
    for r in man.itertuples():
        if not isinstance(r.ckpt, str) or not r.ckpt.strip():
            print(f"SKIP {r.method}+{r.encoder}+{r.dataset}+{r.size}: empty ckpt"); continue
        cfg = ENCODERS[r.encoder]
        print(f"\n=== {r.method}+{r.encoder}+{r.dataset}+{r.size} ===\n  ckpt={r.ckpt}")
        model = load_cp_backbone(r.ckpt, cfg["timm_id"], device)
        tgt_loader = load_target_dataset(r.dataset, args.download_dir, args.processed_dir)
        feat_t, _ = extract_features(model, tgt_loader, device, cfg["pool"])
        unif = wang_isola_uniformity(feat_t, t=2.0, l2_normalize=True)
        overlap = ""
        if do_overlap:
            feat_i, _ = extract_features(model, imn_loader, device, cfg["pool"])
            overlap = round(neighbor_overlap(feat_t, feat_i, k=50), 4)
        rows.append(dict(method=r.method, encoder=r.encoder, dataset=r.dataset, size=r.size,
                         post_uniformity_t2=round(unif, 4), post_overlap_k50=overlap))
        print(f"  post_uniformity={unif:.4f}  post_overlap_k50={overlap}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nsaved {args.out}")
    print("P3.1 post_uniformity decreases (spreads) with size; P3.2 ΔkNN peak precedes overlap rise;")
    print("P3.3 OrganAMNIST zero-crossing coincides with overlap crossing a threshold.")


if __name__ == "__main__":
    main()
