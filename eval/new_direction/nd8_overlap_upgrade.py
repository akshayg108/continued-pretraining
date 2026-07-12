#!/usr/bin/env python
"""
nd8_overlap_upgrade.py — ND8 (GPU pass): the overlap-feature upgrade kit on PRE-CP features.

Round-3 sourcing (papers/new_direction/NEW_DIRECTION_R3.md): the raw-cosine k=50 overlap
operates in a hubness-distorted regime (our own N_10 skewness > 1.4 in 37/60 cells) and
carries a verified centroid-centrality artifact. This pass recomputes the position feature
under four corrected protocols plus one replacement score, per (encoder, dataset):

  overlap_raw       exact reproduction of geometry_metrics.neighbor_overlap (k=50, cosine,
                    target subsample 2000 seed-42, bank = ImageNet-val 5000) — validated
                    against geometry_15.csv in nd8_verdict (ND8-0)
  overlap_centered  same, after subtracting the JOINT centroid (kills the verified
                    centrality component)
  overlap_mp        same, neighbour lists re-ranked by Mutual Proximity (empiric)
  overlap_nicdm     same, neighbour lists re-ranked by NICDM local scaling (k_local=10)
  sun_knn_*         Sun et al. 2022 position score: mean distance to the k-th nearest
                    bank point on L2-normalized features (full bank k=50; small bank
                    500 pts k=5 — the cheap-bank variant)

Diagnostics per cell: k-occurrence skewness of the target-query lists under each protocol
(corrections must reduce it — falsifiable), and hub_centrality_rho = Spearman between
bank-point occurrence and distance-to-joint-centroid (mechanism check: expected negative).

Encoders: the 4 ViT-B (geometry_metrics.ENCODERS) + DINOv3L (for the ViT-L holdout of the
tool-v2 test in nd8_verdict). Algorithms in nd8_position_metrics.py (TDD-tested).

Cluster (one array task per dataset):
  python eval/new_direction/nd8_overlap_upgrade.py --datasets <ds> \
      --imagenet-dir <dir> --download-dir <raw> --processed-dir <arrow> \
      --output eval/outputs/nd8_overlap_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import skew, spearmanr

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                    # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))   # eval/utils

from nd8_position_metrics import (cosine_dist_matrix, knn_indices, mp_empiric_rerank,
                                  nicdm_rerank, center_features, overlap_score,
                                  sun_knn_distance, k_occurrence)

ROOT = Path(__file__).resolve().parent.parent.parent
K_OVERLAP = 50
TARGET_SUB = 2000          # matches geometry_metrics.neighbor_overlap sampling
BANK_N = 5000
SMALL_BANK, SMALL_K = 500, 5
VARIANTS = ["raw", "centered", "mp", "nicdm"]
FIELDS = (["encoder", "dataset", "n_target", "n_bank"]
          + [f"overlap_{v}" for v in VARIANTS]
          + [f"skew_{v}" for v in VARIANTS]
          + ["sun_knn_full", "sun_knn_small", "hub_centrality_rho"])


def position_row(feat_target, feat_bank):
    """All ND8 position statistics for one (encoder, dataset) cell."""
    rng = np.random.RandomState(42)                       # geometry_metrics convention
    ft = feat_target
    if len(ft) > TARGET_SUB:
        ft = ft[rng.choice(len(ft), TARGET_SUB, replace=False)]
    fb = feat_bank
    if len(fb) > BANK_N:
        fb = fb[rng.choice(len(fb), BANK_N, replace=False)]
    combined = np.vstack([ft, fb])
    is_bank = np.zeros(len(combined), bool)
    is_bank[len(ft):] = True
    tgt_rows = np.arange(len(ft))

    out = {"n_target": len(ft), "n_bank": len(fb)}
    D = cosine_dist_matrix(combined)
    nn = {"raw": knn_indices(D, K_OVERLAP)[tgt_rows],
          "centered": knn_indices(cosine_dist_matrix(center_features(combined)),
                                  K_OVERLAP)[tgt_rows],
          "mp": mp_empiric_rerank(D, K_OVERLAP, n_candidates=200, rows=tgt_rows),
          "nicdm": nicdm_rerank(D, K_OVERLAP, k_local=10)[tgt_rows]}
    for v in VARIANTS:
        out[f"overlap_{v}"] = overlap_score(nn[v], is_bank)
        out[f"skew_{v}"] = float(skew(k_occurrence(nn[v], n=len(combined))))

    # mechanism check: are the bank points that dominate raw neighbour lists the
    # centroid-proximal ones? (expected rho < 0)
    occ = k_occurrence(nn["raw"], n=len(combined))
    cent_dist = np.linalg.norm(combined - combined.mean(axis=0), axis=1)
    out["hub_centrality_rho"] = float(
        spearmanr(occ[is_bank], cent_dist[is_bank]).correlation)

    out["sun_knn_full"] = float(sun_knn_distance(feat_target, fb, k=K_OVERLAP).mean())
    small = fb[np.random.RandomState(42).choice(len(fb), SMALL_BANK, replace=False)]
    out["sun_knn_small"] = float(sun_knn_distance(feat_target, small, k=SMALL_K).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-dir", type=str, default=str(ROOT / "eval/data/imagenet_val"))
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/nd8_overlap.csv"))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    import timm
    import torch
    from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                                  load_imagenet_val, extract_features)
    encoders_nd8 = dict(ENCODERS)
    encoders_nd8["DINOv3L"] = {"timm_id": "vit_large_patch16_dinov3.lvd1689m",
                               "pool": "cls"}   # for the ViT-L holdout in nd8_verdict
    encoders = args.encoders or list(encoders_nd8.keys())
    datasets = args.datasets or TARGET_DATASETS
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["encoder"], r["dataset"]) for r in csv.DictReader(f)}
        print(f"Resume: {len(done)} rows present")

    # st_size check: a run killed before its first flush leaves a 0-byte shard that
    # exists() alone would treat as already-headered (audit 2026-07-12)
    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        for enc in encoders:
            todo = [d for d in datasets if (enc, d) not in done]
            if not todo:
                continue
            cfg = encoders_nd8[enc]
            print(f"\n===== {enc} ({cfg['timm_id']}) — {len(todo)} datasets")
            model = timm.create_model(cfg["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            bank_loader = load_imagenet_val(args.imagenet_dir, n_samples=BANK_N)
            bank, _ = extract_features(model, bank_loader, device, cfg["pool"])
            print(f"  bank: {bank.shape}")
            for ds in todo:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
                feat, _ = extract_features(model, loader, device, cfg["pool"])
                row = {"encoder": enc, "dataset": ds}
                row.update(position_row(feat, bank))
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                print(f"  {ds:>14}: raw={row['overlap_raw']:.4f} "
                      f"cen={row['overlap_centered']:.4f} mp={row['overlap_mp']:.4f} "
                      f"nicdm={row['overlap_nicdm']:.4f} "
                      f"skew {row['skew_raw']:.2f}->{row['skew_mp']:.2f}(mp) "
                      f"centrality={row['hub_centrality_rho']:+.2f}")
            del model, bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nNext (local): python eval/new_direction/nd8_verdict.py")


if __name__ == "__main__":
    main()
