#!/usr/bin/env python
"""
geometry_class.py — SECOND-AXIS pre-CP geometry: class-anchored + spectral measures.

Complements geometry_metrics.py (which is class-agnostic: uniformity/overlap/MMD = WHERE the
cloud sits). This script measures HOW the occupied region is organized — the candidate second
axes for the phenomena pure position cannot resolve (within-FG heterogeneity, DTD, MAE
attribution). Labels are used ONLY for measurement (same status as kNN eval), never training.

Per (encoder, dataset), on the SAME <=5000 stratified train subset as geometry_metrics.py:

  Class-manifold organization (P-A, granularity load):
    n_classes                  number of classes present in the subset
    within_spread              mean within-class pairwise cosine distance (L2-normed feats)
    between_spread             mean between-class pairwise cosine distance
    wb_ratio                   within_spread / between_spread  (wide manifolds vs spacing)
    nc1_ratio                  tr(S_W)/tr(S_B) Fisher-style (normalized feats)
    center_margin              mean over classes of nearest-other-class-center cosine distance

  Task-spectral alignment (P-B, DTD mechanism):
    task_energy_top{10,50}     fraction of between-class scatter energy in the top-k PCs of
                               the TARGET cloud (low = class signal lives in the spectral tail)
    task_energy_in_top{10,50}  same, projected on the top-k PCs of the IMAGENET cloud
                               (low = task orthogonal to the encoder's dominant object axes)

  Angular anisotropy / dimensionality (P-C1, gate continuum):
    rankme                     RankMe effective rank, exp(entropy of normalized singular
                               values), on L2-NORMALIZED features (angular anisotropy,
                               complements radial l2_norm_cv)
    rankme_raw                 same on raw features
    alpha_req                  spectral-decay exponent: slope of log lambda_i vs log i over
                               i in [11, d/2] (cov eigenspectrum; larger = faster decay)
    twonn_id                   TwoNN intrinsic dimension (Facco et al.) on normalized feats

Falsifiable predictions this feeds (tested by eval/adjudicate/correlate_second_axis.py):
  P-A  among embedded/FG datasets, the position-residual of dknn is more negative where
       wb_ratio is higher (CUB > Flowers on BOTH sphere encoders), cross-encoder consistent.
  P-B  DTD has LOWER task_energy_in_top-k than the object FG datasets on BOTH sphere encoders.
  P-C1 per-encoder F1 law strength orders inversely with encoder anisotropy
       (rankme: CLIP/DINOv3/SigLIP high, MAE low).

GPU (one forward pass per dataset per encoder + ImageNet per encoder — same cost profile as
geometry_metrics.py). Run on the cluster:
  python eval/geometry_class.py --imagenet-dir <dir> --download-dir <raw> \
      --processed-dir <arrow> --output eval/outputs/geometry_class_15.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                              load_imagenet_val, extract_features)

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.RandomState(42)
MAX_PAIRS_PER_CLASS = 200
MAX_BETWEEN_PAIRS = 20000


def _pairwise_cosdist(a, b):
    return 1.0 - a @ b.T


def class_manifold_stats(feats, labels):
    f = normalize(feats)
    classes = np.unique(labels)
    within, sw = [], 0.0
    mu_global = f.mean(0)
    centers, counts = [], []
    for c in classes:
        fc = f[labels == c]
        counts.append(len(fc))
        centers.append(fc.mean(0))
        if len(fc) >= 2:
            n_pairs = min(MAX_PAIRS_PER_CLASS, len(fc) * (len(fc) - 1) // 2)
            i = RNG.randint(0, len(fc), n_pairs)
            j = RNG.randint(0, len(fc), n_pairs)
            keep = i != j
            within.append((1.0 - (fc[i[keep]] * fc[j[keep]]).sum(1)))
        sw += ((fc - fc.mean(0)) ** 2).sum()
    within_spread = float(np.concatenate(within).mean()) if within else np.nan
    i = RNG.randint(0, len(f), MAX_BETWEEN_PAIRS)
    j = RNG.randint(0, len(f), MAX_BETWEEN_PAIRS)
    keep = labels[i] != labels[j]
    between_spread = float((1.0 - (f[i[keep]] * f[j[keep]]).sum(1)).mean())
    centers = np.stack(centers)
    counts = np.asarray(counts, float)
    sb = float((counts[:, None] * (centers - mu_global) ** 2).sum())
    cd = _pairwise_cosdist(normalize(centers), normalize(centers))
    np.fill_diagonal(cd, np.inf)
    center_margin = float(cd.min(1).mean())
    # CDNV (Galanti et al., ICLR 2022): V(Q1,Q2) = (Var1+Var2) / (2 ||mu1-mu2||^2),
    # averaged over class pairs (lower = more collapsed / better few-shot per their bound).
    variances = np.array([float(((f[labels == c] - centers[k]) ** 2).sum(1).mean())
                          for k, c in enumerate(classes)])
    dd = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(classes), k=1)
    cdnv = float(np.mean((variances[iu[0]] + variances[iu[1]]) / (2 * dd[iu] + 1e-12)))
    return {"n_classes": int(len(classes)), "within_spread": within_spread,
            "between_spread": between_spread,
            "wb_ratio": within_spread / between_spread if between_spread else np.nan,
            "nc1_ratio": sw / sb if sb else np.nan, "center_margin": center_margin,
            "cdnv": cdnv}


def between_class_scatter(f, labels):
    mu = f.mean(0)
    sb = np.zeros((f.shape[1], f.shape[1]))
    for c in np.unique(labels):
        fc = f[labels == c]
        d = (fc.mean(0) - mu)[:, None]
        sb += len(fc) * (d @ d.T)
    return sb / len(f)


def task_energy(feats, labels, basis_feats, ks=(10, 50)):
    """Fraction of between-class scatter energy inside the top-k PCs of basis_feats."""
    f = normalize(feats)
    b = normalize(basis_feats)
    sb = between_class_scatter(f, labels)
    tr_sb = np.trace(sb)
    bc = b - b.mean(0)
    _, _, vt = np.linalg.svd(bc, full_matrices=False)
    out = {}
    for k in ks:
        pk = vt[:k].T                                # (d, k)
        out[k] = float(np.trace(pk.T @ sb @ pk) / tr_sb) if tr_sb > 0 else np.nan
    return out


def rankme(feats):
    s = np.linalg.svd(feats - feats.mean(0), compute_uv=False)
    p = s / (s.sum() + 1e-12) + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def numerical_rank(feats):
    """Tunnel-effect numerical rank (Masarczyk et al. 2023): #singular values > sigma_1*1e-3."""
    s = np.linalg.svd(feats - feats.mean(0), compute_uv=False)
    return int((s > s[0] * 1e-3).sum())


def alpha_req(feats, lo=11, frac_hi=0.5):
    f = feats - feats.mean(0)
    lam = np.linalg.svd(f, compute_uv=False) ** 2 / len(f)
    hi = int(len(lam) * frac_hi)
    idx = np.arange(lo, hi)
    lam_seg = lam[lo:hi]
    keep = lam_seg > 1e-12
    if keep.sum() < 10:
        return np.nan
    slope = np.polyfit(np.log(idx[keep] + 1), np.log(lam_seg[keep]), 1)[0]
    return float(-slope)


def twonn_id(feats, max_n=3000):
    f = normalize(feats)
    if len(f) > max_n:
        f = f[RNG.choice(len(f), max_n, replace=False)]
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(f)
    d, _ = nn.kneighbors(f)
    r1, r2 = d[:, 1], d[:, 2]
    keep = r1 > 1e-12
    mu = r2[keep] / r1[keep]
    # Facco et al. MLE, discarding the top 10% mu tail as recommended
    mu = np.sort(mu)[: int(0.9 * len(mu))]
    return float(len(mu) / np.log(mu).sum())


FIELDNAMES = ["encoder", "dataset", "n_samples", "n_classes",
              "within_spread", "between_spread", "wb_ratio", "nc1_ratio", "center_margin",
              "cdnv",
              "task_energy_top10", "task_energy_top50",
              "task_energy_in_top10", "task_energy_in_top50",
              "rankme", "rankme_raw", "numerical_rank", "alpha_req", "twonn_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-dir", type=str, default=str(ROOT / "eval/data/imagenet_val"))
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str, default=str(ROOT / "eval/outputs/geometry_class_15.csv"))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()))
    ap.add_argument("--datasets", nargs="+", default=TARGET_DATASETS)
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    imagenet_loader = load_imagenet_val(args.imagenet_dir, n_samples=args.imagenet_samples)

    rows = []
    import timm
    for enc_name in args.encoders:
        cfg = ENCODERS[enc_name]
        print(f"\n{'='*60}\nEncoder: {enc_name}\n{'='*60}")
        model = timm.create_model(cfg["timm_id"], pretrained=True, num_classes=0).eval().to(device)
        feat_in, _ = extract_features(model, imagenet_loader, device, cfg["pool"])
        for ds in args.datasets:
            print(f"--- {ds} ---")
            try:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue
            feat, labels = extract_features(model, loader, device, cfg["pool"])
            row = {"encoder": enc_name, "dataset": ds, "n_samples": len(feat)}
            row.update(class_manifold_stats(feat, labels))
            te = task_energy(feat, labels, feat)
            row["task_energy_top10"], row["task_energy_top50"] = te[10], te[50]
            tei = task_energy(feat, labels, feat_in)
            row["task_energy_in_top10"], row["task_energy_in_top50"] = tei[10], tei[50]
            fn = normalize(feat)
            row["rankme"] = rankme(fn)
            row["rankme_raw"] = rankme(feat)
            row["numerical_rank"] = numerical_rank(fn)
            row["alpha_req"] = alpha_req(fn)
            row["twonn_id"] = twonn_id(feat)
            print(f"  wb={row['wb_ratio']:.3f} nc1={row['nc1_ratio']:.3f} "
                  f"E_in10={row['task_energy_in_top10']:.3f} rankme={row['rankme']:.1f} "
                  f"id={row['twonn_id']:.1f}")
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"\nSaved {len(rows)} rows -> {args.output}")
    print("Next: python eval/adjudicate/correlate_second_axis.py")


if __name__ == "__main__":
    main()
