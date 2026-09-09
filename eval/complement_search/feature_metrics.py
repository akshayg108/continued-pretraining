#!/usr/bin/env python
"""
feature_metrics.py — extra pre-CP geometry candidates computed from the frozen feature
dumps in eval/outputs/int1_features/<enc>__<ds>.npz (bank_X = the standard seed-42
<=5000 train cloud, bank_y = labels). CPU only; nothing is trained.

Preprocessing (matches eval/utils/geometry_metrics.py / geometry_class.py):
  * stored features are RAW (uncentred, unnormalised float32);
  * subsample to <= 2000 rows with a fixed seed (42) per cell;
  * L2-normalise every row (sklearn normalize);
  * spectral metrics are computed on the CENTRED normalised cloud (as rankme/alpha_req
    in geometry_class.py); angular/kNN metrics use the uncentred normalised cloud.

Label-free candidates:
  twonn_id_2k          TwoNN intrinsic dimension (Facco et al. 2017), top-10% mu tail dropped
  participation_ratio  (sum lambda)^2 / sum lambda^2 of covariance eigenvalues
  rankme_2k            exp(entropy of L1-normalised singular values)  (Garrido et al. 2023)
  alpha_req_top100     minus the slope of log lambda_i vs log i over the top-100 eigenvalues
                       (larger = faster spectral decay; same sign convention as alpha_req)
  pc1_share, pc5_share top-1 / top-5 PC variance share
  mean_pairwise_cos    mean off-diagonal cosine (anisotropy)
  mean_cos_to_centroid mean cosine between each row and the normalised centroid
  hubness_skew_k10     skewness of the k=10 kNN-graph in-degree distribution
Label-aware candidates (labels used for measurement only, flagged in the screen):
  between_var_share    tr(S_B) / (tr(S_B) + tr(S_W))
  fisher_ratio         tr(S_B) / tr(S_W)   (monotone in between_var_share -> same ranks)
  nc1_papyan           (1/C) tr(S_W pinv(S_B))   (Papyan et al. 2020 NC1)

Usage: python eval/complement_search/feature_metrics.py
Output: eval/complement_search/outputs/feature_metrics.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_DIR = ROOT / "eval/outputs/int1_features"
OUT = ROOT / "eval/complement_search/outputs/feature_metrics.csv"
ENCODERS = ["DINOv3", "CLIP", "MAE", "SigLIP"]
DATASETS = ["breastmnist", "dermamnist", "octmnist", "organamnist", "pathmnist", "galaxy10",
            "eurosat", "plant_village", "dtd",
            "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]
MAX_N = 2000
SEED = 42


def twonn_id(f):
    d, _ = NearestNeighbors(n_neighbors=3).fit(f).kneighbors(f)
    r1, r2 = d[:, 1], d[:, 2]
    keep = r1 > 1e-12
    mu = np.sort(r2[keep] / r1[keep])[: int(0.9 * keep.sum())]
    return float(len(mu) / np.log(mu).sum())


def spectral(f):
    s = np.linalg.svd(f - f.mean(0), compute_uv=False)
    lam = s ** 2 / len(f)
    p = s / s.sum()
    idx = np.arange(1, 101)
    slope = np.polyfit(np.log(idx), np.log(lam[:100]), 1)[0]
    return {"participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "rankme_2k": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
            "alpha_req_top100": float(-slope),
            "pc1_share": float(lam[0] / lam.sum()),
            "pc5_share": float(lam[:5].sum() / lam.sum())}


def angular(f):
    g = f @ f.T
    n = len(f)
    mu = f.mean(0)
    return {"mean_pairwise_cos": float((g.sum() - np.trace(g)) / (n * (n - 1))),
            "mean_cos_to_centroid": float((f @ (mu / np.linalg.norm(mu))).mean())}


def hubness(f, k=10):
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(f).kneighbors(f)
    indeg = np.bincount(idx[:, 1:].ravel(), minlength=len(f))
    return {"hubness_skew_k10": float(skew(indeg))}


def class_scatter(f, y):
    mu = f.mean(0)
    sw = np.zeros((f.shape[1], f.shape[1]))
    sb = np.zeros_like(sw)
    classes = np.unique(y)
    for c in classes:
        fc = f[y == c]
        dc = fc - fc.mean(0)
        sw += dc.T @ dc
        db = (fc.mean(0) - mu)[:, None]
        sb += len(fc) * (db @ db.T)
    sw /= len(f)
    sb /= len(f)
    tr_w, tr_b = np.trace(sw), np.trace(sb)
    return {"between_var_share": float(tr_b / (tr_b + tr_w)),
            "fisher_ratio": float(tr_b / tr_w),
            # S_B has rank <= C-1: cut the pinv at 1e-8 of the top eigenvalue so that
            # floating-point noise directions are not inverted (default rcond=1e-15 blows up).
            "nc1_papyan": float(np.trace(sw @ np.linalg.pinv(sb, rcond=1e-8, hermitian=True))
                                / len(classes))}


def cell_metrics(enc, ds):
    z = np.load(FEAT_DIR / f"{enc}__{ds}.npz")
    X, y = z["bank_X"].astype(np.float64), z["bank_y"]
    if len(X) > MAX_N:
        sel = np.sort(np.random.RandomState(SEED).choice(len(X), MAX_N, replace=False))
        X, y = X[sel], y[sel]
    f = normalize(X)
    row = {"encoder": enc, "dataset": ds, "n_used": len(f), "n_classes_used": int(len(np.unique(y)))}
    row["twonn_id_2k"] = twonn_id(f)
    row.update(spectral(f))
    row.update(angular(f))
    row.update(hubness(f))
    row.update(class_scatter(f, y))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    rows = []
    for enc in ENCODERS:
        for ds in DATASETS:
            rows.append(cell_metrics(enc, ds))
            r = rows[-1]
            print(f"{enc:7s} {ds:14s} n={r['n_used']:4d} id={r['twonn_id_2k']:5.1f} "
                  f"PR={r['participation_ratio']:6.1f} rankme={r['rankme_2k']:6.1f} "
                  f"alpha={r['alpha_req_top100']:.2f} pc1={r['pc1_share']:.3f} "
                  f"cos={r['mean_pairwise_cos']:.3f} hub={r['hubness_skew_k10']:.2f} "
                  f"bvs={r['between_var_share']:.3f} nc1={r['nc1_papyan']:.2f}")
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
