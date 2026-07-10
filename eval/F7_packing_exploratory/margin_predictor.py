#!/usr/bin/env python
"""
margin_predictor.py — CPU-2: does the packing margin improve the C2 decision rule?

Baseline rule (predictor.py, per-encoder standardize): features [neighbor_overlap_k50,
uniformity_t2], LogisticRegression, target sign(dknn@MAX, invariance mean). Frozen numbers:
LOO(DINOv3+CLIP) bal-acc 0.778 / AUC 0.829; cross-encoder -> SigLIP sign(dknn) 13/15.

This script tests the 3-feature variant [+ nmargin = center_margin / between_spread] on the
SAME protocol, plus a HURT-side tie-breaker analysis (margin's signal is embedded-regime-
scoped, so the honest expectation is: little/no global gain, potential gain on the
predicted-HURT subset where DTD/food101 were the misses).

Run: python eval/adjudicate/margin_predictor.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
SPHERE = ["DINOv3", "CLIP"]
INV = ["LeJEPA-CP", "SimCLR-CP"]


def table():
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    gc = pd.read_csv(OUT / "geometry_class_15.csv")
    gc["nmargin"] = gc["center_margin"] / gc["between_spread"]
    geo = geo.merge(gc[["encoder", "dataset", "nmargin", "center_margin"]],
                    on=["encoder", "dataset"])
    df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max]
    y = (inv.groupby(["Backbone", "dataset_key"])["dknn"].mean().reset_index()
         .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    m = geo.merge(y, on=["encoder", "dataset"])
    m["help"] = (m["dknn"] > 0).astype(int)
    return m


def zscore_within(df, feats):
    out = df.copy()
    for f in feats:
        out[f] = df.groupby("encoder")[f].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    return out


def loo(m, feats):
    s = zscore_within(m[m.encoder.isin(SPHERE)].reset_index(drop=True), feats)
    proba = np.full(len(s), np.nan)
    for ds in s.dataset.unique():
        tr, te = s[s.dataset != ds], s[s.dataset == ds]
        clf = LogisticRegression(max_iter=1000).fit(tr[feats], tr.help)
        proba[te.index] = clf.predict_proba(te[feats])[:, 1]
    ok = ~np.isnan(proba)
    yt, pp = s.help.values[ok], proba[ok]
    return (balanced_accuracy_score(yt, (pp >= 0.5).astype(int)),
            roc_auc_score(yt, pp), int(ok.sum()))


def siglip_score(m, feats):
    """Fit on DINOv3+CLIP, score realized SigLIP signs (c2_siglip_score.csv)."""
    pool = zscore_within(m[m.encoder.isin(SPHERE + ["SigLIP"])].copy(), feats)
    tr = pool[pool.encoder.isin(SPHERE)].dropna(subset=["help"])
    te = pool[pool.encoder == "SigLIP"].copy()
    clf = LogisticRegression(max_iter=1000).fit(tr[feats], tr["help"])
    te["p_help"] = clf.predict_proba(te[feats])[:, 1]
    c2 = pd.read_csv(OUT / "c2_siglip_score.csv")[["dataset", "real_dknn"]]
    te = te.merge(c2, on="dataset")
    hit = (te["p_help"] >= 0.5) == (te["real_dknn"] > 0)
    return int(hit.sum()), len(te), te[["dataset", "p_help", "real_dknn"]]


def main():
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    gc = pd.read_csv(OUT / "geometry_class_15.csv")
    gc["nmargin"] = gc["center_margin"] / gc["between_spread"]
    base = geo.merge(gc[["encoder", "dataset", "nmargin"]], on=["encoder", "dataset"])
    df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max]
    y = (inv.groupby(["Backbone", "dataset_key"])["dknn"].mean().reset_index()
         .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    m = base.merge(y, on=["encoder", "dataset"], how="left")
    m["help"] = np.where(m["dknn"].notna(), (m["dknn"] > 0).astype(float), np.nan)

    F2 = ["neighbor_overlap_k50", "uniformity_t2"]
    F3 = F2 + ["nmargin"]
    fit = m.dropna(subset=["help"])
    fit["help"] = fit["help"].astype(int)

    print("=== LOO over datasets, DINOv3+CLIP pooled (per-encoder z-score) ===")
    for name, feats in [("2-feature (baseline)", F2), ("3-feature (+nmargin)", F3)]:
        ba, auc, n = loo(fit, feats)
        print(f"  {name:22s} bal-acc={ba:.3f}  AUC={auc:.3f}  (n={n})   "
              f"[baseline frozen: 0.778 / 0.829]")

    print("\n=== cross-encoder: fit DINOv3+CLIP -> score realized SigLIP sign(dknn) ===")
    for name, feats in [("2-feature", F2), ("3-feature (+nmargin)", F3)]:
        hits, n, te = siglip_score(m, feats)
        print(f"  {name:22s} {hits}/{n}   [frozen rule was 13/15]")
        if name.startswith("3"):
            miss = te[((te.p_help >= 0.5) != (te.real_dknn > 0))]
            print("    misses:", list(miss.dataset))

    print("\n=== HURT-side tie-breaker: among predicted-HURT (2-feat), does nmargin rank "
          "the actual outcomes? ===")
    pool = zscore_within(fit[fit.encoder.isin(SPHERE)].reset_index(drop=True), F2 + ["nmargin"])
    clf = LogisticRegression(max_iter=1000).fit(pool[F2], pool.help)
    pool["p2"] = clf.predict_proba(pool[F2])[:, 1]
    hurt = pool[pool.p2 < 0.5]
    from scipy.stats import spearmanr
    r, p = spearmanr(hurt["nmargin"], hurt["dknn"])
    print(f"  n={len(hurt)} predicted-HURT cells: rho(nmargin, dknn) = {r:+.3f} (p={p:.3f})"
          f"  (positive = looser packing -> less damage, as P-A predicts)")


if __name__ == "__main__":
    main()
