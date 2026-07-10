#!/usr/bin/env python
"""
forces_combined.py — ICLR T5 / C3: the two-force predictor + kNN-vs-LP consistency.

On sphere-native encoders, do the two competing forces — SPREAD (Δuniformity) and COLLISION
(Δoverlap) — predict the frozen-transfer change better TOGETHER than either alone? And do they
predict ΔLP the same way they predict ΔkNN (i.e. are the two frozen metrics geometrically
consistent)? Reports single-force Spearman, combined rank-R² (with the incremental gain over the
best single force), and per-method force weights, for BOTH ΔkNN and ΔLP.

CPU-only, runs on existing data. Run: python eval/f2_mechanism/forces_combined.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))  # eval/ root

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from load_results import load_long, add_size_canon

ROOT = _P(__file__).resolve().parent.parent.parent
SPHERE = ["DINOv3", "CLIP"]


def build():
    sw = pd.read_csv(ROOT / "eval/outputs/postcp_sweep.csv")
    sw = sw[sw.variant == "pretrained"].copy(); sw["method_cp"] = sw.method + "-CP"
    post = (sw.groupby(["method_cp", "encoder", "dataset", "size"])
              .agg(post_unif=("uniformity_t2", "mean"), post_ov=("neighbor_overlap_k50", "mean")).reset_index())
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    pre = {(r.encoder, r.dataset): (r.uniformity_t2, r.neighbor_overlap_k50)
           for r in geo[geo.dataset != "imagenet"].itertuples()}
    post["d_unif"] = post.apply(lambda r: r.post_unif - pre.get((r.encoder, r.dataset), (np.nan, np.nan))[0], axis=1)
    post["d_ov"] = post.apply(lambda r: r.post_ov - pre.get((r.encoder, r.dataset), (np.nan, np.nan))[1], axis=1)
    df = load_long()
    dd = (df.groupby(["Method", "Backbone", "dataset_key", "size"])[["dknn", "dlp"]].mean().reset_index()
            .rename(columns={"Method": "method_cp", "Backbone": "encoder", "dataset_key": "dataset"}))
    post = add_size_canon(post, "dataset", "size"); dd = add_size_canon(dd, "dataset", "size")
    return post.merge(dd[["method_cp", "encoder", "dataset", "size_canon", "dknn", "dlp"]],
                      on=["method_cp", "encoder", "dataset", "size_canon"], how="left")


def rank_r2(X, y):
    Xr = np.column_stack([rankdata(X[:, i]) for i in range(X.shape[1])]); yr = rankdata(y)
    return LinearRegression().fit(Xr, yr).score(Xr, yr)


def main():
    m = build()
    for tgt, name in [("dknn", "ΔkNN"), ("dlp", "ΔLP")]:
        s = m[m.encoder.isin(SPHERE)].dropna(subset=["d_unif", "d_ov", tgt])
        ru, _ = spearmanr(s.d_unif, s[tgt]); ro, _ = spearmanr(s.d_ov, s[tgt])
        r2u = rank_r2(s[["d_unif"]].values, s[tgt].values)
        r2o = rank_r2(s[["d_ov"]].values, s[tgt].values)
        r2b = rank_r2(s[["d_unif", "d_ov"]].values, s[tgt].values)
        print(f"\n== {name}  (sphere pooled, n={len(s)}) ==")
        print(f"  spread   Δuniformity : rho={ru:+.3f}  rankR2={r2u:.3f}")
        print(f"  collision Δoverlap   : rho={ro:+.3f}  rankR2={r2o:.3f}")
        print(f"  COMBINED (both)      : rankR2={r2b:.3f}  (incremental over best single = {r2b - max(r2u, r2o):+.3f})")
        for mth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
            ss = m[(m.method_cp == mth) & (m.encoder.isin(SPHERE))].dropna(subset=["d_unif", "d_ov", tgt])
            if len(ss) >= 8:
                a, _ = spearmanr(ss.d_unif, ss[tgt]); b, _ = spearmanr(ss.d_ov, ss[tgt])
                print(f"    {mth:10s}  Δunif {a:+.2f} | Δoverlap {b:+.2f}")
    m.to_csv(ROOT / "eval/outputs/forces_combined.csv", index=False)
    print(f"\nsaved {ROOT / 'eval/outputs/forces_combined.csv'}")


if __name__ == "__main__":
    main()
