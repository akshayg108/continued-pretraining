#!/usr/bin/env python
"""
dft_snr.py — Adjudication: is the kNN<->FT reversal supported by ΔFT's signal-to-noise?

The reversal (same pre-CP scalar predicts ΔkNN and ΔFT with opposite signs) is a rank claim,
but mean |ΔFT| is only ~1/4-1/8 of |ΔkNN| and FT is near-saturated. This script asks whether
ΔFT@MAX (invariance methods) carries real signal or is seed noise:

  A. Per-cell z = |dft| / sigma_dft, with sigma_dft = sqrt(ft_pre_s^2 + ft_post_s^2)/sqrt(3)
     (seed-std of the mean difference, independence assumption -> conservative).
     Compare the z-distribution of dft vs dknn/dlp.
  B. Method sign-agreement: does sign(dft) agree between LeJEPA-CP and SimCLR-CP per
     (encoder, dataset) more than chance? (If dft were pure noise, agreement ~50%.)
     CANONICAL protocol (CORRECTIONS.md item 7): deltas from cp_long_refreshed.csv
     -> 40/45 = 88.9%. Sections A/C/D stay on results.xlsx via load_long() because the
     refreshed CSV's seed-std/level columns are pre-blend for the 31 blended cells.
  C. Reversal survival: recompute Spearman(geometry, dft) per sphere encoder on
     (i) all 15, (ii) only cells with z >= 1, (iii) weighting by min(z, 3).
  D. Headroom control: is |dft| (and dft itself) explained by FT saturation (1 - ft_pre)?
     partial Spearman(geometry, dft | ft_pre headroom).

CPU-only, existing artifacts. Run: python eval/F1_position_law/dft_snr.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, binomtest
from sklearn.linear_model import LinearRegression

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
INV = ["LeJEPA-CP", "SimCLR-CP"]
GEOMS = ["uniformity_t2", "neighbor_overlap_k50", "mmd_rbf", "cosine_dist_centroid"]


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


def main():
    df = load_long()
    df = df[df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    inv = df[df.Method.isin(INV) & df.is_max].copy()
    # seed-std of the mean difference (3 seeds)
    for m, pre_s, post_s in [("dft", "ft_pre_s", "ft_post_s"),
                             ("dknn", "knn_pre_s", "knn_post_s"),
                             ("dlp", "lp_pre_s", "lp_post_s")]:
        s = np.sqrt(pd.to_numeric(inv[pre_s], errors="coerce") ** 2
                    + pd.to_numeric(inv[post_s], errors="coerce") ** 2) / np.sqrt(3)
        inv[f"z_{m}"] = (inv[m].abs() / s.replace(0, np.nan))

    print("=" * 78)
    print("A. |Δ| / seed-noise (z) at MAX, invariance methods, per metric")
    print("=" * 78)
    for m in ["dknn", "dlp", "dft"]:
        z = inv[f"z_{m}"].dropna()
        print(f"  {m:5s}: median z = {z.median():5.2f} | mean |Δ| = {inv[m].abs().mean():.4f} "
              f"| frac z>=1: {(z >= 1).mean():.2f} | frac z>=2: {(z >= 2).mean():.2f}  (n={len(z)})")
    print("\n  per-encoder dft z:")
    for enc in ["DINOv3", "CLIP", "MAE"]:
        z = inv[inv.Backbone == enc]["z_dft"].dropna()
        print(f"    {enc:7s} median z = {z.median():5.2f} | frac z>=1: {(z >= 1).mean():.2f} (n={len(z)})")

    print("\n" + "=" * 78)
    print("B. sign(dft) agreement between LeJEPA-CP and SimCLR-CP (per encoder x dataset, MAX)")
    print("=" * 78)
    # canonical: refreshed deltas (CORRECTIONS.md item 7 -> 40/45 = 88.9%)
    ref = pd.read_csv(ROOT / "eval/outputs/cp_long_refreshed.csv")
    ref_inv = ref[ref.Backbone.isin(["DINOv3", "CLIP", "MAE"])
                  & ref.Method.isin(INV) & ref.is_max]
    piv = ref_inv.pivot_table(index=["Backbone", "dataset_key"], columns="Method",
                              values="dft").dropna()
    agree = (np.sign(piv[INV[0]]) == np.sign(piv[INV[1]]))
    print(f"  overall: {agree.sum()}/{len(agree)} = {agree.mean():.2f} "
          f"(binom p vs 0.5 = {binomtest(int(agree.sum()), len(agree), 0.5).pvalue:.4f})")
    for enc in ["DINOv3", "CLIP", "MAE"]:
        a = agree.loc[enc]
        print(f"    {enc:7s} {a.sum()}/{len(a)} = {a.mean():.2f}")
    piv_old = inv.pivot_table(index=["Backbone", "dataset_key"], columns="Method",
                              values="dft").dropna()
    a_old = (np.sign(piv_old[INV[0]]) == np.sign(piv_old[INV[1]]))
    print(f"  (legacy pre-refresh protocol: {a_old.sum()}/{len(a_old)} = {a_old.mean():.2f})")
    # same for dknn as reference (refreshed protocol)
    pivk = ref_inv.pivot_table(index=["Backbone", "dataset_key"], columns="Method",
                               values="dknn").dropna()
    ak = (np.sign(pivk[INV[0]]) == np.sign(pivk[INV[1]]))
    print(f"  reference dknn agreement: {ak.sum()}/{len(ak)} = {ak.mean():.2f}")

    print("\n" + "=" * 78)
    print("C. reversal survival under SNR restriction (geometry -> dft, sphere encoders)")
    print("=" * 78)
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    cell = (inv.groupby(["Backbone", "dataset_key"])
            .agg(dft=("dft", "mean"), dknn=("dknn", "mean"), z_dft=("z_dft", "mean"),
                 ft_pre=("ft_pre", "mean")).reset_index())
    for enc in ["DINOv3", "CLIP"]:
        m = geo[geo.encoder == enc].merge(cell[cell.Backbone == enc],
                                          left_on="dataset", right_on="dataset_key")
        print(f"\n--- {enc} ---")
        for g in GEOMS:
            x, y = m[g].values, m["dft"].values
            r_all, p_all = spearmanr(x, y)
            hi = m["z_dft"] >= 1
            r_hi, p_hi = (spearmanr(m.loc[hi, g], m.loc[hi, "dft"])
                          if hi.sum() >= 5 else (np.nan, np.nan))
            w = np.minimum(m["z_dft"].fillna(0), 3)
            # weighted Spearman via weighted Pearson on ranks
            xr, yr = rankdata(x), rankdata(y)
            xw = xr - np.average(xr, weights=w); yw = yr - np.average(yr, weights=w)
            r_w = (np.average(xw * yw, weights=w)
                   / np.sqrt(np.average(xw ** 2, weights=w) * np.average(yw ** 2, weights=w)))
            print(f"  {g:22s} all rho={r_all:+.3f}(p={p_all:.3f}) | z>=1 (n={int(hi.sum())}) "
                  f"rho={r_hi:+.3f}(p={p_hi:.3f}) | z-weighted rho={r_w:+.3f}")

    print("\n" + "=" * 78)
    print("D. headroom control: geometry -> dft controlling for (1 - ft_pre)")
    print("=" * 78)
    for enc in ["DINOv3", "CLIP"]:
        m = geo[geo.encoder == enc].merge(cell[cell.Backbone == enc],
                                          left_on="dataset", right_on="dataset_key")
        head = 1.0 - m["ft_pre"].values
        r_h, p_h = spearmanr(head, m["dft"])
        print(f"\n--- {enc} ---  rho(headroom, dft) = {r_h:+.3f} (p={p_h:.3f})")
        for g in GEOMS:
            r, p = partial_spearman(m[g].values, m["dft"].values, [head])
            print(f"  {g:22s} partial(geom, dft | headroom) rho={r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
