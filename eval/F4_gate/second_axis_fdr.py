#!/usr/bin/env python
"""
second_axis_fdr.py — CPU-3: multiple-testing pass over the second-axis phase.

Recomputes (self-contained) every headline test of the Exp E/F/H phase, groups them into
pre-declared families, applies Benjamini-Hochberg within each family, and writes
eval/outputs/second_axis_stats.csv. Mirrors stats_pass.py's role for the F1 tables.

Families:
  A  pre-CP packing axis, FG-7 residual sweep (6 metrics x 2 encoders)         [exploratory]
  B  class-forces at MAX, sphere pooled (3 deltas: marginal + partial|dunif,dov)
  C  class-forces at MAX, MAE encoder per-method (d_cdnv x 4 methods)
  D  P-C2 SA-readout correlations (3 metrics, MAE encoder)

Run: python eval/adjudicate/second_axis_fdr.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(_P(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from correlate_second_axis import behavioral_cell, position_residual
from class_forces import build, partial_spearman

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
FG = ["dtd", "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]


def bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for k, idx in enumerate(order[::-1]):
        i = len(p) - k
        prev = min(prev, p[idx] * len(p) / i)
        q[idx] = prev
    return q


def main():
    rows = []

    # ---- family A: pre-CP packing sweep on FG-7 residuals ----
    cell = behavioral_cell()
    gc = pd.read_csv(OUT / "geometry_class_15.csv")
    gc["nmargin"] = gc["center_margin"] / gc["between_spread"]
    m = position_residual(cell).merge(gc, on=["encoder", "dataset"])
    for met in ["center_margin", "nmargin", "n_classes", "wb_ratio", "cdnv", "nc1_ratio"]:
        for enc in ["DINOv3", "CLIP"]:
            g = m[(m.encoder == enc) & m.dataset.isin(FG)]
            r, p = spearmanr(g[met], g["resid"])
            rows.append(dict(family="A_preCP_packing_FG7", test=f"{met}|{enc}",
                             rho=r, p=p, n=len(g)))

    # ---- families B & C: class forces at MAX ----
    j = build()
    sph = j[j.encoder.isin(["DINOv3", "CLIP"])]
    for x in ["d_cdnv", "d_margin", "d_nc1"]:
        r, p = spearmanr(sph[x], sph["dknn"])
        rows.append(dict(family="B_classforce_sphere", test=f"{x}_marginal", rho=r, p=p,
                         n=len(sph)))
        r, p = partial_spearman(sph[x].values, sph["dknn"].values,
                                [sph["dunif"].values, sph["dov"].values])
        rows.append(dict(family="B_classforce_sphere", test=f"{x}_partial|dunif,dov",
                         rho=r, p=p, n=len(sph)))
    mae = j[j.encoder == "MAE"]
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        c = mae[mae.Method == meth]
        r, p = spearmanr(c["d_cdnv"], c["dknn"])
        rows.append(dict(family="C_classforce_MAE_permethod", test=f"d_cdnv|{meth}",
                         rho=r, p=p, n=len(c)))

    # ---- family D: SA-readout MAE correlations ----
    sa = pd.read_csv(OUT / "geometry_mae_sa.csv")
    sa["encoder"] = "MAE"
    msa = sa.merge(cell[cell.encoder == "MAE"], on=["encoder", "dataset"])
    for met in ["uniformity_t2", "neighbor_overlap_k50", "mmd_rbf"]:
        r, p = spearmanr(msa[met], msa["dknn"])
        rows.append(dict(family="D_MAE_SA_readout", test=f"{met}_SA", rho=r, p=p, n=len(msa)))

    df = pd.DataFrame(rows)
    df["q"] = np.nan
    for fam, g in df.groupby("family"):
        df.loc[g.index, "q"] = bh(g["p"].values)
    df["survives_q05"] = df["q"] < 0.05
    df = df.sort_values(["family", "q"])
    dst = OUT / "second_axis_stats.csv"
    df.to_csv(dst, index=False)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False,
                       formatters={"rho": "{:+.3f}".format, "p": "{:.4f}".format,
                                   "q": "{:.4f}".format}))
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
