#!/usr/bin/env python
"""
stats_pass_refreshed.py — BH-FDR family for the REFRESHED final protocol.

Replaces reliance on the legacy stats_pass.csv (2-method, pre-refresh target; external
audit 2026-07-09). Declared family (fixed BEFORE computation, 36 tests):
  3 geometry metrics {uniformity_t2, mmd_rbf, neighbor_overlap_k50}
  x 3 deltas {dknn, dlp, dft}
  x 4 encoders {DINOv3, CLIP, MAE (3-method Delta@MAX from cp_long_refreshed),
                SigLIP (realized 2-method from c2_siglip_score)}
BH correction at q = 0.10 within this family.

CPU:  python eval/F1_position_law/stats_pass_refreshed.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
ANG = ["LeJEPA-CP", "SimCLR-CP", "DIET-CP"]
METRICS = ["uniformity_t2", "mmd_rbf", "neighbor_overlap_k50"]


def bh(pvals, q=0.10):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    passed = np.zeros(len(p), bool)
    for rank, idx in enumerate(order, 1):
        if p[idx] <= q * rank / len(p):
            passed[order[:rank]] = True
    return passed


def main():
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    cp = pd.read_csv(OUT / "cp_long_refreshed.csv")
    ang = cp[cp.Method.isin(ANG) & cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    cell = (ang.groupby(["Backbone", "dataset_key"])[["dknn", "dlp", "dft"]].mean()
            .reset_index().rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    c2 = pd.read_csv(OUT / "c2_siglip_score.csv").rename(
        columns={"real_dknn": "dknn", "real_dlp": "dlp", "real_dft": "dft"})
    sig = (geo[geo.encoder == "SigLIP"][["encoder", "dataset"] + METRICS]
           .merge(c2[["dataset", "dknn", "dlp", "dft"]], on="dataset"))
    main_t = geo[["encoder", "dataset"] + METRICS].merge(cell, on=["encoder", "dataset"])
    full = pd.concat([main_t, sig], ignore_index=True)

    rows = []
    for enc in ["DINOv3", "CLIP", "MAE", "SigLIP"]:
        g = full[full.encoder == enc]
        for met in METRICS:
            for ch in ["dknn", "dlp", "dft"]:
                r, p = spearmanr(g[met], g[ch])
                rows.append(dict(encoder=enc, metric=met, delta=ch,
                                 rho=round(r, 4), p=round(p, 5), n=len(g)))
    df = pd.DataFrame(rows)
    df["bh_pass_q10"] = bh(df["p"])
    df.to_csv(OUT / "stats_pass_refreshed.csv", index=False)
    print(f"{df.bh_pass_q10.sum()}/{len(df)} tests pass BH(q=0.10) in the declared family")
    print(df[df.bh_pass_q10].to_string(index=False))
    print(f"\nwrote -> {OUT / 'stats_pass_refreshed.csv'}")


if __name__ == "__main__":
    main()
