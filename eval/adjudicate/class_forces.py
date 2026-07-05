#!/usr/bin/env python
"""
class_forces.py — CPU-1: is there a CLASS-LEVEL collision force?

F2's collision force is drift into the ImageNet cloud (Δoverlap). The packing-margin result
(P-A) suggests a second collision channel: classes colliding with EACH OTHER. This script
differences the class-anchored geometry (Exp H post-CP MAX − Exp E pre-CP) and asks whether
the change in class packing carries force-like signal for ΔkNN (and ΔFT) beyond the two
established forces.

Quantities per (Method, encoder, dataset) at MAX (seed-averaged):
  d_margin  = post center_margin − pre center_margin   (negative = class centers converged)
  d_cdnv    = post cdnv − pre cdnv                     (positive = classes messier vs spacing)
  d_nc1     = post nc1_ratio − pre nc1_ratio
joined with Δuniformity / Δoverlap at MAX from postcp_sweep_fixed.csv and behavioral Δ from
cp_long.csv.

Falsifiable predictions (sphere encoders, rank claims):
  CF1  rho(d_margin, dknn) > 0  (margin collapse accompanies kNN damage)
  CF2  partial(d_margin | dunif) > 0 and partial(d_margin | dunif, dov) > 0
       (class-collision adds signal beyond the two established forces)
  CF3  dose-response: rho(pre_margin, d_margin) — tightly packed datasets lose more margin
       (predict POSITIVE: small pre margin -> more negative d_margin) [FG subgroup focus]
  CF4  exploratory: rho(d_margin, dft) — does class collision speak to the FT side where
       d_within (P-D) failed?

CAVEAT printed: dknn/dft for DIET-MAX cells are stale until test4.
Run: python eval/adjudicate/class_forces.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "MAE": "MAE-CP", "DIET": "DIET-CP"}
FG = ["dtd", "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


def build():
    post = pd.read_csv(OUT / "postcp_class_max.csv")
    post["Method"] = post["method"].map(MMAP)
    post = (post.groupby(["Method", "encoder", "dataset"])
            [["center_margin", "cdnv", "nc1_ratio", "within_spread", "between_spread"]]
            .mean().reset_index())
    pre = pd.read_csv(OUT / "geometry_class_15.csv")
    pre = pre.rename(columns={c: f"{c}_pre" for c in
                              ["center_margin", "cdnv", "nc1_ratio", "within_spread",
                               "between_spread"]})
    m = post.merge(pre[["encoder", "dataset", "center_margin_pre", "cdnv_pre",
                        "nc1_ratio_pre", "within_spread_pre", "between_spread_pre"]],
                   on=["encoder", "dataset"])
    m["d_margin"] = m["center_margin"] - m["center_margin_pre"]
    m["d_cdnv"] = m["cdnv"] - m["cdnv_pre"]
    m["d_nc1"] = m["nc1_ratio"] - m["nc1_ratio_pre"]

    # established forces at MAX from the fixed sweep
    sw = pd.read_csv(OUT / "postcp_sweep_fixed.csv")
    sw = sw[(sw["variant"] == "pretrained") & sw["encoder"].isin(["DINOv3", "CLIP", "MAE"])].copy()
    sw["dataset"] = sw["dataset"].str.lower()
    sw["Method"] = sw["method"].map(MMAP)
    mx = sw.groupby("dataset")["size"].transform("max")
    swm = (sw[sw["size"] == mx].groupby(["Method", "encoder", "dataset"])
           [["uniformity_t2", "neighbor_overlap_k50"]].mean().reset_index())
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"][["encoder", "dataset", "uniformity_t2",
                                          "neighbor_overlap_k50"]]
    geo = geo.rename(columns={"uniformity_t2": "unif_pre", "neighbor_overlap_k50": "ov_pre"})
    swm = swm.merge(geo, on=["encoder", "dataset"])
    swm["dunif"] = swm["uniformity_t2"] - swm["unif_pre"]
    swm["dov"] = swm["neighbor_overlap_k50"] - swm["ov_pre"]
    m = m.merge(swm[["Method", "encoder", "dataset", "dunif", "dov"]],
                on=["Method", "encoder", "dataset"])

    cp = pd.read_csv(OUT / "cp_long.csv")
    beh = (cp[cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
           .groupby(["Method", "Backbone", "dataset_key"])[["dknn", "dlp", "dft"]]
           .mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    return m.merge(beh, on=["Method", "encoder", "dataset"])


def main():
    j = build()
    sph = j[j["encoder"].isin(["DINOv3", "CLIP"])]
    print(f"joined method-cells: {len(j)} total, {len(sph)} sphere")

    print("\n=== CF1/CF2: class-collision force on ΔkNN (sphere @MAX) ===")
    for x in ["d_margin", "d_cdnv", "d_nc1"]:
        r0, p0 = spearmanr(sph[x], sph["dknn"])
        r1, p1 = partial_spearman(sph[x].values, sph["dknn"].values, [sph["dunif"].values])
        r2, p2 = partial_spearman(sph[x].values, sph["dknn"].values,
                                  [sph["dunif"].values, sph["dov"].values])
        print(f"  {x:9s} marginal {r0:+.3f}(p={p0:.1e}) | |dunif {r1:+.3f}(p={p1:.1e}) "
              f"| |dunif,dov {r2:+.3f}(p={p2:.1e})")
    print("  established for reference:")
    r, p = spearmanr(sph["dunif"], sph["dknn"])
    print(f"  dunif     marginal {r:+.3f}(p={p:.1e})")
    r, p = partial_spearman(sph["dov"].values, sph["dknn"].values, [sph["dunif"].values])
    print(f"  dov       partial|dunif {r:+.3f}(p={p:.1e})")

    print("\n  per method (d_margin -> dknn: marginal / partial|dunif):")
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        c = sph[sph["Method"] == meth]
        if len(c) < 10:
            continue
        r0, p0 = spearmanr(c["d_margin"], c["dknn"])
        r1, p1 = partial_spearman(c["d_margin"].values, c["dknn"].values, [c["dunif"].values])
        print(f"    {meth:10s} n={len(c):2d}  {r0:+.3f}(p={p0:.3f}) / {r1:+.3f}(p={p1:.3f})")

    print("\n=== CF3: dose-response — do tightly packed datasets lose more margin? ===")
    inv = sph[sph["Method"].isin(["LeJEPA-CP", "SimCLR-CP"])]
    for enc in ["DINOv3", "CLIP"]:
        g = inv[inv["encoder"] == enc].groupby("dataset")[
            ["center_margin_pre", "d_margin"]].mean().reset_index()
        r, p = spearmanr(g["center_margin_pre"], g["d_margin"])
        gf = g[g["dataset"].isin(FG)]
        rf, pf = spearmanr(gf["center_margin_pre"], gf["d_margin"])
        print(f"  {enc:7s} all15 rho(pre_margin, d_margin)={r:+.3f}(p={p:.3f}) | "
              f"FG7 {rf:+.3f}(p={pf:.3f})")

    print("\n=== CF4 (exploratory): class collision and the FT side ===")
    for x in ["d_margin", "d_cdnv"]:
        r0, p0 = spearmanr(sph[x], sph["dft"])
        r1, p1 = partial_spearman(sph[x].values, sph["dft"].values, [sph["dunif"].values])
        print(f"  {x:9s} -> dft  marginal {r0:+.3f}(p={p0:.1e}) | partial|dunif "
              f"{r1:+.3f}(p={p1:.1e})")
    noD = sph[sph["Method"] != "DIET-CP"]
    r, p = spearmanr(noD["d_margin"], noD["dft"])
    print(f"  [excl DIET, stale dft] d_margin -> dft marginal {r:+.3f}(p={p:.1e}, n={len(noD)})")

    print("\n=== MAE-encoder control (expect no clean law, per gate) ===")
    mae = j[j["encoder"] == "MAE"]
    for x in ["d_margin", "d_cdnv"]:
        r, p = spearmanr(mae[x], mae["dknn"])
        print(f"  {x:9s} -> dknn on MAE encoder: {r:+.3f} (p={p:.3f})")

    print("\nCAVEAT: dknn/dft of DIET-MAX cells stale until test4; re-run after behavior refresh.")


if __name__ == "__main__":
    main()
