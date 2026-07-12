#!/usr/bin/env python
"""
suppressor_robustness.py — Adjudication: how robust are F2's two forces, esp. the collision
suppressor (partial(Δoverlap | Δunif) -> Δknn = -0.40)?

Concerns tested:
  A. SIZE CONFOUND. F3 says post_unif falls and post_overlap rises with CP size, and Δknn also
     moves with size — so pooled-over-sizes correlations may be carried by the shared size axis.
     Recompute the two forces partialling out log(size): partial(Δunif | size), and
     partial(Δov | Δunif, size).
  B. DATASET LOO. Drop one dataset at a time from the pooled sphere set; report min/max of the
     collision partial.
  C. CELL SLICES. Per encoder / per method / MAX-only vs sweep-only.
  D. BROKEN-CKPT SENSITIVITY. Drop every (method, encoder, dataset, size) cell that contains a
     checkpoint listed in rerun_geometry.csv (the 60 retraining cells) and recompute both forces
     on the clean subset.

CPU-only. Run: python eval/adjudicate/suppressor_robustness.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "MAE": "MAE-CP", "DIET": "DIET-CP"}


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


def load_joined():
    """Rebuild the F2 join from raw artifacts (mirrors forces_combined.csv but keeps size + seeds)."""
    # postcp_sweep_fixed.csv = the checkpoint-hygiene-fixed sweep (merge_rest_geometry);
    # the pre-fix postcp_sweep.csv gave -0.552/-0.380/-0.572/-0.346 (audit 2026-07-12).
    sw = pd.read_csv(OUT / "postcp_sweep_fixed.csv")
    sw = sw[(sw["variant"] == "pretrained") & sw["encoder"].isin(["DINOv3", "CLIP", "MAE"])].copy()
    sw["dk"] = sw["dataset"].str.lower()
    sw["Method"] = sw["method"].map(MMAP)
    post = (sw.groupby(["Method", "encoder", "dk", "size"])
            [["l2_norm_cv", "uniformity_t2", "neighbor_overlap_k50"]].mean().reset_index())
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo["dk"] = geo["dataset"].str.lower()
    pre = geo[geo.dataset != "imagenet"][["encoder", "dk", "uniformity_t2", "neighbor_overlap_k50"]]
    pre = pre.rename(columns={"uniformity_t2": "unif_pre", "neighbor_overlap_k50": "ov_pre"})
    m = post.merge(pre, on=["encoder", "dk"])
    m["dunif"] = m["uniformity_t2"] - m["unif_pre"]
    m["dov"] = m["neighbor_overlap_k50"] - m["ov_pre"]
    cp = pd.read_csv(OUT / "cp_long.csv")
    cp["dk"] = cp["dataset_key"].str.lower()
    cpsz = cp.groupby("dk")["size"].apply(lambda s: sorted(set(s))).to_dict()
    m["size_c"] = m.apply(lambda r: r["size"] if r["size"] in cpsz.get(r["dk"], [])
                          else min(cpsz[r["dk"]], key=lambda x: abs(x - r["size"])), axis=1)
    beh = (cp[cp["Backbone"].isin(["DINOv3", "CLIP", "MAE"])]
           [["Method", "Backbone", "dk", "size", "dknn"]]
           .rename(columns={"Backbone": "encoder", "size": "size_c"}))
    return m.merge(beh, on=["Method", "encoder", "dk", "size_c"])


def forces(df, label):
    if len(df) < 12:
        print(f"  {label:42s} n={len(df):4d}  (too few)")
        return
    r_u, p_u = spearmanr(df["dunif"], df["dknn"])
    r_o, p_o = partial_spearman(df["dov"].values, df["dknn"].values, [df["dunif"].values])
    print(f"  {label:42s} n={len(df):4d}  spread rho={r_u:+.3f}(p={p_u:.1e})  "
          f"collision partial={r_o:+.3f}(p={p_o:.1e})")
    return r_u, r_o


def main():
    j = load_joined()
    sph = j[j["encoder"].isin(["DINOv3", "CLIP"])].copy()
    print("=" * 96)
    print("BASELINE (pooled sphere, all sizes)")
    print("=" * 96)
    forces(sph, "pooled sphere")

    print("\nA. SIZE CONFOUND (partial out log10 size)")
    ls = np.log10(sph["size"].values.astype(float))
    r_u, p_u = partial_spearman(sph["dunif"].values, sph["dknn"].values, [ls])
    r_o, p_o = partial_spearman(sph["dov"].values, sph["dknn"].values,
                                [sph["dunif"].values, ls])
    print(f"  spread    partial(dunif | size)        rho={r_u:+.3f} (p={p_u:.1e})")
    print(f"  collision partial(dov | dunif, size)   rho={r_o:+.3f} (p={p_o:.1e})")
    # and within single-size strata
    for s in [1000, "MAXonly"]:
        sub = sph[sph["size_c"].astype(str) == "1000"] if s == 1000 else \
              sph[sph.groupby("dk")["size"].transform("max") == sph["size"]]
        forces(sub, f"stratum: {s}")

    print("\nB. DATASET LEAVE-ONE-OUT (collision partial range)")
    vals = []
    for ds in sorted(sph["dk"].unique()):
        sub = sph[sph["dk"] != ds]
        r_o, _ = partial_spearman(sub["dov"].values, sub["dknn"].values, [sub["dunif"].values])
        vals.append((ds, r_o))
    vals.sort(key=lambda t: t[1])
    print(f"  min: drop {vals[0][0]:14s} -> {vals[0][1]:+.3f} | "
          f"max: drop {vals[-1][0]:14s} -> {vals[-1][1]:+.3f} | "
          f"median {np.median([v for _, v in vals]):+.3f}")

    print("\nC. SLICES")
    for enc in ["DINOv3", "CLIP"]:
        forces(sph[sph["encoder"] == enc], f"encoder = {enc}")
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        forces(sph[sph["Method"] == meth], f"method = {meth} (sphere)")

    print("\nD. BROKEN-CKPT SENSITIVITY (drop the 60 rerun cells)")
    rr = pd.read_csv(OUT / "rerun_geometry.csv", header=None,
                     names=["ckpt", "method", "encoder", "dataset", "size", "seed",
                            "epoch", "step", "blk", "verdict"])
    rr["Method"] = rr["method"].map(MMAP)
    rr["dk"] = rr["dataset"].str.lower()
    bad = set(zip(rr["Method"], rr["encoder"], rr["dk"], rr["size"].astype(int)))
    key = list(zip(sph["Method"], sph["encoder"], sph["dk"], sph["size"].astype(int)))
    clean = sph[[k not in bad for k in key]]
    dropped = len(sph) - len(clean)
    print(f"  dropped {dropped} affected (method,encoder,dataset,size) rows")
    forces(clean, "clean subset (no rerun cells)")
    ls = np.log10(clean["size"].values.astype(float))
    r_o, p_o = partial_spearman(clean["dov"].values, clean["dknn"].values,
                                [clean["dunif"].values, ls])
    print(f"  clean + size control: collision partial rho={r_o:+.3f} (p={p_o:.1e})")


if __name__ == "__main__":
    main()
