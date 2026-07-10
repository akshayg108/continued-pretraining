#!/usr/bin/env python
"""
merge_rest_geometry.py — fold the 59 re-trained cells' geometry (rest_geometry.csv, from
eval/rest/test2_geometry.py) into the production sweep, then recompute the F2/F3 headline
numbers on the fixed sweep.

Never overwrites postcp_sweep.csv: writes eval/outputs/postcp_sweep_fixed.csv.
Rows are matched on (method, encoder, dataset, size, seed) within variant=='pretrained';
matched rows get their l2_norm_cv / uniformity_t2 / neighbor_overlap_k50 replaced; unmatched
rest rows are appended (and reported — expect 0 appended if the old sweep covered all 59).

Recomputed verdicts (compare against the pre-fix numbers frozen in FINDINGS_step4 /
eval/adjudicate/suppressor_robustness.py output of 2026-07-01):
  F2  pooled-sphere spread rho (was -0.552) + collision partial (was -0.380)
      + size-controlled variants (-0.572 / -0.346)
  F2  per method x sphere cells — WATCH: does DIET's collision partial leave null
      now that DIET-MAX ckpts are fully trained? (was -0.114 all / -0.072 clean)
  F3  P3.1 spread counts (was 139/180; angular-3 119/135) and P3.2 collide counts
      (was 150/171; angular-3 125/134)

CAVEAT printed at the end: behavioral dknn for the DIET-MAX (and other rerun) cells is
STALE until eval/rest/test4_behavior_deltas.py refreshes results.xlsx — geometry columns
here are final, the geometry->dknn correlations involving those cells are provisional
until test4 lands.

Run: python eval/adjudicate/merge_rest_geometry.py
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
GEO_COLS = ["l2_norm_cv", "uniformity_t2", "neighbor_overlap_k50"]
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "MAE": "MAE-CP", "DIET": "DIET-CP"}


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


def merge():
    sweep = pd.read_csv(OUT / "postcp_sweep.csv")
    rest = pd.read_csv(OUT / "rest_geometry.csv")
    for c in ("size", "seed"):
        rest[c] = rest[c].astype(int)
        sweep[c] = sweep[c].astype(int)
    rest_keyed = rest.set_index(["method", "encoder", "dataset", "size", "seed"])
    key = list(zip(sweep["method"], sweep["encoder"], sweep["dataset"],
                   sweep["size"], sweep["seed"]))
    is_pre = (sweep["variant"] == "pretrained").values
    updated = 0
    hit = set()
    for i, k in enumerate(key):
        if is_pre[i] and k in rest_keyed.index:
            for c in GEO_COLS:
                v = rest_keyed.loc[k, c]
                if pd.notna(v) and v != "":
                    sweep.iat[i, sweep.columns.get_loc(c)] = float(v)
            sweep.iat[i, sweep.columns.get_loc("ckpt")] = rest_keyed.loc[k, "ckpt"]
            updated += 1
            hit.add(k)
    missing = [k for k in rest_keyed.index if k not in hit]
    appended = 0
    if missing:
        add = rest_keyed.loc[missing].reset_index()
        add["variant"] = "pretrained"
        add["n_samples"] = np.nan
        sweep = pd.concat([sweep, add[[c for c in sweep.columns if c in add.columns]]],
                          ignore_index=True)
        appended = len(add)
    out = OUT / "postcp_sweep_fixed.csv"
    sweep.to_csv(out, index=False)
    print(f"merged: {updated} rows updated, {appended} appended "
          f"(rest rows: {len(rest)}) -> {out}")
    if missing:
        print("  appended keys (not found in old sweep):")
        for k in missing:
            print("   ", k)
    return sweep


def rejoin(sweep):
    sw = sweep[(sweep["variant"] == "pretrained")
               & sweep["encoder"].isin(["DINOv3", "CLIP", "MAE"])].copy()
    sw["dk"] = sw["dataset"].str.lower()
    sw["Method"] = sw["method"].map(MMAP)
    post = (sw.groupby(["Method", "encoder", "dk", "size"])
            [["l2_norm_cv", "uniformity_t2", "neighbor_overlap_k50"]].mean().reset_index())
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo["dk"] = geo["dataset"].str.lower()
    pre = geo[geo.dataset != "imagenet"][["encoder", "dk", "uniformity_t2",
                                          "neighbor_overlap_k50", "l2_norm_cv"]]
    pre = pre.rename(columns={"uniformity_t2": "unif_pre", "neighbor_overlap_k50": "ov_pre",
                              "l2_norm_cv": "cv_pre"})
    m = post.merge(pre, on=["encoder", "dk"])
    m["dunif"] = m["uniformity_t2"] - m["unif_pre"]
    m["dov"] = m["neighbor_overlap_k50"] - m["ov_pre"]
    m["dcv"] = m["l2_norm_cv"] - m["cv_pre"]
    cp = pd.read_csv(OUT / "cp_long.csv")
    cp["dk"] = cp["dataset_key"].str.lower()
    cpsz = cp.groupby("dk")["size"].apply(lambda s: sorted(set(s))).to_dict()
    m["size_c"] = m.apply(lambda r: r["size"] if r["size"] in cpsz.get(r["dk"], [])
                          else min(cpsz[r["dk"]], key=lambda x: abs(x - r["size"])), axis=1)
    beh = (cp[cp["Backbone"].isin(["DINOv3", "CLIP", "MAE"])]
           [["Method", "Backbone", "dk", "size", "dknn"]]
           .rename(columns={"Backbone": "encoder", "size": "size_c"}))
    return m.merge(beh, on=["Method", "encoder", "dk", "size_c"])


def f2_headline(j):
    sph = j[j["encoder"].isin(["DINOv3", "CLIP"])]
    print("\n=== F2 on FIXED sweep (pre-fix values in brackets) ===")
    r, p = spearmanr(sph["dunif"], sph["dknn"])
    print(f"  spread pooled sphere        rho={r:+.3f} (p={p:.1e}, n={len(sph)})   [-0.552]")
    r, p = partial_spearman(sph["dov"].values, sph["dknn"].values, [sph["dunif"].values])
    print(f"  collision partial            rho={r:+.3f} (p={p:.1e})               [-0.380]")
    ls = np.log10(sph["size"].values.astype(float))
    r, p = partial_spearman(sph["dunif"].values, sph["dknn"].values, [ls])
    print(f"  spread | size                rho={r:+.3f} (p={p:.1e})               [-0.572]")
    r, p = partial_spearman(sph["dov"].values, sph["dknn"].values, [sph["dunif"].values, ls])
    print(f"  collision | dunif,size       rho={r:+.3f} (p={p:.1e})               [-0.346]")
    print("\n  per method x sphere (spread rho / collision partial):")
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        c = sph[sph["Method"] == meth]
        if len(c) < 12:
            continue
        ru, pu = spearmanr(c["dunif"], c["dknn"])
        ro, po = partial_spearman(c["dov"].values, c["dknn"].values, [c["dunif"].values])
        note = "  <-- WATCH (was -0.114 n.s. undertrained)" if meth == "DIET-CP" else ""
        print(f"    {meth:10s} n={len(c):3d}  spread {ru:+.3f}(p={pu:.1e})  "
              f"collision {ro:+.3f}(p={po:.1e}){note}")
    mae = j[j["encoder"] == "MAE"]
    r, p = spearmanr(mae["dunif"], mae["dknn"])
    print(f"\n  MAE-encoder control: spread rho={r:+.3f} (p={p:.2f}) — expect n.s. (gate)")


def f3_counts(sweep):
    sw = sweep[(sweep["variant"] == "pretrained")
               & sweep["encoder"].isin(["DINOv3", "CLIP", "MAE"])].copy()
    sw["dk"] = sw["dataset"].str.lower()
    post = (sw.groupby(["method", "encoder", "dk", "size"])
            [["uniformity_t2", "neighbor_overlap_k50"]].mean().reset_index())
    nu = tu = po_ = to = nui = tui = poi = toi = 0
    for (meth, enc, dk), g in post.groupby(["method", "encoder", "dk"]):
        if len(g) < 3:
            continue
        inv = meth in ("LeJEPA", "SimCLR", "DIET")
        ru = spearmanr(g["size"], g["uniformity_t2"]).correlation
        ro = spearmanr(g["size"], g["neighbor_overlap_k50"]).correlation
        if not np.isnan(ru):
            tu += 1; nu += ru < 0
            if inv:
                tui += 1; nui += ru < 0
        if not np.isnan(ro):
            to += 1; po_ += ro > 0
            if inv:
                toi += 1; poi += ro > 0
    print("\n=== F3 on FIXED sweep (pre-fix in brackets) ===")
    print(f"  P3.1 spread : {nu}/{tu} configs rho(size,unif)<0   [139/180]  "
          f"angular-3: {nui}/{tui} [119/135]")
    print(f"  P3.2 collide: {po_}/{to} configs rho(size,ov)>0    [150/171]  "
          f"angular-3: {poi}/{toi} [125/134]")


def main():
    if not (OUT / "rest_geometry.csv").exists():
        sys.exit("eval/outputs/rest_geometry.csv missing — run test2 on the cluster first.")
    sweep = merge()
    j = rejoin(sweep)
    print(f"\njoined rows (geometry x behavior): {len(j)}")
    f2_headline(j)
    f3_counts(sweep)
    print("\nCAVEAT: dknn for the rerun cells (esp. every DIET-MAX cell) is STALE until "
          "test4 refreshes results.xlsx -> re-run this script after updating cp_long.csv.")


if __name__ == "__main__":
    main()
