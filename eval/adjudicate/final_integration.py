#!/usr/bin/env python
"""
final_integration.py — fold test4 (fresh behavior for the 59 rerun cells) into the analysis
and re-run every DIET-affected result. Prints old -> new for each headline number.

Blend rule (disclosed): cp_long stores seed-averaged cell values; test4 gives per-seed deltas
for the RERUN seeds only. For a cell with k rerun seeds (of 3):
    new_mean = (sum(test4 seed deltas) + (3-k) * old_mean) / 3
Exact replacement when k=3; approximation (non-rerun seeds proxied by the old mean) otherwise.

Outputs: eval/outputs/cp_long_refreshed.csv + console report.
Blocks:
  0  replication check (SimCLR/LeJEPA rerun deltas should match old cell means; DIET may shift)
  1  F1 with 3-method Δ@MAX (LeJEPA+SimCLR+DIET, refreshed) vs the 2-method originals
  2  §6.8 per-method position law (fresh DIET vs the epoch-74 provisional numbers)
  3  F2 on fixed sweep with refreshed behavior (pooled + DIET collision FINAL)
  4  class-force d_cdnv with refreshed behavior
  5  packing FG-7 under the 3-method refreshed target (global-fit AND within-FG-fit residuals)
  6  margin predictor under the 3-method refreshed target
  7  P-D recheck (d_within -> dft refreshed; DIET Δwithin largest is geometry-only, unchanged)

Run: python eval/adjudicate/final_integration.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "MAE": "MAE-CP", "DIET": "DIET-CP"}
FG = ["dtd", "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]
SPHERE = ["DINOv3", "CLIP"]


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


# ---------------------------------------------------------------- refresh
def refresh_cp_long():
    cp = pd.read_csv(OUT / "cp_long.csv")
    rb = pd.read_csv(OUT / "rest_behavior.csv")
    rb["Method"] = rb["method"].map(MMAP)
    rb["dk"] = rb["dataset"].str.lower()
    agg = (rb.groupby(["Method", "encoder", "dk", "n"])
           .agg(k=("seed", "size"), t4_dknn=("d_knn", "mean"),
                t4_dlp=("d_lp", "mean"), t4_dft=("d_ft", "mean")).reset_index())
    cp["dk"] = cp["dataset_key"].str.lower()
    idx = {(r.Method, r.encoder, r.dk, int(r.n)): r for r in agg.itertuples()}
    print("=" * 100)
    print("BLOCK 0 — replication check: test4 per-cell mean delta vs old cp_long cell mean")
    print("=" * 100)
    print(f"{'Method':10}{'enc':7}{'dataset':14}{'size':>7}{'k':>3}"
          f"{'old dknn':>10}{'t4 dknn':>9}{'new dknn':>10}{'|shift|':>9}")
    n_upd = 0
    shifts = {"DIET-CP": [], "SimCLR-CP": [], "LeJEPA-CP": []}
    for i, row in cp.iterrows():
        key = (row["Method"], row["Backbone"], row["dk"], int(row["size"]))
        if key not in idx:
            continue
        r = idx[key]
        k = int(r.k)
        for col, t4 in [("dknn", r.t4_dknn), ("dlp", r.t4_dlp), ("dft", r.t4_dft)]:
            old = row[col]
            new = (k * t4 + (3 - k) * old) / 3 if pd.notna(old) else t4
            cp.at[i, col] = round(new, 5)
        old_dknn = row["dknn"]
        new_dknn = cp.at[i, "dknn"]
        shifts[row["Method"]].append(abs(r.t4_dknn - old_dknn))
        print(f"{row['Method']:10}{row['Backbone']:7}{row['dk']:14}{int(row['size']):>7}{k:>3}"
              f"{old_dknn:>10.4f}{r.t4_dknn:>9.4f}{new_dknn:>10.4f}"
              f"{abs(r.t4_dknn - old_dknn):>9.4f}")
        n_upd += 1
    print(f"\ncells updated: {n_upd} (from {len(agg)} test4 cells)")
    for m, s in shifts.items():
        if s:
            print(f"  mean |t4 - old| dknn shift  {m}: {np.mean(s):.4f}  (n={len(s)})"
                  f"  -> {'replication (expected small)' if m != 'DIET-CP' else 'epoch74->150 shift (expected larger)'}")
    dst = OUT / "cp_long_refreshed.csv"
    cp.to_csv(dst, index=False)
    print(f"saved {dst}")
    return cp


# ---------------------------------------------------------------- shared joins
def delta_at_max(cp, methods):
    d = cp[cp.Method.isin(methods) & cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    return (d.groupby(["Backbone", "dk"])[["dknn", "dlp", "dft"]].mean().reset_index()
            .rename(columns={"Backbone": "encoder", "dk": "dataset"}))


def f2_join(cp):
    sw = pd.read_csv(OUT / "postcp_sweep_fixed.csv")
    sw = sw[(sw["variant"] == "pretrained") & sw["encoder"].isin(["DINOv3", "CLIP", "MAE"])].copy()
    sw["dk"] = sw["dataset"].str.lower()
    sw["Method"] = sw["method"].map(MMAP)
    post = (sw.groupby(["Method", "encoder", "dk", "size"])
            [["uniformity_t2", "neighbor_overlap_k50"]].mean().reset_index())
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo["dk"] = geo["dataset"].str.lower()
    pre = geo[geo.dataset != "imagenet"][["encoder", "dk", "uniformity_t2",
                                          "neighbor_overlap_k50"]].rename(
        columns={"uniformity_t2": "unif_pre", "neighbor_overlap_k50": "ov_pre"})
    m = post.merge(pre, on=["encoder", "dk"])
    m["dunif"] = m["uniformity_t2"] - m["unif_pre"]
    m["dov"] = m["neighbor_overlap_k50"] - m["ov_pre"]
    cpsz = cp.groupby("dk")["size"].apply(lambda s: sorted(set(s))).to_dict()
    m["size_c"] = m.apply(lambda r: r["size"] if r["size"] in cpsz.get(r["dk"], [])
                          else min(cpsz[r["dk"]], key=lambda x: abs(x - r["size"])), axis=1)
    beh = (cp[cp["Backbone"].isin(["DINOv3", "CLIP", "MAE"])]
           [["Method", "Backbone", "dk", "size", "dknn"]]
           .rename(columns={"Backbone": "encoder", "size": "size_c"}))
    return m.merge(beh, on=["Method", "encoder", "dk", "size_c"])


def position_residual(cell, fit_scope="all"):
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    m = geo.merge(cell, on=["encoder", "dataset"])
    parts = []
    for enc in SPHERE:
        g = m[m.encoder == enc].copy()
        if fit_scope == "fg":
            g = g[g.dataset.isin(FG)].copy()
        xr = rankdata(g["uniformity_t2"]).reshape(-1, 1)
        yr = rankdata(g["dknn"])
        g["resid"] = yr - LinearRegression().fit(xr, yr).predict(xr)
        parts.append(g)
    return pd.concat(parts)


def main():
    cp = refresh_cp_long()

    # ---- BLOCK 1: F1, 3-method refreshed vs 2-method original ----
    print("\n" + "=" * 100)
    print("BLOCK 1 — F1 position law: Δ@MAX = mean(LeJEPA,SimCLR,DIET) REFRESHED"
          "   [brackets: 2-method original]")
    print("=" * 100)
    OLD_F1 = {("DINOv3", "uniformity_t2"): (0.596, -0.532), ("DINOv3", "mmd_rbf"): (0.693, -0.582),
              ("DINOv3", "neighbor_overlap_k50"): (-0.537, 0.661),
              ("CLIP", "uniformity_t2"): (0.754, -0.486), ("CLIP", "mmd_rbf"): (0.743, -0.661),
              ("CLIP", "neighbor_overlap_k50"): (-0.673, 0.549),
              ("MAE", "uniformity_t2"): (-0.336, -0.500), ("MAE", "mmd_rbf"): (-0.229, -0.574),
              ("MAE", "neighbor_overlap_k50"): (0.500, 0.325)}
    cell3 = delta_at_max(cp, ["LeJEPA-CP", "SimCLR-CP", "DIET-CP"])
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    for enc in ["DINOv3", "CLIP", "MAE"]:
        g = geo[geo.encoder == enc].merge(cell3[cell3.encoder == enc], on="dataset")
        print(f"--- {enc} (n={len(g)}) ---")
        for met in ["uniformity_t2", "mmd_rbf", "neighbor_overlap_k50"]:
            rk, pk = spearmanr(g[met], g["dknn"])
            rf, pf = spearmanr(g[met], g["dft"])
            ok, of = OLD_F1[(enc, met)]
            print(f"  {met:22s} dknn {rk:+.3f}(p={pk:.3f}) [{ok:+.3f}]   "
                  f"dft {rf:+.3f}(p={pf:.3f}) [{of:+.3f}]")

    # ---- BLOCK 2: per-method position law (the section-6.8 table, fresh DIET) ----
    print("\n" + "=" * 100)
    print("BLOCK 2 — per-method position law @MAX (refreshed)"
          "   [brackets: report v-6.8 provisional (DIET = epoch-74)]")
    print("=" * 100)
    OLD_68 = {("LeJEPA-CP", "DINOv3"): (0.59, -0.53), ("SimCLR-CP", "DINOv3"): (0.65, -0.59),
              ("DIET-CP", "DINOv3"): (0.68, -0.63), ("MAE-CP", "DINOv3"): (0.20, -0.30),
              ("LeJEPA-CP", "CLIP"): (0.78, -0.68), ("SimCLR-CP", "CLIP"): (0.77, -0.71),
              ("DIET-CP", "CLIP"): (0.71, -0.54), ("MAE-CP", "CLIP"): (0.43, -0.54)}
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        cellm = delta_at_max(cp, [meth])
        line = f"  {meth:10s}"
        for enc in SPHERE:
            g = geo[geo.encoder == enc].merge(cellm[cellm.encoder == enc], on="dataset")
            ru, _ = spearmanr(g["uniformity_t2"], g["dknn"])
            ro, _ = spearmanr(g["neighbor_overlap_k50"], g["dknn"])
            ou, oo = OLD_68[(meth, enc)]
            line += f"   {enc}: unif {ru:+.2f}[{ou:+.2f}] ov {ro:+.2f}[{oo:+.2f}]"
        print(line)

    # ---- BLOCK 3: F2 refreshed (pooled + DIET collision FINAL) ----
    print("\n" + "=" * 100)
    print("BLOCK 3 — F2 on fixed sweep with REFRESHED behavior   [brackets: pre-refresh]")
    print("=" * 100)
    j = f2_join(cp)
    sph = j[j.encoder.isin(SPHERE)]
    r, p = spearmanr(sph["dunif"], sph["dknn"])
    print(f"  spread pooled          {r:+.3f} (p={p:.1e}, n={len(sph)})  [-0.554]")
    r, p = partial_spearman(sph["dov"].values, sph["dknn"].values, [sph["dunif"].values])
    print(f"  collision partial      {r:+.3f} (p={p:.1e})               [-0.378]")
    OLD_M = {"LeJEPA-CP": (-0.483, -0.442), "SimCLR-CP": (-0.509, -0.418),
             "DIET-CP": (-0.686, -0.110), "MAE-CP": (-0.339, 0.065)}
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        c = sph[sph.Method == meth]
        ru, pu = spearmanr(c["dunif"], c["dknn"])
        ro, po = partial_spearman(c["dov"].values, c["dknn"].values, [c["dunif"].values])
        ou, oo = OLD_M[meth]
        tag = "  <-- DIET FINAL" if meth == "DIET-CP" else ""
        print(f"    {meth:10s} spread {ru:+.3f}[{ou:+.3f}]  collision {ro:+.3f}"
              f"(p={po:.3f})[{oo:+.3f}]{tag}")

    # ---- BLOCK 4: class-force refreshed ----
    print("\n" + "=" * 100)
    print("BLOCK 4 — class-scramble channel d_cdnv with REFRESHED behavior   [pre-refresh]")
    print("=" * 100)
    post = pd.read_csv(OUT / "postcp_class_max.csv")
    post["Method"] = post["method"].map(MMAP)
    post = (post.groupby(["Method", "encoder", "dataset"])[["cdnv", "within_spread"]]
            .mean().reset_index())
    prec = pd.read_csv(OUT / "geometry_class_15.csv")
    m4 = post.merge(prec[["encoder", "dataset", "cdnv", "within_spread"]],
                    on=["encoder", "dataset"], suffixes=("", "_pre"))
    m4["d_cdnv"] = m4["cdnv"] - m4["cdnv_pre"]
    m4["d_within"] = m4["within_spread"] - m4["within_spread_pre"]
    mxj = j[j["size"] == j.groupby(["dk"])["size"].transform("max")]
    mxj = mxj.rename(columns={"dk": "dataset"})[
        ["Method", "encoder", "dataset", "dunif", "dov", "dknn"]]
    beh_ft = (cp[cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
              .groupby(["Method", "Backbone", "dk"])["dft"].mean().reset_index()
              .rename(columns={"Backbone": "encoder", "dk": "dataset"}))
    m4 = m4.merge(mxj, on=["Method", "encoder", "dataset"]).merge(
        beh_ft, on=["Method", "encoder", "dataset"])
    s4 = m4[m4.encoder.isin(SPHERE)]
    r, p = partial_spearman(s4["d_cdnv"].values, s4["dknn"].values,
                            [s4["dunif"].values, s4["dov"].values])
    print(f"  d_cdnv partial|dunif,dov -> dknn: {r:+.3f} (p={p:.1e}, n={len(s4)})  [-0.469]")
    mae4 = m4[m4.encoder == "MAE"]
    for meth in ["LeJEPA-CP", "SimCLR-CP", "DIET-CP", "MAE-CP"]:
        c = mae4[mae4.Method == meth]
        r, p = spearmanr(c["d_cdnv"], c["dknn"])
        print(f"    MAE-enc {meth:10s} d_cdnv->dknn {r:+.3f} (p={p:.3f})")

    # ---- BLOCK 5: packing FG-7, 3-method refreshed target ----
    print("\n" + "=" * 100)
    print("BLOCK 5 — packing axis FG-7, 3-method REFRESHED target"
          "   [2-method stale: DINOv3 +0.786 / CLIP +0.714; 3-method stale: +0.893 / +0.714]")
    print("=" * 100)
    prec["nmargin"] = prec["center_margin"] / prec["between_spread"]
    for scope, label in [("all", "residual from GLOBAL 15-ds fit"),
                         ("fg", "residual from WITHIN-FG-7 fit (verifier objection)")]:
        mr = position_residual(cell3, fit_scope=scope).merge(
            prec[["encoder", "dataset", "center_margin"]], on=["encoder", "dataset"])
        line = f"  [{label}]"
        for enc in SPHERE:
            g = mr[(mr.encoder == enc) & mr.dataset.isin(FG)]
            r, p = spearmanr(g["center_margin"], g["resid"])
            line += f"  {enc}: {r:+.3f}(p={p:.3f})"
        print(line)

    # ---- BLOCK 6: margin predictor, 3-method refreshed target ----
    print("\n" + "=" * 100)
    print("BLOCK 6 — margin predictor, 3-method REFRESHED target   [2-method stale values]")
    print("=" * 100)
    gpred = geo.merge(prec[["encoder", "dataset", "nmargin"]], on=["encoder", "dataset"])
    mp = gpred.merge(cell3, on=["encoder", "dataset"], how="left")
    mp["help"] = np.where(mp["dknn"].notna(), (mp["dknn"] > 0).astype(float), np.nan)

    def zsc(df, feats):
        o = df.copy()
        for f in feats:
            o[f] = df.groupby("encoder")[f].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
        return o

    def loo(df, feats):
        s = zsc(df[df.encoder.isin(SPHERE)].dropna(subset=["help"]).reset_index(drop=True), feats)
        s["help"] = s["help"].astype(int)
        proba = np.full(len(s), np.nan)
        for ds in s.dataset.unique():
            tr, te = s[s.dataset != ds], s[s.dataset == ds]
            clf = LogisticRegression(max_iter=1000).fit(tr[feats], tr.help)
            proba[te.index] = clf.predict_proba(te[feats])[:, 1]
        return (balanced_accuracy_score(s.help, (proba >= 0.5).astype(int)),
                roc_auc_score(s.help, proba))

    F2f = ["neighbor_overlap_k50", "uniformity_t2"]
    for name, feats, old in [("2-feature", F2f, "0.750/0.843"),
                             ("3-feature", F2f + ["nmargin"], "0.778/0.875")]:
        ba, auc = loo(mp, feats)
        print(f"  {name} LOO  bal-acc={ba:.3f}  AUC={auc:.3f}   [2-method stale: {old}]")
    # SigLIP unchanged by refresh (rule frozen; realized SigLIP untouched) — skip rescoring.

    # ---- BLOCK 7: P-D recheck ----
    print("\n" + "=" * 100)
    print("BLOCK 7 — P-D recheck with REFRESHED dft   [pre-refresh]")
    print("=" * 100)
    r, p = spearmanr(s4["d_within"], s4["dft"])
    print(f"  d_within -> dft pooled sphere@MAX: {r:+.3f} (p={p:.2f}, n={len(s4)})  [-0.132]")
    dw = s4.groupby("Method")["d_within"].mean().round(4)
    print("  Δwithin by method (geometry-only, unchanged):", dw.to_dict())

    print("\nDone. Exp B (test3): overall recovery 0.594 [0.609], CLIP 0.329 [0.373], "
          "octmnist-CLIP 0.755 [1.161 capped] — conclusion intact (aggregation failure dominant).")


if __name__ == "__main__":
    main()
