#!/usr/bin/env python
"""
correlate_second_axis.py — CPU scorer for the second-axis proposals, run AFTER the GPU
scripts land their CSVs. Each block is a pre-registered falsifiable test; each prints
SUPPORTED / REFUTED verdicts with the honesty guards used throughout the project
(cross-encoder sign consistency; rank claims only; n disclosed).

Inputs (produced by the GPU handover scripts):
  eval/outputs/geometry_class_15.csv   <- eval/geometry_class.py            (P-A, P-B, P-C1)
  eval/outputs/postcp_class_max.csv    <- eval/adjudicate/postcp_class_sweep.py  (P-D)
  eval/outputs/geometry_mae_sa.csv     <- eval/adjudicate/mae_sa_geometry.py     (P-C2)
Missing files -> that block is skipped with a note.

Tests:
  P-A  granularity load: among the 7 embedded/FG datasets, position-residual of dknn@MAX is
       negatively rank-correlated with wb_ratio (and nc1_ratio) on BOTH sphere encoders;
       specifically wb_ratio(CUB200) > wb_ratio(Flowers102) on both.
  P-B  task-spectral alignment: DTD has the LOWEST (or bottom-2) task_energy_in_top50 among
       the 7 FG datasets on BOTH sphere encoders; across FG, |position residual| shrinks as
       task_energy_in_top50 falls (muting, not sign flip).
  P-C1 gate continuum: encoder-level mean rankme orders CLIP/DINOv3/SigLIP >> MAE, and the
       per-encoder F1 law strength |rho(uniformity_t2, dknn)| orders with it.
  P-C2 readout confound: F1 correlations for the MAE encoder recomputed on SA-pooled
       geometry — report side-by-side with the mean-pool numbers.
  P-D  reversal mediation: Δwithin_spread -> dft negative on sphere@MAX; partialling it out
       weakens geometry->dft; DIET-CP has the largest Δwithin among angular methods.

Run: python eval/adjudicate/correlate_second_axis.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
FG = ["dtd", "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]
INV = ["LeJEPA-CP", "SimCLR-CP"]


def partial_spearman(x, y, controls):
    xr, yr = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in controls])
    xres = xr - LinearRegression().fit(C, xr).predict(C)
    yres = yr - LinearRegression().fit(C, yr).predict(C)
    return spearmanr(xres, yres)


def behavioral_cell():
    df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    return (inv.groupby(["Backbone", "dataset_key"])
            [["dknn", "dlp", "dft"]].mean().reset_index()
            .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))


def position_residual(cell):
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    m = geo.merge(cell, on=["encoder", "dataset"])
    parts = []
    for enc in ["DINOv3", "CLIP"]:
        g = m[m.encoder == enc].copy()
        xr = rankdata(g["uniformity_t2"]).reshape(-1, 1)
        yr = rankdata(g["dknn"])
        g["resid"] = yr - LinearRegression().fit(xr, yr).predict(xr)
        parts.append(g)
    return pd.concat(parts)


def main():
    cell = behavioral_cell()

    # ---------- P-A + P-B (need geometry_class_15.csv) ----------
    f = OUT / "geometry_class_15.csv"
    if f.exists():
        gc = pd.read_csv(f)
        m = position_residual(cell).merge(gc, on=["encoder", "dataset"])
        print("=" * 88)
        print("P-A granularity load (FG-only, position residual vs class-manifold organization)")
        print("=" * 88)
        for metric in ["wb_ratio", "nc1_ratio", "cdnv", "center_margin", "n_classes"]:
            ok = []
            for enc in ["DINOv3", "CLIP"]:
                g = m[(m.encoder == enc) & m.dataset.isin(FG)]
                r, p = spearmanr(g[metric], g["resid"])
                ok.append(r)
                print(f"  {enc:7s} rho({metric}, resid) = {r:+.3f} (p={p:.3f}, n={len(g)})")
            cons = np.sign(ok[0]) == np.sign(ok[1])
            print(f"  -> cross-encoder consistent: {cons}")
        for enc in ["DINOv3", "CLIP"]:
            g = m[m.encoder == enc].set_index("dataset")
            if {"cub200", "flowers102"} <= set(g.index):
                v = g.loc["cub200", "wb_ratio"], g.loc["flowers102", "wb_ratio"]
                print(f"  {enc}: wb_ratio CUB={v[0]:.3f} vs Flowers={v[1]:.3f} -> "
                      f"{'SUPPORTED' if v[0] > v[1] else 'REFUTED'} (predict CUB > Flowers)")
        print()
        print("=" * 88)
        print("P-B task-spectral alignment (DTD muting)")
        print("=" * 88)
        for enc in ["DINOv3", "CLIP"]:
            g = m[(m.encoder == enc) & m.dataset.isin(FG)].sort_values("task_energy_in_top50")
            ranks = list(g.dataset)
            pos = ranks.index("dtd") + 1 if "dtd" in ranks else -1
            print(f"  {enc:7s} FG ranking by task_energy_in_top50 (low->high): {ranks}")
            print(f"          DTD rank {pos}/{len(ranks)} -> "
                  f"{'SUPPORTED' if pos in (1, 2) else 'REFUTED'} (predict bottom-2)")
            r, p = spearmanr(g["task_energy_in_top50"], g["resid"].abs())
            print(f"          rho(task_energy, |resid|) = {r:+.3f} (p={p:.3f}) "
                  f"(muting predicts NEGATIVE... i.e. low energy -> large positive resid)")
        print()
        print("=" * 88)
        print("P-C1 gate continuum (encoder anisotropy vs law strength)")
        print("=" * 88)
        geo15 = pd.read_csv(OUT / "geometry_15.csv")
        geo15 = geo15[geo15.dataset != "imagenet"]
        strengths, aniso = {}, {}
        for enc in ["DINOv3", "CLIP", "MAE"]:
            g = geo15[geo15.encoder == enc].merge(cell[cell.encoder == enc], on="dataset")
            strengths[enc], _ = spearmanr(g["uniformity_t2"], g["dknn"])
            aniso[enc] = gc[gc.encoder == enc]["rankme"].mean()
        for enc in strengths:
            print(f"  {enc:7s} law strength rho = {strengths[enc]:+.3f} | mean rankme = "
                  f"{aniso.get(enc, np.nan):.1f}")
        print("  predict: rankme(MAE) << rankme(sphere encoders) AND |rho| orders with rankme")
    else:
        print(f"[skip P-A/P-B/P-C1] {f} missing — run eval/geometry_class.py first")

    # ---------- P-C2 (needs geometry_mae_sa.csv) ----------
    f = OUT / "geometry_mae_sa.csv"
    if f.exists():
        sa = pd.read_csv(f)
        sa["encoder"] = "MAE"
        m = sa.merge(cell[cell.encoder == "MAE"], on=["encoder", "dataset"])
        print("\n" + "=" * 88)
        print("P-C2 MAE readout confound: F1 correlations on SA-pooled geometry vs mean-pool")
        print("=" * 88)
        geo15 = pd.read_csv(OUT / "geometry_15.csv")
        mp = geo15[(geo15.encoder == "MAE") & (geo15.dataset != "imagenet")].merge(
            cell[cell.encoder == "MAE"], on="dataset")
        for metric in ["uniformity_t2", "neighbor_overlap_k50", "mmd_rbf"]:
            r_mp, p_mp = spearmanr(mp[metric], mp["dknn"])
            r_sa, p_sa = spearmanr(m[metric], m["dknn"])
            print(f"  {metric:22s} mean-pool rho={r_mp:+.3f}(p={p_mp:.3f}) | "
                  f"SA rho={r_sa:+.3f}(p={p_sa:.3f})")
        print("  sphere-law reference signs: uniformity +, overlap -, mmd + "
              "(if SA flips MAE to these signs -> readout artifact; if not -> genuine)")
    else:
        print(f"\n[skip P-C2] {f} missing — run eval/adjudicate/mae_sa_geometry.py first")

    # ---------- P-D (needs postcp_class_max.csv + geometry_class_15.csv) ----------
    f1, f2 = OUT / "postcp_class_max.csv", OUT / "geometry_class_15.csv"
    if f1.exists() and f2.exists():
        post = pd.read_csv(f1)
        post["Method"] = post["method"].map({"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP",
                                             "MAE": "MAE-CP", "DIET": "DIET-CP"})
        post = (post.groupby(["Method", "encoder", "dataset"])
                [["within_spread", "between_spread"]].mean().reset_index())
        pre = pd.read_csv(f2)[["encoder", "dataset", "within_spread", "between_spread"]]
        pre = pre.rename(columns={"within_spread": "w_pre", "between_spread": "b_pre"})
        m = post.merge(pre, on=["encoder", "dataset"])
        m["d_within"] = m["within_spread"] - m["w_pre"]
        m["d_between"] = m["between_spread"] - m["b_pre"]
        df = load_long()
        beh = (df[df.is_max & df.Backbone.isin(["DINOv3", "CLIP"])]
               .groupby(["Method", "Backbone", "dataset_key"])[["dft", "dknn"]].mean()
               .reset_index().rename(columns={"Backbone": "encoder",
                                              "dataset_key": "dataset"}))
        j = m.merge(beh, on=["Method", "encoder", "dataset"])
        sph = j[j.encoder.isin(["DINOv3", "CLIP"])]
        print("\n" + "=" * 88)
        print("P-D reversal mediation (Δwithin-class spread -> ΔFT, sphere @MAX)")
        print("=" * 88)
        r, p = spearmanr(sph["d_within"], sph["dft"])
        print(f"  pooled rho(d_within, dft) = {r:+.3f} (p={p:.1e}, n={len(sph)})  "
              f"(predict NEGATIVE)")
        r2, p2 = spearmanr(sph["d_within"], sph["dknn"])
        print(f"  reference rho(d_within, dknn) = {r2:+.3f} (p={p2:.1e})")
        geo15 = pd.read_csv(OUT / "geometry_15.csv")[["encoder", "dataset",
                                                      "neighbor_overlap_k50"]]
        jj = sph.merge(geo15, on=["encoder", "dataset"])
        r3, _ = spearmanr(jj["neighbor_overlap_k50"], jj["dft"])
        r4, _ = partial_spearman(jj["neighbor_overlap_k50"].values, jj["dft"].values,
                                 [jj["d_within"].values])
        print(f"  mediation: rho(overlap, dft) = {r3:+.3f} -> partial(| d_within) = {r4:+.3f} "
              f"(predict shrink toward 0)")
        print("  Δwithin by angular method (predict DIET largest):")
        print(sph.groupby("Method")["d_within"].mean().round(4).to_string())
    else:
        print(f"\n[skip P-D] need {f1.name} + {f2.name}")


if __name__ == "__main__":
    main()
