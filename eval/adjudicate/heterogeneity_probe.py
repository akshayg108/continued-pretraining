#!/usr/bin/env python
"""
heterogeneity_probe.py — Adjudication + exploration: can ANY covariate available in the
existing artifacts resolve (a) the within-fine-grained heterogeneity (CUB200 -0.475 vs
Flowers102 -0.002 at near-identical angular position) or (b) the DTD exception?

EXPLORATORY by design: every correlation is reported with BH-FDR over the full sweep, and the
honesty guard is CROSS-ENCODER SIGN CONSISTENCY (a covariate only counts if it moves the same
way on DINOv3 and CLIP, the standard the DTD diagnostic already used). n is tiny (15 / 6-7 FG);
this can only NOMINATE second-axis candidates, not confirm them.

Covariates (existing data only):
  task scale:   n_classes, samples_per_class_at_MAX (= MAX size / n_classes), log10 MAX size
  baseline:     knn_pre, lp_pre, ft_pre, lp_minus_knn_gap
  geometry ext: uniformity_at_gamma, mmd_gamma (median-heuristic bandwidth ~ density scale),
                mmd_m_pp_target (self-kernel mass ~ cluster tightness), mmd_m_pq_cross,
                overlap_k20/overlap_k50 ratio (neighborhood-scale profile),
                l2_norm_cv, l2_norm_mean, uniformity_t2_raw

Targets:
  T1 position residual of dknn@MAX (invariance mean) after per-encoder rank fit on
     uniformity_t2 (the same residual construction FINDINGS_step2 used for DTD).
  T2 dknn@MAX within the FG subgroup only.

Run: python eval/adjudicate/heterogeneity_probe.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
INV = ["LeJEPA-CP", "SimCLR-CP"]
FG = ["dtd", "food101", "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet"]

N_CLASSES = {"breastmnist": 2, "dermamnist": 7, "octmnist": 4, "organamnist": 11,
             "pathmnist": 9, "galaxy10": 10, "eurosat": 10, "plant_village": 38,
             "dtd": 47, "food101": 101, "fgvc_aircraft": 100, "cars196": 196,
             "cub200": 200, "flowers102": 102, "oxford_pet": 37}


def bh_fdr(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank_i, idx in enumerate(order[::-1]):
        i = len(p) - rank_i
        prev = min(prev, p[idx] * len(p) / i)
        q[idx] = prev
    return q


def main():
    df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max & df.Backbone.isin(["DINOv3", "CLIP"])]
    cell = (inv.groupby(["Backbone", "dataset_key"])
            .agg(dknn=("dknn", "mean"), knn_pre=("knn_pre", "mean"), lp_pre=("lp_pre", "mean"),
                 ft_pre=("ft_pre", "mean"), size=("size", "max")).reset_index())
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"].copy()
    m = geo.merge(cell, left_on=["encoder", "dataset"],
                  right_on=["Backbone", "dataset_key"])
    m["n_classes"] = m["dataset"].map(N_CLASSES)
    m["samples_per_class"] = m["size"] / m["n_classes"]
    m["log_size"] = np.log10(m["size"])
    m["lp_knn_gap"] = m["lp_pre"] - m["knn_pre"]
    m["ov_ratio_k20_k50"] = m["neighbor_overlap_k20"] / m["neighbor_overlap_k50"].replace(0, np.nan)

    # T1: position residual (rank-fit dknn ~ uniformity_t2 per encoder, as in FINDINGS_step2)
    res = []
    for enc in ["DINOv3", "CLIP"]:
        g = m[m.encoder == enc].copy()
        xr = rankdata(g["uniformity_t2"]).reshape(-1, 1)
        yr = rankdata(g["dknn"])
        fit = LinearRegression().fit(xr, yr)
        g["resid"] = yr - fit.predict(xr)
        res.append(g)
    m = pd.concat(res)

    COVS = ["n_classes", "samples_per_class", "log_size", "knn_pre", "lp_pre", "ft_pre",
            "lp_knn_gap", "uniformity_at_gamma", "mmd_gamma", "mmd_m_pp_target",
            "mmd_m_pq_cross", "ov_ratio_k20_k50", "l2_norm_cv", "l2_norm_mean",
            "uniformity_t2_raw"]

    for target, sub_label, sel in [("resid", "all-15 position residual", None),
                                   ("dknn", "FG-only dknn", FG)]:
        print("=" * 96)
        print(f"TARGET: {target} ({sub_label})  —  covariate sweep, per encoder + consistency")
        print("=" * 96)
        rows = []
        for cov in COVS:
            r = {}
            for enc in ["DINOv3", "CLIP"]:
                g = m[m.encoder == enc]
                if sel is not None:
                    g = g[g.dataset.isin(sel)]
                gg = g[[cov, target]].dropna()
                if len(gg) < 5 or gg[cov].nunique() < 3:
                    r[enc] = (np.nan, np.nan, len(gg))
                    continue
                rho, p = spearmanr(gg[cov], gg[target])
                r[enc] = (rho, p, len(gg))
            rows.append((cov, r))
        ps = [v for _, r in rows for enc in ["DINOv3", "CLIP"]
              for v in [r[enc][1]] if not np.isnan(v)]
        qmap = dict(zip([(c, e) for c, r in rows for e in ["DINOv3", "CLIP"]
                         if not np.isnan(r[e][1])], bh_fdr(ps)))
        print(f"{'covariate':22s} {'DINOv3 rho(p)[q]':>26s} {'CLIP rho(p)[q]':>26s}  sign-consistent?")
        for cov, r in rows:
            cells = []
            for enc in ["DINOv3", "CLIP"]:
                rho, p, n = r[enc]
                q = qmap.get((cov, enc), np.nan)
                cells.append(f"{rho:+.2f}(p={p:.3f})[q={q:.2f}]" if not np.isnan(rho) else "n/a")
            s1, s2 = r["DINOv3"][0], r["CLIP"][0]
            cons = ("YES" if (not np.isnan(s1) and not np.isnan(s2)
                              and np.sign(s1) == np.sign(s2)
                              and min(abs(s1), abs(s2)) > 0.3) else "-")
            print(f"{cov:22s} {cells[0]:>26s} {cells[1]:>26s}  {cons}")
        print()

    # MATCHED-DOSE contrast (added 2026-07-01): the size sweeps DO exist for cub200/flowers102
    # (the "new-7 ran MAX-only" note in load_results.py was outdated), so the CP-dose confound
    # can be settled on existing data — no new runs needed.
    print("=" * 96)
    print("MATCHED-DOSE: dknn (invariance mean) at comparable CP-set sizes, CUB200 vs FLOWERS102")
    print("=" * 96)
    inv_all = df[df.Method.isin(INV) & df.Backbone.isin(["DINOv3", "CLIP"])]
    tab = (inv_all[inv_all.dataset_key.isin(["cub200", "flowers102"])]
           .groupby(["Backbone", "dataset_key", "size"])["dknn"].mean().round(3))
    print(tab.to_string())
    print("\nVerdict: if CUB@{200,500,1000} damage >> Flowers@{500,1020} damage at matched or "
          "smaller dose, the dose confound is REFUTED and the heterogeneity needs a second "
          "axis (P-A).")

    # The specific CUB-vs-Flowers contrast: print their full covariate rows
    print("=" * 96)
    print("CUB200 vs FLOWERS102 raw covariates (both encoders)")
    print("=" * 96)
    show = ["encoder", "dataset", "dknn", "resid", "uniformity_t2", "neighbor_overlap_k50",
            "n_classes", "samples_per_class", "size", "knn_pre", "lp_knn_gap",
            "mmd_gamma", "mmd_m_pp_target", "ov_ratio_k20_k50"]
    print(m[m.dataset.isin(["cub200", "flowers102", "dtd", "food101"])][show]
          .sort_values(["encoder", "dataset"]).to_string(index=False))


if __name__ == "__main__":
    main()
