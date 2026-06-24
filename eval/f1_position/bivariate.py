#!/usr/bin/env python
"""F2.1: does (angular position, pre-CP baseline) predict ΔkNN better than either alone,
and does position survive controlling for baseline (and vice versa)? Resolves DTD + the
fine-grained-internal heterogeneity.

NOTE vs PLAN snippet: in-sample R^2 trivially increases when a predictor is added, so the
rigorous test is the TWO-DIRECTION partial Spearman (does each variable add signal beyond the
other) plus adjusted R^2 (penalises the extra predictor). Both reported here.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # eval/ root for shared modules
from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent  # continued-pretraining
GEOM = str(ROOT / "eval/outputs/geometry_15.csv")
INV = ["LeJEPA-CP", "SimCLR-CP"]


def partial(x, y, ctrl):
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(ctrl).reshape(-1, 1)
    xres = xr - LinearRegression().fit(cr, xr).predict(cr)
    yres = yr - LinearRegression().fit(cr, yr).predict(cr)
    r, p = spearmanr(xres, yres)
    return float(r), float(p)


def adj_r2(X, y):
    yr = rankdata(y)
    r2 = LinearRegression().fit(X, yr).score(X, yr)
    n, p = X.shape[0], X.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    g = pd.read_csv(GEOM)
    df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max]
    agg = inv.groupby(["Backbone", "dataset_key"]).agg(
        dknn=("dknn", "mean"), knn_pre=("knn_pre", "mean")).reset_index()
    rows = []
    for enc in ["DINOv3", "CLIP", "MAE"]:
        m = g[(g.encoder == enc) & (g.dataset != "imagenet")].merge(
            agg[agg.Backbone == enc], left_on="dataset", right_on="dataset_key")
        for pos in ["uniformity_t2", "neighbor_overlap_k50"]:
            P = rankdata(m[pos].values).reshape(-1, 1)
            B = rankdata(m.knn_pre.values).reshape(-1, 1)
            r_pos, _ = spearmanr(m[pos], m.dknn)
            r_base, _ = spearmanr(m.knn_pre, m.dknn)
            pr_pos, pp_pos = partial(m[pos].values, m.dknn.values, m.knn_pre.values)   # pos | base
            pr_base, pp_base = partial(m.knn_pre.values, m.dknn.values, m[pos].values)  # base | pos
            rows.append(dict(
                encoder=enc, position=pos, n=len(m),
                rho_pos=round(r_pos, 3), rho_base=round(r_base, 3),
                partial_pos_given_base=round(pr_pos, 3), p_pos=round(pp_pos, 3),
                partial_base_given_pos=round(pr_base, 3), p_base=round(pp_base, 3),
                adjR2_pos=round(adj_r2(P, m.dknn.values), 3),
                adjR2_base=round(adj_r2(B, m.dknn.values), 3),
                adjR2_both=round(adj_r2(np.hstack([P, B]), m.dknn.values), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "eval/outputs/bivariate.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False))
    print("\nDTD vs FG-cluster pre-CP baseline (DINOv3) — does low baseline explain DTD's +Δ?:")
    d = agg[agg.Backbone == "DINOv3"].set_index("dataset_key")
    for k in ["dtd", "cub200", "cars196", "food101", "flowers102", "oxford_pet"]:
        print(f"  {k:12s} knn_pre={d.loc[k,'knn_pre']:.3f}  Δknn={d.loc[k,'dknn']:+.3f}")


if __name__ == "__main__":
    main()
