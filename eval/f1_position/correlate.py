#!/usr/bin/env python
"""
correlate.py — Join recomputed geometry (geometry_metrics.py) to CP outcomes
(results.xlsx) and test Hypothesis 1 on the full 15-dataset benchmark.

What it does, per ENCODER (this is FIX vs the original, which pooled a single
encoder-averaged Δ across all encoders):
  A. Spearman / partial-Spearman(|norm) / LOO of each geometry metric vs
     Δ kNN / Δ LP / Δ FT, using that encoder's Δ at MAX averaged over the
     chosen CP methods (default: invariance = LeJEPA-CP + SimCLR-CP).
  B. New-dataset prediction check: does each new dataset's Δ kNN land where
     hypothesis_1.md Finding 5 predicted? (point check + the proper rank check).
  C. Claim 4: does neighbor_overlap predict the kNN-vs-FT sign reversal?
  D. MMD-identity diagnostic: is MMD^2 = M_PP + exp(L_uniform(Q)) - 2 M_PQ
     actually true as computed? (hypothesis_1.md asserts it as exact.)

Join key: geometry is per (encoder, dataset) on a <=5000 train subset; Δ is at
MAX. Both are the "full-data" regime, so they are comparable.

Usage:
  python eval/correlate.py --geometry eval/outputs/geometry_15.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # eval/ root for shared modules
from load_results import load_long, NEW_DATASETS

ROOT = Path(__file__).resolve().parent.parent.parent
INVARIANCE = ["LeJEPA-CP", "SimCLR-CP"]

GEOM_METRICS = ["cosine_dist_centroid", "mmd_rbf", "neighbor_overlap_k20",
                "neighbor_overlap_k50", "uniformity_t2", "uniformity_t2_raw"]

# hypothesis_1.md Finding 5 point predictions for the 7 new datasets' Δ kNN.
FINDING5_PRED = {
    "dtd": (-0.10, 0.02), "cars196": (-0.08, 0.02), "cub200": (-0.08, 0.02),
    "flowers102": (-0.08, 0.02), "oxford_pet": (-0.08, 0.02),
    "eurosat": (0.02, 0.08), "plant_village": (0.02, 0.08),
}


def partial_spearman(xs, ys, controls):
    xs, ys = np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()
    C = np.column_stack([np.asarray(c, float).ravel() for c in controls])
    xr, yr = rankdata(xs), rankdata(ys)
    Cr = np.column_stack([rankdata(C[:, i]) for i in range(C.shape[1])])
    x_res = xr - LinearRegression().fit(Cr, xr).predict(Cr)
    y_res = yr - LinearRegression().fit(Cr, yr).predict(Cr)
    rho, p = spearmanr(x_res, y_res)
    return float(rho), float(p)


def loo_predict_spearman(xs, ys):
    xs, ys = np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()
    n = len(xs)
    if n < 5:
        return float("nan"), float("nan")
    preds, actuals = [], []
    full_xr, full_yr = rankdata(xs), rankdata(ys)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        lr = LinearRegression().fit(rankdata(xs[mask]).reshape(-1, 1), rankdata(ys[mask]))
        preds.append(float(lr.predict([[full_xr[i]]])[0]))
        actuals.append(float(full_yr[i]))
    rho, p = spearmanr(preds, actuals)
    return float(rho), float(p)


def build_delta(df, methods=INVARIANCE):
    """Per (encoder, dataset_key) Δ at MAX, mean over `methods`."""
    d = df[df.Method.isin(methods) & df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    g = (d.groupby(["Backbone", "dataset_key"])[["dknn", "dlp", "dft"]]
         .mean().reset_index().rename(columns={"Backbone": "encoder"}))
    return g


def correlations(geom, delta):
    rows = []
    for enc in ["DINOv3", "CLIP", "MAE"]:
        g = geom[(geom.encoder == enc) & (geom.dataset != "imagenet")]
        d = delta[delta.encoder == enc]
        m = g.merge(d, left_on="dataset", right_on="dataset_key", how="inner")
        have_norm = {"l2_norm_mean", "l2_norm_cv"}.issubset(m.columns)
        for metric in GEOM_METRICS:
            if metric not in m.columns:
                continue
            sub = m[[metric, "dknn", "dlp", "dft", "l2_norm_mean", "l2_norm_cv"]].dropna(subset=[metric])
            sub = sub[pd.to_numeric(sub[metric], errors="coerce").notna()]
            if len(sub) < 4:
                continue
            xs = sub[metric].astype(float).values
            out = {"encoder": enc, "metric": metric, "n": len(sub)}
            for tgt in ["dknn", "dlp", "dft"]:
                ys = sub[tgt].astype(float).values
                rho, p = spearmanr(xs, ys)
                out[f"rho_{tgt}"], out[f"p_{tgt}"] = rho, p
                if len(sub) >= 5 and have_norm:
                    pr, _ = partial_spearman(xs, ys, [sub.l2_norm_mean.values, sub.l2_norm_cv.values])
                    lr, _ = loo_predict_spearman(xs, ys)
                else:
                    pr = lr = float("nan")
                out[f"partial_{tgt}"], out[f"loo_{tgt}"] = pr, lr
            rows.append(out)
    return pd.DataFrame(rows)


def new_dataset_check(delta):
    print("\n" + "=" * 78)
    print("B. NEW-DATASET PREDICTION CHECK (hypothesis_1 Finding 5 vs actual Δ kNN@MAX)")
    print("=" * 78)
    for enc in ["DINOv3", "CLIP"]:
        d = delta[delta.encoder == enc].set_index("dataset_key")
        print(f"\n--- {enc} ---  (predicted range -> actual; PASS if in range)")
        for ds, (lo, hi) in FINDING5_PRED.items():
            if ds not in d.index:
                continue
            actual = d.loc[ds, "dknn"]
            ok = "PASS" if lo <= actual <= hi else "FAIL"
            print(f"  {ds:14s} pred[{lo:+.2f},{hi:+.2f}] -> {actual:+.3f}  {ok}")


def claim4_reversal(geom, delta, overlap_col="neighbor_overlap_k50"):
    print("\n" + "=" * 78)
    print(f"C. CLAIM 4: {overlap_col} vs Δ kNN and Δ FT per encoder (sign should flip)")
    print("=" * 78)
    for enc in ["DINOv3", "CLIP", "MAE"]:
        g = geom[(geom.encoder == enc) & (geom.dataset != "imagenet")]
        if overlap_col not in g.columns:
            print(f"  {enc}: no overlap column (ImageNet-skipped run?)"); continue
        m = g.merge(delta[delta.encoder == enc], left_on="dataset", right_on="dataset_key")
        m = m[pd.to_numeric(m[overlap_col], errors="coerce").notna()]
        if len(m) < 4:
            print(f"  {enc}: n={len(m)} too few"); continue
        x = m[overlap_col].astype(float).values
        rk, _ = spearmanr(x, m.dknn.values)
        rf, _ = spearmanr(x, m.dft.values)
        flip = "REVERSAL CONFIRMED" if (rk < 0 < rf) else "no clean reversal"
        print(f"  {enc:7s} n={len(m):2d}  ρ(overlap,Δknn)={rk:+.3f}  ρ(overlap,Δft)={rf:+.3f}  -> {flip}")


def mmd_identity_diagnostic(geom):
    print("\n" + "=" * 78)
    print("D. MMD-IDENTITY DIAGNOSTIC  (hypothesis_1: MMD² = M_PP + exp(L_unif(Q)) − 2 M_PQ, 'exact')")
    print("    mapping: target-self mmd_m_pp_target == M_QQ == exp(L_unif(Q));"
          " imagenet-self mmd_m_qq_imagenet == M_PP ('constant per encoder')")
    print("=" * 78)
    need = {"mmd_m_pp_target", "uniformity_at_gamma", "mmd_m_qq_imagenet", "mmd_rbf",
            "mmd_m_pq_cross"}
    if not need.issubset(geom.columns):
        print("  geometry CSV lacks MMD components (ImageNet-skipped run?) — diagnostic skipped")
        return
    g = geom[geom.dataset != "imagenet"].copy()
    for c in need:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=list(need))
    # (i) target-self vs exp(uniformity_at_gamma): equal only modulo diagonal inclusion
    g["exp_unif_gamma"] = np.exp(g["uniformity_at_gamma"])
    g["gap_selfkernel_vs_expunif"] = g["mmd_m_pp_target"] - g["exp_unif_gamma"]
    # (ii) is imagenet-self (claimed-constant M_PP) actually constant per encoder?
    for enc in ["DINOv3", "CLIP", "MAE"]:
        sub = g[g.encoder == enc]
        if len(sub) < 2:
            continue
        mpp = sub["mmd_m_qq_imagenet"]
        mean_gap = sub["gap_selfkernel_vs_expunif"].abs().mean()
        print(f"  {enc:7s} M_PP(imagenet self) across datasets: mean={mpp.mean():.4f} "
              f"std={mpp.std():.4f} (CV={mpp.std()/mpp.mean():.3f}) "
              f"| |target_self − exp(unif_γ)| mean={mean_gap:.4f}")
    print("  -> nonzero std(M_PP) means M_PP is NOT constant (γ is data-dependent);")
    print("     nonzero |target_self − exp(unif_γ)| means the 'exact' identity fails (diagonal + t≠γ).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", type=str, default=str(ROOT / "eval/outputs/geometry_15.csv"))
    ap.add_argument("--methods", nargs="+", default=INVARIANCE)
    ap.add_argument("--results", type=str, default=None,
                    help="path to results.xlsx (default: ../results.xlsx next to eval/)")
    ap.add_argument("--out", type=str, default=str(ROOT / "eval/outputs/correlations_15.csv"))
    args = ap.parse_args()

    geom = pd.read_csv(args.geometry)
    long_df = load_long(args.results) if args.results else load_long()
    delta = build_delta(long_df, methods=args.methods)

    print("=" * 78)
    print(f"A. GEOMETRY → Δ CORRELATIONS (per-encoder; Δ@MAX mean over {args.methods})")
    print("=" * 78)
    corr = correlations(geom, delta)
    if corr.empty:
        print("  no overlap between geometry CSV and Δ — check dataset keys / geometry run")
    else:
        for enc in ["DINOv3", "CLIP", "MAE"]:
            sub = corr[corr.encoder == enc]
            if sub.empty:
                continue
            print(f"\n--- {enc} ---")
            for _, r in sub.iterrows():
                print(f"  {r.metric:22s} n={int(r.n):2d}  "
                      f"ρknn={r.rho_dknn:+.3f}(⟂{r.partial_dknn:+.3f},LOO{r.loo_dknn:+.3f})  "
                      f"ρlp={r.rho_dlp:+.3f}  ρft={r.rho_dft:+.3f}")
        corr.to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")

    new_dataset_check(delta)
    claim4_reversal(geom, delta)
    mmd_identity_diagnostic(geom)


if __name__ == "__main__":
    main()
