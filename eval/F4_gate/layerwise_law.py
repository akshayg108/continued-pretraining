#!/usr/bin/env python
"""
layerwise_law.py — Exp I scorer: does the position law's strength vary CONTINUOUSLY with
layer geometry across ~48 (encoder, layer) virtual-encoder states?

Consumes eval/outputs/layerwise_pre.csv + layerwise_postcp.csv (concat shards first).
Produces eval/outputs/layerwise_curve.csv + the L1-L3 verdict block of
eval/DESIGN_spectrum_transport.md.

Per (encoder, layer):
  theta_coupling = |Spearman(unif_pre_l, cdnv_pre_l)| across the 15 datasets
  law_rho        = Spearman(unif_pre_l(D), dknn_l(D)) across datasets, where dknn_l is the
                   seed- and method-averaged post-CP internal kNN minus the pre-CP one.

Verdicts:
  L1 sanity  : rankme depth profiles (rise-then-tunnel on sphere encoders; MAE low) — printed
               for inspection, no hard gate.
  L2 MAIN    : Spearman(theta_l, law_rho_l) over ALL (encoder, layer) points; encoder-block
               bootstrap CI (resample encoders) + per-encoder within-curve Spearman.
               Positive & CI>0  -> continuous-spectrum evidence.
               Two clusters / null -> the cliff conclusion is reinforced with 12x the points.
  L3 upside  : MAE layers whose theta reaches the sphere floor (25th pct of sphere thetas):
               does law_rho locally recover (> 0.3 while the MAE final layer is <= 0)?

CPU:  python eval/adjudicate/layerwise_law.py
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
SPHERE = ["DINOv3", "CLIP", "SigLIP"]


def bh_fdr(pvals, q=0.10):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    passed = np.zeros(len(p), bool)
    for rank, idx in enumerate(order, 1):
        if p[idx] <= q * rank / len(p):
            passed[order[:rank]] = True
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(OUT / "layerwise_pre.csv"))
    ap.add_argument("--post", default=str(OUT / "layerwise_postcp.csv"))
    ap.add_argument("--out", default=str(OUT / "layerwise_curve.csv"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.RandomState(42)

    pre = pd.read_csv(args.pre)
    post = pd.read_csv(args.post)
    print(f"pre: {len(pre)} rows ({pre.encoder.nunique()} encoders x "
          f"{pre.dataset.nunique()} datasets x {pre.layer.nunique()} layers)")
    print(f"post: {len(post)} rows, methods={sorted(post.method.unique())}")

    # ---- per-layer delta: method+seed-averaged post knn minus pre knn ------------------
    post_cell = (post.groupby(["encoder", "dataset", "layer"])["knn_internal"]
                 .mean().reset_index().rename(columns={"knn_internal": "knn_post"}))
    m = pre.merge(post_cell, on=["encoder", "dataset", "layer"], how="inner")
    m["dknn_l"] = m["knn_post"] - m["knn_internal"]

    # ---- frozen (dataset, layer) rows: the recipe couples unfreeze depth to dataset size
    # (main grid: n<10k -> last 2 blocks, 10-25k -> 4, 25-50k -> 6, >50k -> all;
    # SigLIP grid: 2 everywhere; verified against run scripts AND empirically 2026-07-08).
    # A frozen layer has delta == 0 identically; keeping those rows poisons law_rho with
    # tie artifacts (this retracted the MAE depth-gradient claim). Drop per ROW, not per
    # layer: detect from the per-ckpt deltas, max over methods/seeds.
    raw = post.merge(pre[["encoder", "dataset", "layer", "knn_internal"]],
                     on=["encoder", "dataset", "layer"], suffixes=("", "_pre"))
    raw["d1"] = raw["knn_internal"] - raw["knn_internal_pre"]
    fro = (raw.groupby(["encoder", "dataset", "layer"])["d1"]
           .apply(lambda s: s.abs().max() < 1e-6).reset_index(name="frozen"))
    m = m.merge(fro, on=["encoder", "dataset", "layer"], how="left")
    n_frozen = int(m["frozen"].sum())
    m = m[~m["frozen"].fillna(False)]
    print(f"dropped {n_frozen} frozen (dataset, layer) rows (unfreeze-depth schedule)")

    # ---- curve: one row per (encoder, layer) -------------------------------------------
    rows = []
    dropped = []
    for (enc, layer), g in m.groupby(["encoder", "layer"]):
        if len(g) < 7 or g.dknn_l.isna().any():
            dropped.append((enc, int(layer), len(g)))
            continue
        rows.append(dict(
            encoder=enc, layer=int(layer),
            theta_coupling=abs(spearmanr(g.uniformity_t2, g.cdnv).correlation),
            law_rho=spearmanr(g.uniformity_t2, g.dknn_l).correlation,
            rankme_med=float(g.rankme.median()),
            n_datasets=len(g),
            n_methods=int(post[(post.encoder == enc)].method.nunique())))
    curve = pd.DataFrame(rows).sort_values(["encoder", "layer"])
    curve = curve.dropna(subset=["law_rho", "theta_coupling"])
    curve.to_csv(args.out, index=False)
    print(f"\ncurve: {len(curve)} valid (encoder, layer) points -> {args.out}")
    print("NOTE: depth range below L9 is untestable BY DESIGN (unfreeze depth is coupled to")
    print("dataset size, so shallow layers are trained for <=5 datasets). The spectrum test")
    print("runs on L9-L12 only; L11/L12 have all 15 datasets.")
    if dropped:
        print(f"dropped {len(dropped)} under-covered (encoder, layer) points: {dropped}")

    # ---- L1: depth profiles --------------------------------------------------------------
    print("\nL1 rankme depth profiles (median across datasets):")
    for enc, g in curve.groupby("encoder"):
        prof = " ".join(f"{v:5.0f}" for v in g.sort_values("layer").rankme_med)
        print(f"  {enc:8} {prof}")
    print("  (expect: sphere encoders rise then collapse in the tunnel segment; MAE low/flat)")

    # ---- L2: main test --------------------------------------------------------------------
    r_all, p_all = spearmanr(curve.theta_coupling, curve.law_rho)
    encs = curve.encoder.unique()
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(encs, size=len(encs), replace=True)
        sub = pd.concat([curve[curve.encoder == e] for e in pick])
        if sub.theta_coupling.nunique() > 2:
            boots.append(spearmanr(sub.theta_coupling, sub.law_rho).correlation)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"\nL2 MAIN pooled Spearman(theta_l, law_rho_l) = {r_all:+.3f} (p={p_all:.4f}, "
          f"n={len(curve)}; NOT independent — layers share an encoder)")
    print(f"   encoder-block bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    per_enc = {}
    pvals, names = [], []
    for enc, g in curve.groupby("encoder"):
        r, p = spearmanr(g.theta_coupling, g.law_rho)
        per_enc[enc] = (r, p, len(g))
        pvals.append(p)
        names.append(enc)
        print(f"   within-{enc:8}: rho={r:+.3f} (p={p:.3f}, n={len(g)})")
    fdr = bh_fdr(pvals)
    print(f"   BH-FDR(q=0.10) within-encoder passes: "
          f"{[n for n, ok in zip(names, fdr) if ok] or 'none'}")
    l2 = "CONTINUOUS-SPECTRUM EVIDENCE" if (r_all > 0 and lo > 0) else \
         "NO spectrum signal on the valid L9-L12 range (deeper range untestable by design)"
    print(f"   L2 verdict: {l2}")

    # ---- L3: MAE local recovery ------------------------------------------------------------
    if "MAE" in per_enc:
        sphere_floor = np.percentile(
            curve[curve.encoder.isin(SPHERE)].theta_coupling, 25)
        mae = curve[curve.encoder == "MAE"].sort_values("layer")
        final_rho = mae.law_rho.iloc[-1]
        cand = mae[mae.theta_coupling >= sphere_floor]
        print(f"\nL3 MAE local recovery (sphere theta floor = {sphere_floor:.3f}; "
              f"MAE final-layer law_rho = {final_rho:+.3f}):")
        if cand.empty:
            print("   no MAE layer reaches the sphere coupling floor -> L3 not triggered")
        else:
            for _, r in cand.iterrows():
                mark = " <-- RECOVERY" if (r.law_rho > 0.3 and final_rho <= 0) else ""
                print(f"   layer {int(r.layer):2d}: theta={r.theta_coupling:.3f} "
                      f"law_rho={r.law_rho:+.3f}{mark}")
            hit = ((cand.law_rho > 0.3) & (final_rho <= 0)).any()
            verdict = ("RECOVERY — gate is a function of geometry, not identity" if hit
                       else "no recovery at sphere-level layers")
            print(f"   L3 verdict: {verdict}")

    print("\nDisclosures: layers within an encoder are NOT independent samples (block "
          "bootstrap is the honest CI); internal kNN protocol differs from production kNN; "
          "SigLIP curve is 2-method (no DIET SigLIP).")


if __name__ == "__main__":
    main()
