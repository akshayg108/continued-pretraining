#!/usr/bin/env python
"""
nd7_verdict.py — ND7 (CPU adjudicator): does CP move class information up the ranking,
and is that movement the mediator of the kNN benefit?

The story's spine ("CP reranks label placement") has so far only indirect evidence
(Delta-rank <-> Delta-kNN + kernel theory). ND7 measures it directly: Delta cC_K =
post-CP label placement (nd7_placement.csv) minus pre-CP placement (nd6_alignment.csv),
per MAX cell, seed-averaged.

Pre-registered readouts (declared 2026-07-11, BEFORE nd7_placement.csv existed).
Primary scope = main grid (DINOv3/CLIP/MAE x LeJEPA/SimCLR/DIET, dknn from results.xlsx
is_max rows); SigLIP is supplementary (dataset-level 2-method mean vs c2_siglip_score).
Primary placement variant = cC_K (class-centered label power in the top-#classes modes;
ND6's strongest non-circular scalar). caucC_log reported as robustness.

  ND7-1 SPINE: rho(d_cC_K, dknn) > 0 on ALL THREE main encoders (n=45 cells each:
        3 methods x 15 datasets). PASS = "CP helps kNN where it lifts label placement"
        becomes a measured statement.
  ND7-2 MEDIATION: pooled over the main grid (n=135), the partial Spearman
        rho(d_rankme, dknn | d_cC_K) shrinks below HALF of the raw rho(d_rankme, dknn).
        PASS = the shape channel (rank) acts on kNN THROUGH placement.
  ND7-3 PLACEMENT ATTRACTOR: rho(cC_K_pre, d_cC_K) < 0 on all three main encoders,
        WITH the honesty checks that separate an attractor from regression-to-mean
        (rho(pre, post) strongly positive AND std(post) < std(pre)) — mirroring the
        rank-attractor analysis of 2026-07-11.
n=45/135 point estimates; sign-pattern evidence, no FDR family (diagnostic round).

Inputs: eval/outputs/nd7_placement.csv (cluster), nd6_alignment.csv, nd1_precp_spectral.csv,
results.xlsx, c2_siglip_score.csv.
Run (local): python eval/new_direction/nd7_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MAIN_ENCS = ["DINOv3", "CLIP", "MAE"]
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP"}


def partial_spearman(x, y, control):
    """Spearman of x,y after removing the rank-linear effect of the control
    (same construction as eval/F3_dynamics/merge_rest_geometry.partial_spearman)."""
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(control)
    A = np.column_stack([cr, np.ones_like(cr)])
    xres = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    yres = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return spearmanr(xres, yres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default=str(OUT / "nd7_placement.csv"))
    ap.add_argument("--pre-align", default=str(OUT / "nd6_alignment.csv"))
    ap.add_argument("--pre-spec", default=str(OUT / "nd1_precp_spectral.csv"))
    args = ap.parse_args()
    if not _P(args.placement).exists():
        sys.exit(f"MISSING {args.placement} — run the ND7 GPU pass first "
                 f"(run/slurm/new_direction/nd7_placement.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd7_placement_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd7_placement.csv', index=False)\"")

    post = pd.read_csv(args.placement)
    cell = (post.groupby(["method", "encoder", "dataset"])
            [["cC_K", "caucC_log", "rankme"]].mean().reset_index()
            .rename(columns={"cC_K": "cC_K_post", "caucC_log": "caucC_post",
                             "rankme": "rankme_post"}))
    pre_a = pd.read_csv(args.pre_align)[["encoder", "dataset", "cC_K", "caucC_log"]]
    pre_a = pre_a.rename(columns={"cC_K": "cC_K_pre", "caucC_log": "caucC_pre"})
    pre_s = pd.read_csv(args.pre_spec)[["encoder", "dataset", "rankme"]]
    pre_s = pre_s.rename(columns={"rankme": "rankme_pre"})
    t = cell.merge(pre_a, on=["encoder", "dataset"]).merge(pre_s, on=["encoder", "dataset"])
    t["d_cC_K"] = t.cC_K_post - t.cC_K_pre
    t["d_caucC"] = t.caucC_post - t.caucC_pre
    t["d_rankme"] = t.rankme_post - t.rankme_pre

    beh = load_long()
    beh = (beh[beh.is_max & beh.Method.isin(MMAP.values())
               & beh.Backbone.isin(MAIN_ENCS)]
           .groupby(["Backbone", "Method", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    main_t = (t.assign(Method=t.method.map(MMAP))
              .merge(beh, on=["encoder", "Method", "dataset"]))
    main_t.to_csv(OUT / "nd7_joined.csv", index=False)
    print(f"main grid joined: {len(main_t)} cells "
          f"(expect ~135 = 3 methods x 3 encoders x 15 datasets)")

    # ---- ND7-1: the spine --------------------------------------------------------------
    print("\n" + "=" * 88)
    print("ND7-1 SPINE  rho(d_cC_K, dknn) per encoder  [robustness: d_caucC]")
    print("=" * 88)
    ok = 0
    for enc in MAIN_ENCS:
        g = main_t[main_t.encoder == enc]
        r, p = spearmanr(g.d_cC_K, g.dknn)
        r2, _ = spearmanr(g.d_caucC, g.dknn)
        ok += r > 0
        print(f"  {enc:>7}: rho={r:+.3f} (p={p:.4f}, n={len(g)})   [caucC: {r2:+.3f}]")
    print(f"  verdict: {'PASS' if ok == 3 else 'FAIL'} (need >0 on all 3)")

    # ---- ND7-2: mediation --------------------------------------------------------------
    print("\nND7-2 MEDIATION (pooled main grid):")
    raw, praw = spearmanr(main_t.d_rankme, main_t.dknn)
    part, ppart = partial_spearman(main_t.d_rankme, main_t.dknn, main_t.d_cC_K)
    print(f"  raw     rho(d_rankme, dknn)          = {raw:+.3f} (p={praw:.4f})")
    print(f"  partial rho(d_rankme, dknn | d_cC_K) = {part:+.3f} (p={ppart:.4f})")
    v2 = abs(part) < 0.5 * abs(raw)
    print(f"  verdict: {'PASS — rank acts through placement' if v2 else 'FAIL — rank has a placement-independent channel'} "
          f"(pre-registered: |partial| < 0.5 x |raw|)")
    rev, prev = partial_spearman(main_t.d_cC_K, main_t.dknn, main_t.d_rankme)
    print(f"  (reverse control: rho(d_cC_K, dknn | d_rankme) = {rev:+.3f}, p={prev:.4f} — "
          f"placement should RETAIN signal when rank is controlled)")

    # ---- ND7-3: placement attractor + regression-to-mean honesty checks -----------------
    print("\nND7-3 PLACEMENT ATTRACTOR:")
    ok3 = 0
    for enc in MAIN_ENCS:
        g = main_t[main_t.encoder == enc]
        r, p = spearmanr(g.cC_K_pre, g.d_cC_K)
        ok3 += r < 0
        print(f"  {enc:>7}: rho(cC_K_pre, d_cC_K) = {r:+.3f} (p={p:.4f})")
    r_pp = spearmanr(main_t.cC_K_pre, main_t.cC_K_post).correlation
    print(f"  honesty: rho(pre, post) = {r_pp:+.3f}; std pre {main_t.cC_K_pre.std():.4f} "
          f"-> post {main_t.cC_K_post.std():.4f}")
    q = main_t.assign(b=pd.qcut(main_t.cC_K_pre, 4,
                                labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]))
    print(q.groupby("b", observed=True)[["cC_K_pre", "cC_K_post"]].mean().round(3).to_string())
    v3 = ok3 == 3 and r_pp > 0.5 and main_t.cC_K_post.std() < main_t.cC_K_pre.std()
    print(f"  verdict: {'PASS (attractor, not pure RTM)' if v3 else 'FAIL / not attractor-shaped'}")

    # ---- SigLIP supplementary (dataset-level, 2-method mean both sides) -----------------
    sig = t[(t.encoder == "SigLIP") & t.method.isin(["LeJEPA", "SimCLR"])]
    c2_path = OUT / "c2_siglip_score.csv"
    if len(sig) and c2_path.exists():
        s = sig.groupby("dataset")[["d_cC_K"]].mean().reset_index()
        c2 = pd.read_csv(c2_path)[["dataset", "real_dknn"]]
        j = s.merge(c2, on="dataset")
        r, p = spearmanr(j.d_cC_K, j.real_dknn)
        print(f"\nSigLIP supplementary (dataset-level, n={len(j)}): "
              f"rho(d_cC_K, real_dknn) = {r:+.3f} (p={p:.4f})")
    print(f"\nwrote -> {OUT / 'nd7_joined.csv'}")


if __name__ == "__main__":
    main()
