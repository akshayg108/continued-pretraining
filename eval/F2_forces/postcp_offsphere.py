#!/usr/bin/env python
"""
postcp_offsphere.py — analyse the scaled Exp A (postcp_sweep.py output).

Tests the quantitative F2 mechanism law: on sphere-native encoders, how far CP pushes the
features OFF the sphere (Δ L2-norm CV = post_cv − pre_cv) predicts the kNN degradation.
  A. mean Δcv by (method, encoder): does MAE-CP raise CV on DINOv3/CLIP while invariance-CP doesn't?
  B. for MAE-CP on {DINOv3,CLIP}: Spearman(Δcv, ΔkNN) over (dataset,size) — expect strong negative
     (more off-sphere → worse ΔkNN). This is the new quantitative result.
  C. data-size dependence: does Δcv grow with CP data for MAE-CP on sphere encoders?

Runs anywhere (pure pandas). Needs postcp_sweep.csv + geometry_15.csv (pre-CP CV) + results.xlsx.

  python eval/postcp_offsphere.py --sweep eval/outputs/postcp_sweep.csv \
      --geometry eval/outputs/geometry_15.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))  # eval/ root for shared modules
from load_results import load_long, add_size_canon

ROOT = Path(__file__).resolve().parent.parent.parent
SPHERE = ["DINOv3", "CLIP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "eval/outputs/postcp_sweep.csv"))
    ap.add_argument("--geometry", default=str(ROOT / "eval/outputs/geometry_15.csv"))
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    sw = pd.read_csv(args.sweep)
    sw = sw[sw.variant == "pretrained"].copy()
    sw["method_cp"] = sw.method + "-CP"
    # post_cv averaged over seeds per (method, encoder, dataset, size)
    post = (sw.groupby(["method_cp", "encoder", "dataset", "size"])
              .agg(post_cv=("l2_norm_cv", "mean"),
                   post_unif=("uniformity_t2", "mean")).reset_index())

    geom = pd.read_csv(args.geometry)
    pre_cv = {(r.encoder, r.dataset): r.l2_norm_cv
              for r in geom[geom.dataset != "imagenet"].itertuples()}
    post["pre_cv"] = post.apply(lambda r: pre_cv.get((r.encoder, r.dataset), np.nan), axis=1)
    post["delta_cv"] = post.post_cv - post.pre_cv

    df = load_long(args.results) if args.results else load_long()
    dknn = (df.groupby(["Method", "Backbone", "dataset_key", "size"])["dknn"].mean()
              .reset_index().rename(columns={"Method": "method_cp", "Backbone": "encoder",
                                             "dataset_key": "dataset"}))
    # Reconcile MAX-label drift (FGVC: ckpt n3334 vs results n3400) by joining on size_canon.
    post = add_size_canon(post, "dataset", "size")
    dknn = add_size_canon(dknn, "dataset", "size")
    m = post.merge(dknn[["method_cp", "encoder", "dataset", "size_canon", "dknn"]],
                   on=["method_cp", "encoder", "dataset", "size_canon"], how="left")
    m.to_csv(ROOT / "eval/outputs/postcp_offsphere.csv", index=False)

    print("=" * 72)
    print("A. mean Δ(L2-norm CV) by method × encoder  (>0 = pushed OFF the sphere)")
    print("=" * 72)
    tab = m.pivot_table(index="method_cp", columns="encoder", values="delta_cv", aggfunc="mean")
    print(tab.round(4).to_string())

    print("\n" + "=" * 72)
    print("B. MAE-CP: does off-sphere movement predict kNN degradation? (sphere encoders)")
    print("=" * 72)
    for enc in SPHERE:
        sub = m[(m.method_cp == "MAE-CP") & (m.encoder == enc)].dropna(subset=["delta_cv", "dknn"])
        if len(sub) >= 4:
            r, p = spearmanr(sub.delta_cv, sub.dknn)
            print(f"  {enc:7s} n={len(sub):3d}  ρ(Δcv, Δknn) = {r:+.3f} (p={p:.3g})  "
                  f"[expect strongly negative]")
        else:
            print(f"  {enc}: n={len(sub)} too few")
    # contrast: invariance methods barely move CV
    print("\n  contrast — mean |Δcv| (sphere encoders):")
    for mth in ["MAE-CP", "LeJEPA-CP", "SimCLR-CP", "DIET-CP"]:
        sub = m[(m.method_cp == mth) & (m.encoder.isin(SPHERE))]
        print(f"    {mth:11s} mean Δcv={sub.delta_cv.mean():+.4f}  mean Δknn={sub.dknn.mean():+.4f}")

    print("\n" + "=" * 72)
    print("C. MAE-CP on sphere encoders: Δcv vs CP data size (does pushing-off grow with data?)")
    print("=" * 72)
    sub = m[(m.method_cp == "MAE-CP") & (m.encoder.isin(SPHERE))].dropna(subset=["delta_cv"])
    if len(sub) >= 4:
        r, p = spearmanr(sub["size"], sub.delta_cv)
        print(f"  ρ(size, Δcv) = {r:+.3f} (p={p:.3g})  [expect positive: more data → more off-sphere]")
    print(f"\nsaved eval/outputs/postcp_offsphere.csv")


if __name__ == "__main__":
    main()
