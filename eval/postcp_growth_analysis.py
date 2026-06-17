#!/usr/bin/env python
"""
postcp_growth_analysis.py — NEW. Exp C / Finding 3 (growth dynamics) with NO extra GPU.

Reuses the post-CP geometry the sweep already computed (uniformity + neighbor_overlap at EVERY
size — the sweep ran with --imagenet-dir) instead of re-running postcp_growth.py on the GPU. Per
(method, encoder, dataset, size) it joins:
  - post_unif, post_overlap   seed-averaged, from postcp_sweep.csv
  - pre_overlap               from geometry_15.csv   ->   delta_overlap = post - pre
  - dknn                      from results.xlsx (Δ kNN)
and tests Finding-3's predictions on every config with >= min-sizes distinct CP sizes:
  P3.1  cloud SPREADS as CP data grows   ->  rho(size, post_unif)    < 0
  P3.2  cloud COLLIDES with ImageNet     ->  rho(size, post_overlap) > 0, and the Δknn peak sits
        at an EARLIER (interior) size while overlap keeps rising afterwards (peak precedes collision)
  P3.3  zero-crossing case (OrganAMNIST + DINOv3): print the size-resolved trajectory.

CPU-only (pure pandas/scipy). MAE *method* is INCLUDED — the sweep was re-run with the fixed
loader, so its post-CP geometry is valid now. Sizes are reconciled to results.xlsx via size_canon.

  python eval/postcp_growth_analysis.py --sweep eval/outputs/postcp_sweep.csv \
      --geometry eval/outputs/geometry_15.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from load_results import load_long, add_size_canon

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "eval/outputs/postcp_sweep.csv"))
    ap.add_argument("--geometry", default=str(ROOT / "eval/outputs/geometry_15.csv"))
    ap.add_argument("--results", default=None)
    ap.add_argument("--min-sizes", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/postcp_growth_analysis.csv"))
    args = ap.parse_args()

    sw = pd.read_csv(args.sweep)
    sw = sw[sw.variant == "pretrained"].copy()
    sw["method_cp"] = sw.method + "-CP"
    g = (sw.groupby(["method_cp", "encoder", "dataset", "size"])
           .agg(post_unif=("uniformity_t2", "mean"),
                post_overlap=("neighbor_overlap_k50", "mean")).reset_index())

    geo = pd.read_csv(args.geometry)
    pre_ov = {(r.encoder, r.dataset): r.neighbor_overlap_k50
              for r in geo[geo.dataset != "imagenet"].itertuples()}
    g["pre_overlap"] = g.apply(lambda r: pre_ov.get((r.encoder, r.dataset), np.nan), axis=1)
    g["delta_overlap"] = g.post_overlap - g.pre_overlap

    df = load_long(args.results) if args.results else load_long()
    dk = (df.groupby(["Method", "Backbone", "dataset_key", "size"])["dknn"].mean().reset_index()
            .rename(columns={"Method": "method_cp", "Backbone": "encoder", "dataset_key": "dataset"}))
    g = add_size_canon(g, "dataset", "size")
    dk = add_size_canon(dk, "dataset", "size")
    m = g.merge(dk[["method_cp", "encoder", "dataset", "size_canon", "dknn"]],
                on=["method_cp", "encoder", "dataset", "size_canon"], how="left")
    m = m.sort_values(["method_cp", "encoder", "dataset", "size"])
    m.to_csv(args.out, index=False)

    rows = []
    for (mth, enc, ds), sub in m.groupby(["method_cp", "encoder", "dataset"]):
        sub = sub.dropna(subset=["size"]).drop_duplicates("size").sort_values("size")
        if sub["size"].nunique() < args.min_sizes:
            continue

        def rho(col):
            s = sub.dropna(subset=[col])
            if s["size"].nunique() < 3 or s[col].nunique() < 2:  # need a trajectory + non-constant
                return (np.nan, np.nan)
            return spearmanr(s["size"], s[col])

        ru, _ = rho("post_unif")
        ro, _ = rho("post_overlap")
        peak_size, interior, ov_rise_after = np.nan, np.nan, np.nan
        d = sub.dropna(subset=["dknn"])
        if d["size"].nunique() >= 3:
            peak_size = d.loc[d.dknn.idxmax(), "size"]
            interior = bool(peak_size < d["size"].max())  # benefit peaks before the largest size
            after = sub[sub["size"] > peak_size]
            at_peak = sub.loc[sub["size"] == peak_size, "post_overlap"]
            if len(after) and len(at_peak) and pd.notna(at_peak.iloc[0]) and after.post_overlap.notna().any():
                ov_rise_after = bool(after.post_overlap.max() > at_peak.iloc[0])
        rows.append(dict(method=mth, encoder=enc, dataset=ds, n_sizes=int(sub["size"].nunique()),
                         rho_size_unif=ru, rho_size_overlap=ro, dknn_peak_size=peak_size,
                         peak_interior=interior, overlap_rises_after_peak=ov_rise_after))
    t = pd.DataFrame(rows)

    print("=" * 78)
    print(f"Finding 3 — growth dynamics   ({len(t)} configs with >={args.min_sizes} CP sizes)")
    print("=" * 78)

    u = t.dropna(subset=["rho_size_unif"])
    print("\nP3.1  cloud SPREADS  (rho(size, post_uniformity) < 0):")
    if len(u):
        print(f"   {(u.rho_size_unif < 0).sum()}/{len(u)} configs negative   mean rho = {u.rho_size_unif.mean():+.3f}")

    o = t.dropna(subset=["rho_size_overlap"])
    print("\nP3.2  cloud COLLIDES with ImageNet  (rho(size, post_overlap) > 0):")
    if len(o):
        print(f"   {(o.rho_size_overlap > 0).sum()}/{len(o)} configs positive   mean rho = {o.rho_size_overlap.mean():+.3f}")
    else:
        print("   (no post_overlap in sweep — was --imagenet-dir set during the sweep?)")
    pk = t.dropna(subset=["peak_interior"])
    if len(pk):
        inter = pk[pk.peak_interior]
        print(f"   Δknn peak is INTERIOR (benefit peaks before max data) in {len(inter)}/{len(pk)} configs;")
        ra = inter.dropna(subset=["overlap_rises_after_peak"])
        if len(ra):
            print(f"   of those, overlap KEEPS RISING after the Δknn peak in "
                  f"{int(ra.overlap_rises_after_peak.sum())}/{len(ra)}  (supports 'peak precedes collision').")

    print("\nP3.3  zero-crossing case — OrganAMNIST × DINOv3 trajectory:")
    org = m[(m.encoder == "DINOv3") & (m.dataset == "organamnist")].sort_values(["method_cp", "size"])
    if len(org):
        print(org[["method_cp", "size", "dknn", "post_unif", "post_overlap", "delta_overlap"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    else:
        print("   (no OrganAMNIST×DINOv3 multi-size rows in sweep)")

    print(f"\nper-(method,encoder,dataset,size) join saved -> {args.out}")
    if len(t):
        print("\n--- per-config trends ---")
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
