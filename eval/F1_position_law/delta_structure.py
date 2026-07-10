#!/usr/bin/env python
"""
delta_structure.py — Pure Δ-structure analysis from results.xlsx.

Runs WITHOUT any geometry recompute (no torch/timm/ImageNet) — it only needs
the CP outcomes already in results.xlsx. It quantifies the three empirical
phenomena the geometry framework must explain:

  1. The kNN/LP-vs-FT reversal (quadrant counts by encoder x dataset type).
  2. MAE-CP catastrophic degradation (vs the same objective on MAE backbone).
  3. The 15-dataset Δ@MAX ranking per encoder (the ground-truth ordering that
     the recomputed geometry must reproduce in correlate.py).

Convention for cross-dataset comparison: size = MAX (the only size the 7 new
datasets ran), and "invariance" = mean over {LeJEPA-CP, SimCLR-CP} (the main
reversal carriers). Both are overridable in the functions.
"""

import numpy as np
import pandas as pd

import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))  # eval/ root for shared modules
from load_results import load_long

INVARIANCE = ["LeJEPA-CP", "SimCLR-CP"]


def reversal_quadrants(df, methods=INVARIANCE, thresh=0.01):
    """Count kNN/LP vs FT sign agreement per (encoder, dataset_type) at MAX.

    Quadrant by (sign Δknn, sign Δft):
      'kNN+/FT-'  OOD-style reversal (frozen up, finetune down)
      'kNN-/FT+'  FG-style reversal  (frozen down, finetune up)
      'aligned+'  both up    'aligned-' both down   'flat' within +-thresh
    """
    d = df[df.Method.isin(methods) & df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    g = d.groupby(["Backbone", "dataset_type", "dataset_key"])[["dknn", "dft"]].mean().reset_index()

    def quad(r):
        k, f = r.dknn, r.dft
        if abs(k) < thresh and abs(f) < thresh:
            return "flat"
        if k > 0 and f < -thresh:
            return "kNN+/FT-"
        if k < -thresh and f > 0:
            return "kNN-/FT+"
        return "aligned+" if k > 0 else "aligned-"

    g["quad"] = g.apply(quad, axis=1)
    tab = (g.groupby(["Backbone", "dataset_type", "quad"]).size()
             .unstack(fill_value=0))
    return tab


def mae_cp_catastrophe(df, cat_thresh=-0.30):
    """MAE-CP Δknn at MAX on every backbone x dataset; flag catastrophic (< cat_thresh)."""
    d = df[(df.Method == "MAE-CP") & df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    g = d.groupby(["dataset_key", "Backbone"])["dknn"].mean().unstack()
    flagged = (g < cat_thresh)
    return g, flagged


def delta_at_max_ranking(df, methods=INVARIANCE):
    """Per-encoder 15-dataset Δ@MAX ranking (mean over `methods`)."""
    d = df[df.Method.isin(methods) & df.is_max]
    out = {}
    for bb in ["DINOv3", "CLIP", "MAE"]:
        g = (d[d.Backbone == bb].groupby(["dataset_key", "dataset_type"])
             [["dknn", "dlp", "dft"]].mean().sort_values("dknn"))
        out[bb] = g
    return out


def gap_effect(df, configs):
    """Δknn vs size trajectories for representative (backbone, method, dataset) configs."""
    rows = []
    for bb, m, ds in configs:
        sel = df[(df.Backbone == bb) & (df.Method == m) & (df.dataset_key == ds)]
        sel = sel.sort_values("size")
        # NOTE: use r["size"], not r.size — Series.size is the element-count attribute.
        traj = "  ".join(f"{'MAX' if r['is_max'] else int(r['size'])}:{r['dknn']:+.3f}"
                         for _, r in sel.iterrows() if pd.notna(r["dknn"]))
        rows.append((f"{bb}+{m}+{ds}", traj))
    return rows


if __name__ == "__main__":
    df = load_long()

    print("=" * 78)
    print("1. kNN/FT REVERSAL QUADRANTS (invariance methods, MAX) — counts of datasets")
    print("=" * 78)
    print(reversal_quadrants(df).to_string())

    print("\n" + "=" * 78)
    print("2. MAE-CP Δknn@MAX (flag < -0.30 = catastrophic)")
    print("=" * 78)
    g, flagged = mae_cp_catastrophe(df)
    print(g.round(3).to_string())
    cats = [(d, b) for d in g.index for b in g.columns if flagged.loc[d, b]]
    print("catastrophic:", cats if cats else "none")

    print("\n" + "=" * 78)
    print("3. 15-DATASET Δ@MAX RANKING per encoder (invariance mean) — GROUND TRUTH for correlate.py")
    print("=" * 78)
    for bb, g in delta_at_max_ranking(df).items():
        print(f"\n--- {bb} ---")
        for (k, t), r in g.iterrows():
            print(f"  {k:14s} [{t:3s}] Δknn={r.dknn:+.3f}  Δlp={r.dlp:+.3f}  Δft={r.dft:+.3f}")

    print("\n" + "=" * 78)
    print("4. GAP EFFECT: Δknn vs size (small-gap decay vs large-gap growth)")
    print("=" * 78)
    cfgs = [("DINOv3", "LeJEPA-CP", "pathmnist"),
            ("DINOv3", "LeJEPA-CP", "galaxy10"),
            ("DINOv3", "LeJEPA-CP", "organamnist"),
            ("MAE", "SimCLR-CP", "food101"),
            ("DINOv3", "LeJEPA-CP", "octmnist")]
    for name, traj in gap_effect(df, cfgs):
        print(f"  {name:34s} {traj}")
