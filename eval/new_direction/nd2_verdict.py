#!/usr/bin/env python
"""
nd2_verdict.py — ND2 (CPU adjudicator): expansion / consolidation along the CP size axis.

Li et al. 2025 (papers/new_direction/Li2025_ThreePhaseSpectral.pdf) find LLM pretraining
passes through entropy-seeking EXPANSION (effective rank rises) then compression-seeking
ANISOTROPIC consolidation (rank falls while dominant directions keep their variance),
measured with RankMe + alpha-ReQ. This adjudicator asks whether the CP benchmark's two
forces show that spectral signature along OUR size axis.

AXIS CAVEAT (disclosed up front): our x-axis is CP data SIZE across independent completed
runs, not training time within one run — a phase reading is an analogy, not a replication.

Pre-registered readouts (declared 2026-07-10, before nd2 data existed):
  ND2-1 EXPANSION: on sphere encoders (DINOv3/CLIP) x angular methods, rank rises with
        size: per-(method,encoder,dataset) rho(size, rankme) > 0 in a clear majority of
        configs with >= --min-sizes distinct sizes (directional count, like F3's P3.1).
  ND2-2 ALPHA mirror: alpha falls with size (spectrum flattens = spread force) in the same
        majority sense; uniformity_t2's rho(size, .) reproduces F3's spread sign on the
        same rows (internal consistency check between the sphere and spectral vocabularies).
  ND2-3 CONSOLIDATION (exploratory): fraction of configs whose rankme trajectory has an
        INTERIOR maximum (rank rises then falls). If substantial, the Li phase sequence has
        a size-axis analog; if rank is monotone, CP's spread stays entropy-seeking at every
        scale we probed and 'suppression' is NOT rank compression (a finding either way).
  ND2-4 (exploratory): pooled per encoder, rho(d_rankme, dknn) where d_rankme = post - pre
        (pre from nd1_precp_spectral.csv) — does rank expansion track the kNN benefit the
        way uniformity spread does?

Inputs: eval/outputs/nd2_spectral_sweep.csv (concat of shards), nd1_precp_spectral.csv,
results.xlsx. Run (local): python eval/new_direction/nd2_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long, add_size_canon

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
ANGULAR = ["LeJEPA", "SimCLR", "DIET"]
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP", "MAE": "MAE-CP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(OUT / "nd2_spectral_sweep.csv"))
    ap.add_argument("--pre", default=str(OUT / "nd1_precp_spectral.csv"))
    ap.add_argument("--min-sizes", type=int, default=4)
    args = ap.parse_args()

    if not _P(args.sweep).exists():
        sys.exit(f"MISSING {args.sweep} — run the ND2 GPU pass, then concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd2_spectral_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd2_spectral_sweep.csv', index=False)\"")
    sw = pd.read_csv(args.sweep)
    sw = sw[sw.variant == "pretrained"]
    cell = (sw.groupby(["method", "encoder", "dataset", "size"])
            [["rankme", "alpha", "uniformity_t2"]].mean().reset_index())

    # ---- ND2-1/2/3: per-config trajectories over the size axis -------------------------
    rows = []
    for (m, e, d), g in cell.groupby(["method", "encoder", "dataset"]):
        g = g.sort_values("size")
        if g["size"].nunique() < args.min_sizes:
            continue
        r_rank = spearmanr(g["size"], g["rankme"]).correlation
        r_alpha = spearmanr(g["size"], g["alpha"]).correlation
        r_unif = spearmanr(g["size"], g["uniformity_t2"]).correlation
        peak = int(np.argmax(g["rankme"].values))
        rows.append(dict(method=m, encoder=e, dataset=d, n_sizes=g["size"].nunique(),
                         rho_size_rankme=r_rank, rho_size_alpha=r_alpha,
                         rho_size_unif=r_unif,
                         interior_rank_peak=0 < peak < len(g) - 1,
                         peak_pos=peak, n_points=len(g)))
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "nd2_trajectories.csv", index=False)

    sph = tr[tr.encoder.isin(["DINOv3", "CLIP"]) & tr.method.isin(ANGULAR)]
    n1 = (sph.rho_size_rankme > 0).sum()
    n2a = (sph.rho_size_alpha < 0).sum()
    n2b = ((np.sign(sph.rho_size_unif) == -np.sign(sph.rho_size_rankme)) |
           (sph.rho_size_rankme == 0)).sum()
    n3 = sph.interior_rank_peak.sum()

    print("=" * 88)
    print(f"ND2 trajectories: {len(tr)} configs with >= {args.min_sizes} sizes "
          f"({len(sph)} sphere x angular)")
    print("=" * 88)
    print(f"ND2-1 EXPANSION   rho(size, rankme) > 0        : {n1}/{len(sph)} sphere-angular configs")
    print(f"ND2-2 ALPHA       rho(size, alpha)  < 0        : {n2a}/{len(sph)}")
    print(f"      consistency unif spread opposes rank sign : {n2b}/{len(sph)} "
          f"(uniformity more negative = more spread, so expect opposite signs)")
    nd23_msg = ("size-axis analog of the Li phase sequence EXISTS" if n3 > len(sph) * 0.3
                else "rank is mostly monotone in size — no phase analog")
    print(f"ND2-3 CONSOLIDATION interior rankme maximum     : {n3}/{len(sph)} ({nd23_msg})")
    print("\nPer-encoder/method medians of rho(size, rankme):")
    print(tr.groupby(["encoder", "method"]).rho_size_rankme.median().round(3).to_string())

    # ---- ND2-4: d_rankme vs dknn (pooled per encoder) ----------------------------------
    pre = pd.read_csv(args.pre)[["encoder", "dataset", "rankme"]]
    pre = pre.rename(columns={"rankme": "rankme_pre"})
    beh = load_long()
    beh = beh[beh.Method.isin(MMAP.values())]
    beh = (beh.groupby(["Backbone", "Method", "dataset_key", "size"]).dknn.mean()
           .reset_index().rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    # join on size_canon, not raw size: MAX labels drift between ckpt filenames and
    # results.xlsx (e.g. fgvc 3334 vs 3400) — see load_results.add_size_canon.
    beh = add_size_canon(beh, dataset_col="dataset", size_col="size").drop(columns=["size"])
    cell_c = add_size_canon(cell, dataset_col="dataset", size_col="size")
    j = cell_c.assign(Method=cell_c.method.map(MMAP)).merge(
        beh, on=["encoder", "Method", "dataset", "size_canon"]).merge(
        pre, on=["encoder", "dataset"])
    j["d_rankme"] = j.rankme - j.rankme_pre
    print("\nND2-4 (exploratory) pooled rho(d_rankme, dknn) per encoder:")
    for e, g in j.groupby("encoder"):
        r, p = spearmanr(g.d_rankme, g.dknn)
        print(f"  {e:>7}: rho={r:+.3f} (p={p:.4f}, n={len(g)})")
    j.to_csv(OUT / "nd2_rank_vs_dknn.csv", index=False)
    print(f"\nwrote -> {OUT / 'nd2_trajectories.csv'}, {OUT / 'nd2_rank_vs_dknn.csv'}")


if __name__ == "__main__":
    main()
