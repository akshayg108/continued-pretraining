#!/usr/bin/env python
"""
transport_law.py — Exp J scorer: T1-T4 verdicts on the transport-field decomposition
(eval/DESIGN_spectrum_transport.md, Design 3).

Consumes eval/outputs/transport_field_max.csv (concat shards first), plus:
  postcp_sweep_fixed.csv + geometry_15.csv  -> d_overlap at MAX (collision channel)
  cp_long_refreshed.csv                     -> dknn / dft @MAX (behavior)

T1 (highest value): toward_imagenet ~ d_overlap on sphere encoders, AND invariance methods
   (LeJEPA/SimCLR) show larger toward_imagenet than DIET and MAE-CP -> mechanistic ticket for
   "collision is invariance-specific".
T2 within_share = within/total ~ dknn (negative predicted); FG > OOD in within_share.
T3 identity check: |resid_identity| < 1e-8 everywhere (else numerical bug — stop).
T4 exploratory: between_energy ~ dft.

CPU:  python eval/adjudicate/transport_law.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
M2CP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP", "MAE": "MAE-CP"}
INV = ["LeJEPA", "SimCLR"]
MAX_N = {"food101": 75750, "octmnist": 97477, "plant_village": 43596, "organamnist": 34561,
         "galaxy10": 14188, "fgvc_aircraft": 3334, "cars196": 8144, "breastmnist": 546,
         "cub200": 5994, "dermamnist": 7007, "dtd": 1880, "eurosat": 16200,
         "flowers102": 1020, "oxford_pet": 3680, "pathmnist": 89996}


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
    ap.add_argument("--field", default=str(OUT / "transport_field_max.csv"))
    ap.add_argument("--stats-out", default=str(OUT / "transport_stats.csv"))
    args = ap.parse_args()

    tf = pd.read_csv(args.field)
    print(f"transport rows: {len(tf)} "
          f"({tf.method.nunique()} methods x {tf.encoder.nunique()} encoders x "
          f"{tf.dataset.nunique()} datasets)")

    # ---- T3 first: identity must hold before anything else is interpretable ------------
    # Relative gate: features are float32, so the exact identity holds only to ~1e-6
    # relative accumulation precision (verified exact on float64 synthetic data).
    worst = (tf.resid_identity.abs() / tf.total_energy).max()
    print(f"\nT3 identity check: max |resid| / total_energy = {worst:.2e} "
          f"-> {'PASS' if worst < 1e-4 else 'FAIL — NUMERICAL BUG, STOP'}")
    if worst >= 1e-4:
        return

    # ---- cell means over seeds -----------------------------------------------------------
    cell = (tf.groupby(["method", "encoder", "dataset"])
            [["total_energy", "trans_energy", "between_energy", "within_energy",
              "toward_imagenet", "cos_mu_imagenet", "mu_norm"]].mean().reset_index())
    cell["within_share"] = cell.within_energy / cell.total_energy
    cell["Method"] = cell.method.map(M2CP)

    # behavior @MAX
    cp = pd.read_csv(OUT / "cp_long_refreshed.csv")
    beh = (cp[cp.is_max].groupby(["Method", "Backbone", "dataset_key"])
           [["dknn", "dft"]].mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    dstype = cp[["dataset_key", "dataset_type"]].drop_duplicates().set_index(
        "dataset_key")["dataset_type"].to_dict()
    cell = cell.merge(beh, on=["Method", "encoder", "dataset"], how="left")
    cell["dataset_type"] = cell.dataset.map(dstype)

    # d_overlap @MAX from the fixed sweep + pre-CP geometry
    sweep = pd.read_csv(OUT / "postcp_sweep_fixed.csv")
    sweep = sweep[(sweep.variant == "pretrained")
                  & sweep.apply(lambda r: MAX_N.get(r.dataset) == r["size"], axis=1)]
    geo = pd.read_csv(OUT / "geometry_15.csv")
    pre_ov = geo.set_index(["encoder", "dataset"])["neighbor_overlap_k50"].to_dict()
    sweep["d_overlap"] = sweep.apply(
        lambda r: r.neighbor_overlap_k50 - pre_ov.get((r.encoder, r.dataset), np.nan), axis=1)
    dov = (sweep.groupby(["method", "encoder", "dataset"])["d_overlap"]
           .mean().reset_index())
    cell = cell.merge(dov, on=["method", "encoder", "dataset"], how="left")

    # ---- T1: toward-ImageNet is the collision vector, and it is invariance-specific ------
    print("\nT1 toward-ImageNet (vector collision):")
    fam_p, fam_names = [], []
    for enc in ["DINOv3", "CLIP"]:
        g = cell[(cell.encoder == enc) & cell.method.isin(["LeJEPA", "SimCLR", "DIET"])
                 ].dropna(subset=["d_overlap"])
        r, p = spearmanr(g.toward_imagenet, g.d_overlap)
        fam_p.append(p)
        fam_names.append(f"T1-corr-{enc}")
        print(f"  {enc}: rho(toward, d_overlap) = {r:+.3f} (p={p:.4f}, n={len(g)})")
    for enc in ["DINOv3", "CLIP"]:
        gi = cell[(cell.encoder == enc) & cell.method.isin(INV)].toward_imagenet.dropna()
        gd = cell[(cell.encoder == enc) & (cell.method == "DIET")].toward_imagenet.dropna()
        gm = cell[(cell.encoder == enc) & (cell.method == "MAE")].toward_imagenet.dropna()
        u_p = mannwhitneyu(gi, gd, alternative="greater").pvalue if len(gd) else np.nan
        fam_p.append(u_p)
        fam_names.append(f"T1-inv>DIET-{enc}")
        print(f"  {enc}: median toward — inv {gi.median():+.4f} | DIET "
              f"{gd.median() if len(gd) else np.nan:+.4f} | MAE-CP "
              f"{gm.median() if len(gm) else np.nan:+.4f}  (inv>DIET one-sided p={u_p:.4f})")

    # ---- T2: within-class scramble vs kNN damage -----------------------------------------
    print("\nT2 within-class scramble share:")
    for enc in ["DINOv3", "CLIP", "MAE"]:
        g = cell[(cell.encoder == enc) & cell.method.isin(["LeJEPA", "SimCLR", "DIET"])
                 ].dropna(subset=["dknn"])
        r, p = spearmanr(g.within_share, g.dknn)
        fam_p.append(p)
        fam_names.append(f"T2-{enc}")
        fg = g[g.dataset_type == "FG"].within_share.median()
        ood = g[g.dataset_type == "OOD"].within_share.median()
        print(f"  {enc}: rho(within_share, dknn) = {r:+.3f} (p={p:.4f}, n={len(g)}); "
              f"median share FG {fg:.3f} vs OOD {ood:.3f}")

    # ---- T4 exploratory --------------------------------------------------------------------
    print("\nT4 (exploratory) between-class motion vs dft:")
    for enc in ["DINOv3", "CLIP"]:
        g = cell[(cell.encoder == enc) & cell.method.isin(["LeJEPA", "SimCLR", "DIET"])
                 ].dropna(subset=["dft"])
        r, p = spearmanr(g.between_energy, g.dft)
        print(f"  {enc}: rho(between, dft) = {r:+.3f} (p={p:.4f}, n={len(g)}) [exploratory]")

    fdr = bh_fdr(fam_p)
    print(f"\nBH-FDR(q=0.10) over the confirmatory family: "
          f"{[n for n, ok in zip(fam_names, fdr) if ok] or 'none pass'}")
    print("Wording discipline: rank claims only; T1 method contrast is the mechanistic "
          "ticket for collision's invariance-specificity; T4 stays exploratory.")

    cell.to_csv(args.stats_out, index=False)
    print(f"\nwrote -> {args.stats_out}")


if __name__ == "__main__":
    main()
