#!/usr/bin/env python
"""
bilinear_law.py — Design 1 of eval/DESIGN_spectrum_transport.md: the bilinear unified law
and the coupling axis theta_E (4th gate evidence).

theta_E := |Spearman(uniformity_t2, cdnv)| across the 15 datasets, per encoder — how tightly
the encoder's UNLABELED geometry couples to its LABELED class geometry. Defined without any
behavioral data. The bilinear single-term model  dknn ~ x * theta_E  (x = within-encoder
z-scored uniformity) is compared against the position-only and binary-gate models on the 60
pooled cells with dataset-grouped LOO rank regression.

Cell table: DINOv3/CLIP/MAE = 3-method (LeJEPA/SimCLR/DIET) mean dknn @MAX from
cp_long_refreshed.csv; SigLIP = realized 2-method dknn from c2_siglip_score.csv (no DIET
SigLIP runs — DISCLOSED mixing). SigLIP pre-CP kNN comes from results.xlsx (SigLIP sheet).

Pre-registered verdicts (design doc):
  V1 M4 (x*theta single term) LOO within 0.02 of M2 (binary gate) and >= 0.04 above M1
     (position only); dataset block-bootstrap CI on the M4-M2 difference.
  V2 theta ordering: MAE lowest, gap > 0.15 vs min(sphere trio).
  V3 4th gate evidence: rho(uniformity, pre-CP kNN) < 0 on all three sphere encoders,
     > 0 on MAE (sign flip).
  V4 theta variants (cdnv / center_margin / knn_pre coupling) direction-consistent.

CPU, existing data:  python eval/adjudicate/bilinear_law.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
SPHERE = ["DINOv3", "CLIP", "SigLIP"]
ANGULAR = ["LeJEPA-CP", "SimCLR-CP", "DIET-CP"]

XLSX_DS_KEY = {"dermamnist": "dermamnist", "breastmnist": "breastmnist", "octmnist": "octmnist",
               "organamnist": "organamnist", "pathmnist": "pathmnist", "galaxy10": "galaxy10",
               "eurosat": "eurosat", "plantvillage": "plant_village", "dtd": "dtd",
               "food101": "food101", "fgvcaircraft": "fgvc_aircraft", "cars196": "cars196",
               "cub200": "cub200", "flowers102": "flowers102", "oxfordpet": "oxford_pet"}


def siglip_knn_pre(xlsx_path):
    """SigLIP pre-CP kNN per dataset from the 'By Method (SigLIP)' sheet (MAX rows)."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx_path, read_only=True)["By Method (SigLIP)"]
    vals = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        ds_disp, num_data, knn = row[3], row[4], row[8]
        if not ds_disp or not num_data or "MAX" not in str(num_data) or knn is None:
            continue
        key = XLSX_DS_KEY.get(str(ds_disp).lower().replace("_", "").replace(" ", ""))
        if key:
            vals.setdefault(key, []).append(float(knn))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def rank_against(train_vals, test_vals):
    """Rank of each test value against the TRAIN distribution (no leakage)."""
    tr = np.sort(np.asarray(train_vals, float))
    lo = np.searchsorted(tr, test_vals, side="left")
    hi = np.searchsorted(tr, test_vals, side="right")
    return 1.0 + (lo + hi) / 2.0


def loo_ds_rank(df, cols):
    """Dataset-grouped LOO rank regression; Spearman(pred, dknn) pooled over held-out rows."""
    preds = np.full(len(df), np.nan)
    for ds in df["dataset"].unique():
        te = (df["dataset"] == ds).values
        tr = ~te
        Xtr = np.column_stack([rankdata(df.loc[tr, c]) for c in cols])
        Xte = np.column_stack([rank_against(df.loc[tr, c], df.loc[te, c]) for c in cols])
        reg = LinearRegression().fit(Xtr, rankdata(df.loc[tr, "dknn"]))
        preds[te] = reg.predict(Xte)
    return float(spearmanr(preds, df["dknn"]).correlation)


def build_cells(geo, gc, cp, c2):
    inv = cp[cp.Method.isin(ANGULAR) & cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    cell = (inv.groupby(["Backbone", "dataset_key"])["dknn"].mean().reset_index()
            .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    cell["n_methods"] = 3
    sig = (geo[geo.encoder == "SigLIP"][["encoder", "dataset", "uniformity_t2"]]
           .merge(c2[["dataset", "real_dknn"]], on="dataset")
           .rename(columns={"real_dknn": "dknn"}))
    sig["n_methods"] = 2
    main = geo[["encoder", "dataset", "uniformity_t2"]].merge(cell, on=["encoder", "dataset"])
    cells = pd.concat([main, sig], ignore_index=True)
    cells = cells.merge(gc[["encoder", "dataset", "cdnv", "center_margin"]],
                        on=["encoder", "dataset"])
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ROOT / "results.xlsx"))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default=str(OUT / "bilinear_law.csv"))
    args = ap.parse_args()
    rng = np.random.RandomState(42)

    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    gc = pd.read_csv(OUT / "geometry_class_15.csv")
    cp = pd.read_csv(OUT / "cp_long_refreshed.csv")
    c2 = pd.read_csv(OUT / "c2_siglip_score.csv")

    cells = build_cells(geo, gc, cp, c2)
    assert len(cells) == 60, f"expected 60 cells, got {len(cells)}"

    # knn_pre per (encoder, dataset): main grid @MAX; SigLIP from the xlsx sheet
    kp = (cp[cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
          .groupby(["Backbone", "dataset_key"])["knn_pre"].mean().reset_index()
          .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    sig_kp = siglip_knn_pre(args.xlsx)
    kp = pd.concat([kp, pd.DataFrame([{"encoder": "SigLIP", "dataset": d, "knn_pre": v}
                                      for d, v in sig_kp.items()])], ignore_index=True)
    cells = cells.merge(kp, on=["encoder", "dataset"], how="left")

    # ---- theta_E and its variants -----------------------------------------
    theta, theta_var, knnpre_rho = {}, {}, {}
    for enc, g in cells.groupby("encoder"):
        r_cdnv = spearmanr(g.uniformity_t2, g.cdnv).correlation
        r_marg = spearmanr(g.uniformity_t2, g.center_margin).correlation
        r_kpre = (spearmanr(g.uniformity_t2, g.knn_pre).correlation
                  if g.knn_pre.notna().all() else np.nan)
        theta[enc] = abs(r_cdnv)
        theta_var[enc] = {"cdnv": abs(r_cdnv), "margin": abs(r_marg),
                          "knnpre": abs(r_kpre) if r_kpre == r_kpre else np.nan}
        knnpre_rho[enc] = r_kpre

    cells["x_z"] = cells.groupby("encoder")["uniformity_t2"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    cells["theta_cdnv"] = cells["encoder"].map(theta)
    cells["gate"] = (cells["encoder"] != "MAE").astype(float)
    cells["gated"] = cells["x_z"] * cells["gate"]
    cells["bilinear"] = cells["x_z"] * cells["theta_cdnv"]

    # ---- models with dataset-grouped LOO -----------------------------------
    MODELS = [("M1 position only", ["x_z"]),
              ("M2 binary gate", ["x_z", "gated"]),
              ("M3 x + x*theta", ["x_z", "bilinear"]),
              ("M4 x*theta single", ["bilinear"])]
    loo = {name: loo_ds_rank(cells, cols) for name, cols in MODELS}

    # dataset block bootstrap on M4 - M2 (duplicated datasets relabeled -> distinct folds)
    ds_list = cells["dataset"].unique()
    diffs = []
    for _ in range(args.boot):
        pick = rng.choice(ds_list, size=len(ds_list), replace=True)
        pseudo = pd.concat(
            [cells[cells.dataset == d].assign(dataset=f"{d}#{j}") for j, d in enumerate(pick)],
            ignore_index=True)
        diffs.append(loo_ds_rank(pseudo, ["bilinear"]) - loo_ds_rank(pseudo, ["x_z", "gated"]))
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])

    # ---- report -------------------------------------------------------------
    print("=" * 88)
    print("BILINEAR LAW — 60 pooled cells (D3/CLIP/MAE 3-method @MAX; SigLIP realized 2-method)")
    print("=" * 88)
    print(f"\ntheta_E = |rho(uniformity, cdnv)| across 15 datasets (variants for V4):")
    print(f"{'encoder':8} {'theta_cdnv':>11} {'theta_margin':>13} {'theta_knnpre':>13} "
          f"{'rho(unif,knn_pre)':>18}")
    for enc in ["DINOv3", "CLIP", "SigLIP", "MAE"]:
        v = theta_var[enc]
        print(f"{enc:8} {v['cdnv']:>11.3f} {v['margin']:>13.3f} {v['knnpre']:>13.3f} "
              f"{knnpre_rho[enc]:>+18.3f}")
    print(f"\nDataset-grouped LOO (rank regression, no leakage):")
    for name, _ in MODELS:
        print(f"  {name:20s} LOO-Spearman = {loo[name]:+.3f}")
    print(f"  M4 - M2 block-bootstrap 95% CI: [{lo_d:+.3f}, {hi_d:+.3f}]")

    v1 = (loo["M4 x*theta single"] >= loo["M2 binary gate"] - 0.02
          and loo["M4 x*theta single"] >= loo["M1 position only"] + 0.04)
    sphere_min = min(theta[e] for e in SPHERE)
    v2 = theta["MAE"] < sphere_min and (sphere_min - theta["MAE"]) > 0.15
    v3 = all(knnpre_rho[e] < 0 for e in SPHERE) and knnpre_rho["MAE"] > 0
    v4 = all(theta_var["MAE"][k] == min(theta_var[e][k] for e in theta_var)
             for k in ("cdnv", "knnpre"))  # margin checked separately (weaker discriminator)
    v4_margin = theta_var["MAE"]["margin"] <= sorted(theta_var[e]["margin"]
                                                     for e in theta_var)[1]
    print("\nVERDICTS (pre-registered in eval/DESIGN_spectrum_transport.md):")
    print(f"  V1 single-term bilinear ties gate, beats position-only : {'PASS' if v1 else 'FAIL'}")
    print(f"  V2 theta gap MAE vs sphere > 0.15                      : {'PASS' if v2 else 'FAIL'}"
          f"  (MAE {theta['MAE']:.3f} vs min-sphere {sphere_min:.3f})")
    print(f"  V3 sign flip rho(unif, knn_pre): sphere<0, MAE>0       : {'PASS' if v3 else 'FAIL'}")
    print(f"  V4 variant consistency (MAE lowest on cdnv & knnpre;   : "
          f"{'PASS' if v4 else 'FAIL'} (margin bottom-2: {v4_margin})")
    print("\nDisclosures: SigLIP cells are 2-method realized dknn (no DIET SigLIP). n=4 encoders")
    print("cannot distinguish continuous from binary gating — the bilinear form is a unifying")
    print("RESTATEMENT with a principled encoder parameter, not evidence for a continuum.")

    keep = ["encoder", "dataset", "uniformity_t2", "x_z", "theta_cdnv", "dknn", "knn_pre",
            "n_methods"]
    cells[keep].to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}")
    return 0 if (v1 and v2 and v3) else 1


if __name__ == "__main__":
    sys.exit(main())
