#!/usr/bin/env python
"""
vitl_score.py — R1-R4 pre-registered verdicts for the ViT-L scale-robustness spot check
(eval/DESIGN_vitL_robustness.md).

Inputs: eval/outputs/geometry_vitL.csv (DINOv3L pre-CP geometry, all 15 datasets),
eval/outputs/cpL_behavior.csv (pre/post kNN/LP/SFT for 3 methods x 7 datasets x 3 seeds),
plus geometry_15.csv + cp_long_refreshed.csv for the frozen ViT-B rule.

The decision rule is the FROZEN protocol of predictor.py / the 2026-06-19 freeze:
features [neighbor_overlap_k50, uniformity_t2] z-scored WITHIN encoder, LogisticRegression
(max_iter=1000, default C), target sign(invariance-mean dknn@MAX), fit on DINOv3+CLIP
(30 cells) — then applied unchanged to DINOv3 ViT-L (z-scores over ITS 15 datasets).

CPU:  python eval/adjudicate/vitl_score.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
INV = ["LeJEPA-CP", "SimCLR-CP"]
DISPLAY2KEY = {"Galaxy10": "galaxy10", "DermaMNIST": "dermamnist", "EuroSAT": "eurosat",
               "FGVC_Aircraft": "fgvc_aircraft", "Cars196": "cars196", "CUB200": "cub200",
               "DTD": "dtd"}


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    # ---- ViT-L deltas ------------------------------------------------------------------
    beh = pd.read_csv(OUT / "cpL_behavior.csv")
    beh["dataset"] = beh["display"].map(DISPLAY2KEY)
    pre = beh[beh.kind == "pre"].groupby("dataset")[["pre_knn", "pre_lp", "pre_sft"]].mean()
    post = beh[beh.kind == "cp"].groupby(["method", "dataset"])[
        ["post_knn", "post_lp", "post_sft"]].mean()
    d = post.join(pre, on="dataset")
    for ch in ["knn", "lp", "sft"]:
        d[f"d{ch}"] = d[f"post_{ch}"] - d[f"pre_{ch}"]
    per_method = d[["dknn", "dlp", "dsft"]].reset_index()
    cell = per_method.groupby("dataset")[["dknn", "dlp", "dsft"]].mean()  # 3-method mean

    print("=" * 88)
    print("ViT-L realized deltas (3-method mean @MAX; per-method table below)")
    print("=" * 88)
    print(cell.round(3).to_string())
    print("\nper-method:")
    print(per_method.pivot(index="dataset", columns="method", values="dknn").round(3).to_string())

    # ---- R1: geometry rank stability (local recompute of the cluster verdict) ----------
    gB = pd.read_csv(OUT / "geometry_15.csv")
    gB = gB[(gB.encoder == "DINOv3") & (gB.dataset != "imagenet")]
    gL = pd.read_csv(OUT / "geometry_vitL.csv")
    m15 = gB.merge(gL, on="dataset", suffixes=("_B", "_L"))
    r_u = spearmanr(m15.uniformity_t2_B, m15.uniformity_t2_L).correlation
    r_o = spearmanr(m15.neighbor_overlap_k50_B, m15.neighbor_overlap_k50_L).correlation
    print(f"\nR1 geometry rank stability (n={len(m15)}): unif rho={r_u:+.3f}, "
          f"overlap rho={r_o:+.3f} -> {'PASS' if (r_u > 0.8 and r_o > 0.8) else 'FAIL'} "
          f"(pre-registered > 0.8 both)")

    # ---- R2: the GENUINE frozen ViT-B rule applied to ViT-L -----------------------------
    # Frozen (2026-06-19) target = PRE-refresh two-method mean from results.xlsx via
    # load_long(), exactly as predictor.py / preregister_siglip.py built it. An earlier
    # version of this script refit on cp_long_refreshed.csv while calling itself frozen
    # (caught by external audit 2026-07-09; labels were identical on all 15+7 datasets,
    # so the 6/7 result was unaffected, but the coefficients were not the frozen ones).
    # Fixed: fit from results.xlsx; assert coefficients AND frozen-label reproduction.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    from load_results import load_long
    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False) & df.is_max]
    tgt = (inv[inv.Backbone.isin(["DINOv3", "CLIP"])]
           .groupby(["Backbone", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    geoB = pd.read_csv(OUT / "geometry_15.csv")
    geoB = geoB[geoB.dataset != "imagenet"]
    tr = geoB.merge(tgt, on=["encoder", "dataset"])
    tr = tr[tr.encoder.isin(["DINOv3", "CLIP"])].copy()
    for f in ["neighbor_overlap_k50", "uniformity_t2"]:
        tr[f + "_z"] = tr.groupby("encoder")[f].transform(zscore)
    tr["help"] = (tr.dknn > 0).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(
        tr[["neighbor_overlap_k50_z", "uniformity_t2_z"]], tr["help"])
    b_ov, b_un, b0 = clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0]
    FROZEN = (0.194967, 1.717809, 0.639062)   # audited genuine frozen coefficients
    assert max(abs(b_ov - FROZEN[0]), abs(b_un - FROZEN[1]), abs(b0 - FROZEN[2])) < 1e-3, \
        f"frozen fit drifted: ({b_ov:+.6f}, {b_un:+.6f}, {b0:+.6f}) vs {FROZEN}"
    print(f"\nGENUINE frozen coefficients (results.xlsx 2-method target, n=30): "
          f"b_overlap={b_ov:+.6f}, b_unif={b_un:+.6f}, b0={b0:+.6f}")

    # self-check: frozen model must reproduce the archived frozen SigLIP labels
    prereg = pd.read_csv(OUT / "preregister_siglip.csv")
    lab_col = next(c for c in prereg.columns if "pred" in c.lower() and "knn" in c.lower())
    gS = geoB[geoB.encoder == "SigLIP"].copy()
    for f in ["neighbor_overlap_k50", "uniformity_t2"]:
        gS[f + "_z"] = zscore(gS[f])
    pS = clf.predict_proba(gS[["neighbor_overlap_k50_z", "uniformity_t2_z"]])[:, 1]
    predS = np.where(pS >= 0.5, "HELP", "HURT")
    stored = prereg.set_index("dataset")[lab_col].reindex(gS.dataset).astype(str).values
    n_match = int((predS == stored).sum())
    print(f"frozen-label self-check vs preregister_siglip.csv: {n_match}/{len(gS)}")
    assert n_match == len(gS), "frozen model fails to reproduce archived frozen labels"

    gL = gL.copy()
    gL["neighbor_overlap_k50_z"] = zscore(gL["neighbor_overlap_k50"])
    gL["uniformity_t2_z"] = zscore(gL["uniformity_t2"])
    gL["p_help"] = clf.predict_proba(
        gL[["neighbor_overlap_k50_z", "uniformity_t2_z"]])[:, 1]
    gL["pred"] = np.where(gL.p_help >= 0.5, "HELP", "HURT")

    sc = gL.set_index("dataset").join(cell, how="inner")
    print(f"\nR2 frozen-coefficient transfer to ViT-L ({len(sc)}/7 datasets):")
    print(f"{'dataset':>14} {'p_help':>7} {'pred':>5} {'dknn':>7} {'dlp':>7} "
          f"{'hit_knn':>7} {'hit_lp':>6}")
    hits_k = hits_l = 0
    for ds, r in sc.iterrows():
        hk = (r.pred == "HELP") == (r.dknn > 0)
        hl = (r.pred == "HELP") == (r.dlp > 0)
        hits_k += hk
        hits_l += hl
        print(f"{ds:>14} {r.p_help:>7.3f} {r.pred:>5} {r.dknn:>+7.3f} {r.dlp:>+7.3f} "
              f"{'Y' if hk else 'MISS':>7} {'Y' if hl else 'MISS':>6}")
    print(f"  sign(dkNN): {hits_k}/7   sign(dLP): {hits_l}/7   "
          f"pre-registered pass >= 5/7 (dtd expected miss)")
    print(f"  R2 verdict: {'PASS' if hits_k >= 5 else 'FAIL'}")

    # ---- R3 / R4 -----------------------------------------------------------------------
    r3 = spearmanr(sc.uniformity_t2, sc.dknn).correlation
    r4 = spearmanr(sc.uniformity_t2, sc.dsft).correlation
    print(f"\nR3 position-law direction (n=7, point estimate): rho(unif, dknn) = {r3:+.3f} "
          f"-> {'PASS (direction >0)' if r3 > 0 else 'FAIL'}")
    print(f"R4 reversal direction (exploratory, n=7): rho(unif, dsft) = {r4:+.3f} "
          f"(predicted < 0)")

    sc.reset_index()[["dataset", "p_help", "pred", "dknn", "dlp", "dsft"]].to_csv(
        OUT / "vitl_score.csv", index=False)
    print(f"\nwrote -> {OUT / 'vitl_score.csv'}")


if __name__ == "__main__":
    main()
