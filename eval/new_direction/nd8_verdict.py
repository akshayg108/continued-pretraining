#!/usr/bin/env python
"""
nd8_verdict.py — ND8 (CPU adjudicator): does any corrected position feature cure overlap's
measured deficiencies, and does swapping it into the frozen tool beat v1 on BOTH holdouts?

Measured deficiencies being cured (2026-07-12 analysis): raw overlap's partial correlation
with label placement given uniformity is NEGATIVE on all four encoders
(D3 -0.171 / CLIP -0.321 / SigLIP -0.257 / MAE -0.574) — it is a shape shadow, not a
placement axis — and it operates at N_k skewness > 1.4 in 37/60 cells.

Pre-registered readouts (declared 2026-07-12, BEFORE nd8_overlap.csv existed):
  ND8-0 VALIDATION: overlap_raw reproduces geometry_15.neighbor_overlap_k50 on the 60
        ViT-B cells (max |diff| < 0.02). Fails -> stop, protocol drifted.
  ND8-1 PRIMARY (orthogonality cure): for each variant, partial Spearman
        rho(variant, cC_K | uniformity) per ViT-B encoder. A variant CURES the deficiency
        if its partial exceeds raw's on >= 3/4 encoders AND is positive on >= 2.
        Orientation convention (declared before data): sun_knn_* are DISTANCES
        (higher = farther from the bank) and enter ND8-1 sign-flipped (as proximity),
        so the signed criterion is direction-coherent for all six features.
  ND8-2 (level utility): |rho(variant, knn_pre)| >= |rho(raw, knn_pre)| on >= 2/4 encoders
        (the fix must not destroy the level signal).
  ND8-3 (mechanism): median skew_variant < median skew_raw for centered/MP/NICDM, and
        median hub_centrality_rho < 0 (hubs are centroid-proximal, as the theory says).
  ND8-4 (TOOL v2, decision test): rebuild the F5 frozen-protocol fit (2-method pre-refresh
        dknn target, DINOv3+CLIP, features [feat_z, unif_z] z-scored within encoder,
        LogisticRegression) with each variant in place of overlap; evaluate FROZEN on the
        SigLIP holdout (sign(dknn)+sign(dlp) hits vs c2_siglip_score, 15 datasets) and the
        ViT-L holdout (sign hits vs cpL_behavior 3-method means, 7 datasets). The v1
        (raw-overlap) row must reproduce the recorded numbers (internal sanity). A variant
        QUALIFIES as v2 candidate iff its total hits >= v1 on BOTH holdouts. sun_knn_*
        enter as-is (logistic learns the sign).
n=15 per cell; sign/rank evidence; no FDR family (engineering round).

Inputs: eval/outputs/nd8_overlap.csv (cluster), geometry_15.csv, nd6_alignment.csv,
geometry_vitL.csv, c2_siglip_score.csv, cpL_behavior.csv, results.xlsx.
Run (local): python eval/new_direction/nd8_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
ENCS_B = ["DINOv3", "CLIP", "SigLIP", "MAE"]
FEATS = ["overlap_raw", "overlap_centered", "overlap_mp", "overlap_nicdm",
         "sun_knn_full", "sun_knn_small"]
DISPLAY2KEY = {"Galaxy10": "galaxy10", "DermaMNIST": "dermamnist", "EuroSAT": "eurosat",
               "FGVC_Aircraft": "fgvc_aircraft", "Cars196": "cars196", "CUB200": "cub200",
               "DTD": "dtd"}   # cpL_behavior display names (as in vitl_score.py)


def partial_spearman(x, y, control):
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(control)
    A = np.column_stack([cr, np.ones_like(cr)])
    xres = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    yres = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return spearmanr(xres, yres).correlation


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def frozen_fit(train, feat_col):
    """F5 frozen-protocol fit with `feat_col` in place of overlap (cf. vitl_score.py)."""
    tr = train.copy()
    for c in [feat_col, "uniformity_t2"]:
        tr[c + "_z"] = tr.groupby("encoder")[c].transform(zscore)
    tr["help"] = (tr.dknn > 0).astype(int)
    return LogisticRegression(max_iter=1000).fit(
        tr[[feat_col + "_z", "uniformity_t2_z"]].values, tr["help"].values)


def holdout_hits(clf, feats, unif, d_knn, d_lp):
    z = np.column_stack([zscore(feats), zscore(unif)])
    pred = clf.predict_proba(z)[:, 1] >= 0.5
    return int((pred == (d_knn > 0)).sum()), int((pred == (d_lp > 0)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", default=str(OUT / "nd8_overlap.csv"))
    args = ap.parse_args()
    if not _P(args.overlap).exists():
        sys.exit(f"MISSING {args.overlap} — run the ND8 GPU pass first "
                 f"(run/slurm/new_direction/nd8_overlap.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd8_overlap_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd8_overlap.csv', index=False)\"")

    nd8 = pd.read_csv(args.overlap)
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"][["encoder", "dataset", "uniformity_t2",
                                          "neighbor_overlap_k50"]]
    nd6 = pd.read_csv(OUT / "nd6_alignment.csv")[["encoder", "dataset", "cC_K"]]
    b = nd8[nd8.encoder.isin(ENCS_B)].merge(geo, on=["encoder", "dataset"]) \
                                     .merge(nd6, on=["encoder", "dataset"])
    knn = load_long()
    knn = (knn[knn.is_max & knn.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
           .groupby(["Backbone", "dataset_key"]).knn_pre.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))

    # ---- ND8-0 validation ----------------------------------------------------------------
    diff = (b.overlap_raw - b.neighbor_overlap_k50).abs()
    print(f"ND8-0 protocol reproduction: max |overlap_raw - geometry_15| = {diff.max():.4f} "
          f"({'PASS' if diff.max() < 0.02 else 'FAIL — STOP, protocol drifted'})")

    # ---- ND8-1 primary: orthogonality cure ------------------------------------------------
    print("\nND8-1 partial rho(feature, cC_K | uniformity) per encoder "
          "[raw baseline should be negative; sun_knn rows sign-flipped: distance -> proximity]:")
    part = {}
    for v in FEATS:
        sgn = -1.0 if v.startswith("sun_knn") else 1.0
        part[v] = {e: sgn * partial_spearman(b[b.encoder == e][v], b[b.encoder == e].cC_K,
                                             b[b.encoder == e].uniformity_t2) for e in ENCS_B}
    tab = pd.DataFrame(part).T.round(3)
    print(tab.to_string())
    raw = tab.loc["overlap_raw"]
    for v in FEATS[1:]:
        beats = int((tab.loc[v] > raw).sum())
        pos = int((tab.loc[v] > 0).sum())
        print(f"  {v:>18}: beats raw on {beats}/4, positive on {pos}/4 -> "
              f"{'CURES' if beats >= 3 and pos >= 2 else 'no cure'}")

    # ---- ND8-2 level utility ---------------------------------------------------------------
    bk = b.merge(knn, on=["encoder", "dataset"])   # 3 encoders (SigLIP knn via xlsx omitted)
    print("\nND8-2 |rho(feature, knn_pre)| per encoder (D3/CLIP/MAE):")
    lev = {v: {e: abs(spearmanr(bk[bk.encoder == e][v], bk[bk.encoder == e].knn_pre)
                      .correlation) for e in ["DINOv3", "CLIP", "MAE"]} for v in FEATS}
    print(pd.DataFrame(lev).T.round(3).to_string())

    # ---- ND8-3 mechanism -------------------------------------------------------------------
    print("\nND8-3 mechanism: median k-occurrence skewness per protocol + centrality:")
    for v in ["raw", "centered", "mp", "nicdm"]:
        print(f"  skew_{v:>8}: median {b['skew_' + v].median():.2f}")
    print(f"  hub_centrality_rho: median {b.hub_centrality_rho.median():+.3f} "
          f"({'hubs ARE centroid-proximal' if b.hub_centrality_rho.median() < 0 else 'mechanism NOT confirmed'})")

    # ---- ND8-4 tool v2 ---------------------------------------------------------------------
    print("\nND8-4 frozen-tool v2 test (train D3+CLIP 2-method pre-refresh; "
          "holdouts SigLIP 15 + ViT-L 7):")
    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False) & df.is_max]
    tgt = (inv[inv.Backbone.isin(["DINOv3", "CLIP"])]
           .groupby(["Backbone", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    train = b.merge(tgt, on=["encoder", "dataset"])

    c2 = pd.read_csv(OUT / "c2_siglip_score.csv")
    sig = b[b.encoder == "SigLIP"].merge(c2[["dataset", "real_dknn", "real_dlp"]],
                                         on="dataset")

    geoL = pd.read_csv(OUT / "geometry_vitL.csv")[["dataset", "uniformity_t2"]]
    L = nd8[nd8.encoder == "DINOv3L"].merge(geoL, on="dataset", suffixes=("", "_g"))
    assert len(L) == 15, f"DINOv3L rows incomplete ({len(L)}/15) — partial nd8 shards?"
    # frozen protocol (vitl_score.py): z-scores over DINOv3L's 15 datasets, THEN join to 7
    for c in FEATS + ["uniformity_t2"]:
        L[c + "_z"] = zscore(L[c])
    beh = pd.read_csv(OUT / "cpL_behavior.csv")
    beh["dataset"] = beh["display"].map(DISPLAY2KEY)
    pre = beh[beh.kind == "pre"].groupby("dataset")[["pre_knn", "pre_lp"]].mean()
    post = beh[beh.kind == "cp"].groupby("dataset")[["post_knn", "post_lp"]].mean()
    dL = (post.join(pre).assign(dknn=lambda x: x.post_knn - x.pre_knn,
                                dlp=lambda x: x.post_lp - x.pre_lp).reset_index())

    print(f"{'feature':>18} {'SigLIP kNN':>10} {'SigLIP LP':>9} {'ViT-L kNN':>9} "
          f"{'ViT-L LP':>8} {'total':>6}")
    totals = {}
    Lj = L.merge(dL, on="dataset")
    for v in FEATS:
        clf = frozen_fit(train, v)
        sk, sl = holdout_hits(clf, sig[v].values, sig.uniformity_t2.values,
                              sig.real_dknn.values, sig.real_dlp.values)
        predL = clf.predict_proba(
            Lj[[v + "_z", "uniformity_t2_z"]].values)[:, 1] >= 0.5
        lk = int((predL == (Lj.dknn > 0)).sum())
        ll = int((predL == (Lj.dlp > 0)).sum())
        totals[v] = (sk + sl, lk + ll)
        print(f"{v:>18} {sk:>7}/15 {sl:>6}/15 {lk:>6}/{len(Lj)} {ll:>5}/{len(Lj)} "
              f"{sk + sl + lk + ll:>6}")
    v1_sig, v1_L = totals["overlap_raw"]
    for v in FEATS[1:]:
        s, l = totals[v]
        q = s >= v1_sig and l >= v1_L
        print(f"  {v:>18}: {'QUALIFIES as v2 candidate' if q else 'does not qualify'} "
              f"(SigLIP {s} vs v1 {v1_sig}; ViT-L {l} vs v1 {v1_L})")
    print("\nNOTE: v1 row is an internal sanity check — it must reproduce the recorded "
          "SigLIP/ViT-L hit counts; if it does not, distrust this table before anything else.")


if __name__ == "__main__":
    main()
