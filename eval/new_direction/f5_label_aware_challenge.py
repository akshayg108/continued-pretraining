#!/usr/bin/env python
"""
f5_label_aware_challenge.py — local replication of the label-aware feature challenge
first run by the v3 re-audit (verification/v3_reaudit_2026-07-15, F5 section).

Question: target labels ARE available at decision time — does a label-aware placement
feature (cC_K) or the naive baseline (knn_pre) beat the frozen 2-feature tool v1?

Protocol (identical to nd8_verdict's ND8-4 frozen-protocol machinery): train on the
DINOv3+CLIP 30 cells with the 2-method (LeJEPA/SimCLR) pre-refresh MAX dknn>0 target;
features z-scored WITHIN encoder; LogisticRegression(max_iter=1000); evaluate sign hits
on the SigLIP holdout (c2_siglip_score real_dknn / real_dlp, 15 datasets). The ViT-L
holdout is skipped: no cC_K exists for DINOv3L (would need a small GPU pass; only
warranted if a challenger wins here).

This is a REPLICATION of an already-run analysis, not a new claim: pass = reproducing
the re-audit's numbers (v1 13/15 kNN / 12/15 LP; cC_K-only 10/15; knn_pre-only 8/15;
v1+cC_K 13/15 kNN). Run (local): python eval/new_direction/f5_label_aware_challenge.py
"""
import sys
from pathlib import Path as _P

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))

from load_results import load_long
from nd1_verdict import knn_pre_levels

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"

VARIANTS = {  # name -> feature list (uniformity_t2 is part of v1, not automatic)
    "v1 (overlap+unif)": ["neighbor_overlap_k50", "uniformity_t2"],
    "cC_K only": ["cC_K"],
    "knn_pre only": ["knn_pre"],
    "v1 + cC_K": ["neighbor_overlap_k50", "uniformity_t2", "cC_K"],
}


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    geo = pd.read_csv(OUT / "geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"][["encoder", "dataset", "uniformity_t2",
                                          "neighbor_overlap_k50"]]
    nd6 = pd.read_csv(OUT / "nd6_alignment.csv")[["encoder", "dataset", "cC_K"]]
    base = geo.merge(nd6, on=["encoder", "dataset"]) \
              .merge(knn_pre_levels(), on=["encoder", "dataset"])

    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False) & df.is_max]
    tgt = (inv[inv.Backbone.isin(["DINOv3", "CLIP"])]
           .groupby(["Backbone", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    train = base[base.encoder.isin(["DINOv3", "CLIP"])].merge(tgt, on=["encoder", "dataset"])
    train["help"] = (train.dknn > 0).astype(int)

    c2 = pd.read_csv(OUT / "c2_siglip_score.csv")
    sig = base[base.encoder == "SigLIP"].merge(c2[["dataset", "real_dknn", "real_dlp"]],
                                               on="dataset")
    assert len(sig) == 15, f"SigLIP holdout incomplete ({len(sig)}/15)"

    print("Label-aware challenge — frozen protocol, SigLIP holdout "
          "(replication target: v1 13/15,12/15; cC_K 10/15; knn_pre 8/15; v1+cC_K 13/15)")
    print(f"{'variant':>20} {'kNN hits':>9} {'LP hits':>8}")
    for name, feats in VARIANTS.items():
        tr = train.copy()
        for c in feats:
            tr[c + "_z"] = tr.groupby("encoder")[c].transform(zscore)
        clf = LogisticRegression(max_iter=1000).fit(
            tr[[c + "_z" for c in feats]].values, tr["help"].values)
        sg = sig.copy()
        for c in feats:
            sg[c + "_z"] = zscore(sg[c])              # SigLIP z over its own 15 datasets
        pred = clf.predict_proba(sg[[c + "_z" for c in feats]].values)[:, 1] >= 0.5
        k = int((pred == (sg.real_dknn > 0)).sum())
        l = int((pred == (sg.real_dlp > 0)).sum())
        print(f"{name:>20} {k:>6}/15 {l:>5}/15")
    print("\nReading: if no challenger exceeds v1's kNN hits, the T2 slot is nailed shut —"
          "\neven label-aware placement does not beat the label-free position+shape tool.")


if __name__ == "__main__":
    main()
