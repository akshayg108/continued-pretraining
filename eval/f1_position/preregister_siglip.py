#!/usr/bin/env python
"""
preregister_siglip.py — ICLR C2 rigor upgrade: PRE-REGISTERED out-of-sample predictions for SigLIP-2.

Before any SigLIP CP run exists, fit the decision rule on the two fitted sphere encoders
(DINOv3 + CLIP, invariance methods LeJEPA-CP + SimCLR-CP) and predict, for each of SigLIP-2's 15
datasets, whether CP will HELP vs HURT frozen transfer (sign of ΔkNN / ΔLP) and fine-tune (ΔFT,
opposite sign). These predictions are FROZEN here; when the SigLIP CP runs complete, score them
against the realized signs (cross-encoder generalization = the headline C2 evidence).

Key methodological choice: features are standardized WITHIN each encoder (z-score over its own 15
datasets) before fit/predict, so the predictor transfers the *relationship* (rank of overlap /
uniformity -> help/hurt) and is NOT confounded by SigLIP-2's higher absolute overlap scale
(~3x DINOv3/CLIP). Rank-consistency of SigLIP-2 geometry vs DINOv3/CLIP is ~0.9 (≈ DINOv3↔CLIP),
which is what makes this transfer well-posed.

CPU-only. Run: python eval/f1_position/preregister_siglip.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # eval/ root

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
FIT_ENCODERS = ["DINOv3", "CLIP"]   # encoders WITH CP data, used to fit
TEST_ENCODER = "SigLIP"             # held-out encoder, geometry only (no CP data yet)
FEATS = ["neighbor_overlap_k50", "uniformity_t2"]
FROZEN = "2026-06-19"               # pre-registration date


def zscore_within(df, feats, group="encoder"):
    out = df.copy()
    for f in feats:
        out[f] = df.groupby(group)[f].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    return out


def main():
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False) & df.is_max]
    agg = (inv.groupby(["Backbone", "dataset_key"])[["dknn", "dlp", "dft"]].mean()
              .reset_index().rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    m = geo.merge(agg, on=["encoder", "dataset"], how="left")

    fit = m[m.encoder.isin(FIT_ENCODERS)].copy()
    test = m[m.encoder == TEST_ENCODER].dropna(subset=FEATS).copy()
    # standardize features within each encoder (removes per-encoder scale offset)
    pool = pd.concat([fit, test])
    pool = zscore_within(pool, FEATS)
    fitz = pool[pool.encoder.isin(FIT_ENCODERS)]
    testz = pool[pool.encoder == TEST_ENCODER].sort_values("dataset")

    out = test.sort_values("dataset")[["dataset"] + FEATS].reset_index(drop=True)
    print(f"\n{'='*72}\nPRE-REGISTERED SigLIP-2 predictions  (fit on {FIT_ENCODERS}; frozen {FROZEN})\n{'='*72}")
    for tgt, name in [("dknn", "ΔkNN"), ("dlp", "ΔLP"), ("dft", "ΔFT")]:
        tr = fitz.dropna(subset=[tgt])
        y = (tr[tgt] > 0).astype(int)
        if y.nunique() < 2:
            print(f"  {name}: fit encoders have single class -> skip"); continue
        clf = LogisticRegression(max_iter=1000).fit(tr[FEATS], y)
        proba = clf.predict_proba(testz[FEATS])[:, 1]
        out[f"{name}_p_help"] = np.round(proba, 3)
        out[f"{name}_pred"] = np.where(proba >= 0.5, "HELP", "HURT")
        print(f"\n-- {name}  (fit help-rate on DINOv3+CLIP = {y.mean():.2f}, n={len(tr)}) --")
        for d, p in zip(testz.dataset, proba):
            print(f"    {d:16s}  P(help)={p:.2f}  -> {'HELP' if p>=0.5 else 'HURT'}")
        print(f"    predicted HELP on {int((proba>=0.5).sum())}/{len(proba)} SigLIP datasets")

    out.insert(0, "frozen_date", FROZEN)
    out.insert(0, "encoder", TEST_ENCODER)
    dst = ROOT / "eval/outputs/preregister_siglip.csv"
    out.to_csv(dst, index=False)
    print(f"\nsaved {dst}\nWhen SigLIP CP completes: score realized sign(Δ) vs these frozen predictions.")


if __name__ == "__main__":
    main()
