#!/usr/bin/env python
"""
predictor.py — ICLR C2: the actionable pre-CP decision rule.

From PRE-CP geometry ONLY (computed before any continued pretraining), predict whether CP will
HELP kNN (sign of ΔkNN) on sphere-native encoders, evaluated OUT-OF-SAMPLE and compared to
baselines. This converts the F1 correlational law into a tool ("should I CP this target? which way
will frozen quality vs fine-tune move?"). Repeat for ΔFT (expected opposite-sign rule).

Features (pre-CP, per encoder×dataset): neighbor_overlap_k50, uniformity_t2  (from geometry_15.csv)
Target: sign(Δ@MAX), Δ averaged over LeJEPA-CP + SimCLR-CP  (from results.xlsx)
Eval: leave-one-DATASET-out (pooled, sphere encoders) + cross-encoder transfer; baselines =
majority and "Sorkhei-style" (pre-CP frozen kNN quality knn_pre as the score).

CPU-only. Run: python eval/f1_position/predictor.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # eval/ root for shared modules

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent  # continued-pretraining
SPHERE = ["DINOv3", "CLIP"]
FEATS = ["neighbor_overlap_k50", "uniformity_t2"]


def dataset_table(target):
    geo = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    geo = geo[geo.dataset != "imagenet"]
    df = load_long()
    inv = df[df.Method.str.contains("LeJEPA|SimCLR", case=False, na=False) & df.is_max]
    y = (inv.groupby(["Backbone", "dataset_key"])[[target, "knn_pre"]].mean()
            .reset_index().rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    m = geo.merge(y, on=["encoder", "dataset"]).dropna(subset=FEATS + [target, "knn_pre"])
    m["help"] = (m[target] > 0).astype(int)
    return m


def loo_dataset(m, feats):
    """Pooled leave-one-dataset-out over sphere encoders -> balanced-acc + AUC."""
    s = m[m.encoder.isin(SPHERE)].reset_index(drop=True)
    proba = np.full(len(s), np.nan)
    for ds in s.dataset.unique():
        tr, te = s[s.dataset != ds], s[s.dataset == ds]
        if tr.help.nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[feats], tr.help)
        proba[te.index] = clf.predict_proba(te[feats])[:, 1]
    ok = ~np.isnan(proba)
    yt, pp = s.help.values[ok], proba[ok]
    ba = balanced_accuracy_score(yt, (pp >= 0.5).astype(int))
    auc = roc_auc_score(yt, pp) if len(np.unique(yt)) == 2 else float("nan")
    return ba, auc, int(ok.sum())


def cross_encoder(m, feats, fit_enc, test_enc):
    tr, te = m[m.encoder.isin(fit_enc)], m[m.encoder == test_enc]
    if len(te) == 0 or tr.help.nunique() < 2 or te.help.nunique() < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[feats], tr.help)
    pp = clf.predict_proba(te[feats])[:, 1]
    return balanced_accuracy_score(te.help, (pp >= 0.5).astype(int)), roc_auc_score(te.help, pp), len(te)


def report(target, name):
    m = dataset_table(target)
    sph = m[m.encoder.isin(SPHERE)]
    print(f"\n===== target: {name}  (help = {name} > 0) =====")
    print(f"sphere points: {len(sph)}  | help rate: {sph.help.mean():.2f}")
    ba, auc, n = loo_dataset(m, FEATS)
    print(f"  OURS  overlap+uniformity   LOO  bal-acc={ba:.3f}  AUC={auc:.3f}  (n={n})")
    bk, ak, _ = loo_dataset(m, ["knn_pre"])
    print(f"  Sorkhei-style  knn_pre      LOO  bal-acc={bk:.3f}  AUC={ak:.3f}")
    print(f"  majority baseline                bal-acc=0.500")
    for fit, te in [(["CLIP"], "DINOv3"), (["DINOv3"], "CLIP")]:
        r = cross_encoder(m, FEATS, fit, te)
        if r:
            print(f"  cross-encoder fit{fit}->{te:7s}  bal-acc={r[0]:.3f}  AUC={r[1]:.3f}  (n={r[2]})")
    if "SigLIP" in m.encoder.unique():
        r = cross_encoder(m, FEATS, SPHERE, "SigLIP")
        if r:
            print(f"  cross-encoder fit[DINOv3,CLIP]->SigLIP  bal-acc={r[0]:.3f}  AUC={r[1]:.3f}  (n={r[2]})")
    else:
        print("  (SigLIP not in geometry_15 yet -> run Task 2 for the headline cross-encoder test)")
    return m


if __name__ == "__main__":
    mk = report("dknn", "ΔkNN")
    report("dlp", "ΔLP")
    report("dft", "ΔFT")
    mk.to_csv(ROOT / "eval/outputs/predictor.csv", index=False)
    print(f"\nsaved {ROOT / 'eval/outputs/predictor.csv'}")
