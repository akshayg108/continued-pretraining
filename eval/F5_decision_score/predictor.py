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
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))  # eval/ root for shared modules

import argparse

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


def _per_encoder_zscore(tr, te, feats):
    """Per-encoder z-score ABLATION (LOO-safe): standardize each feature WITHIN each
    encoder group, using mean/std fit on TRAIN rows only, then applied to TRAIN+TEST.

    Returns (tr_x, te_x) feature frames. Unlike the default global StandardScaler
    (one mean/std over all encoders pooled), this removes each encoder's own location
    and scale before pooling, so the LogisticRegression sees encoder-relative geometry.
    Encoders present only in `te` (no train rows) fall back to their own test stats.
    """
    tr_x, te_x = tr[feats].copy(), te[feats].copy()
    stats = {}
    for enc, g in tr.groupby(tr.encoder):
        mu, sd = g[feats].mean(), g[feats].std(ddof=0).replace(0.0, 1.0)
        stats[enc] = (mu, sd)
        idx = tr.index[tr.encoder == enc]
        tr_x.loc[idx] = (tr.loc[idx, feats] - mu) / sd
    for enc in te.encoder.unique():
        idx = te.index[te.encoder == enc]
        if enc in stats:
            mu, sd = stats[enc]
        else:  # encoder unseen in train -> use its own test stats so columns are comparable
            mu = te.loc[idx, feats].mean()
            sd = te.loc[idx, feats].std(ddof=0).replace(0.0, 1.0)
        te_x.loc[idx] = (te.loc[idx, feats] - mu) / sd
    return tr_x, te_x


def loo_dataset(m, feats, per_encoder=False):
    """Pooled leave-one-dataset-out over sphere encoders -> balanced-acc + AUC.

    per_encoder=False (default): features standardized GLOBALLY by the pipeline's
        StandardScaler (existing behavior, unchanged).
    per_encoder=True (ablation): features z-scored WITHIN each encoder (train-fit)
        before the pipeline; the global scaler then becomes a near-identity rescale.
    """
    s = m[m.encoder.isin(SPHERE)].reset_index(drop=True)
    proba = np.full(len(s), np.nan)
    for ds in s.dataset.unique():
        tr, te = s[s.dataset != ds], s[s.dataset == ds]
        if tr.help.nunique() < 2:
            continue
        tr_x, te_x = (_per_encoder_zscore(tr, te, feats) if per_encoder
                      else (tr[feats], te[feats]))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr_x, tr.help)
        proba[te.index] = clf.predict_proba(te_x)[:, 1]
    ok = ~np.isnan(proba)
    yt, pp = s.help.values[ok], proba[ok]
    ba = balanced_accuracy_score(yt, (pp >= 0.5).astype(int))
    auc = roc_auc_score(yt, pp) if len(np.unique(yt)) == 2 else float("nan")
    return ba, auc, int(ok.sum())


def cross_encoder(m, feats, fit_enc, test_enc, per_encoder=False):
    tr, te = m[m.encoder.isin(fit_enc)], m[m.encoder == test_enc]
    if len(te) == 0 or tr.help.nunique() < 2 or te.help.nunique() < 2:
        return None
    tr_x, te_x = (_per_encoder_zscore(tr, te, feats) if per_encoder
                  else (tr[feats], te[feats]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr_x, tr.help)
    pp = clf.predict_proba(te_x)[:, 1]
    return balanced_accuracy_score(te.help, (pp >= 0.5).astype(int)), roc_auc_score(te.help, pp), len(te)


def report(target, name, per_encoder=False):
    m = dataset_table(target)
    sph = m[m.encoder.isin(SPHERE)]
    tag = "per-encoder-standardize" if per_encoder else "global-standardize"
    print(f"\n===== target: {name}  (help = {name} > 0)  [{tag}] =====")
    print(f"sphere points: {len(sph)}  | help rate: {sph.help.mean():.2f}")
    ba, auc, n = loo_dataset(m, FEATS, per_encoder=per_encoder)
    print(f"  OURS  overlap+uniformity   LOO  bal-acc={ba:.3f}  AUC={auc:.3f}  (n={n})")
    bk, ak, _ = loo_dataset(m, ["knn_pre"], per_encoder=per_encoder)
    print(f"  Sorkhei-style  knn_pre      LOO  bal-acc={bk:.3f}  AUC={ak:.3f}")
    print(f"  majority baseline                bal-acc=0.500")
    for fit, te in [(["CLIP"], "DINOv3"), (["DINOv3"], "CLIP")]:
        r = cross_encoder(m, FEATS, fit, te, per_encoder=per_encoder)
        if r:
            print(f"  cross-encoder fit{fit}->{te:7s}  bal-acc={r[0]:.3f}  AUC={r[1]:.3f}  (n={r[2]})")
    if "SigLIP" in m.encoder.unique():
        r = cross_encoder(m, FEATS, SPHERE, "SigLIP", per_encoder=per_encoder)
        if r:
            print(f"  cross-encoder fit[DINOv3,CLIP]->SigLIP  bal-acc={r[0]:.3f}  AUC={r[1]:.3f}  (n={r[2]})")
    else:
        print("  (SigLIP has no CP results to score against -> cross-encoder transfer cannot be evaluated)")
    return m


def standardize_ablation(target="dknn", name="ΔkNN"):
    """C2 ablation: global vs per-encoder feature standardization for one target.

    Returns a long-format DataFrame (one row per variant×metric block) and saves it to
    eval/outputs/predictor_standardize_ablation.csv. SigLIP cross-encoder rows are only
    emitted when SigLIP survives into the merged table (i.e. has CP results to score).
    """
    m = dataset_table(target)
    siglip_in_geo = "SigLIP" in pd.read_csv(ROOT / "eval/outputs/geometry_15.csv").encoder.unique()
    siglip_scorable = "SigLIP" in m.encoder.unique()
    rows = []
    for per_encoder in (False, True):
        variant = "per_encoder" if per_encoder else "global"
        ba, auc, n = loo_dataset(m, FEATS, per_encoder=per_encoder)
        rows.append(dict(target=target, target_name=name, variant=variant,
                         block="loo_sphere_DINOv3+CLIP", bal_acc=ba, auc=auc, n=n))
        r = cross_encoder(m, FEATS, SPHERE, "SigLIP", per_encoder=per_encoder)
        if r:
            rows.append(dict(target=target, target_name=name, variant=variant,
                             block="cross_encoder_DINOv3+CLIP->SigLIP",
                             bal_acc=r[0], auc=r[1], n=r[2]))
        else:
            note = ("SigLIP_in_geometry_15_but_no_CP_results_to_score"
                    if siglip_in_geo else "SigLIP_not_in_geometry_15")
            rows.append(dict(target=target, target_name=name, variant=variant,
                             block="cross_encoder_DINOv3+CLIP->SigLIP",
                             bal_acc=np.nan, auc=np.nan, n=0, note=note))
    out = pd.DataFrame(rows)
    path = ROOT / "eval/outputs/predictor_standardize_ablation.csv"
    out.to_csv(path, index=False)
    print(f"\nSigLIP in geometry_15: {siglip_in_geo}  | SigLIP scorable (has CP results): {siglip_scorable}")
    print(out.to_string(index=False))
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-encoder-standardize", action="store_true",
                    help="ABLATION: z-score features WITHIN each encoder before pooling "
                         "(default OFF = global StandardScaler, existing behavior).")
    ap.add_argument("--ablation", action="store_true",
                    help="Run the global-vs-per-encoder standardize comparison for ΔkNN "
                         "and save eval/outputs/predictor_standardize_ablation.csv.")
    args = ap.parse_args()

    if args.ablation:
        standardize_ablation("dknn", "ΔkNN")
    else:
        pe = args.per_encoder_standardize
        mk = report("dknn", "ΔkNN", per_encoder=pe)
        report("dlp", "ΔLP", per_encoder=pe)
        report("dft", "ΔFT", per_encoder=pe)
        # Only the default (global-standardize) run owns the canonical predictor.csv.
        if not pe:
            mk.to_csv(ROOT / "eval/outputs/predictor.csv", index=False)
            print(f"\nsaved {ROOT / 'eval/outputs/predictor.csv'}")
