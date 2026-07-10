#!/usr/bin/env python
"""
nd1_verdict.py — ND1 (CPU adjudicator): is the SigLIP-2 level-channel sign flip a
metric artifact or a model property?

Baseline anomaly (findings/FINDINGS_step9 addendum): the level channel
rho(uniformity_t2, pre-CP kNN) across 15 datasets is NEGATIVE on DINOv3 (-0.25) and
CLIP (-0.13) but POSITIVE on SigLIP-2 (+0.171), with readout variants excluded as the
cause. Tsitsulin et al. 2023 (papers/new_direction/) show 5 of 7 spectral metrics can
reverse correlation sign from the metric-by-condition interaction alone; coherence was
their only sign-stable axis. VCI (Xu et al. 2023) is invariant to invertible linear
transforms, which uniformity is not.

Pre-registered readout (declared 2026-07-10, BEFORE nd1_precp_spectral.csv existed):
  ND1-A  Baseline reproduces from the new feature pass: sign(rho_unif) SigLIP != DINOv3/CLIP.
  ND1-B  Primary: under coherence_mu, does SigLIP-2 carry the SAME sign as DINOv3 and CLIP?
         YES -> the flip is metric-specific (Tsitsulin artifact-template supported);
         NO  -> the flip survives a sign-stable axis (model-property reading strengthened).
  ND1-C  Secondary, same question for vci / rankme / alpha (descriptive; no single-metric
         verdict — reported as the count of axes on which SigLIP aligns with the spheres).
n=15 Spearman correlations; point estimates with p-values, no FDR family (diagnostic, not
a confirmatory claim).

Inputs: eval/outputs/nd1_precp_spectral.csv (cluster), results.xlsx (pre-CP kNN levels).
Run (local): python eval/new_direction/nd1_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "F4_gate"))
from load_results import load_long
from bilinear_law import siglip_knn_pre

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
METRICS = ["uniformity_t2", "coherence_mu", "coherence_mu99", "vci", "rankme", "alpha"]
SPHERE = ["DINOv3", "CLIP"]


def knn_pre_levels():
    """Pre-CP kNN per (encoder, dataset): results.xlsx via load_long for D3/CLIP/MAE
    (mean of knn_pre over MAX rows, same convention as bilinear_law), SigLIP sheet direct."""
    df = load_long()
    lv = (df[df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
          .groupby(["Backbone", "dataset_key"]).knn_pre.mean().reset_index()
          .rename(columns={"Backbone": "encoder", "dataset_key": "dataset",
                           "knn_pre": "knn_pre"}))
    sig = pd.DataFrame([{"encoder": "SigLIP", "dataset": k, "knn_pre": v}
                        for k, v in siglip_knn_pre(ROOT.parent / "results.xlsx").items()])
    return pd.concat([lv, sig], ignore_index=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(OUT / "nd1_precp_spectral.csv"))
    args = ap.parse_args()
    spec_path = _P(args.spec)
    if not spec_path.exists():
        sys.exit(f"MISSING {spec_path} — run the ND1 GPU pass first "
                 f"(run/slurm/new_direction/nd1_precp_spectral.sh)")
    spec = pd.read_csv(spec_path)
    t = spec.merge(knn_pre_levels(), on=["encoder", "dataset"])

    rows = []
    for enc in ["DINOv3", "CLIP", "SigLIP", "MAE"]:
        g = t[t.encoder == enc]
        for m in METRICS:
            r, p = spearmanr(g[m], g.knn_pre)
            rows.append(dict(encoder=enc, metric=m, rho=round(r, 4), p=round(p, 5), n=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "nd1_level_channel.csv", index=False)

    print("=" * 88)
    print("ND1 — level channel rho(metric, pre-CP kNN) per encoder (n per cell below)")
    print("=" * 88)
    print(res.pivot(index="metric", columns="encoder", values="rho").round(3).to_string())

    piv = res.pivot(index="metric", columns="encoder", values="rho")
    sgn = np.sign(piv)

    a = sgn.loc["uniformity_t2", "SigLIP"] != sgn.loc["uniformity_t2", SPHERE].iloc[0] \
        and (sgn.loc["uniformity_t2", SPHERE] < 0).all() and sgn.loc["uniformity_t2", "SigLIP"] > 0
    print(f"\nND1-A baseline anomaly reproduces (unif: D3/CLIP<0, SigLIP>0): "
          f"{'PASS' if a else 'FAIL'}")

    aligned = (sgn.loc["coherence_mu", "SigLIP"] == sgn.loc["coherence_mu", SPHERE]).all()
    print(f"ND1-B PRIMARY  coherence_mu: SigLIP sign == DINOv3 == CLIP: "
          f"{'YES' if aligned else 'NO'}")
    print("      -> " + ("flip is METRIC-SPECIFIC: supports the Tsitsulin "
                         "metric-by-condition artifact template for A4"
                         if aligned else
                         "flip SURVIVES the sign-stable axis: strengthens the "
                         "model-property reading of A4"))

    n_align = sum(int((sgn.loc[m, "SigLIP"] == sgn.loc[m, SPHERE]).all())
                  for m in ["vci", "rankme", "alpha", "coherence_mu99"])
    print(f"ND1-C secondary: SigLIP aligns with the spheres on {n_align}/4 further axes "
          f"(vci / rankme / alpha / coherence_mu99)")

    bad = spec[spec.alpha_r2 < 0.8]
    if len(bad):
        print(f"\nWARN: {len(bad)} cells with alpha fit R2 < 0.8 — treat their alpha "
              f"entries as unreliable:\n{bad[['encoder', 'dataset', 'alpha_r2']].to_string(index=False)}")
    print(f"\nwrote -> {OUT / 'nd1_level_channel.csv'}")


if __name__ == "__main__":
    main()
