#!/usr/bin/env python
"""
nd6_verdict.py — ND6 (CPU adjudicator): can round-2 theory quantities upgrade the level
channel of F1/F5?

Three theory imports being tested (papers/new_direction/NEW_DIRECTION_R2.md §A/§C):
task-model alignment C(rho) (Canatar 2021), the omniscient LP risk estimate (Wei 2022),
and hubness (Radovanovic 2010). Baselines: the nd1 spectral metrics (rankme / alpha /
uniformity / vci) on the same features.

Pre-registered readouts (declared 2026-07-11, BEFORE nd6_alignment.csv existed):
  ND6-1 PRIMARY (alignment universality): rho(caucC_log, pre-CP kNN) > 0 on ALL four
        encoders. C(rho) is label-aware, so theory predicts an encoder-UNIVERSAL level
        law (like VCI, unlike uniformity/rankme/alpha which split the encoders).
  ND6-2 (R3 reframing — coupling = alignment): SigLIP-2 shows an alignment DEFICIT
        relative to CLIP despite CLIP-like geometry: caucC_log(SigLIP) < caucC_log(CLIP)
        on >= 11/15 datasets, AND SigLIP's per-dataset alignment is closer to MAE's than
        to CLIP's (median |SigLIP-MAE| < median |SigLIP-CLIP|). Either failing = the
        coupling anomaly is NOT an alignment deficit.
  ND6-3 (F5 upgrade, LP channel): |rho(omni_risk_n1000, lp_pre)| exceeds BOTH
        |rho(rankme, lp_pre)| and |rho(alpha, lp_pre)| on >= 3/4 encoders (expected sign:
        negative — lower predicted risk, higher LP). PASS = the theory-grounded predictor
        beats the spectral scalars on the level channel.
  ND6-4 (hubness, exploratory): (a) rho(skew_n10, knn_pre) < 0 on all four encoders;
        (b) Radovanovic template: |rho(bad_frac_n10, knn_pre)| > |rho(skew_n10, knn_pre)|
        on >= 3/4 encoders (label-geometry misalignment dominates pure hubness).
n=15 Spearman per cell; sign-pattern evidence, no FDR family (diagnostic round).

Inputs: eval/outputs/nd6_alignment.csv (cluster), nd1_precp_spectral.csv, results.xlsx.
Run (local): python eval/new_direction/nd6_verdict.py
"""
import argparse
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from nd1_verdict import knn_pre_levels
from load_results import load_long, RESULTS_XLSX

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
ENCS = ["DINOv3", "CLIP", "SigLIP", "MAE"]
NEW_METRICS = ["caucC_log", "cC100", "cC_K", "omni_risk_n500", "omni_risk_n1000",
               "skew_n10", "bad_frac_n10"]
BASELINES = ["uniformity_t2", "rankme", "alpha", "vci"]


def siglip_lp_pre(xlsx_path):
    """SigLIP pre-CP LP per dataset from the 'By Method (SigLIP)' sheet (MAX rows).
    Same extraction as bilinear_law.siglip_knn_pre but column 10 (lp_m) instead of 8
    (layout: PRE-CP 8 knn_m 9 knn_s 10 lp_m — see load_results.py header comment);
    the display-name map is copied verbatim from bilinear_law.XLSX_DS_KEY (proven
    against this exact sheet)."""
    import openpyxl
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "F4_gate"))
    from bilinear_law import XLSX_DS_KEY
    ws = openpyxl.load_workbook(xlsx_path, read_only=True)["By Method (SigLIP)"]
    vals = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        ds_disp, num_data, lp = row[3], row[4], row[10]
        if not ds_disp or not num_data or "MAX" not in str(num_data) or lp is None:
            continue
        key = XLSX_DS_KEY.get(str(ds_disp).lower().replace("_", "").replace(" ", ""))
        if key:
            vals.setdefault(key, []).append(float(lp))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def lp_pre_levels():
    df = load_long()
    lv = (df[df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
          .groupby(["Backbone", "dataset_key"]).lp_pre.mean().reset_index()
          .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    sig = pd.DataFrame([{"encoder": "SigLIP", "dataset": k, "lp_pre": v}
                        for k, v in siglip_lp_pre(RESULTS_XLSX).items()])
    return pd.concat([lv, sig], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", default=str(OUT / "nd6_alignment.csv"))
    ap.add_argument("--pre", default=str(OUT / "nd1_precp_spectral.csv"))
    args = ap.parse_args()
    if not _P(args.alignment).exists():
        sys.exit(f"MISSING {args.alignment} — run the ND6 GPU pass first "
                 f"(run/slurm/new_direction/nd6_alignment.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd6_alignment_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd6_alignment.csv', index=False)\"")

    a = pd.read_csv(args.alignment)
    base = pd.read_csv(args.pre)[["encoder", "dataset"] + BASELINES]
    t = (a.merge(base, on=["encoder", "dataset"])
          .merge(knn_pre_levels(), on=["encoder", "dataset"])
          .merge(lp_pre_levels(), on=["encoder", "dataset"]))

    rows = []
    for enc in ENCS:
        g = t[t.encoder == enc]
        for m in NEW_METRICS + BASELINES:
            for tgt in ["knn_pre", "lp_pre"]:
                r, p = spearmanr(g[m], g[tgt])
                rows.append(dict(encoder=enc, metric=m, target=tgt,
                                 rho=round(r, 4), p=round(p, 5), n=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "nd6_level_channel.csv", index=False)

    for tgt in ["knn_pre", "lp_pre"]:
        print("=" * 88)
        print(f"ND6 — level channel rho(metric, {tgt}) per encoder")
        print("=" * 88)
        print(res[res.target == tgt].pivot(index="metric", columns="encoder",
                                           values="rho").round(3)
              .reindex(NEW_METRICS + BASELINES).to_string())

    piv_k = res[res.target == "knn_pre"].pivot(index="metric", columns="encoder", values="rho")
    piv_l = res[res.target == "lp_pre"].pivot(index="metric", columns="encoder", values="rho")

    # ---- ND6-1: alignment universality ---------------------------------------------------
    v1 = (piv_k.loc["caucC_log"] > 0).all()
    print(f"\nND6-1 PRIMARY  rho(caucC_log, knn_pre) > 0 on all 4 encoders: "
          f"{'PASS' if v1 else 'FAIL'}  ({piv_k.loc['caucC_log'].round(3).to_dict()})")

    # ---- ND6-2: coupling anomaly = alignment deficit? ------------------------------------
    wide = a.pivot(index="dataset", columns="encoder", values="caucC_log").dropna()
    n_lower = int((wide["SigLIP"] < wide["CLIP"]).sum())
    d_mae = (wide["SigLIP"] - wide["MAE"]).abs().median()
    d_clip = (wide["SigLIP"] - wide["CLIP"]).abs().median()
    v2a, v2b = n_lower >= 11, d_mae < d_clip
    print(f"ND6-2 alignment deficit: SigLIP < CLIP on {n_lower}/15 datasets "
          f"({'OK' if v2a else 'MISS'}, need >=11); median |SigLIP-MAE|={d_mae:.4f} vs "
          f"|SigLIP-CLIP|={d_clip:.4f} ({'OK' if v2b else 'MISS'})")
    print(f"      verdict: {'PASS — coupling anomaly reads as an alignment deficit' if v2a and v2b else 'FAIL — alignment does not explain the coupling anomaly'}")

    # ---- ND6-3: theory predictor vs spectral baselines on the LP channel -----------------
    wins = 0
    for enc in ENCS:
        omni = abs(piv_l.loc["omni_risk_n1000", enc])
        beat = omni > abs(piv_l.loc["rankme", enc]) and omni > abs(piv_l.loc["alpha", enc])
        wins += beat
        print(f"ND6-3 {enc:>7}: |rho(omni,lp)|={omni:.3f} vs rankme "
              f"{abs(piv_l.loc['rankme', enc]):.3f} / alpha {abs(piv_l.loc['alpha', enc]):.3f} "
              f"{'WIN' if beat else 'lose'}")
    print(f"      verdict: {'PASS' if wins >= 3 else 'FAIL'} ({wins}/4, need >=3; "
          f"expected sign negative: {(piv_l.loc['omni_risk_n1000'] < 0).sum()}/4 negative)")

    # ---- ND6-4: hubness ------------------------------------------------------------------
    v4a = (piv_k.loc["skew_n10"] < 0).all()
    n_tmpl = sum(int(abs(piv_k.loc["bad_frac_n10", e]) > abs(piv_k.loc["skew_n10", e]))
                 for e in ENCS)
    print(f"ND6-4 (a) rho(skew_n10, knn_pre) < 0 on all 4: {'PASS' if v4a else 'FAIL'} "
          f"({piv_k.loc['skew_n10'].round(3).to_dict()})")
    print(f"      (b) |bad_frac| > |skew| (Radovanovic template): {n_tmpl}/4 "
          f"({'PASS' if n_tmpl >= 3 else 'FAIL'}, need >=3)")
    print(f"\nwrote -> {OUT / 'nd6_level_channel.csv'}")


if __name__ == "__main__":
    main()
