#!/usr/bin/env python
"""
nd10_verdict.py — ND10+ND11 (CPU adjudicator): does the operator decomposition separate
the two CP motions, and does a LOCAL graph quantity carry the kNN benefit sign that the
global quantities provably cannot?

Pre-registered readouts (declared 2026-07-16, BEFORE nd10_operator.csv existed).
Conventions mirror nd7_verdict: cells seed-averaged to (method, encoder, dataset); main
grid = DINOv3/CLIP/MAE x LeJEPA/SimCLR/DIET with per-method dknn from results.xlsx is_max
rows; SigLIP = supplementary panel (2-method dataset means vs c2_siglip_score.real_dknn);
partial Spearman = project rank-residual convention. n=45/135 point estimates, sign/rank
evidence, no FDR family (diagnostic round). knn_purity is label-aware and needs POST
features: it is a MECHANISM quantity, not a decision-tool feature; the pre-only panel at
the end is exploratory.

  ND10-1 (two motions at operator level): rotation share s = |dA_rot|/(|dA_spec|+|dA_rot|)
         has median in [0.2, 0.8] on >= 2/3 main encoders -> both motions are real
         operator components; otherwise the dominant side is named and C3 wording changes.
  ND10-2 (operator two-channel): pooled main grid — partial rho(dA_spec, dknn | dA_rot)
         retains >= half of raw rho(dA_spec, dknn) with p < 0.05, AND the reverse control
         retains >= half with p < 0.05 -> the two components track dknn independently
         (operator version of the step13 two-channel result).
  ND10-3 (structure validation, expected by construction): pooled |rho(dA_rot, dcC_K)| >
         |rho(dA_spec, dcC_K)| — basis rotation is where placement change lives.
  ND11-1 PRIMARY (local beats global): per main encoder (n=45),
         |rho(d_purity, dknn)| > |rho(d_rankme, dknn)| AND > |rho(dcC_K, dknn)| on
         >= 2/3 encoders -> local-carrier candidate CONFIRMED.
  ND11-2 (sign carrier): pooled main grid sign-agreement fraction
         mean[sign(d_purity) == sign(dknn)] >= 0.75 -> PASS (T3 candidate);
         in [0.65, 0.75) -> weak; < 0.65 -> FAIL. Baseline for context: the same
         fraction for sign(dcC_K) (the tide lifts ~85% of cells, so it should do
         poorly on hurt cells — that contrast is the point).
  ND11-3 (operator matching): per main encoder, |rho(d_graph_cC_K, dknn)| >
         |rho(dcC_K, dknn)| on >= 2/3 encoders -> the kNN-matched graph operator beats
         the global feature operator for the kNN channel.

Inputs: eval/outputs/nd10_operator.csv (cluster), nd7_placement.csv,
nd1_precp_spectral.csv, results.xlsx, c2_siglip_score.csv.
Run (local): python eval/new_direction/nd10_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MAIN_ENCS = ["DINOv3", "CLIP", "MAE"]
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP"}


def partial_spearman(x, y, control):
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(control)
    A = np.column_stack([cr, np.ones_like(cr)])
    xres = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
    yres = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return spearmanr(xres, yres)


def main():
    path = OUT / "nd10_operator.csv"
    if not path.exists():
        sys.exit(f"MISSING {path} — run the ND10 GPU pass first "
                 f"(run/slurm/new_direction/nd10_operator.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd10_operator_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd10_operator.csv', index=False)\"")
    nd10 = pd.read_csv(path)
    nd10["d_purity"] = nd10.knn_purity_post - nd10.knn_purity_pre
    nd10["d_graph_cC_K"] = nd10.graph_cC_K_post - nd10.graph_cC_K_pre
    cell = (nd10.groupby(["method", "encoder", "dataset"], as_index=False)
            [["dA_total", "dA_spec", "dA_rot", "dcC_K", "affinity_topK",
              "d_purity", "d_graph_cC_K"]].mean())

    # delta-rank per cell (same sources as nd7_verdict: nd7 post rankme - nd1 pre rankme)
    post = pd.read_csv(OUT / "nd7_placement.csv")
    postr = (post.groupby(["method", "encoder", "dataset"], as_index=False)
             .rankme.mean().rename(columns={"rankme": "rankme_post"}))
    pre = pd.read_csv(OUT / "nd1_precp_spectral.csv")[["encoder", "dataset", "rankme"]] \
        .rename(columns={"rankme": "rankme_pre"})
    cell = cell.merge(postr, on=["method", "encoder", "dataset"], how="left") \
               .merge(pre, on=["encoder", "dataset"], how="left")
    cell["d_rankme"] = cell.rankme_post - cell.rankme_pre

    df = load_long()
    beh = (df[df.is_max & df.Backbone.isin(MAIN_ENCS)]
           .groupby(["Backbone", "Method", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    cell["Method"] = cell.method.map(MMAP)
    t = cell[cell.encoder.isin(MAIN_ENCS)].merge(
        beh, on=["encoder", "Method", "dataset"])
    print(f"main-grid cells joined: {len(t)} (expected ~135)")
    # guard (review 2026-07-16): the nd7/nd1 rank join is a LEFT merge; a NaN d_rankme
    # would make spearmanr return nan and silently count as an ND11-1 loss
    assert t.d_rankme.notna().all(), \
        f"NaN d_rankme in {int(t.d_rankme.isna().sum())} cells — nd7/nd1 coverage drifted"

    # ---- ND10-1 two motions --------------------------------------------------------------
    t["rot_share"] = t.dA_rot.abs() / (t.dA_spec.abs() + t.dA_rot.abs() + 1e-30)
    print("\nND10-1 rotation share of |dA| per encoder [both motions real if median in "
          "[0.2, 0.8] on >=2/3]:")
    hits = 0
    for e in MAIN_ENCS:
        m = t[t.encoder == e].rot_share.median()
        ok = 0.2 <= m <= 0.8
        hits += int(ok)
        print(f"  {e:>8}: median rot_share = {m:.3f} {'in-band' if ok else 'OUT'}")
    print(f"  -> {'PASS — two motions both real at operator level' if hits >= 2 else 'FAIL — one motion dominates (name it in C3)'}")

    # ---- ND10-2 operator two-channel ------------------------------------------------------
    print("\nND10-2 pooled operator two-channel (n={}):".format(len(t)))
    raw_s, praw_s = spearmanr(t.dA_spec, t.dknn)
    part_s, ppart_s = partial_spearman(t.dA_spec, t.dknn, t.dA_rot)
    raw_r, praw_r = spearmanr(t.dA_rot, t.dknn)
    part_r, ppart_r = partial_spearman(t.dA_rot, t.dknn, t.dA_spec)
    print(f"  dA_spec: raw {raw_s:+.3f} (p={praw_s:.2g})  | dA_rot partial {part_s:+.3f} (p={ppart_s:.2g})")
    print(f"  dA_rot : raw {raw_r:+.3f} (p={praw_r:.2g})  | dA_spec partial {part_r:+.3f} (p={ppart_r:.2g})")
    ok2 = (abs(part_s) >= 0.5 * abs(raw_s) and ppart_s < 0.05
           and abs(part_r) >= 0.5 * abs(raw_r) and ppart_r < 0.05)
    print(f"  -> {'PASS — independent operator channels' if ok2 else 'FAIL — components are not independent trackers'}")

    # ---- ND10-3 structure check ----------------------------------------------------------
    r_rot = spearmanr(t.dA_rot, t.dcC_K).correlation
    r_spec = spearmanr(t.dA_spec, t.dcC_K).correlation
    print(f"\nND10-3 |rho(dA_rot, dcC_K)|={abs(r_rot):.3f} vs |rho(dA_spec, dcC_K)|="
          f"{abs(r_spec):.3f} -> {'OK (rotation carries placement, as constructed)' if abs(r_rot) > abs(r_spec) else 'ANOMALY — inspect before trusting the pass'}")

    # ---- ND11-1 local beats global --------------------------------------------------------
    print("\nND11-1 per-encoder |rho(., dknn)| [local carrier if d_purity beats BOTH "
          "d_rankme and dcC_K on >=2/3]:")
    wins = 0
    for e in MAIN_ENCS:
        g = t[t.encoder == e]
        rp = spearmanr(g.d_purity, g.dknn).correlation
        rr = spearmanr(g.d_rankme, g.dknn).correlation
        rc = spearmanr(g.dcC_K, g.dknn).correlation
        win = abs(rp) > abs(rr) and abs(rp) > abs(rc)
        wins += int(win)
        print(f"  {e:>8}: d_purity {rp:+.3f}  d_rankme {rr:+.3f}  dcC_K {rc:+.3f}  "
              f"{'WIN' if win else '—'}")
    print(f"  -> {'LOCAL CARRIER CANDIDATE CONFIRMED' if wins >= 2 else 'not confirmed'}")

    # ---- ND11-2 sign carrier ---------------------------------------------------------------
    nz = t[t.dknn.abs() > 1e-6]
    agree_p = float((np.sign(nz.d_purity) == np.sign(nz.dknn)).mean())
    agree_c = float((np.sign(nz.dcC_K) == np.sign(nz.dknn)).mean())
    verdict = ("PASS (T3 candidate)" if agree_p >= 0.75 else
               "weak" if agree_p >= 0.65 else "FAIL")
    print(f"\nND11-2 pooled sign agreement: sign(d_purity)==sign(dknn) {agree_p:.1%} "
          f"[baseline sign(dcC_K): {agree_c:.1%}] -> {verdict}")

    # ---- ND11-3 operator matching ----------------------------------------------------------
    print("\nND11-3 per-encoder graph placement vs feature placement:")
    wins3 = 0
    for e in MAIN_ENCS:
        g = t[t.encoder == e]
        rg = spearmanr(g.d_graph_cC_K, g.dknn).correlation
        rc = spearmanr(g.dcC_K, g.dknn).correlation
        win = abs(rg) > abs(rc)
        wins3 += int(win)
        print(f"  {e:>8}: d_graph_cC_K {rg:+.3f} vs dcC_K {rc:+.3f} {'WIN' if win else '—'}")
    print(f"  -> {'PASS — operator matching matters for kNN' if wins3 >= 2 else 'FAIL'}")

    # ---- SigLIP supplementary + exploratory pre-only panel ---------------------------------
    c2_path = OUT / "c2_siglip_score.csv"
    sig = cell[cell.encoder == "SigLIP"]
    if len(sig) and c2_path.exists():
        s = sig.groupby("dataset")[["d_purity", "dA_spec", "dA_rot"]].mean().reset_index()
        c2 = pd.read_csv(c2_path)[["dataset", "real_dknn"]]
        sj = s.merge(c2, on="dataset")
        print(f"\nSigLIP supplementary (n={len(sj)}, 2-method means): "
              f"rho(d_purity, dknn)={spearmanr(sj.d_purity, sj.real_dknn).correlation:+.3f}")
    g0 = nd10.groupby(["encoder", "dataset"], as_index=False)[["knn_purity_pre"]].mean()
    lev = t.groupby(["encoder", "dataset"], as_index=False).dknn.mean() \
        .merge(g0, on=["encoder", "dataset"])
    print("\nexploratory pre-only panel (decision-relevance; no criterion):")
    for e in MAIN_ENCS:
        g = lev[lev.encoder == e]
        print(f"  {e:>8}: rho(knn_purity_pre, dknn) = "
              f"{spearmanr(g.knn_purity_pre, g.dknn).correlation:+.3f}")


if __name__ == "__main__":
    main()
