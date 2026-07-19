#!/usr/bin/env python
"""
nd9_verdict.py — ND9 (CPU adjudicator): is the SigLIP-2 alignment deficit a CAPTURE
deficit, a conditional-PLACEMENT deficit, or both — and does the readout-weighted
accessibility serve the LP level better than the placement share alone?

Pre-registered readouts (declared 2026-07-16, BEFORE nd9_capture.csv existed; rank-aware
rerun amendments declared 2026-07-16 before nd9_capture_rankaware.csv existed):
  ND9-R RANK ACCEPTANCE (rerun gate, Codex follow-up): 60 rows, 60 unique cells, no
        NaN; numerical_rank == min(n_samples, embed_dim) on EVERY cell (maximal numerical
        rank: 768 on the 56 non-degenerate cells, n_samples on breastmnist's 4).
        Any true rank deficiency -> capture keeps its pending tag AND the nd6-era cC
        definitions must be revisited before further use. Per-row capture diff vs the
        pre-mask nd9_capture.csv is reported (expected ~0 when full-rank).
  ND9-0 VALIDATION: cC_K_check reproduces nd6_alignment.cC_K on the 60 ViT-B cells
        (max |diff| < 0.01). Fails -> stop, protocol drifted.
  ND9-1 PRIMARY (deficit decomposition): sign test SigLIP vs CLIP on capture_cen,
        EXCLUDING degenerate cells (n_samples <= embed_dim, where the feature span is
        complete and capture == 1 identically for every encoder — the difference is float
        noise; breastmnist n=546 < 768 is the known case, review 2026-07-16). Deficit =
        SigLIP < CLIP on >= 12 of the remaining datasets (at 14 remaining: two-sided
        binomial p ~= 0.013). Interpretation table (declared):
          capture deficit YES  -> the ND6-2 alignment deficit has a CAPTURE component;
                                  C4 wording becomes "capture + placement deficit"
                                  (relative weight = exploratory readout below).
          capture deficit NO   -> the deficit is conditional-placement-only; current C4
                                  wording stands, sharpened ("capture is intact").
        The conditional-placement side is NOT retested here (frozen fact: caucC_log
        SigLIP < CLIP 15/15, ND6-2). Exploratory (no criterion): per-dataset gap
        correlation — which of {capture_cen, caucC_log deficit} tracks the per-dataset
        SigLIP-CLIP knn_pre gap better.
  ND9-2 (accessibility serves LP): within-encoder Spearman rho(acc_r1, lp_pre) >=
        rho(cC_K, lp_pre) on >= 3/4 encoders (theory: A(kappa) is the ridge/LP-matched
        functional; kappa* = mean(lam), i.e. column acc_r1, declared here). kNN columns
        are reported without criterion — per the operator view kNN belongs to the local
        graph (ND11), not to K.
  ND9-3 (robustness of the deficit to the centering convention): kta_cen sign test
        SigLIP < CLIP on >= 12/15.
n=15 per cell; sign/rank evidence; no FDR family (engineering round).

Inputs: eval/outputs/nd9_capture.csv (cluster), nd6_alignment.csv, and knn/lp levels via
nd1_verdict.knn_pre_levels + nd6_verdict's SigLIP LP extraction.
Run (local): python eval/new_direction/nd9_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
ENCS = ["DINOv3", "CLIP", "SigLIP", "MAE"]


def sign_test(a_minus_b):
    """Count of strictly negative (SigLIP-CLIP) differences + two-sided binomial p."""
    neg = int((a_minus_b < 0).sum())
    n = len(a_minus_b)
    p = binomtest(neg, n, 0.5).pvalue
    return neg, n, p


def main():
    path = OUT / "nd9_capture_rankaware.csv"
    if not path.exists():
        sys.exit(f"MISSING {path} — run the rank-aware ND9 pass first "
                 f"(run/slurm/new_direction/nd9_capture.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd9_capture_rankaware_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd9_capture_rankaware.csv', index=False)\"")
    nd9 = pd.read_csv(path)

    # ---- ND9-R rank acceptance (rerun gate) -------------------------------------------------
    assert len(nd9) == 60, f"expected 60 rows, got {len(nd9)}"
    assert not nd9.duplicated(["encoder", "dataset"]).any(), "duplicate cells"
    assert not nd9.isna().any().any(), "NaN cells in the rank-aware pass"
    expected = np.minimum(nd9.n_samples, nd9.embed_dim)
    deficient = nd9[nd9.numerical_rank != expected]
    print(f"ND9-R rank acceptance: rank == min(n, d) on {len(nd9) - len(deficient)}/60 "
          f"(s_min range {nd9.s_min.min():.3g}..{nd9.s_min.max():.3g})")
    if len(deficient):
        print(deficient[["encoder", "dataset", "n_samples", "embed_dim",
                         "numerical_rank"]].to_string(index=False))
        print("  -> TRUE RANK DEFICIENCY — STOP (hardened 2026-07-16, Codex round 3): "
              "capture keeps its pending tag, the ND9-1..3 verdicts below would be "
              "void, and the nd6-era cC definitions must be revisited first.")
        sys.exit(1)
    print("  -> PASS — maximal numerical rank min(n, d) everywhere; capture's pending tag is lifted")
    old_path = OUT / "nd9_capture.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)[["encoder", "dataset", "capture_cen"]] \
            .rename(columns={"capture_cen": "capture_cen_old"})
        d = nd9.merge(old, on=["encoder", "dataset"])
        d["diff"] = (d.capture_cen - d.capture_cen_old).abs()
        print(f"  per-row capture diff vs pre-mask pass: max {d['diff'].max():.2e}, "
              f"mean {d['diff'].mean():.2e}")
        big = d[d["diff"] > 1e-6]
        if len(big):
            print("  rows with |diff| > 1e-6:")
            print(big[["encoder", "dataset", "capture_cen_old", "capture_cen",
                       "diff"]].to_string(index=False))
        else:
            print("  all 60 rows within 1e-6 of the pre-mask pass")
    nd6_path = OUT / "nd6_alignment.csv"
    if not nd6_path.exists():
        sys.exit(f"MISSING {nd6_path} — the ND9-0 gate needs the ND6 baselines locally")
    nd6 = pd.read_csv(nd6_path)[["encoder", "dataset", "cC_K", "caucC_log"]]
    b = nd9.merge(nd6, on=["encoder", "dataset"])
    assert len(b) == 60, f"expected 60 ViT-B cells after the nd6 join, got {len(b)} — partial shards?"

    # ---- ND9-0 protocol gate ---------------------------------------------------------------
    diff = (b.cC_K_check - b.cC_K).abs()
    gate = diff.max() < 0.01
    print(f"ND9-0 protocol reproduction: max |cC_K_check - nd6.cC_K| = {diff.max():.4f} "
          f"({'PASS' if gate else 'FAIL — STOP, protocol drifted'})")
    if not gate:
        sys.exit(1)   # pre-registered: everything downstream is void on gate failure

    # ---- ND9-1 capture deficit -------------------------------------------------------------
    wide = b.pivot(index="dataset", columns="encoder",
                   values=["capture_cen", "kta_cen", "caucC_log"])
    assert wide.notna().all().all(), "NaN in the encoder pivot — incomplete cells"
    # degenerate cells: n <= d makes the span complete and capture identically 1 for
    # every encoder — the SigLIP-CLIP difference there is float noise (review 2026-07-16)
    degen = sorted(b.loc[b.n_samples <= b.embed_dim, "dataset"].unique())
    if degen:
        print(f"  (ND9-1 excludes degenerate n<=d datasets: {', '.join(degen)})")
    d_cap = (wide["capture_cen"]["SigLIP"] - wide["capture_cen"]["CLIP"]).drop(degen)
    neg, n, p = sign_test(d_cap)
    deficit = neg >= 12
    print(f"\nND9-1 capture: SigLIP < CLIP on {neg}/{n} datasets (binomial p={p:.4g}) -> "
          f"{'CAPTURE DEFICIT — C4 becomes capture+placement' if deficit else 'no capture deficit — placement-only, C4 stands sharpened'}")
    print("  per-dataset capture_cen (SigLIP - CLIP):")
    for ds, v in d_cap.sort_values().items():
        print(f"    {ds:>14}: {v:+.4f}")

    # exploratory: which factor tracks the per-dataset knn_pre gap (no criterion)
    from nd1_verdict import knn_pre_levels
    lev = knn_pre_levels().pivot(index="dataset", columns="encoder", values="knn_pre")
    gap = (lev["SigLIP"] - lev["CLIP"]).reindex(d_cap.index)
    d_pl = (wide["caucC_log"]["SigLIP"] - wide["caucC_log"]["CLIP"]).reindex(d_cap.index)
    m = gap.notna()
    print(f"  exploratory gap tracking (n={int(m.sum())}): "
          f"rho(d_capture, knn_gap)={spearmanr(d_cap[m], gap[m]).correlation:+.3f}  "
          f"rho(d_caucC, knn_gap)={spearmanr(d_pl[m], gap[m]).correlation:+.3f}")

    # ---- ND9-2 accessibility serves LP -----------------------------------------------------
    import openpyxl  # noqa: F401  (nd6_verdict's SigLIP LP extraction needs it)
    from nd6_verdict import lp_pre_levels
    lp = lp_pre_levels()
    bl = b.merge(lp, on=["encoder", "dataset"])
    print("\nND9-2 within-encoder rho(feature, lp_pre) [criterion: acc_r1 >= cC_K on >=3/4]:")
    wins = 0
    for e in ENCS:
        g = bl[bl.encoder == e]
        r_acc = spearmanr(g.acc_r1, g.lp_pre).correlation
        r_cck = spearmanr(g.cC_K, g.lp_pre).correlation
        win = r_acc >= r_cck
        wins += int(win)
        print(f"  {e:>8}: acc_r1 {r_acc:+.3f} vs cC_K {r_cck:+.3f} "
              f"{'WIN' if win else 'loss'}  (capture {spearmanr(g.capture_cen, g.lp_pre).correlation:+.3f}, "
              f"kta {spearmanr(g.kta_cen, g.lp_pre).correlation:+.3f})")
    print(f"  -> acc_r1 >= cC_K on {wins}/4: {'PASS' if wins >= 3 else 'FAIL'}")
    print("  (kNN columns, reported without criterion — kNN belongs to the local graph:)")
    knn = knn_pre_levels()
    bk = b.merge(knn, on=["encoder", "dataset"])
    for e in ENCS:
        g = bk[bk.encoder == e]
        print(f"  {e:>8}: rho(acc_r1, knn_pre)={spearmanr(g.acc_r1, g.knn_pre).correlation:+.3f} "
              f"rho(cC_K, knn_pre)={spearmanr(g.cC_K, g.knn_pre).correlation:+.3f}")

    # ---- ND9-3 KTA robustness --------------------------------------------------------------
    d_kta = wide["kta_cen"]["SigLIP"] - wide["kta_cen"]["CLIP"]
    neg3, n3, p3 = sign_test(d_kta)
    print(f"\nND9-3 kta_cen: SigLIP < CLIP on {neg3}/{n3} (binomial p={p3:.4g}) -> "
          f"{'PASS (deficit robust to centering)' if neg3 >= 12 else 'FAIL (convention-sensitive — flag)'}")


if __name__ == "__main__":
    main()
