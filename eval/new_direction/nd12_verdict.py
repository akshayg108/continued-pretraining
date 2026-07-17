#!/usr/bin/env python
"""
nd12_verdict.py — ND12 (CPU adjudicator). Definition, split rule and criteria are
FROZEN in eval/new_direction/ND12_PREREG.md (2026-07-16, before any ND12 data existed);
this file implements them one-to-one. Summary:

  G1     nd10 consistency gate (legacy graph_cC_K_pre: rho > 0.99, max diff < 0.02).
  G2     evaluator-reproduction gate: knn_f1_hat_pre vs results.xlsx knn_pre —
         pooled Spearman > 0.95, per-encoder > 0.90 (levels reproduce up to
         bank/query subsampling; mean |diff| reported without threshold).
  ND12-1 PRIMARY: leave-one-dataset-out incremental value. 15 dataset folds; cells at
         (method, encoder, dataset); LogisticRegression on z-scored features;
         target sign(dknn > 0). GLOBAL = {d_rankme, d_cC_K, dA_spec, dA_rot}
         (round-3 fix: includes the strongest tested globals; dA_total = spec+rot
         not duplicated); GLOBAL+GRAPH adds d graph_cC_K_m20b; GRAPH-ONLY alone.
         PASS iff pooled LODO accuracy(GLOBAL+GRAPH) - accuracy(GLOBAL) >= +0.02
         AND dataset-block bootstrap (2000 draws, seed 0) P(diff > 0) >= 0.90
         (round-3 fix: bootstrap promoted to conjunct). Population asserts:
         exactly 135 cells = 15 datasets x 9. vote-margin NEVER a feature (kinship).
  ND12-2 descriptive bridge: d vote_margin vs dknn (reported, never evidence).
  ND12-3 sensitivity: ND12-1 re-run with m20wb / legacy k10 in place of m20b
         (reported, never criterion-bearing; no best-variant picking).

Inputs: eval/outputs/nd12_operator.csv, nd10_operator.csv (G1), nd7_placement.csv +
nd1_precp_spectral.csv (d_rankme), results.xlsx.
Run (local): python eval/new_direction/nd12_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long
from nd1_verdict import knn_pre_levels

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
MAIN_ENCS = ["DINOv3", "CLIP", "MAE"]
MMAP = {"LeJEPA": "LeJEPA-CP", "SimCLR": "SimCLR-CP", "DIET": "DIET-CP"}
KEY = ["method", "encoder", "dataset", "size", "seed"]
GLOBAL_FEATS = ["d_rankme", "d_cC_K", "dA_spec", "dA_rot"]   # round-3 fix: include the
# strongest tested global quantities (dA_spec ~80.7% sign agreement); dA_total is
# their sum and is not duplicated. PASS wording (round-4): "beyond the
# pre-registered four-feature linear global baseline".
GRAPH_FEAT = "d_graph_m20b"                      # frozen evidence variant (PREREG I2)


def lodo_accuracy(t, feats):
    """Pooled leave-one-dataset-out sign accuracy for one feature set."""
    hits = []
    for ds in sorted(t.dataset.unique()):
        tr, te = t[t.dataset != ds], t[t.dataset == ds]
        mu, sd = tr[feats].mean(), tr[feats].std(ddof=0).replace(0, 1.0)
        clf = LogisticRegression(max_iter=1000).fit(
            ((tr[feats] - mu) / sd).values, (tr.dknn > 0).astype(int).values)
        pred = clf.predict(((te[feats] - mu) / sd).values)
        hits.append(pd.Series(pred == (te.dknn > 0).values, index=te.index))
    return pd.concat(hits).sort_index()


def main():
    path = OUT / "nd12_operator.csv"
    if not path.exists():
        sys.exit(f"MISSING {path} — run the ND12 GPU pass first "
                 f"(run/slurm/new_direction/nd12_operator.sh), concat shards:\n"
                 f"  python -c \"import glob,pandas as pd; pd.concat([pd.read_csv(f) for f in "
                 f"sorted(glob.glob('eval/outputs/nd12_operator_shards/*.csv'))])"
                 f".to_csv('eval/outputs/nd12_operator.csv', index=False)\"")
    nd12 = pd.read_csv(path)

    # ---- G0 key census (round-4 fix): raw keys must be one-to-one with the ND10 pass ----
    old = pd.read_csv(OUT / "nd10_operator.csv")[KEY + ["graph_cC_K_pre"]]
    assert not nd12.duplicated(KEY).any(), "duplicate cell keys in nd12_operator.csv"
    assert not old.duplicated(KEY).any(), "duplicate cell keys in nd10_operator.csv"
    k12 = set(map(tuple, nd12[KEY].astype(str).values))
    k10 = set(map(tuple, old[KEY].astype(str).values))
    assert k12 == k10, (f"raw key sets differ: nd12-only {len(k12 - k10)}, "
                        f"nd10-only {len(k10 - k12)} — seed coverage drifted; the "
                        f"135-cell check would silently average incomplete seeds")
    print(f"G0 key census: {len(k12)} raw keys, one-to-one with the ND10 pass (PASS)")

    # ---- G1 nd10 consistency ----------------------------------------------------------------
    j = nd12.merge(old, on=KEY, suffixes=("", "_nd10"), validate="one_to_one")
    rho = spearmanr(j.graph_cC_K_pre, j.graph_cC_K_pre_nd10).correlation
    md = float((j.graph_cC_K_pre - j.graph_cC_K_pre_nd10).abs().max())
    ok1 = rho > 0.99 and md < 0.02
    print(f"G1 nd10 consistency: rho={rho:.4f}, max diff={md:.4f} "
          f"({'PASS' if ok1 else 'FAIL — STOP'}) [join {len(j)}]")
    if not ok1:
        sys.exit(1)

    # ---- G2 evaluator reproduction ------------------------------------------------------------
    # Declared unit (round-4 clarification): the proxy's PRE side is method/seed-
    # invariant, so G2 is evaluated on the 45 (encoder, dataset) pre-CP LEVELS
    # (pooled n=45; per-encoder n=15), not on method-cells.
    pre = (nd12[nd12.encoder.isin(MAIN_ENCS)]
           .groupby(["encoder", "dataset"], as_index=False).knn_f1_hat_pre.mean()
           .merge(knn_pre_levels(), on=["encoder", "dataset"]))
    assert len(pre) == 45, f"expected 45 (encoder, dataset) levels, got {len(pre)}"
    rp = spearmanr(pre.knn_f1_hat_pre, pre.knn_pre).correlation
    print(f"\nG2 evaluator reproduction: pooled rho(knn_f1_hat_pre, knn_pre)={rp:+.3f} "
          f"(mean |diff|={float((pre.knn_f1_hat_pre - pre.knn_pre).abs().mean()):.4f}, "
          f"no threshold on the level)")
    ok2 = rp > 0.95
    for e in MAIN_ENCS:
        g = pre[pre.encoder == e]
        r = spearmanr(g.knn_f1_hat_pre, g.knn_pre).correlation
        ok2 &= r > 0.90
        print(f"  {e:>8}: rho={r:+.3f}")
    print(f"  -> {'PASS — the proxy is protocol-faithful (standardized bank)' if ok2 else 'FAIL — STOP, not evaluator-faithful; nothing downstream is interpretable'}")
    if not ok2:
        sys.exit(1)

    # ---- cell table -----------------------------------------------------------------------------
    for v, col in [("", "d_graph_k10"), ("_m20b", "d_graph_m20b"), ("_m20wb", "d_graph_m20wb")]:
        nd12[col] = nd12[f"graph_cC_K{v}_post"] - nd12[f"graph_cC_K{v}_pre"]
    nd12["d_vote_margin"] = nd12.vote_margin_post - nd12.vote_margin_pre
    nd12["d_cC_K"] = nd12.dcC_K
    cell = nd12.groupby(["method", "encoder", "dataset"], as_index=False)[
        ["d_graph_k10", "d_graph_m20b", "d_graph_m20wb", "d_vote_margin", "d_cC_K",
         "dA_spec", "dA_rot"]].mean()
    post = pd.read_csv(OUT / "nd7_placement.csv").groupby(
        ["method", "encoder", "dataset"], as_index=False).rankme.mean() \
        .rename(columns={"rankme": "rankme_post"})
    prer = pd.read_csv(OUT / "nd1_precp_spectral.csv")[["encoder", "dataset", "rankme"]] \
        .rename(columns={"rankme": "rankme_pre"})
    cell = cell.merge(post, on=["method", "encoder", "dataset"], how="left") \
               .merge(prer, on=["encoder", "dataset"], how="left")
    cell["d_rankme"] = cell.rankme_post - cell.rankme_pre
    df = load_long()
    beh = (df[df.is_max & df.Backbone.isin(MAIN_ENCS)]
           .groupby(["Backbone", "Method", "dataset_key"]).dknn.mean().reset_index()
           .rename(columns={"Backbone": "encoder", "dataset_key": "dataset"}))
    cell["Method"] = cell.method.map(MMAP)
    t = cell[cell.encoder.isin(MAIN_ENCS)].merge(beh, on=["encoder", "Method", "dataset"])
    assert t.d_rankme.notna().all(), "NaN d_rankme — nd7/nd1 coverage drifted"
    t = t.reset_index(drop=True)
    # round-3 fix: hard population gate — LODO folds must be complete dataset blocks
    assert len(t) == 135, f"expected exactly 135 main-grid cells, got {len(t)}"
    per_ds = t.groupby("dataset").size()
    assert len(per_ds) == 15 and (per_ds == 9).all(), \
        f"expected 15 datasets x 9 cells, got:\n{per_ds.to_string()}"
    print(f"\nmain-grid cells: {len(t)} (15 datasets x 9 cells verified)")

    # ---- ND12-1 primary: LODO incremental value ------------------------------------------------
    acc = {}
    hitmaps = {}
    for name, feats in [("GLOBAL", GLOBAL_FEATS),
                        ("GLOBAL+GRAPH", GLOBAL_FEATS + [GRAPH_FEAT]),
                        ("GRAPH-ONLY", [GRAPH_FEAT])]:
        hitmaps[name] = lodo_accuracy(t, feats)
        acc[name] = float(hitmaps[name].mean())
        print(f"  LODO accuracy {name:>13}: {acc[name]:.1%}")
    diff = acc["GLOBAL+GRAPH"] - acc["GLOBAL"]
    wins = 0
    for ds in sorted(t.dataset.unique()):
        m = t.dataset == ds
        wins += int(hitmaps["GLOBAL+GRAPH"][m.values].mean() > hitmaps["GLOBAL"][m.values].mean())
    rng = np.random.RandomState(0)
    datasets = sorted(t.dataset.unique())
    boot = 0
    for _ in range(2000):
        pick = rng.choice(datasets, len(datasets), replace=True)
        idx = np.concatenate([np.flatnonzero((t.dataset == d).values) for d in pick])
        boot += int(hitmaps["GLOBAL+GRAPH"].values[idx].mean()
                    > hitmaps["GLOBAL"].values[idx].mean())
    pboot = boot / 2000
    ok = diff >= 0.02 and pboot >= 0.90            # round-3 fix: conjunctive criterion
    verdict = ("PASS — the graph operator carries held-out sign information beyond "
               "the pre-registered four-feature linear global baseline" if ok else
               "FAIL — only claimable: stronger marginal correlation than feature placement")
    print(f"ND12-1 PRIMARY: diff = {diff:+.1%}, bootstrap P(diff > 0) = {pboot:.3f} "
          f"[need >= +2.0pp AND >= 0.90] -> {verdict}")
    print(f"  fold wins (GLOBAL+GRAPH > GLOBAL): {wins}/15")

    # ---- ND12-2 descriptive bridge (never evidence) ---------------------------------------------
    nz = t[t.dknn.abs() > 1e-6]
    print(f"\nND12-2 bridge (kin to outcome — descriptive only): "
          f"sign agreement d_vote_margin {float((np.sign(nz.d_vote_margin) == np.sign(nz.dknn)).mean()):.1%}, "
          f"pooled rho {spearmanr(nz.d_vote_margin, nz.dknn).correlation:+.3f}")

    # ---- ND12-3 sensitivity (reported, no criteria, no picking) ---------------------------------
    print("\nND12-3 sensitivity (LODO diff vs GLOBAL, reported only):")
    for gf in ["d_graph_m20wb", "d_graph_k10"]:
        a = float(lodo_accuracy(t, GLOBAL_FEATS + [gf]).mean())
        print(f"  GLOBAL+{gf:>14}: {a:.1%} (diff {a - acc['GLOBAL']:+.1%})")


if __name__ == "__main__":
    main()
