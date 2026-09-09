#!/usr/bin/env python
"""
repulsion_share.py — one extra label-free candidate, derived (not published) from the
Wang & Liu (2021) gradient of the contrastive / Gaussian-potential objective:
  rho_k(beta) = mean_i  sum_{j in kNN_k(i)} P_ij ,  P_ij = softmax_j(beta * cos_ij), j != i
= the share of an anchor's initial repulsion that lands on its own k nearest neighbours.
High share -> the objective's first updates re-sort local neighbourhoods (neighbour-targeted
repulsion); low share -> near-uniform repulsion = dilation about the centroid.
Also the mean normalised entropy of P_i. beta in {2, 4, 10} (4 = Gaussian potential t=2;
2 = this repo's SimCLR temperature 0.5). Same preprocessing as feature_metrics.py.
Runs the candidate through the screen bar (screen.py) without touching the earlier outputs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feature_metrics as FM                    # noqa: E402
import screen as S                              # noqa: E402

K = 20
BETAS = (2, 4, 10)


def repulsion_metrics(f):
    g = f @ f.T
    n = len(f)
    np.fill_diagonal(g, -np.inf)                # exclude self
    nn_idx = np.argpartition(-g, K, axis=1)[:, :K]
    out = {}
    for beta in BETAS:
        logits = beta * g
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        share = np.take_along_axis(p, nn_idx, axis=1).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1) / np.log(n - 1)
        out[f"rep_share_k{K}_b{beta}"] = float(share.mean())
        out[f"rep_entropy_b{beta}"] = float(ent.mean())
    return out


def main():
    rows = []
    for enc in FM.ENCODERS:
        for ds in FM.DATASETS:
            z = np.load(FM.FEAT_DIR / f"{enc}__{ds}.npz")
            X = z["bank_X"].astype(np.float64)
            if len(X) > FM.MAX_N:
                sel = np.sort(np.random.RandomState(FM.SEED).choice(len(X), FM.MAX_N, replace=False))
                X = X[sel]
            row = {"encoder": enc, "dataset": ds}
            row.update(repulsion_metrics(normalize(X)))
            rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(HERE / "outputs/repulsion_share.csv", index=False)
    w = S.wide_table().merge(m, on=["encoder", "dataset"], how="left")
    cands = [c for c in m.columns if c.startswith("rep_")]
    res = S.screen_all(w, [(c, "features", False) for c in cands])
    if "available_on_siglip" not in res.columns:
        res["available_on_siglip"] = True
    res.to_csv(HERE / "outputs/repulsion_share_screen.csv", index=False)
    cols = ["pooled_rho", "within_OOD_rho", "within_FG_rho", "type_partial_rho",
            "partial_vs_unif_rho", "rho_with_unif"]
    for c in cands:
        r = res[res.candidate == c].set_index("encoder")
        print(f"\n== {c}")
        print(r[cols].round(3).to_string())
        v = S.evaluate_bar(r)
        print("  bar:", {k: v[k] for k in ["a_partial_both", "b_siglip_sign", "c_not_type",
                                            "d_not_redundant", "passes_bar"]})


if __name__ == "__main__":
    main()
