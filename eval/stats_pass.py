"""Multiple-comparison-aware stats pass on geometry->Delta correlations.

Reads eval/outputs/correlations_15.csv (one row per (encoder, metric)).
Treats the Spearman p-values as families, one per Delta target
(dkNN, dLP, dFT), and applies Benjamini-Hochberg FDR correction
within each family. Writes eval/outputs/stats_pass.csv with the
original rho/p plus BH-adjusted q-values.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "outputs", "correlations_15.csv")
OUT_CSV = os.path.join(HERE, "outputs", "stats_pass.csv")

DELTAS = ["dknn", "dlp", "dft"]


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values. Uses scipy when available,
    else a hand-rolled implementation as a cross-check."""
    p = np.asarray(pvals, dtype=float)
    q_scipy = false_discovery_control(p, method="bh")

    # Hand-rolled BH for verification.
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_manual = np.empty(n)
    q_manual[order] = q

    assert np.allclose(q_scipy, q_manual, atol=1e-12), (
        "scipy BH and manual BH disagree", q_scipy, q_manual
    )
    return q_scipy


def main():
    df = pd.read_csv(IN_CSV)
    n_rows = len(df)

    # Per-Delta BH FDR across all (encoder, metric) rows.
    for d in DELTAS:
        pcol = f"p_{d}"
        qcol = f"q_{d}"
        df[qcol] = bh_fdr(df[pcol].values)

    total_tests = n_rows * len(DELTAS)

    # Assemble tidy output: original rho/p + adjusted q per Delta.
    out_cols = ["encoder", "metric", "n"]
    for d in DELTAS:
        out_cols += [f"rho_{d}", f"p_{d}", f"q_{d}"]
    out = df[out_cols].copy()
    out.to_csv(OUT_CSV, index=False)

    # ---- Reporting ----
    print(f"Rows (encoder,metric): {n_rows}")
    print(f"Families (Deltas): {len(DELTAS)} -> {DELTAS}")
    print(f"Total Spearman tests: {total_tests}")
    print()

    # Survivors at q < 0.05, per Delta.
    surv_records = []
    for d in DELTAS:
        qcol = f"q_{d}"
        rcol = f"rho_{d}"
        sub = df[df[qcol] < 0.05]
        for _, row in sub.iterrows():
            surv_records.append(
                (row["encoder"], row["metric"], d.upper(),
                 row[rcol], row[f"p_{d}"], row[qcol])
            )

    print(f"Survivors at BH-FDR q<0.05 (per-Delta families): {len(surv_records)}")
    print(f"{'encoder':<8} {'metric':<24} {'D':<5} {'rho':>7} {'p':>10} {'q':>10}")
    for enc, met, d, rho, p, q in sorted(surv_records, key=lambda x: (x[2], x[0], x[1])):
        print(f"{enc:<8} {met:<24} {d:<5} {rho:>7.3f} {p:>10.4g} {q:>10.4g}")

    # ---- Headline focus: sphere encoders x decision-rule metrics ----
    print()
    print("=" * 70)
    print("HEADLINE: DINOv3/CLIP x {neighbor_overlap_k50, uniformity_t2}")
    print("Decision-rule deltas: dkNN, dLP")
    print("=" * 70)
    enc_focus = ["DINOv3", "CLIP"]
    met_focus = ["neighbor_overlap_k50", "uniformity_t2"]
    d_focus = ["dknn", "dlp"]
    print(f"{'encoder':<8} {'metric':<22} {'D':<5} {'rho':>7} {'p':>10} {'q':>10} {'q<.05':>6}")
    for enc in enc_focus:
        for met in met_focus:
            r = df[(df.encoder == enc) & (df.metric == met)]
            if r.empty:
                continue
            r = r.iloc[0]
            for d in d_focus:
                rho = r[f"rho_{d}"]
                p = r[f"p_{d}"]
                q = r[f"q_{d}"]
                surv = "YES" if q < 0.05 else "no"
                print(f"{enc:<8} {met:<22} {d.upper():<5} {rho:>7.3f} "
                      f"{p:>10.4g} {q:>10.4g} {surv:>6}")

    print()
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
