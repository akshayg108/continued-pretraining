#!/usr/bin/env python
"""
nd6_alignment.py — ND6 (GPU pass): task-model alignment C(rho), the omniscient LP risk
estimate, and hubness metrics on the PRE-CP frozen features.

Theory sources (papers/new_direction/, round 2 — formulas extracted verbatim 2026-07-11):
  C(rho)     Canatar/Bordelon/Pehlevan 2021 Eq. 6: cumulative fraction of target power in
             the top-rho kernel eigenmodes ("task-model alignment").
  omni risk  Wei/Hu/Steinhardt 2022 Eq. 1 + Eq. 4: R_omni = (dkappa/dlambda) * kappa^2 *
             sum_i lambda_i (beta^T v_i)^2 / (kappa+lambda_i)^2, with kappa(lambda, N) the
             unique positive solution of 1 = lambda/kappa + (1/N) sum_i lambda_i/(kappa+lambda_i).
             beta proxy = full-sample least-squares fit, under which
             lambda_i (beta^T v_i)^2 = ||u_i^T Y||^2 / n  (u_i = left singular vectors of X)
             — the same per-mode label powers C(rho) uses; one SVD serves both.
  hubness    Radovanovic 2010: N_k skewness (k-occurrence right-skew), bad-occurrence
             fraction (label-disagreeing share of kNN inclusions — the CAV-linked factor),
             and centrality coupling rho(N_k, ||x - mean||^2) (hubs near the data mean).

Protocol identical to nd1_precp_spectral.py: timm pretrained encoders, <=5000 stratified
samples, eval transform, per-encoder pool. Adjudication in nd6_verdict.py (local; joins
knn_pre/lp_pre levels + the nd1 spectral baselines).

Self-test (CPU, validates the omniscient-risk implementation against Monte-Carlo ridge
regression on synthetic Gaussians):  python eval/new_direction/nd6_alignment.py --selftest

Cluster (one array task per dataset):
  python eval/new_direction/nd6_alignment.py --datasets <ds> \
      --download-dir <raw> --processed-dir <arrow> \
      --output eval/outputs/nd6_alignment_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                    # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))   # eval/utils

ROOT = Path(__file__).resolve().parent.parent.parent
C_RHOS = [10, 50, 100]
OMNI_N = [500, 1000]          # training-set sizes the risk is predicted at
OMNI_RIDGE = 1e-4             # lambda = OMNI_RIDGE * mean(eigenvalues); near-ridgeless
FIELDS = (["encoder", "dataset", "n_samples", "embed_dim", "n_classes"]
          + [f"C{r}" for r in C_RHOS] + ["C_K", "aucC_log"]
          + [f"cC{r}" for r in C_RHOS] + ["cC_K", "caucC_log"]
          + [f"omni_risk_n{n}" for n in OMNI_N]
          + ["skew_n10", "skew_n20", "bad_frac_n10", "hub_centrality_rho"])


# ---------------------------------------------------------------------------------------
# Task-model alignment + omniscient risk (shared SVD)
# ---------------------------------------------------------------------------------------
def mode_label_powers(X, y):
    """Per-eigenmode label powers p_i = ||u_i^T Y||^2 (raw and class-centered Y), plus the
    second-moment eigenvalues lambda_i = s_i^2 / n. X is used UNCENTERED (second moment,
    matching Wei Eq. 1 and the uncentered kernel-PCA of Canatar's Methods)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    lam = (s ** 2) / n
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)      # one-hot (n x K)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    p_raw = ((U.T @ Y) ** 2).sum(axis=1)
    p_cen = ((U.T @ Yc) ** 2).sum(axis=1)
    return lam, p_raw, p_cen, len(classes)


def c_rho_summary(p, n_classes, prefix=""):
    """C(rho) at fixed rho values + log-spaced AUC (Canatar Eq. 6 cumulative alignment)."""
    C = np.cumsum(p) / max(p.sum(), 1e-30)
    out = {}
    for r in C_RHOS:
        out[f"{prefix}C{r}"] = float(C[min(r, len(C)) - 1])
    out[f"{prefix}C_K"] = float(C[min(n_classes, len(C)) - 1])
    grid = np.unique(np.logspace(0, np.log10(len(C)), 50).astype(int)) - 1
    out[f"{prefix}aucC_log"] = float(C[grid].mean())
    return out


def _solve_kappa(lam, n, ridge):
    """kappa(lambda, N): unique positive solution of 1 = ridge/kappa + (1/N) sum lam/(kappa+lam)
    (Wei Eq. 4), by bisection on a monotone-decreasing RHS."""
    def rhs(k):
        return ridge / k + (lam / (k + lam)).sum() / n
    lo, hi = 1e-15, ridge + lam.sum() / n + 1e-9   # rhs(hi) < 1 guaranteed
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rhs(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def omni_risk(lam, lam_beta2, n, ridge):
    """Wei Eq. 1 with the extracted dkappa/dlambda. lam_beta2[i] = lambda_i (beta^T v_i)^2."""
    kappa = _solve_kappa(lam, n, ridge)
    s2 = (lam / (kappa + lam) ** 2).sum()
    dkappa = (1.0 / kappa) / (ridge / kappa ** 2 + s2 / n)      # implicit differentiation
    return float(dkappa * kappa ** 2 * (lam_beta2 / (kappa + lam) ** 2).sum())


def omni_risk_rows(lam, p_raw, n_full):
    """Risk predictions at the OMNI_N training sizes; beta proxy = full-sample LS fit, so
    lambda_i (beta^T v_i)^2 = p_raw_i / n_full."""
    lam_beta2 = p_raw / n_full
    ridge = OMNI_RIDGE * lam.mean()
    return {f"omni_risk_n{n}": omni_risk(lam, lam_beta2, n, ridge) for n in OMNI_N}


# ---------------------------------------------------------------------------------------
# Hubness (Radovanovic 2010)
# ---------------------------------------------------------------------------------------
def hubness_metrics(X, y, k_max=20):
    from scipy.stats import skew, spearmanr
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    Xn = normalize(np.asarray(X, dtype=np.float64))              # cosine, matching kNN eval
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn)
    idx = idx[:, 1:]                                             # drop self
    n = len(Xn)
    n10 = np.bincount(idx[:, :10].ravel(), minlength=n)
    n20 = np.bincount(idx.ravel(), minlength=n)
    bad10 = float((y[idx[:, :10]] != y[:, None]).mean())
    sq_dist_mean = ((Xn - Xn.mean(axis=0)) ** 2).sum(axis=1)
    rho_cent = spearmanr(n10, sq_dist_mean).correlation
    return {"skew_n10": float(skew(n10)), "skew_n20": float(skew(n20)),
            "bad_frac_n10": bad10, "hub_centrality_rho": float(rho_cent)}


# ---------------------------------------------------------------------------------------
def _selftest():
    """Validate omni_risk end-to-end against Monte-Carlo ridge regression: synthetic
    Gaussian features with a power-law second moment and a known beta."""
    rng = np.random.RandomState(0)
    d, n_pop = 200, 8000
    lam_true = np.arange(1, d + 1, dtype=float) ** -1.5
    beta = rng.randn(d) * np.sqrt(lam_true)         # target with power in leading modes
    beta /= np.linalg.norm(beta)

    def sample(n):
        return rng.randn(n, d) * np.sqrt(lam_true)

    Xpop = sample(n_pop)
    ypop = Xpop @ beta
    lam, p_raw, _, _ = mode_label_powers(Xpop, np.zeros(n_pop, dtype=int))
    # regression target: replace one-hot powers with powers of the continuous target
    U, s, _ = np.linalg.svd(Xpop, full_matrices=False)
    p_y = (U.T @ ypop) ** 2
    lam_beta2 = p_y / n_pop

    for n_train in (100, 400):
        ridge = OMNI_RIDGE * lam.mean()
        pred = omni_risk(lam, lam_beta2, n_train, ridge)
        # Monte-Carlo ridge regression at the SAME effective ridge (lambda*n convention:
        # Wei's objective is (1/N)||y-Xb||^2 + lambda||b||^2  =>  sklearn alpha = N*lambda)
        from sklearn.linear_model import Ridge
        risks = []
        for _ in range(30):
            Xtr, Xte = sample(n_train), sample(4000)
            m = Ridge(alpha=n_train * ridge, fit_intercept=False).fit(Xtr, Xtr @ beta)
            risks.append(np.mean((Xte @ m.coef_ - Xte @ beta) ** 2))
        mc = float(np.mean(risks))
        rel = abs(pred - mc) / mc
        print(f"  n={n_train}: omni={pred:.5f}  MC={mc:.5f}  rel.err={rel:.2%}")
        assert rel < 0.15, f"omniscient risk mismatch at n={n_train}: {rel:.2%}"
    # C(rho) sanity: labels aligned with the top eigenvector give high C10
    Xa = sample(3000)
    ya = (Xa[:, 0] > 0).astype(int)                 # mode-1 dominated labels
    lam_a, pr, pc, K = mode_label_powers(Xa, ya)
    ca = c_rho_summary(pc, K, "c")
    yr = rng.randint(0, 2, 3000)                    # random labels
    _, _, pc_r, _ = mode_label_powers(Xa, yr)
    cr = c_rho_summary(pc_r, 2, "c")
    assert ca["cC10"] > cr["cC10"] + 0.1, f"C10 aligned {ca['cC10']:.3f} <= random {cr['cC10']:.3f}"
    print(f"  C10 aligned={ca['cC10']:.3f} > random={cr['cC10']:.3f}  OK")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--output", type=str,
                    default=str(ROOT / "eval/outputs/nd6_alignment.csv"))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import timm
    import torch
    from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                                  extract_features)
    encoders = args.encoders or list(ENCODERS.keys())
    datasets = args.datasets or TARGET_DATASETS
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["encoder"], r["dataset"]) for r in csv.DictReader(f)}
        print(f"Resume: {len(done)} rows present")

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for enc in encoders:
            todo = [d for d in datasets if (enc, d) not in done]
            if not todo:
                continue
            cfg = ENCODERS[enc]
            print(f"\n===== {enc} ({cfg['timm_id']}) — {len(todo)} datasets")
            model = timm.create_model(cfg["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            for ds in todo:
                loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
                feat, labels = extract_features(model, loader, device, cfg["pool"])
                lam, p_raw, p_cen, n_cls = mode_label_powers(feat, labels)
                row = {"encoder": enc, "dataset": ds, "n_samples": len(feat),
                       "embed_dim": feat.shape[1], "n_classes": n_cls}
                row.update(c_rho_summary(p_raw, n_cls))
                row.update(c_rho_summary(p_cen, n_cls, "c"))
                row.update(omni_risk_rows(lam, p_raw, len(feat)))
                row.update(hubness_metrics(feat, labels))
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                print(f"  {ds:>14}: cC100={row['cC100']:.3f} aucC={row['caucC_log']:.3f} "
                      f"omni_n1000={row['omni_risk_n1000']:.4f} skew10={row['skew_n10']:.2f} "
                      f"bad10={row['bad_frac_n10']:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nNext (local): python eval/new_direction/nd6_verdict.py")


if __name__ == "__main__":
    main()
