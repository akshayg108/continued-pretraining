#!/usr/bin/env python
"""
spectral_metrics.py — shared spectral-geometry metrics for the new-direction experiments
(ND1-ND4; see papers/new_direction/NEW_DIRECTION.md §H and eval/new_direction/README.md).

Definitions are implemented VERBATIM from the primary papers (extracted from the PDFs in
papers/new_direction/ on 2026-07-10):

  rankme(X)      Garrido et al. 2023 (ICML), Eq. 1: p_k = sigma_k / ||sigma||_1 over the
                 singular values of the (uncentered) feature matrix; RankMe = exp(-sum p log p).
  alpha_req(X)   Agrawal et al. 2022 (NeurIPS): eigenvalues lambda_i of the empirical feature
                 COVARIANCE follow lambda_i ~ i^(-alpha); alpha = -slope of an OLS fit of
                 log lambda_i on log i. Fit protocol here: skip the first ALPHA_HEAD_SKIP
                 eigenvalues (head), drop the numerical-noise tail (< 1e-12 * lambda_1),
                 cap at ALPHA_MAX_IDX; emit the fit R^2 so bad fits are visible.
  coherence(X)   Tsitsulin et al. 2023, Def. 3.1 (standard incoherence, Candes-Recht):
                 for centered X with left singular vectors U (n x r),
                 mu = (n / r) * max_i ||U_i.||^2 (row leverage). r = numerical rank.
                 Also emits the r-at-99%-energy variant (convention robustness).
  vci(X, y)      Xu et al. 2023 (ICML), Def. 5.3: VCI = 1 - Tr[Sigma_T^+ Sigma_B] / rank(Sigma_B).
                 Paper states the balanced-class (1/K) form (their Eq. 2-3); we use
                 class-frequency weights so Sigma_T = Sigma_W + Sigma_B holds exactly for
                 imbalanced datasets (coincides with the paper when balanced). VCI = 0 at
                 exact neural collapse; higher = more within-class variability retained.

All metrics work for d > n (e.g. the DIET head output) via the Gram trick.
Self-test (CPU, synthetic): python eval/new_direction/spectral_metrics.py
"""
import numpy as np

ALPHA_HEAD_SKIP = 10     # eigenvalue indices 1..10 excluded from the power-law fit
ALPHA_MAX_IDX = 512      # fit at most up to this eigenvalue index
RANK_TOL = 1e-6          # numerical-rank threshold, relative to sigma_1


def _left_svd(X):
    """U (n x r) and singular values s of X without forming a d x d matrix when d > n."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d <= n:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        return U, s
    G = X @ X.T                                    # n x n Gram
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)
    return V[:, order], np.sqrt(w)


def rankme(X):
    """Effective rank of the RAW (uncentered) feature matrix (Garrido et al. 2023, Eq. 1)."""
    _, s = _left_svd(X)
    s = s[s > 0]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def alpha_req(X):
    """(alpha, r2) of the covariance-eigenspectrum power-law fit (Agrawal et al. 2022)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s = _left_svd(Xc)
    lam = (s ** 2) / max(len(X) - 1, 1)            # covariance eigenvalues, descending
    lam = lam[lam > lam[0] * 1e-12] if lam[0] > 0 else lam
    lo, hi = ALPHA_HEAD_SKIP, min(len(lam), ALPHA_MAX_IDX)
    if hi - lo < 30:
        return float("nan"), float("nan")
    idx = np.arange(lo + 1, hi + 1)                # 1-based eigenvalue index
    logx, logy = np.log(idx), np.log(lam[lo:hi])
    A = np.column_stack([logx, np.ones_like(logx)])
    coef, res, *_ = np.linalg.lstsq(A, logy, rcond=None)
    ss_tot = ((logy - logy.mean()) ** 2).sum()
    r2 = 1.0 - (res[0] / ss_tot if len(res) and ss_tot > 0 else np.nan)
    return float(-coef[0]), float(r2)


def coherence(X):
    """(mu_numrank, mu_99energy): Def. 3.1 leverage coherence of the CENTERED feature matrix."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s = _left_svd(Xc)
    n = len(Xc)
    if s[0] <= 0:
        return float("nan"), float("nan")

    def mu_at(r):
        lev = (U[:, :r] ** 2).sum(axis=1)          # ||U_r^T e_i||^2 row leverage
        return float(n / r * lev.max())

    r_num = int((s > s[0] * RANK_TOL).sum())
    energy = np.cumsum(s ** 2) / (s ** 2).sum()
    r_99 = int(np.searchsorted(energy, 0.99) + 1)
    return mu_at(max(r_num, 1)), mu_at(max(r_99, 1))


def class_scatter(X, y):
    """(Sigma_B, Sigma_T, mu_global, class_means, weights) with class-frequency weights."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    classes, counts = np.unique(y, return_counts=True)
    mu = X.mean(axis=0)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    w = counts / counts.sum()
    dm = means - mu
    sigma_b = (w[:, None] * dm).T @ dm
    Xc = X - mu
    sigma_t = Xc.T @ Xc / len(X)
    return sigma_b, sigma_t, mu, means, w


def vci(X, y, max_dim=4096):
    """(vci, rank_b) — Xu et al. 2023 Def. 5.3. NaN for d > max_dim (only used on backbones)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] > max_dim:
        return float("nan"), 0
    sigma_b, sigma_t, mu, means, w = class_scatter(X, y)
    lam_t, V = np.linalg.eigh(sigma_t)
    tol = max(lam_t.max(), 0) * 1e-10
    inv = np.where(lam_t > tol, 1.0 / np.maximum(lam_t, tol), 0.0)
    dm = means - mu                                # K x d
    proj = dm @ V                                  # class-mean deviations in Sigma_T eigenbasis
    trace = float((w[:, None] * proj * proj * inv[None, :]).sum())
    lam_b = np.linalg.eigvalsh(sigma_b)
    rank_b = int((lam_b > max(lam_b.max(), 0) * 1e-10).sum())
    if rank_b == 0:
        return float("nan"), 0
    return float(1.0 - trace / rank_b), rank_b


def spectral_row(X, y=None):
    """All ND metrics for one feature matrix, as a flat dict (the shared CSV vocabulary)."""
    a, a_r2 = alpha_req(X)
    mu, mu99 = coherence(X)
    row = {"rankme": rankme(X), "alpha": a, "alpha_r2": a_r2,
           "coherence_mu": mu, "coherence_mu99": mu99}
    if y is not None:
        v, rb = vci(X, y)
        row.update({"vci": v, "vci_rank_b": rb})
    return row


def _selftest():
    rng = np.random.RandomState(0)
    # isotropic gaussian: rankme ~ d, coherence small, alpha ~ 0
    X = rng.randn(2000, 64)
    r = rankme(X)
    assert 55 < r <= 64.01, f"rankme isotropic: {r}"
    # exact power-law spectrum: alpha recovered
    d, n, alpha_true = 256, 4000, 1.0
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha_true)
    Xp = rng.randn(n, d) * np.sqrt(lam)
    a, r2 = alpha_req(Xp)
    assert abs(a - alpha_true) < 0.15 and r2 > 0.95, f"alpha: {a}, r2={r2}"
    # exact neural collapse: VCI = 0
    means = rng.randn(5, 32) * 10
    y = np.repeat(np.arange(5), 100)
    Xc = means[y]
    v, rb = vci(Xc + rng.randn(*Xc.shape) * 1e-9, y)
    assert abs(v) < 1e-3 and rb == 4, f"vci collapse: {v}, rank_b={rb}"
    # no class structure: VCI ~ 1
    v2, _ = vci(rng.randn(1000, 32), rng.randint(0, 5, 1000))
    assert v2 > 0.9, f"vci random: {v2}"
    # coherence: a spiky embedding (one dominant row) is more coherent than gaussian
    Xs = rng.randn(1000, 32)
    Xs[0] *= 50
    mu_spiky, _ = coherence(Xs)
    mu_gauss, _ = coherence(rng.randn(1000, 32))
    assert mu_spiky > mu_gauss, f"coherence: spiky {mu_spiky} <= gauss {mu_gauss}"
    # gram-trick path (d > n) agrees with direct path
    Xw = rng.randn(200, 500)
    r_wide = rankme(Xw)
    U, s = _left_svd(Xw)
    s_direct = np.linalg.svd(Xw, compute_uv=False)
    assert np.allclose(s, s_direct, atol=1e-8), "gram-trick singular values mismatch"
    print(f"selftest OK  (rankme_iso={r:.1f}, alpha={a:.3f}, vci_collapse={v:.2e}, "
          f"vci_random={v2:.3f}, rankme_wide={r_wide:.1f})")


if __name__ == "__main__":
    _selftest()
