#!/usr/bin/env python3
"""int1_surgery.py — counterfactual surgery on frozen features (INT1_PREREG.md).

Every surgery is a fixed FEATURE-SPACE linear map M (d x d), fitted from the bank
features' SVD only (no labels), applied identically to bank and query rows (X @ M).
With bank X = U S V^T:
  rotation  M = Q (Haar orthogonal)                — negative control, exact isometry
  power     M = V diag(s^(alpha-1)) V^T            — bank becomes U S^alpha V^T:
             spectrum reshaped, mode identity/order preserved (alpha > 0)
  demote    M = V diag(s'/s) V^T with s' = s after swapping the top `block` values
             with the block at `depth`               — sigma multiset preserved,
             placement (which directions carry large scale) changed
  shuffle   same with a full random permutation of s (seed)
  combo     demote(block, depth) then power(alpha) from ONE bank SVD — the factorial
             cell for the spectrum x placement interaction (v1.4): bank spectrum
             multiset equals pure power's, placement equals the demoted assignment

v1.4 (Codex round-2): non-rotation maps act as the IDENTITY on the orthogonal
complement of the bank row space (M += I - V^T V). For n >= d banks V^T V = I and
nothing changes; for n < d banks (breastmnist, 546 x 768) the query component
outside the row space now passes through instead of being annihilated, so
identity-vs-surgery deltas isolate the in-span effect. query_oos_frac stays as a
reported diagnostic, no longer an exclusion rule.

Small-metric helpers (rankme_raw / capture_of / cC_K_of) reuse the audited
implementations in spectral_metrics / nd9_task_operator.

Tests (TDD, written first): eval/new_direction/test_int1_surgery.py
"""
import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))

from spectral_metrics import rankme as _rankme                     # noqa: E402
from nd9_task_operator import _svd_powers                          # noqa: E402


def rankme_raw(X):
    return float(_rankme(np.asarray(X, dtype=np.float64)))


def capture_of(X, y):
    _, _, _, p_cen, _, y2c, _ = _svd_powers(X, y)
    return float(p_cen.sum() / y2c)


def cC_K_of(X, y):
    _, _, _, p_cen, _, _, n_cls = _svd_powers(X, y)
    C = np.cumsum(p_cen) / max(p_cen.sum(), 1e-30)
    return float(C[min(n_cls, len(C)) - 1])


def _bank_svd(X):
    X = np.asarray(X, dtype=np.float64)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return s, Vt


def _demote_order(s, block, depth):
    order = np.arange(len(s))
    b = min(block, len(s))
    dep = min(int(depth), len(s) - b)
    order[:b], order[dep:dep + b] = order[dep:dep + b].copy(), order[:b].copy()
    return order


def fit_surgery(bank_X, kind, alpha=None, seed=None, block=16, depth=None,
                s_target=None):
    """Return the fixed feature-space map M (d x d) for one pre-registered surgery."""
    bank_X = np.asarray(bank_X, dtype=np.float64)
    d = bank_X.shape[1]
    if kind == "rotation":
        g = np.random.RandomState(seed).randn(d, d)
        Q, _ = np.linalg.qr(g)
        return Q
    s, Vt = _bank_svd(bank_X)
    safe = np.maximum(s, 1e-12)
    if kind == "power":
        scale = safe ** (float(alpha) - 1.0)
    elif kind == "demote":
        scale = np.maximum(s[_demote_order(s, block, depth)], 1e-12) / safe
    elif kind == "shuffle":
        perm = np.random.RandomState(seed).permutation(len(s))
        scale = np.maximum(s[perm], 1e-12) / safe
    elif kind == "combo":
        s_dem = np.maximum(s[_demote_order(s, block, depth)], 1e-12)
        scale = (s_dem ** float(alpha)) / safe
    elif kind == "transplant":
        # INT2 A5: replace singular values IN RANK ORDER by the target spectrum
        # (directions unchanged); ranks beyond the target length keep their own.
        # Numerically-null directions are left untouched (round-6: a rank-
        # deficient post bank must not have its null space amplified by t/eps)
        t = np.asarray(s_target, dtype=np.float64)
        tol = s.max() * max(bank_X.shape) * np.finfo(np.float64).eps
        m = min(len(t), len(s))
        s_new = s.copy()
        s_new[:m] = np.maximum(t[:m], 1e-12)
        scale = np.where(s > tol, s_new / safe, 1.0)
    else:
        raise ValueError(f"unknown surgery kind: {kind}")
    M = Vt.T @ np.diag(scale) @ Vt
    # v1.4: identity on the complement of the bank row space (no-op when n >= d)
    return M + (np.eye(d) - Vt.T @ Vt)


def apply_surgery(X, M):
    return np.asarray(X, dtype=np.float64) @ M


def rankme_from_s(s):
    """RankMe directly from a singular-value vector (same formula as
    spectral_metrics.rankme: entropy exponent of p = s / sum(s))."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def calibrate_alpha_from_s(s, target_rankme, lo=0.05, hi=4.0, iters=40):
    """INT2 fast path: for a power surgery the surgered bank spectrum is exactly
    s^alpha, so calibration needs no matrix SVD per bisection step."""
    s = np.asarray(s, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rankme_from_s(s ** mid) > target_rankme:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_alpha(bank_X, target_rankme, lo=0.05, hi=4.0, iters=40):
    """Bisect alpha so rankme(bank after power surgery) hits target_rankme.
    rankme is monotone decreasing in alpha (larger alpha = sharper spectrum)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = rankme_raw(apply_surgery(bank_X, fit_surgery(bank_X, "power", alpha=mid)))
        if got > target_rankme:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
