#!/usr/bin/env python3
"""TDD tests for int1_surgery.py — written BEFORE the implementation.

INT1 counterfactual surgery (INT1_PREREG.md): every transform is a fixed linear map
in FEATURE space, fitted on bank features only, applied identically to bank and query.
Preservation contracts are the physics of the experiment — each gets a test.

Run: python3 eval/new_direction/test_int1_surgery.py
"""
import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))

from int1_surgery import (fit_surgery, apply_surgery, rankme_raw, capture_of, cC_K_of)

rng = np.random.RandomState(0)


def _cloud(n=600, d=48, decay=-1.2):
    lam = np.arange(1, d + 1, dtype=float) ** decay
    X = rng.randn(n, d) * np.sqrt(lam)
    y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)   # labels ride the top mode
    return X, y


# ------------------------------------------------------------------ T1 rotation
def test_rotation_preserves_pairwise_cosines_exactly():
    X, _ = _cloud()
    Q = fit_surgery(X, kind="rotation", seed=0)
    Xr = apply_surgery(X, Q)
    a, b = X[:50], Xr[:50]
    ca = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (a / np.linalg.norm(a, axis=1, keepdims=True)).T
    cb = (b / np.linalg.norm(b, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
    assert np.abs(ca - cb).max() < 1e-10


def test_rotation_same_map_applies_to_held_out_queries():
    X, _ = _cloud()
    Q = fit_surgery(X, kind="rotation", seed=0)
    Xq = rng.randn(100, X.shape[1])
    out = apply_surgery(Xq, Q)
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(Xq, axis=1))


# --------------------------------------------------------------- T2 spectral power
def test_spectral_power_moves_rankme_in_expected_direction():
    X, _ = _cloud()
    r1 = rankme_raw(X)
    flat = apply_surgery(X, fit_surgery(X, kind="power", alpha=0.5))
    sharp = apply_surgery(X, fit_surgery(X, kind="power", alpha=2.0))
    assert rankme_raw(flat) > r1 + 1, (rankme_raw(flat), r1)     # flatter -> higher rank
    assert rankme_raw(sharp) < r1 - 1, (rankme_raw(sharp), r1)   # sharper -> lower rank


def test_spectral_power_preserves_capture_and_placement_pre_l2():
    X, y = _cloud()
    M = fit_surgery(X, kind="power", alpha=0.5)
    Xs = apply_surgery(X, M)
    assert abs(capture_of(Xs, y) - capture_of(X, y)) < 1e-8
    assert abs(cC_K_of(Xs, y) - cC_K_of(X, y)) < 1e-8            # order preserved


def test_spectral_power_identity_at_alpha_one():
    X, _ = _cloud()
    Xs = apply_surgery(X, fit_surgery(X, kind="power", alpha=1.0))
    assert np.abs(Xs - X).max() < 1e-8


def test_calibrate_alpha_hits_target_rank():
    from int1_surgery import calibrate_alpha
    X, _ = _cloud(n=800, d=64)
    target = rankme_raw(X) * 0.6
    alpha = calibrate_alpha(X, target_rankme=target)
    got = rankme_raw(apply_surgery(X, fit_surgery(X, kind="power", alpha=alpha)))
    assert abs(got - target) / target < 0.02, (got, target, alpha)


# ------------------------------------------------------------ T3 iso-spectral demotion
def test_demotion_preserves_sigma_multiset_rankme_and_capture():
    X, y = _cloud()
    M = fit_surgery(X, kind="demote", block=4, depth=16)
    Xs = apply_surgery(X, M)
    s0 = np.sort(np.linalg.svd(X, compute_uv=False))
    s1 = np.sort(np.linalg.svd(Xs, compute_uv=False))
    assert np.allclose(s0, s1, rtol=1e-8)                        # sigma multiset
    assert abs(rankme_raw(Xs) - rankme_raw(X)) < 1e-6            # RankMe
    assert abs(capture_of(Xs, y) - capture_of(X, y)) < 1e-8      # capture


def test_demotion_moves_placement_down():
    """Labels ride the top mode; demoting the top block's scale must push label power
    out of the top-K eigenmodes (cC_K drops)."""
    X, y = _cloud()
    Xs = apply_surgery(X, fit_surgery(X, kind="demote", block=4, depth=16))
    assert cC_K_of(Xs, y) < cC_K_of(X, y) - 0.05, (cC_K_of(Xs, y), cC_K_of(X, y))


def test_full_shuffle_preserves_sigma_multiset():
    X, y = _cloud()
    Xs = apply_surgery(X, fit_surgery(X, kind="shuffle", seed=0))
    s0 = np.sort(np.linalg.svd(X, compute_uv=False))
    s1 = np.sort(np.linalg.svd(Xs, compute_uv=False))
    assert np.allclose(s0, s1, rtol=1e-8)
    assert abs(capture_of(Xs, y) - capture_of(X, y)) < 1e-8


# ----------------------------------------- v1.4: identity on the unobserved complement
def _thin_cloud(n=40, d=64):
    """n < d bank (breastmnist regime): row space is a proper subspace of R^d."""
    lam = np.arange(1, n + 1, dtype=float) ** -1.0
    B = rng.randn(n, d)
    U, _, Vt = np.linalg.svd(B, full_matrices=False)
    X = (U * lam) @ Vt                                  # exact rank-n cloud
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    return X, y


def _oos_split(q, bank_X):
    """Split a query vector into (in-span, out-of-span) parts w.r.t. the bank row space."""
    _, _, Vt = np.linalg.svd(np.asarray(bank_X, float), full_matrices=False)
    P = Vt.T @ Vt
    return q @ P, q - q @ P


def test_power_acts_as_identity_on_query_complement_when_n_lt_d():
    """v1.4 (Codex round-2): with n < d the old construction annihilated the query
    component outside the bank row space. The map must now pass it through EXACTLY,
    so identity-vs-surgery deltas isolate the in-span effect."""
    X, _ = _thin_cloud()
    q = rng.randn(5, X.shape[1])
    _, q_out = _oos_split(q, X)
    assert np.linalg.norm(q_out) > 1e-3                  # the regime actually bites
    M = fit_surgery(X, kind="power", alpha=2.0)
    _, out_after = _oos_split(apply_surgery(q, M), X)
    assert np.abs(out_after - q_out).max() < 1e-8


def test_demote_and_shuffle_act_as_identity_on_query_complement_when_n_lt_d():
    X, _ = _thin_cloud()
    q = rng.randn(5, X.shape[1])
    _, q_out = _oos_split(q, X)
    for kind, kw in (("demote", dict(block=4, depth=16)), ("shuffle", dict(seed=0))):
        M = fit_surgery(X, kind=kind, **kw)
        _, out_after = _oos_split(apply_surgery(q, M), X)
        assert np.abs(out_after - q_out).max() < 1e-8, kind


def test_demote_still_preserves_sigma_multiset_when_n_lt_d():
    X, _ = _thin_cloud()
    Xs = apply_surgery(X, fit_surgery(X, kind="demote", block=4, depth=16))
    s0 = np.sort(np.linalg.svd(X, compute_uv=False))
    s1 = np.sort(np.linalg.svd(Xs, compute_uv=False))
    assert np.allclose(s0, s1, rtol=1e-7, atol=1e-10)


# ------------------------------------------------------- v1.4: interaction arm (combo)
def test_combo_matches_power_spectrum_but_differs_in_placement():
    """combo = demote(block, depth) then power(alpha) built from ONE bank SVD: the bank
    spectrum multiset equals the pure-power one (same RankMe), but WHICH directions
    carry the large scales differs — the factorial cell for spectrum x placement."""
    X, y = _cloud()
    Mc = fit_surgery(X, kind="combo", alpha=2.0, block=4, depth=16)
    Mp = fit_surgery(X, kind="power", alpha=2.0)
    Xc, Xp = apply_surgery(X, Mc), apply_surgery(X, Mp)
    sc = np.sort(np.linalg.svd(Xc, compute_uv=False))
    sp = np.sort(np.linalg.svd(Xp, compute_uv=False))
    assert np.allclose(sc, sp, rtol=1e-7)                        # same sigma multiset
    assert abs(rankme_raw(Xc) - rankme_raw(Xp)) < 1e-6           # same RankMe
    assert cC_K_of(Xc, y) < cC_K_of(Xp, y) - 0.05                # placement moved
    assert np.abs(Mc - Mp).max() > 1e-3                          # genuinely different map


def test_combo_at_alpha_one_equals_demote():
    X, _ = _cloud()
    Mc = fit_surgery(X, kind="combo", alpha=1.0, block=4, depth=16)
    Md = fit_surgery(X, kind="demote", block=4, depth=16)
    assert np.abs(Mc - Md).max() < 1e-8


def test_combo_acts_as_identity_on_query_complement_when_n_lt_d():
    X, _ = _thin_cloud()
    q = rng.randn(5, X.shape[1])
    _, q_out = _oos_split(q, X)
    M = fit_surgery(X, kind="combo", alpha=2.0, block=4, depth=16)
    _, out_after = _oos_split(apply_surgery(q, M), X)
    assert np.abs(out_after - q_out).max() < 1e-8


# ---------------------------------------------- INT2: spectrum transplant (A5)
def test_transplant_replaces_singular_values_in_rank_order():
    X, _ = _cloud()
    target = np.linspace(200.0, 100.0, X.shape[1])          # well above post values
    M = fit_surgery(X, kind="transplant", s_target=target)
    s_new = np.sort(np.linalg.svd(apply_surgery(X, M), compute_uv=False))[::-1]
    assert np.allclose(s_new, target, rtol=1e-6)


def test_transplant_shorter_target_keeps_post_tail():
    X, _ = _cloud()
    s_post = np.sort(np.linalg.svd(X, compute_uv=False))[::-1]
    target = np.linspace(300.0, 200.0, 10)                  # top-10 only, above tail
    M = fit_surgery(X, kind="transplant", s_target=target)
    s_new = np.sort(np.linalg.svd(apply_surgery(X, M), compute_uv=False))[::-1]
    assert np.allclose(s_new[:10], target, rtol=1e-6)
    assert np.allclose(np.sort(s_new[10:]), np.sort(s_post[10:]), rtol=1e-6)


def test_transplant_acts_as_identity_on_query_complement_when_n_lt_d():
    X, _ = _thin_cloud()
    q = rng.randn(5, X.shape[1])
    _, q_out = _oos_split(q, X)
    M = fit_surgery(X, kind="transplant", s_target=np.linspace(5.0, 1.0, 20))
    _, out_after = _oos_split(apply_surgery(q, M), X)
    assert np.abs(out_after - q_out).max() < 1e-8


# ------------------------------------- INT2: fast spectrum-side RankMe calibration
def test_rankme_from_s_matches_matrix_rankme():
    from int1_surgery import rankme_from_s
    X, _ = _cloud()
    s = np.linalg.svd(X, compute_uv=False)
    assert abs(rankme_from_s(s) - rankme_raw(X)) < 1e-9


def test_calibrate_alpha_from_s_matches_full_recompute():
    from int1_surgery import calibrate_alpha_from_s, calibrate_alpha, rankme_from_s
    X, _ = _cloud(n=800, d=64)
    s = np.linalg.svd(X, compute_uv=False)
    target = rankme_raw(X) * 0.6
    a_fast = calibrate_alpha_from_s(s, target_rankme=target)
    a_slow = calibrate_alpha(X, target_rankme=target)
    assert abs(a_fast - a_slow) < 1e-3
    assert abs(rankme_from_s(s ** a_fast) - target) / target < 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
