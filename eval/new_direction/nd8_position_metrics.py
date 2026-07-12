#!/usr/bin/env python3
"""
nd8_position_metrics.py — position-metric algorithms for the ND8 overlap upgrade.

Implemented verbatim from the round-3 sources (papers/new_direction/NEW_DIRECTION_R3.md):
  mp_empiric_rerank   Schnitzer et al., JMLR 2012 (MP-empiric): similarity
                      MP(d_xy) = |{j != x,y : d_xj > d_xy AND d_yj > d_yx}| / (n-2).
  nicdm_rerank        local scaling (NICDM): d'(x,y) = d(x,y) / sqrt(mu(x) * mu(y)),
                      mu = mean distance to the k_local nearest neighbours.
  center_features     subtract the JOINT centroid (removes the verified cosine
                      centroid-centrality hub component — Radovanovic/Feldbauer).
  sun_knn_distance    Sun et al., ICML 2022: distance to the k-th nearest reference
                      point on L2-normalized features.

Tests (TDD, written first): eval/new_direction/test_nd8_position_metrics.py
"""
import numpy as np


def _l2n(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cosine_dist_matrix(A, B=None):
    """Pairwise cosine distances; zero diagonal when B is None (self-distances)."""
    An = _l2n(A)
    Bn = An if B is None else _l2n(B)
    D = 1.0 - An @ Bn.T
    np.clip(D, 0.0, None, out=D)
    if B is None:
        np.fill_diagonal(D, 0.0)
    return D


def knn_indices(D, k):
    """Top-k neighbour indices per row of a SELF distance matrix (self excluded)."""
    Dw = D.copy()
    np.fill_diagonal(Dw, np.inf)
    part = np.argpartition(Dw, k, axis=1)[:, :k]
    order = np.take_along_axis(Dw, part, axis=1).argsort(axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def k_occurrence(nn_idx, n):
    """N_k: how often each of the n points appears in others' neighbour lists."""
    return np.bincount(nn_idx.ravel(), minlength=n)


def mp_empiric_rerank(D, k, n_candidates=200, rows=None):
    """k-NN lists under Mutual Proximity (empiric) similarity.

    For each query x, only its n_candidates raw-nearest points are re-ranked (MP is
    monotone-ranking within any candidate set; re-ranking far points cannot enter the
    top-k in practice). Diagonal of D must be 0 and off-diagonal > 0, which makes the
    j = x and j = y terms drop out of the count automatically. `rows` restricts the
    queries (returns len(rows) lists in the given order).
    """
    n = len(D)
    m = min(n_candidates, n - 1)
    raw = knn_indices(D, m)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    out = np.empty((len(rows), k), dtype=np.int64)
    for i, x in enumerate(rows):
        cand = raw[x]
        mask_x = D[x][None, :] > D[x, cand][:, None]     # (m, n)
        mask_y = D[cand] > D[cand, x][:, None]           # (m, n)
        sims = (mask_x & mask_y).sum(axis=1) / (n - 2)
        top = np.argsort(-sims, kind="stable")[:k]
        out[i] = cand[top]
    return out


def nicdm_rerank(D, k, k_local=10):
    """k-NN lists under NICDM-rescaled distances."""
    Dw = D.copy()
    np.fill_diagonal(Dw, np.inf)
    mu = np.sort(Dw, axis=1)[:, :k_local].mean(axis=1)
    Dn = Dw / np.sqrt(np.outer(mu, mu))
    part = np.argpartition(Dn, k, axis=1)[:, :k]
    order = np.take_along_axis(Dn, part, axis=1).argsort(axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def center_features(X):
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def overlap_score(nn_idx, is_bank):
    """Fraction of bank points among the neighbour lists (the overlap statistic)."""
    return float(np.asarray(is_bank)[nn_idx].mean())


def sun_knn_distance(X_target, X_bank, k):
    """Per-target distance to the k-th nearest bank point, L2-normalized features
    (Sun et al. 2022; normalization is mandatory per their ablation)."""
    T, B = _l2n(X_target), _l2n(X_bank)
    D = 1.0 - T @ B.T                    # cosine distance == squared-L2/2 on the sphere
    return np.sort(D, axis=1)[:, k - 1]
