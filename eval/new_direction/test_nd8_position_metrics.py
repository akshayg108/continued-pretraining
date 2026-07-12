#!/usr/bin/env python3
"""
test_nd8_position_metrics.py — TDD tests for the ND8 position-metric algorithms.

Definitions under test (papers/new_direction/NEW_DIRECTION_R3.md):
  MP-empiric   Schnitzer et al., JMLR 2012: MP(d_xy) = |{j != x,y : d_xj > d_xy
               AND d_yj > d_yx}| / (n-2)  — higher = closer. Property: reduces
               k-occurrence skewness on hub-prone data.
  NICDM        local scaling: d'(x,y) = d(x,y) / sqrt(mu_k(x) * mu_k(y)),
               mu_k = mean distance to the k_local nearest neighbours.
  centering    subtract the JOINT centroid before cosine — removes the verified
               centroid-centrality hub component (Radovanovic / Feldbauer).
  Sun kNN dist Sun et al., ICML 2022: distance to the k-th nearest reference point
               on L2-normalized features.

Run:  python3 eval/new_direction/test_nd8_position_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nd8_position_metrics import (cosine_dist_matrix, knn_indices, mp_empiric_rerank,
                                  nicdm_rerank, center_features, overlap_score,
                                  sun_knn_distance, k_occurrence)

rng = np.random.RandomState(0)


def brute_mp(D, x, y):
    n = len(D)
    js = [j for j in range(n) if j != x and j != y]
    hits = sum(1 for j in js if D[x, j] > D[x, y] and D[y, j] > D[y, x])
    return hits / (n - 2)


def hubby_cloud(n=400, d=120):
    """Hub-prone configuration: anisotropic high-d gaussian with a strong common
    component (the centroid offset produces cosine centrality hubs)."""
    scales = np.linspace(2.0, 0.02, d)
    return rng.randn(n, d) * scales + 1.5


def test_cosine_dist_matrix_basics():
    A = rng.randn(20, 8)
    D = cosine_dist_matrix(A)
    assert D.shape == (20, 20)
    assert np.allclose(np.diag(D), 0, atol=1e-8)
    assert np.allclose(D, D.T, atol=1e-8)
    v = A[3] * 2.7                       # same direction -> distance 0
    D2 = cosine_dist_matrix(np.vstack([A[3], v]))
    assert abs(D2[0, 1]) < 1e-8
    print("  cosine_dist_matrix OK")


def test_knn_indices_excludes_self():
    A = rng.randn(30, 5)
    D = cosine_dist_matrix(A)
    idx = knn_indices(D, k=4)
    assert idx.shape == (30, 4)
    for i in range(30):
        assert i not in idx[i]
    print("  knn_indices OK")


def test_mp_matches_bruteforce():
    A = rng.randn(30, 6)
    D = cosine_dist_matrix(A)
    idx = mp_empiric_rerank(D, k=5, n_candidates=29)
    # rebuild the MP similarity for row 0 by brute force and check the top-5 agree
    sims = np.array([brute_mp(D, 0, y) if y != 0 else -1 for y in range(30)])
    expect = np.argsort(-sims, kind="stable")[:5]
    assert set(idx[0]) == set(expect), f"{sorted(idx[0])} vs {sorted(expect)}"
    # rows= subset must reproduce the corresponding full rows
    sub = mp_empiric_rerank(D, k=5, n_candidates=29, rows=np.array([0, 7]))
    assert np.array_equal(sub[0], idx[0]) and np.array_equal(sub[1], idx[7])
    print("  mp_empiric_rerank == brute force OK (+rows subset)")


def test_mp_reduces_hubness():
    X = hubby_cloud()
    D = cosine_dist_matrix(X)
    raw = knn_indices(D, k=10)
    mp = mp_empiric_rerank(D, k=10, n_candidates=100)
    s_raw = skew(k_occurrence(raw, n=len(X)))
    s_mp = skew(k_occurrence(mp, n=len(X)))
    assert s_raw > 1.0, f"test premise broken: cloud not hub-prone (skew {s_raw:.2f})"
    assert s_mp < s_raw / 2, f"MP skew {s_mp:.2f} not << raw {s_raw:.2f}"
    print(f"  MP hubness reduction OK (skew {s_raw:.2f} -> {s_mp:.2f})")


def test_nicdm_matches_formula():
    A = rng.randn(25, 6)
    D = cosine_dist_matrix(A)
    idx = nicdm_rerank(D, k=4, k_local=5)
    # brute force row 0
    Dinf = D + np.diag(np.full(len(D), np.inf))
    mu = np.sort(Dinf, axis=1)[:, :5].mean(axis=1)
    d0 = Dinf[0] / np.sqrt(mu[0] * mu)
    expect = np.argsort(d0, kind="stable")[:4]
    assert set(idx[0]) == set(expect)
    print("  nicdm_rerank == formula OK")


def test_centering_removes_centroid_hub():
    # points on a shifted shell: the point nearest the centroid becomes a cosine hub
    X = rng.randn(300, 40) + 4.0        # strong common component -> centroid artifact
    D = cosine_dist_matrix(X)
    occ_raw = k_occurrence(knn_indices(D, k=10), n=len(X))
    Xc = center_features(X)
    occ_c = k_occurrence(knn_indices(cosine_dist_matrix(Xc), k=10), n=len(X))
    assert skew(occ_c) < skew(occ_raw), \
        f"centering did not reduce hub skew ({skew(occ_raw):.2f} -> {skew(occ_c):.2f})"
    print(f"  centering hub reduction OK (skew {skew(occ_raw):.2f} -> {skew(occ_c):.2f})")


def test_overlap_score_counts_bank_fraction():
    # 10 targets whose neighbours are 3 bank points out of 5 -> overlap 0.6
    nn = np.array([[0, 1, 5, 6, 7]] * 10)      # indices into combined set
    is_bank = np.zeros(20, bool)
    is_bank[5:] = True                          # ids >=5 are bank
    assert abs(overlap_score(nn, is_bank) - 0.6) < 1e-9
    print("  overlap_score OK")


def test_sun_knn_distance_geometry():
    bank = rng.randn(500, 16)
    target_near = bank[:50] + 0.01 * rng.randn(50, 16)
    target_far = rng.randn(50, 16) + 6.0
    d_near = sun_knn_distance(target_near, bank, k=5).mean()
    d_far = sun_knn_distance(target_far, bank, k=5).mean()
    assert d_near < d_far, "near targets must score smaller kth-NN distance"
    # smaller bank -> kth-NN distance can only grow (fewer candidates)
    d_small = sun_knn_distance(target_near, bank[:50], k=5).mean()
    assert d_small >= d_near - 1e-9
    print("  sun_knn_distance geometry OK")


if __name__ == "__main__":
    test_cosine_dist_matrix_basics()
    test_knn_indices_excludes_self()
    test_mp_matches_bruteforce()
    test_mp_reduces_hubness()
    test_nicdm_matches_formula()
    test_centering_removes_centroid_hub()
    test_overlap_score_counts_bank_fraction()
    test_sun_knn_distance_geometry()
    print("ALL ND8 METRIC TESTS PASS")
