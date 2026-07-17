#!/usr/bin/env python3
"""TDD tests for nd9_task_operator.py — written BEFORE the implementation.

Covers the ND9/ND10/ND11 metric layer:
  task_capture                     total task power captured by the feature span
  accessibility_curve              ridge-weighted spectral accessibility A(kappa)
  centered_kta                     centered kernel-target alignment (linear CKA vs labels)
  subspace_affinity                top-r eigenspace overlap (mean cos^2 principal angles)
  spectrum_transplant_decomposition  dA split into spectral-flow vs basis-rotation parts
  graph_label_metrics              cosine-kNN graph label purity + graph placement

Run: python3 eval/new_direction/test_nd9_task_operator.py
"""
import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))

from nd9_task_operator import (task_capture, accessibility_curve, centered_kta,
                               subspace_affinity, spectrum_transplant_decomposition,
                               graph_label_metrics)

rng = np.random.RandomState(0)


def _onehot_centered(y):
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    return Y - Y.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------- task_capture
def test_capture_equals_insample_r2():
    """capture_cen must equal the in-sample R^2 of unregularized least squares on the
    centered one-hot target (both are ||P_col(X) Yc||^2 / ||Yc||^2)."""
    X = rng.randn(300, 40)
    y = rng.randint(0, 5, 300)
    Yc = _onehot_centered(y)
    coef, *_ = np.linalg.lstsq(X, Yc, rcond=None)
    r2 = 1.0 - ((Yc - X @ coef) ** 2).sum() / (Yc ** 2).sum()
    cap = task_capture(X, y)
    assert abs(cap["capture_cen"] - r2) < 1e-8, (cap["capture_cen"], r2)


def test_capture_is_one_when_labels_in_span():
    """If the centered one-hot columns lie inside col(X), capture_cen == 1."""
    y = rng.randint(0, 4, 200)
    Yc = _onehot_centered(y)
    X = np.hstack([Yc, rng.randn(200, 10)])          # labels exactly representable
    cap = task_capture(X, y)
    assert cap["capture_cen"] > 1.0 - 1e-10, cap["capture_cen"]


def test_capture_in_unit_interval_and_low_for_orthogonal_features():
    """Random narrow features can only capture a small fraction; bounds hold."""
    X = rng.randn(1000, 5)                            # 5 dims vs 1000 samples
    y = rng.randint(0, 10, 1000)
    cap = task_capture(X, y)
    for k in ("capture_raw", "capture_cen"):
        assert 0.0 <= cap[k] <= 1.0 + 1e-12, (k, cap[k])
    assert cap["capture_cen"] < 0.2, cap["capture_cen"]


def test_capture_zero_features_is_zero():
    """An all-zero feature matrix spans nothing: capture must be 0, not the energy of
    Y along arbitrary null-direction basis vectors (Codex audit 2026-07-16)."""
    X = np.zeros((200, 50))
    y = rng.randint(0, 4, 200)
    cap = task_capture(X, y)
    assert abs(cap["capture_raw"]) < 1e-10, cap
    assert abs(cap["capture_cen"]) < 1e-10, cap


def test_capture_identity_holds_for_rank_deficient_features():
    """Rank-5 features embedded in 50 dims: capture must still equal the lstsq R^2
    (directions with zero singular value are NOT in col(X) and must not contribute)."""
    A = rng.randn(200, 5) @ rng.randn(5, 50)
    y = rng.randint(0, 4, 200)
    Yc = _onehot_centered(y)
    coef, *_ = np.linalg.lstsq(A, Yc, rcond=None)
    r2 = 1.0 - ((Yc - A @ coef) ** 2).sum() / (Yc ** 2).sum()
    cap = task_capture(A, y)
    assert abs(cap["capture_cen"] - r2) < 1e-8, (cap["capture_cen"], r2)


# ------------------------------------------------------- accessibility_curve
def test_accessibility_limits_and_monotonicity():
    """A(kappa->0) == capture; A is nonincreasing in kappa; A >= 0."""
    X = rng.randn(400, 30)
    y = rng.randint(0, 3, 400)
    Yc = _onehot_centered(y)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    lam = s ** 2 / len(X)
    p = ((U.T @ Yc) ** 2).sum(axis=1)
    y2 = (Yc ** 2).sum()
    kappas = np.array([1e-12, 1e-3, 1e-1, 1e1]) * lam.mean()
    A = accessibility_curve(lam, p, y2, kappas)
    cap = task_capture(X, y)["capture_cen"]
    assert abs(A[0] - cap) < 1e-6, (A[0], cap)
    assert np.all(np.diff(A) <= 1e-12), A
    assert np.all(A >= -1e-12), A


# ---------------------------------------------------------------- centered_kta
def test_kta_is_one_for_rotated_scaled_labels():
    """X that is exactly the centered one-hot target (rotated, scaled) has KTA == 1
    (linear CKA is invariant to orthogonal transforms and isotropic scaling)."""
    y = rng.randint(0, 4, 300)
    Yc = _onehot_centered(y)
    R = np.linalg.qr(rng.randn(4, 4))[0]
    X = 3.7 * (Yc @ R)
    assert centered_kta(X, y) > 1.0 - 1e-8


def test_kta_orthogonal_invariance_and_random_near_zero():
    X = rng.randn(300, 20)
    y = rng.randint(0, 5, 300)
    R = np.linalg.qr(rng.randn(20, 20))[0]
    a, b = centered_kta(X, y), centered_kta(X @ R, y)
    assert abs(a - b) < 1e-10, (a, b)
    assert a < 0.15, a                                # random labels: near zero


# ------------------------------------------------------------ subspace_affinity
def test_affinity_one_for_same_subspace_zero_for_orthogonal():
    Q = np.linalg.qr(rng.randn(60, 20))[0]
    U1, U2, U3 = Q[:, :5], Q[:, :5] @ np.linalg.qr(rng.randn(5, 5))[0], Q[:, 5:10]
    assert abs(subspace_affinity(U1, U2, 5) - 1.0) < 1e-10   # same span, mixed within
    assert abs(subspace_affinity(U1, U3, 5)) < 1e-10         # orthogonal spans


# --------------------------------------- spectrum_transplant_decomposition
def _cloud(n=400, d=30):
    lam_true = np.arange(1, d + 1, dtype=float) ** -1.2
    X = rng.randn(n, d) * np.sqrt(lam_true)
    y = (X[:, 0] > 0).astype(int)                     # labels ride the top mode
    return X, y


def test_decomposition_is_exact_partition():
    """dA_total == dA_spec + dA_rot for the declared path order."""
    X, y = _cloud()
    Xp = rng.randn(400, 30) * 0.9
    out = spectrum_transplant_decomposition(X, Xp, y)
    assert abs(out["dA_total"] - (out["dA_spec"] + out["dA_rot"])) < 1e-10, out


def test_pure_rescale_is_all_spectral():
    """Rescaling singular values (order-preserving), same singular vectors: the whole
    dA must land in the spectral part; rotation part == 0."""
    X, y = _cloud()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2 = s * np.linspace(1.0, 0.4, len(s))            # monotone squeeze, order preserved
    Xp = U @ np.diag(s2) @ Vt
    out = spectrum_transplant_decomposition(X, Xp, y)
    assert abs(out["dA_rot"]) < 1e-8, out
    assert abs(out["dA_total"]) > 1e-4                # the rescale actually moved A
    assert out["affinity_topK"] > 1.0 - 1e-8


def test_pure_rotation_is_all_rotational():
    """Left-orthogonal transform of the sample cloud keeps the spectrum, rotates the
    eigenbasis relative to the fixed labels: dA_spec == 0."""
    X, y = _cloud()
    R = np.linalg.qr(rng.randn(400, 400))[0]
    Xp = R @ X
    out = spectrum_transplant_decomposition(X, Xp, y)
    assert abs(out["dA_spec"]) < 1e-8, out
    assert abs(out["dA_total"] - out["dA_rot"]) < 1e-10


# ------------------------------------------------------------ graph_label_metrics
def test_graph_metrics_separated_clusters():
    """Two well-separated clusters with matching labels: purity ~ 1 and the label
    signal lives in the lowest graph-frequency modes (graph_cC_K high)."""
    a = rng.randn(150, 10) * 0.05 + np.eye(10)[0]
    b = rng.randn(150, 10) * 0.05 + np.eye(10)[1]
    X = np.vstack([a, b])
    y = np.array([0] * 150 + [1] * 150)
    out = graph_label_metrics(X, y, k=10, seed=1)
    assert out["knn_purity"] > 0.95, out
    assert out["graph_cC_K"] > 0.5, out


def test_graph_metrics_random_labels():
    """Random labels on the same cloud: purity ~ chance, graph placement low."""
    X = rng.randn(300, 10)
    y = rng.randint(0, 2, 300)
    out = graph_label_metrics(X, y, k=10, seed=1)
    assert out["knn_purity"] < 0.65, out
    assert out["graph_cC_K"] < 0.2, out


# ---------------------------------------------- evaluator-matched graph variants (ND12)
def test_graph_matched_defaults_reproduce_legacy():
    """weighted=False, class_balanced=False, k=10 must reproduce the ND10-era numbers
    exactly (backward compatibility of the extended signature)."""
    X = rng.randn(300, 10)
    y = rng.randint(0, 2, 300)
    a = graph_label_metrics(X, y, k=10, seed=1)
    b = graph_label_metrics(X, y, k=10, seed=1, weighted=False, class_balanced=False)
    assert a["graph_cC_K"] == b["graph_cC_K"] and a["knn_purity"] == b["knn_purity"]


def test_class_balanced_equals_unbalanced_for_equal_classes():
    """With perfectly balanced classes, per-class energy normalization is a no-op."""
    a = rng.randn(100, 8) * 0.05 + np.eye(8)[0]
    b = rng.randn(100, 8) * 0.05 + np.eye(8)[1]
    X = np.vstack([a, b]); y = np.array([0] * 100 + [1] * 100)
    u = graph_label_metrics(X, y, k=10, seed=1, class_balanced=False)
    v = graph_label_metrics(X, y, k=10, seed=1, class_balanced=True)
    assert abs(u["graph_cC_K"] - v["graph_cC_K"]) < 1e-9, (u, v)


def test_class_balanced_lifts_minority_class_signal():
    """THREE classes (binary is a provable no-op: the two centered one-hot columns are
    collinear with equal energy): a tight minority cluster (30 pts, clean low-frequency
    label signal) vs two diffuse majority classes (135 pts each, labels assigned at
    random = graph noise). Balancing must lift the minority's share of low-frequency
    label energy."""
    minority = rng.randn(30, 10) * 0.03 + 3 * np.eye(10)[0]
    majority = rng.randn(270, 10)
    X = np.vstack([minority, majority])
    y = np.concatenate([np.zeros(30, int), rng.randint(1, 3, 270)])
    u = graph_label_metrics(X, y, k=10, seed=1, class_balanced=False)
    v = graph_label_metrics(X, y, k=10, seed=1, class_balanced=True)
    assert v["graph_cC_K"] > u["graph_cC_K"] + 0.05, (u["graph_cC_K"], v["graph_cC_K"])


def test_macro_purity_discounts_majority_class_inflation():
    """Random labels at 90/10 imbalance: micro purity ~ 0.82 (majority self-matches),
    macro purity ~ 0.5 — the macro column must not inherit the inflation."""
    X = rng.randn(400, 10)
    y = (rng.rand(400) < 0.1).astype(int)
    out = graph_label_metrics(X, y, k=20, seed=1)
    assert out["knn_purity"] > out["knn_purity_macro"] + 0.2, out


def test_weighted_graph_separated_clusters_still_clean():
    """Similarity-weighted Laplacian on well-separated clusters keeps placement high."""
    a = rng.randn(150, 10) * 0.05 + np.eye(10)[0]
    b = rng.randn(150, 10) * 0.05 + np.eye(10)[1]
    X = np.vstack([a, b]); y = np.array([0] * 150 + [1] * 150)
    out = graph_label_metrics(X, y, k=20, seed=1, weighted=True, class_balanced=True)
    assert out["graph_cC_K"] > 0.5, out
    assert out["knn_purity"] > 0.95, out


# ------------------------------------------------- true vote operator (ND12, I1)
def test_vote_margin_sign_matches_sklearn_prediction():
    """For every query: margin > 0 iff the sklearn evaluator-protocol prediction is
    correct (weights='distance', metric='cosine', k=20) — the margin must be the
    signed version of exactly the evaluator's argmax."""
    from nd9_task_operator import vote_operator_metrics
    Xtr, ytr = rng.randn(400, 16), rng.randint(0, 5, 400)
    Xte, yte = rng.randn(150, 16), rng.randint(0, 5, 150)
    out = vote_operator_metrics(Xtr, ytr, Xte, yte, k=20, return_per_query=True)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize
    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance")
    knn.fit(normalize(Xtr), ytr)
    pred = knn.predict(normalize(Xte))
    agree = ((out["margins"] > 0) == (pred == yte))
    assert agree.mean() > 0.99, agree.mean()          # ties may cost isolated points


def test_vote_f1_equals_evaluator_protocol_f1():
    """knn_f1_hat must equal macro-F1 over ALL classes of the sklearn evaluator
    prediction (torchmetrics convention: absent classes count as zero)."""
    from nd9_task_operator import vote_operator_metrics
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize
    from sklearn.metrics import f1_score
    Xtr, ytr = rng.randn(300, 12), rng.randint(0, 4, 300)
    Xte, yte = rng.randn(120, 12), rng.randint(0, 4, 120)
    out = vote_operator_metrics(Xtr, ytr, Xte, yte, k=20)
    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance")
    knn.fit(normalize(Xtr), ytr)
    ref = f1_score(yte, knn.predict(normalize(Xte)), average="macro",
                   labels=np.arange(4), zero_division=0)
    assert abs(out["knn_f1_hat"] - ref) < 1e-10, (out["knn_f1_hat"], ref)


def test_vote_separated_clusters_full_margin():
    """Clean clusters: F1 == 1 and margins ~ +1 (true class takes all the weight)."""
    from nd9_task_operator import vote_operator_metrics
    Xtr = np.vstack([rng.randn(100, 8) * 0.03 + np.eye(8)[i] for i in (0, 1)])
    ytr = np.array([0] * 100 + [1] * 100)
    Xte = np.vstack([rng.randn(40, 8) * 0.03 + np.eye(8)[i] for i in (0, 1)])
    yte = np.array([0] * 40 + [1] * 40)
    out = vote_operator_metrics(Xtr, ytr, Xte, yte, k=20)
    assert out["knn_f1_hat"] == 1.0
    assert out["vote_margin_mean"] > 0.95, out


def test_vote_k_capped_at_bank_size():
    """k > n_train must not crash (evaluator caps k = min(k, n_train))."""
    from nd9_task_operator import vote_operator_metrics
    Xtr, ytr = rng.randn(12, 6), rng.randint(0, 2, 12)
    Xte, yte = rng.randn(9, 6), rng.randint(0, 2, 9)
    out = vote_operator_metrics(Xtr, ytr, Xte, yte, k=20)
    assert out["n_bank"] == 12 and np.isfinite(out["vote_margin_mean"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
