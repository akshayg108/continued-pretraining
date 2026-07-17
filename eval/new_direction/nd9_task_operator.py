#!/usr/bin/env python3
"""nd9_task_operator.py — metric layer for ND9 (capture), ND10 (delta-K decomposition)
and ND11 (local graph carrier). Theory source: the task-coupled operator-transport
proposal (verification/results/theory_unification_2026-07-15/); conventions match
nd6_alignment.py (UNCENTERED second-moment eigenbasis; one-hot Y; class-centered Yc
variants carry the 'cen' suffix, as in nd6's cC* columns).

  task_capture          total task power in the feature span: sum_i ||u_i^T Y||^2/||Y||^2
                        — the denominator nd6's cC(rho) discards (a SHARE conditional on
                        capture); two encoders can share cC_K yet capture differently.
  accessibility_curve   A(kappa) = sum_i lam_i/(lam_i+kappa) p_i / ||Y||^2 — ridge-weighted
                        spectral accessibility; kappa->0 recovers capture.
  centered_kta          linear CKA between features and one-hot labels (both centered).
  subspace_affinity     mean cos^2 principal angles between top-r eigenspaces.
  spectrum_transplant_decomposition
                        dA split along the declared path: transplant the POST spectrum
                        onto the PRE basis (rank-matched, both descending) ->
                        dA_spec = A(lam_post, U_pre) - A(lam_pre, U_pre)   (eigenvalue flow)
                        dA_rot  = A(lam_post, U_post) - A(lam_post, U_pre) (basis rotation)
                        Exact partition: dA_total == dA_spec + dA_rot. kappa is anchored on
                        the PRE spectrum (kappa_rel * mean(lam_pre)), declared pre-data.
  graph_label_metrics   cosine-kNN graph (the operator kNN actually consumes): neighbour
                        label purity + graph placement (label energy in the K
                        lowest-frequency modes of the normalized Laplacian).

Tests (TDD, written first): eval/new_direction/test_nd9_task_operator.py
"""
import numpy as np


def _onehot(y):
    classes = np.unique(y)
    Y = (np.asarray(y)[:, None] == classes[None, :]).astype(np.float64)
    return Y, len(classes)


def _svd_powers(X, y):
    """Shared SVD block: eigenvalues lam_i = s_i^2/n of the uncentered second moment and
    per-mode label powers for raw and class-centered one-hot Y (nd6 convention).

    FIX 2026-07-16 (Codex audit): directions whose singular value is below the standard
    numerical-rank tolerance are NOT in col(X); their U columns are arbitrary basis
    filler and their label powers are spurious (a rank-5 cloud in 50 dims inflated
    capture 0.04 -> 0.19). Such entries are zeroed IN PLACE (lam and p), which keeps
    array lengths — and therefore the transplant decomposition's rank matching —
    unchanged. Zeroed entries contribute nothing to capture, accessibility, or C(rho)."""
    X = np.asarray(X, dtype=np.float64)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = s.max(initial=0.0) * max(X.shape) * np.finfo(np.float64).eps
    live = s > tol
    lam = np.where(live, s ** 2 / len(X), 0.0)
    Y, n_cls = _onehot(y)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    p_raw = np.where(live, ((U.T @ Y) ** 2).sum(axis=1), 0.0)
    p_cen = np.where(live, ((U.T @ Yc) ** 2).sum(axis=1), 0.0)
    return U, lam, p_raw, p_cen, (Y ** 2).sum(), (Yc ** 2).sum(), n_cls


def task_capture(X, y):
    """Fraction of task power inside the feature span (in-sample R^2 of unregularized
    least squares); 'cen' = class-centered target (primary, mean direction removed)."""
    _, _, p_raw, p_cen, y2_raw, y2_cen, _ = _svd_powers(X, y)
    return {"capture_raw": float(p_raw.sum() / y2_raw),
            "capture_cen": float(p_cen.sum() / y2_cen)}


def accessibility_curve(lam, p, y_norm2, kappas):
    """A(kappa) = sum_i lam_i/(lam_i+kappa) * p_i / y_norm2 for each kappa."""
    lam = np.asarray(lam, dtype=np.float64)[None, :]
    kap = np.asarray(kappas, dtype=np.float64)[:, None]
    w = lam / (lam + kap)
    return (w @ np.asarray(p, dtype=np.float64)) / y_norm2


def centered_kta(X, y):
    """Linear CKA between centered features and centered one-hot labels."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Y, _ = _onehot(y)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Yc.T @ Xc) ** 2
    return float(cross / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)))


def subspace_affinity(U1, U2, r):
    """Mean cos^2 of the principal angles between span(U1[:, :r]) and span(U2[:, :r])."""
    M = U1[:, :r].T @ U2[:, :r]
    return float((M ** 2).sum() / r)


def spectrum_transplant_decomposition(X_pre, X_post, y, kappa_rel=1.0):
    """Split dA (pre -> post accessibility change) into eigenvalue-flow and
    basis-rotation parts along the declared transplant path (see module docstring)."""
    U0, lam0, _, p0, _, y2, n_cls = _svd_powers(X_pre, y)
    U1, lam1, _, p1, _, _, _ = _svd_powers(X_post, y)
    assert len(lam0) == len(lam1), "pre/post spectra must be rank-matched (same n, d)"
    kappa = kappa_rel * lam0.mean()

    def A(lam, p):
        return float((lam / (lam + kappa) * p).sum() / y2)

    A_pre, A_post, A_spec = A(lam0, p0), A(lam1, p1), A(lam1, p0)
    C0 = np.cumsum(p0) / max(p0.sum(), 1e-30)
    C1 = np.cumsum(p1) / max(p1.sum(), 1e-30)
    K = min(n_cls, len(C0))
    return {"dA_total": A_post - A_pre,
            "dA_spec": A_spec - A_pre,
            "dA_rot": A_post - A_spec,
            "A_pre": A_pre, "A_post": A_post,
            "dcC_K": float(C1[K - 1] - C0[K - 1]),
            "affinity_topK": subspace_affinity(U0, U1, K)}


def vote_operator_metrics(train_X, train_y, test_X, test_y, k=20,
                          return_per_query=False):
    """ND12 I1 — the vote-operator PROXY: directed test-to-train kNN voting
    with the EXACT protocol of stable_cp/evaluation/zero_shot_eval.py::knn_evaluate
    (L2-normalize both sides; KNeighborsClassifier(min(k, n_train), metric='cosine',
    weights='distance') — inverse-distance weights, sklearn zero-distance convention).

    Returns knn_f1_hat (macro-F1 over ALL train classes, absent classes count 0 —
    torchmetrics convention) and the signed vote margin per query: weighted share of
    the true class minus the largest other-class share (margin > 0 <=> the evaluator's
    argmax is correct, up to exact ties). ROLE RESTRICTION (ND12_PREREG.md): these are
    gate/bridge quantities, kin to the outcome — never model features, never evidence."""
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize
    train_X = normalize(np.asarray(train_X, dtype=np.float64))
    test_X = normalize(np.asarray(test_X, dtype=np.float64))
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    kk = min(k, len(train_y))
    knn = KNeighborsClassifier(n_neighbors=kk, metric="cosine", weights="distance")
    knn.fit(train_X, train_y)
    pred = knn.predict(test_X)
    classes = np.unique(train_y)
    f1 = f1_score(test_y, pred, average="macro", labels=classes, zero_division=0)

    dist, idx = knn.kneighbors(test_X)
    w = np.zeros_like(dist)
    zero = dist <= 1e-12
    has_zero = zero.any(axis=1)
    w[has_zero] = zero[has_zero].astype(float)               # sklearn zero-distance rule
    with np.errstate(divide="ignore"):
        w[~has_zero] = 1.0 / dist[~has_zero]
    labels = train_y[idx]
    shares = np.stack([(w * (labels == c)).sum(axis=1) for c in classes], axis=1)
    shares = shares / np.maximum(shares.sum(axis=1, keepdims=True), 1e-30)
    true_col = np.searchsorted(classes, test_y)
    true_share = shares[np.arange(len(test_y)), true_col]
    others = shares.copy()
    others[np.arange(len(test_y)), true_col] = -np.inf
    margins = true_share - others.max(axis=1)
    out = {"knn_f1_hat": float(f1),
           "vote_margin_mean": float(margins.mean()),
           "vote_margin_pos_frac": float((margins > 0).mean()),
           "n_bank": int(len(train_y)), "n_test": int(len(test_y))}
    if return_per_query:
        out["margins"] = margins
    return out


def graph_label_metrics(X, y, k=10, n_max=2000, seed=42, weighted=False,
                        class_balanced=False):
    """Cosine-kNN graph quantities on (a subsample of) X: neighbour label purity (micro
    and macro) and graph placement = label energy in the K lowest modes of the
    normalized Laplacian.

    ND12 evaluator-matching knobs (defaults reproduce the ND10-era numbers exactly):
      weighted        adjacency carries cosine similarity (1 - d, clipped at 0) instead
                      of 0/1 edges — distance-weighted graph.
      class_balanced  each class's one-hot column is normalized to unit energy before
                      the placement projection, so minority classes count equally
                      (macro convention, matching the macro-F1 evaluator).
    knn_purity_macro = per-class mean of same-label neighbour fractions (always
    reported; the micro column keeps its historical name)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) > n_max:
        sel = np.random.RandomState(seed).choice(len(X), n_max, replace=False)
        X, y = X[sel], y[sel]
    n = len(X)
    Xn = normalize(X)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
    dist, idx = nn.kneighbors(Xn)
    dist, idx = dist[:, 1:], idx[:, 1:]
    same = (y[idx] == y[:, None])
    purity = float(same.mean())
    per_point = same.mean(axis=1)
    purity_macro = float(np.mean([per_point[y == c].mean() for c in np.unique(y)]))

    A = np.zeros((n, n))
    A[np.arange(n)[:, None], idx] = np.clip(1.0 - dist, 0.0, None) if weighted else 1.0
    A = np.maximum(A, A.T)                                   # symmetrize
    d = A.sum(axis=1)
    Dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(n) - Dinv[:, None] * A * Dinv[None, :]
    evals, evecs = np.linalg.eigh(L)                         # ascending = low frequency first
    Y, n_cls = _onehot(y)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if class_balanced:
        Yc = Yc / np.maximum(np.sqrt((Yc ** 2).sum(axis=0, keepdims=True)), 1e-12)
    p = ((evecs.T @ Yc) ** 2).sum(axis=1)                    # complete basis: sums to ||Yc||^2
    K = min(n_cls, n)
    return {"knn_purity": purity,
            "knn_purity_macro": purity_macro,
            "graph_cC_K": float(p[:K].sum() / max(p.sum(), 1e-30)),
            "n_used": n}
