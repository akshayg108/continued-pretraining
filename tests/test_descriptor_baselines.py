"""Checks for the cache-only, matched-sample descriptor comparison."""

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist


def runner():
    path = Path(__file__).resolve().parents[1] / "eval/descriptor_baselines.py"
    assert path.is_file(), "Cache-only descriptor runner is missing"
    return importlib.import_module("eval.descriptor_baselines")


def test_sampling_matches_published_uniformity():
    module = runner()
    np.testing.assert_array_equal(module.angular_indices(546), np.arange(546))
    np.testing.assert_array_equal(
        module.angular_indices(5000),
        np.random.RandomState(42).choice(5000, 3000, replace=False),
    )


def test_angular_descriptors_match_direct_formulas():
    module = runner()
    x = np.random.RandomState(8).normal(size=(12, 5))
    z = x / np.linalg.norm(x, axis=1, keepdims=True)
    values = module.target_descriptors(x)
    upper = np.triu_indices(len(z), 1)
    singular = np.linalg.svd(z, compute_uv=False)
    p = singular / singular.sum()
    assert values["uniformity_t2"] == pytest.approx(
        np.log(np.exp(-2 * pdist(z, "sqeuclidean")).mean()), abs=1e-12)
    assert values["mean_pairwise_cos"] == pytest.approx((z @ z.T)[upper].mean(), abs=1e-12)
    assert values["rankme_l2_uncentered"] == pytest.approx(np.exp(-np.sum(p * np.log(p))))
    assert values["n_angular"] == 12


def test_row_lengths_do_not_change_angular_or_spectral_values():
    module = runner()
    x = np.random.RandomState(10).normal(size=(40, 8))
    a = module.target_descriptors(x)
    b = module.target_descriptors(x * np.linspace(0.1, 20, len(x))[:, None])
    for key in module.TARGET_METRICS:
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_rankme_is_not_centered_and_excludes_no_singular_directions():
    module = runner()
    identical = module.target_descriptors(np.ones((6, 4)))
    orthogonal = module.target_descriptors(np.eye(4))
    assert identical["rankme_l2_uncentered"] == pytest.approx(1)
    assert identical["mean_pairwise_cos"] == pytest.approx(1)
    assert identical["uniformity_t2"] == pytest.approx(0)
    assert orthogonal["rankme_l2_uncentered"] == pytest.approx(4)
    assert orthogonal["mean_pairwise_cos"] == pytest.approx(0)
    assert orthogonal["uniformity_t2"] == pytest.approx(-4)


@pytest.mark.parametrize("bad", [np.zeros((4, 8)), np.full((4, 8), np.nan),
                                 np.ones((1, 8)), np.ones(5)])
def test_invalid_features_are_rejected(bad):
    with pytest.raises(ValueError):
        runner().target_descriptors(bad)


def test_reference_metrics_match_existing_estimators():
    module = runner()
    rng = np.random.RandomState(11)
    x, ref = rng.normal(size=(80, 7)), rng.normal(size=(91, 7))
    z = x / np.linalg.norm(x, axis=1, keepdims=True)
    q = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    gamma = 1 / np.median(pdist(np.vstack([z[:500], q[:500]]), "sqeuclidean"))
    pp = np.exp(-gamma * cdist(z, z, "sqeuclidean")).mean()
    qq = np.exp(-gamma * cdist(q, q, "sqeuclidean")).mean()
    pq = np.exp(-gamma * cdist(z, q, "sqeuclidean")).mean()
    distances = cdist(z, np.vstack([z, q]), "cosine")
    distances[np.arange(len(z)), np.arange(len(z))] = np.inf
    indices = distances.argsort(axis=1)[:, :50]
    values = module.reference_descriptors(x, ref)
    assert values["mmd_gamma"] == pytest.approx(gamma)
    assert values["mmd_rbf"] == pytest.approx(pp + qq - 2 * pq, abs=1e-12)
    assert values["neighbor_overlap_k50"] == pytest.approx((indices >= len(z)).mean())
    assert values["n_overlap_queries"] == len(z)


def test_cache_hash_and_shapes_are_checked(tmp_path):
    module = runner()
    path = tmp_path / "bank.npz"
    np.savez(path, bank_X=np.ones((4, 8)), bank_y=np.arange(4))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    x = module.read_bank(path, expected_hash=digest, expected_n=4, expected_dim=8)
    assert x.shape == (4, 8)
    with pytest.raises(ValueError, match="checksum"):
        module.read_bank(path, expected_hash="wrong")
    with pytest.raises(ValueError, match="shape"):
        module.read_bank(path, expected_n=5, expected_dim=8)


def native_rows(module):
    rows = []
    for dataset, seed in module.native_keys():
        rows.append(dict(encoder="SigLIP", dataset=dataset, seed=seed,
                         n_bank=546, n_angular=546,
                         **{key: float(seed) for key in module.DESCRIPTORS}))
    return rows


def test_native_aggregation_keeps_all_three_galaxy_splits():
    module = runner()
    rows = native_rows(module)
    result = module.aggregate_native(rows)
    assert len(result) == 15
    galaxy = next(row for row in result if row["dataset"] == "galaxy10")
    assert galaxy["n_splits"] == 3
    assert galaxy["mean_pairwise_cos"] == 43
    assert sum(row["n_splits"] for row in result) == 17


def test_native_aggregation_rejects_partial_or_duplicate_splits():
    module = runner()
    rows = native_rows(module)
    for invalid in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="coverage"):
            module.aggregate_native(invalid)


def test_outputs_are_never_overwritten(tmp_path):
    module = runner()
    target = tmp_path / "result"
    module.write_outputs(target, [{"x": 1}], [{"x": 1}], {"test": True})
    before = (target / "descriptors.csv").read_bytes()
    with pytest.raises(FileExistsError):
        module.write_outputs(target, [{"x": 2}], [{"x": 2}], {})
    assert (target / "descriptors.csv").read_bytes() == before
