"""Pre-CP geometry on one shared target sample and a fixed ImageNet reference."""

import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

GEOMETRY_PROTOCOL = "precp_geometry_5000_v1"
GEOMETRY_METRICS = (
    "uniformity_t2",
    "mean_pairwise_cos",
    "rankme_l2_uncentered",
    "mmd_rbf",
    "neighbor_overlap_k50",
)
_BANK_SIZE = 5000
_BLOCK_SIZE = 256


def select_geometry_indices(n_samples):
    """Select a label-independent, seed-independent subset in source-row order."""
    if n_samples < 2:
        raise ValueError("Geometry requires at least two target samples")
    if n_samples > _BANK_SIZE:
        return np.sort(np.random.RandomState(42).choice(n_samples, _BANK_SIZE, replace=False))
    return np.arange(n_samples)


def _unit_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite feature rows with at least two samples")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Geometry feature norms must be finite and positive")
    return x / norms


def _kernel_mean(x, y, gamma):
    total = 0.0
    for start in range(0, len(x), _BLOCK_SIZE):
        squared = np.clip(2 - 2 * (x[start : start + _BLOCK_SIZE] @ y.T), 0, 4)
        total += float(np.exp(-gamma * squared).sum())
    return total / (len(x) * len(y))


def geometry_descriptors(features, reference):
    """Measure all five descriptors using every supplied target feature row."""
    target, imagenet = _unit_features(features), _unit_features(reference)
    n = len(target)
    if n > _BANK_SIZE or len(imagenet) != _BANK_SIZE:
        raise ValueError("Geometry requires at most 5000 target and exactly 5000 reference rows")
    if target.shape[1] != imagenet.shape[1]:
        raise ValueError("Target and reference feature dimensions must match")

    kernel_sum = 0.0
    for start in range(0, n, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, n)
        squared = np.clip(2 - 2 * (target[start:end] @ target.T), 0, 4)
        kernel = np.exp(-2 * squared)
        kernel[np.arange(end - start), np.arange(start, end)] = 0
        kernel_sum += float(kernel.sum())
    total = target.sum(axis=0)
    mean_cosine = (float(total @ total) - n) / (n * (n - 1))
    singular = np.linalg.svd(target, compute_uv=False)
    probabilities = singular[singular > 0] / singular.sum()
    rankme = float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    combined = np.vstack([target, imagenet])
    distances = pdist(combined, metric="sqeuclidean")
    gamma = 1 / max(float(np.median(distances, overwrite_input=True)), 1e-8)
    del distances
    pp = _kernel_mean(target, target, gamma)
    qq = _kernel_mean(imagenet, imagenet, gamma)
    pq = _kernel_mean(target, imagenet, gamma)

    neighbors = NearestNeighbors(n_neighbors=51, metric="cosine", algorithm="brute").fit(combined)
    hits = 0
    for start in range(0, n, _BLOCK_SIZE):
        indices = neighbors.kneighbors(target[start : start + _BLOCK_SIZE], return_distance=False)
        # Exclude the actual query row, not an arbitrary first neighbor under ties.
        hits += sum(
            np.count_nonzero(row[row != start + offset][:50] >= n)
            for offset, row in enumerate(indices)
        )

    return {
        "n_geometry": n,
        "n_reference": len(imagenet),
        "n_bandwidth_samples": len(combined),
        "uniformity_t2": float(np.log(kernel_sum / (n * (n - 1)))),
        "mean_pairwise_cos": float(mean_cosine),
        "rankme_l2_uncentered": rankme,
        "mmd_rbf": pp + qq - 2 * pq,
        "mmd_gamma": gamma,
        "mmd_m_pp_target": pp,
        "mmd_m_qq_imagenet": qq,
        "mmd_m_pq_cross": pq,
        "neighbor_overlap_k50": hits / (50 * n),
    }


def evaluate_geometry(features, reference_path, output_path=None, *, metadata):
    """Measure selected clean target features against a compatible reference archive."""
    reference_path = Path(reference_path).expanduser().resolve()
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["features"]
        reference_metadata = json.loads(archive["metadata"].item())
    expected = {
        "protocol": GEOMETRY_PROTOCOL,
        "n_reference": _BANK_SIZE,
        **{key: metadata[key] for key in ("backbone", "pool_strategy", "normalization")},
    }
    if "feature_readout" in metadata:
        expected["feature_readout"] = metadata["feature_readout"]
    for key, value in expected.items():
        if reference_metadata.get(key) != value:
            raise ValueError(f"Reference metadata mismatch for {key}: {reference_path}")

    indices = select_geometry_indices(len(features))
    bank = np.asarray(features)[indices]
    if bank.dtype.hasobject:
        raise ValueError("Geometry features must not contain Python objects")
    results = {
        "protocol": GEOMETRY_PROTOCOL,
        "n_train": len(features),
        "sampling_seed": 42,
        "reference_file": str(reference_path),
        **geometry_descriptors(bank, reference),
    }
    if "feature_readout" in metadata:
        results["feature_readout"] = metadata["feature_readout"]
    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results["features_file"] = str(output_path)
        feature_metadata = dict(metadata, **results)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            np.savez_compressed(
                handle,
                features=bank,
                indices=indices,
                metadata=json.dumps(feature_metadata, allow_nan=False),
            )
        Path(handle.name).replace(output_path)
    return results
