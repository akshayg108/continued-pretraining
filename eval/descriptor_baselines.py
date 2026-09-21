"""Recompute pre-CP descriptors from saved features on CPU, without training.

Uniformity, mean pairwise cosine, and uncentered RankMe share the published
seed-42 sample of at most 3000 row-normalized features. MMD and overlap retain
their existing bank sizes and estimators. No outcomes are used in this module.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from eval.full_ft.manifest import DATASET_META, ENCODERS


TARGET_METRICS = ("uniformity_t2", "mean_pairwise_cos", "rankme_l2_uncentered")
REFERENCE_METRICS = ("mmd_rbf", "neighbor_overlap_k50")
DESCRIPTORS = (*TARGET_METRICS, *REFERENCE_METRICS)
MAIN_ENCODERS = ("DINOv3", "CLIP", "MAE")
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "name": "descriptor_baselines_v1",
    "angular_sample": "min(n_bank, 3000); RandomState(42), without replacement",
    "normalization": "row-wise L2 normalization; float64 arithmetic",
    "uniformity": "t=2; distinct pairs only; natural logarithm",
    "mean_pairwise_cos": "mean cosine over distinct within-target pairs",
    "rankme": "exp(entropy(singular_values/sum)); normalized, UNCENTERED matrix",
    "mmd": "squared biased RBF-MMD; full target bank <=5000; reference 5000",
    "gamma": "1/max(median squared distances in pooled first 500 per bank, 1e-8)",
    "overlap": "k=50; min(n_bank,2000) seed-42 queries; reference 5000; exclude self",
    "no_training": True,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite, nonempty feature matrix with at least two rows")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Zero or nonfinite feature norms")
    return x / norms


def angular_indices(n_bank):
    if n_bank < 2:
        raise ValueError("At least two target features are required")
    if n_bank > 3000:
        return np.random.RandomState(42).choice(n_bank, 3000, replace=False)
    return np.arange(n_bank)


def read_bank(path, *, expected_hash=None, expected_n=None, expected_dim=768):
    path = Path(path)
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        x, y = arrays["bank_X"], arrays["bank_y"]
    if (x.ndim != 2 or x.shape[1] != expected_dim or y.shape != (len(x),)
            or (expected_n is not None and len(x) != expected_n)):
        raise ValueError(f"Unexpected feature/label shape: {path}")
    if not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError(f"Invalid bank labels: {path}")
    unit_features(x)
    return x


def target_descriptors(features):
    indices = angular_indices(len(features))
    z = unit_features(features)[indices]
    n = len(z)
    kernel_sum = 0.0
    for start in range(0, n, 256):
        end = min(start + 256, n)
        sq = np.clip(2 - 2 * (z[start:end] @ z.T), 0, 4)
        kernel = np.exp(-2 * sq)
        kernel[np.arange(end - start), np.arange(start, end)] = 0
        kernel_sum += float(kernel.sum())
    total = z.sum(axis=0)
    cosine = (float(total @ total) - n) / (n * (n - 1))
    # Keep the common direction: centering would measure a different spectrum.
    singular = np.linalg.svd(z, compute_uv=False)
    p = singular[singular > 0] / singular.sum()
    return dict(n_bank=len(features), n_angular=n,
                angular_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                uniformity_t2=float(np.log(kernel_sum / (n * (n - 1)))),
                mean_pairwise_cos=float(cosine),
                rankme_l2_uncentered=float(np.exp(-np.sum(p * np.log(p)))))


def reference_descriptors(features, reference):
    a, b = unit_features(features), unit_features(reference)
    distances = pdist(np.vstack([a[:500], b[:500]]), metric="sqeuclidean")
    gamma = 1 / max(float(np.median(distances)), 1e-8)

    def kernel_mean(x, y):
        total = 0.0
        for start in range(0, len(x), 256):
            sq = np.clip(2 - 2 * (x[start:start + 256] @ y.T), 0, 4)
            total += float(np.exp(-gamma * sq).sum())
        return total / (len(x) * len(y))

    pp, qq, pq = kernel_mean(a, a), kernel_mean(b, b), kernel_mean(a, b)
    query = a
    if len(query) > 2000:
        query = query[np.random.RandomState(42).choice(len(query), 2000, replace=False)]
    combined = np.vstack([query, b])
    if len(combined) < 51:
        raise ValueError("Overlap k=50 requires at least 51 pooled features")
    neighbors = NearestNeighbors(n_neighbors=51, metric="cosine").fit(combined)
    indices = neighbors.kneighbors(query, return_distance=False)
    # Remove the actual query index, including when exact duplicates tie with it.
    hits = sum(np.count_nonzero(row[row != i][:50] >= len(query))
               for i, row in enumerate(indices))
    return dict(mmd_rbf=pp + qq - 2 * pq, mmd_gamma=gamma,
                mmd_m_pp_target=pp, mmd_m_qq_imagenet=qq, mmd_m_pq_cross=pq,
                neighbor_overlap_k50=hits / (50 * len(query)),
                n_overlap_queries=len(query), n_reference=len(b))


def native_keys():
    return [(dataset, seed) for dataset in DATASET_META
            for seed in ((42, 43, 44) if dataset == "galaxy10" else (42,))]


def aggregate_native(rows):
    keys = [(row["dataset"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(native_keys()):
        raise ValueError("Native descriptor coverage must be all 17 unique target splits")
    result = []
    for dataset in DATASET_META:
        group = [row for row in rows if row["dataset"] == dataset]
        first = group[0]
        result.append(dict(encoder="SigLIP", dataset=dataset, n_splits=len(group),
                           n_bank=first["n_bank"], n_angular=first["n_angular"],
                           reference_source="recomputed_from_hashed_native_features",
                           **{key: float(np.mean([r[key] for r in group])) for key in DESCRIPTORS}))
    return result


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_outputs(outdir, per_split, summary, manifest):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("descriptors_per_split.csv", per_split), ("descriptors.csv", summary)):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (outdir / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main_grid(feature_dir, geometry_csv):
    archived = {(r["encoder"], r["dataset"]): r for r in read_csv(geometry_csv)}
    rows = []
    for encoder in MAIN_ENCODERS:
        for dataset in DATASET_META:
            path = Path(feature_dir) / f"{encoder}__{dataset}.npz"
            x = read_bank(path, expected_n=min(DATASET_META[dataset][2], 5000))
            measured = target_descriptors(x)
            old = archived[(encoder, dataset)]
            row = dict(encoder=encoder, dataset=dataset, seed=42, n_splits=1, **measured,
                       reference_source="archived_geometry_15_csv",
                       **{key: float(old[key]) for key in REFERENCE_METRICS},
                       published_uniformity_t2=float(old["uniformity_t2"]),
                       uniformity_difference=measured["uniformity_t2"] - float(old["uniformity_t2"]),
                       feature_file=str(path.resolve()), feature_sha256=sha256(path))
            rows.append(row)
            print(f"DONE encoder={encoder} dataset={dataset} n={measured['n_angular']} "
                  "angular=recomputed reference_metrics=archived", flush=True)
    if not np.isfinite([[row[key] for key in DESCRIPTORS] for row in rows]).all():
        raise ValueError("Nonfinite main-grid descriptors")
    return rows, rows, {"geometry_csv": str(Path(geometry_csv).resolve()),
                        "geometry_csv_sha256": sha256(geometry_csv)}


def native_feature_path(root, metadata):
    relative = Path(metadata["feature_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Native feature paths must stay inside the geometry directory")
    return Path(root) / relative


def siglip_native(geometry_dir):
    root = Path(geometry_dir)
    reference_meta = json.loads((root / "imagenet.json").read_text())
    identity = dict(protocol="siglip_native_geometry_v1", status="complete", encoder="SigLIP",
                    backbone=ENCODERS["SigLIP"][0], pool_strategy="map",
                    initialization="public_pretrained", no_cp=True, no_ft=True, no_lp=True,
                    normalization={"mean": [.5] * 3, "std": [.5] * 3})
    for key, value in dict(identity, dataset="imagenet", n_samples=5000).items():
        if reference_meta.get(key) != value:
            raise ValueError(f"Invalid native reference identity: {key}")
    ref_path = native_feature_path(root, reference_meta)
    ref = read_bank(ref_path, expected_hash=reference_meta["feature_file_sha256"], expected_n=5000)
    rows = []
    for dataset, seed in native_keys():
        meta_path = root / "results" / f"{dataset}__seed{seed}.json"
        meta = json.loads(meta_path.read_text())
        expected = dict(identity, dataset=dataset, seed=seed, n_samples=DATASET_META[dataset][2],
                        n_bank=min(DATASET_META[dataset][2], 5000),
                        weights_sha256=reference_meta["weights_sha256"],
                        reference_feature_sha256=reference_meta["feature_file_sha256"])
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Invalid native target identity: {dataset} seed={seed} {key}")
        path = native_feature_path(root, meta)
        x = read_bank(path, expected_hash=meta["feature_file_sha256"], expected_n=meta["n_bank"])
        measured = dict(target_descriptors(x), **reference_descriptors(x, ref))
        for new_key, old_key, tolerance in (("uniformity_t2", "uniformity_t2_subset", 1e-5),
                                            ("mmd_rbf", "mmd_rbf", 1e-5),
                                            ("neighbor_overlap_k50", "neighbor_overlap_k50", 1e-4)):
            if not np.isclose(measured[new_key], meta[old_key], atol=tolerance, rtol=0):
                raise ValueError(f"Native geometry replication mismatch: {dataset} s{seed} {new_key}")
        rows.append(dict(encoder="SigLIP", dataset=dataset, seed=seed, **measured,
                         reference_source="recomputed_from_hashed_native_features",
                         published_uniformity_t2=meta["uniformity_t2_subset"],
                         uniformity_difference=measured["uniformity_t2"] - meta["uniformity_t2_subset"],
                         mmd_difference=measured["mmd_rbf"] - meta["mmd_rbf"],
                         overlap_difference=measured["neighbor_overlap_k50"] - meta["neighbor_overlap_k50"],
                         feature_file=meta["feature_file"], feature_sha256=meta["feature_file_sha256"]))
        print(f"DONE encoder=SigLIP dataset={dataset} seed={seed} n={measured['n_angular']} "
              "all_descriptors=recomputed", flush=True)
    return rows, aggregate_native(rows), {"geometry_dir": str(root.resolve()),
                                          "reference_sha256": reference_meta["feature_file_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("main-grid", "siglip-native"))
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "eval/outputs/int1_features")
    parser.add_argument("--geometry-csv", type=Path, default=ROOT / "eval/outputs/geometry_15.csv")
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.mode == "siglip-native" and args.geometry_dir is None:
        parser.error("--geometry-dir is required for native SigLIP")
    if args.outdir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.outdir}")
    with threadpool_limits(limits=args.threads):
        if args.mode == "main-grid":
            rows, summary, source = main_grid(args.feature_dir, args.geometry_csv)
        else:
            rows, summary, source = siglip_native(args.geometry_dir)
    manifest = dict(protocol=PROTOCOL, source=source, n_splits=len(rows), n_targets=len(summary),
                    script_sha256=sha256(__file__), python=platform.python_version(),
                    numpy=np.__version__)
    write_outputs(args.outdir, rows, summary, manifest)
    print(f"COMPLETE target_splits={len(rows)} target_summaries={len(summary)} outdir={args.outdir}")
    writer = csv.DictWriter(sys.stdout,
                            fieldnames=["encoder", "dataset", "n_splits", *DESCRIPTORS],
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)


if __name__ == "__main__":
    main()
