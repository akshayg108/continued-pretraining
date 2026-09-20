"""Recompute SigLIP-2 pre-CP geometry with native mean/std, without training."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from eval.full_ft.manifest import DATASET_META, ENCODERS
from eval.precp_official_norm import (
    _array_hash, _check_features, _provenance, _weights_hash, check_gpu,
    full_uniformity, official_normalization, subset_indices, write_new_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "siglip_native_geometry_v1"
MODEL_ID, POOL = ENCODERS["SigLIP"]
EXPECTED = Path(__file__).with_name("siglip_native_geometry_expected.csv")
U_TOLERANCE = 1e-5
FIELDS = (
    "uniformity_t2", "uniformity_t2_subset", "l2_norm_mean", "l2_norm_std",
    "l2_norm_cv", "full_l2_norm_mean", "full_l2_norm_std", "full_l2_norm_cv",
    "cosine_dist_centroid", "mmd_rbf", "mmd_m_pp_target", "mmd_m_qq_imagenet",
    "mmd_m_pq_cross", "mmd_gamma", "neighbor_overlap_k20", "neighbor_overlap_k50",
)


def build_plan():
    return [dict(dataset=dataset, seed=seed, n_samples=meta[2])
            for dataset, meta in DATASET_META.items()
            for seed in ((42, 43, 44) if dataset == "galaxy10" else (42,))]


def load_expected_uniformity():
    with EXPECTED.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["dataset"], int(row["seed"])): float(row["uniformity_t2"])
              for row in rows}
    expected = {(t["dataset"], t["seed"]) for t in build_plan()}
    if len(rows) != len(result) or set(result) != expected:
        raise ValueError("Invalid baseline uniformity coverage")
    return result


def check_uniformity(dataset, seed, value):
    expected = load_expected_uniformity().get((dataset, seed))
    if (expected is None or not math.isfinite(value)
            or not math.isclose(value, expected, abs_tol=U_TOLERANCE, rel_tol=0)):
        raise ValueError(f"Native baseline uniformity mismatch: {dataset} seed={seed} "
                         f"expected={expected}, observed={value}, tolerance={U_TOLERANCE}")


def bank_indices(labels):
    """Keep the existing <=5000-image, seed-42 stratified geometry bank."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    if len(indices) > 5000:
        indices, _ = train_test_split(indices, train_size=5000,
                                      stratify=labels, random_state=42)
    return np.sort(indices)


def check_features(features, labels, expected_n):
    if (features.shape != (expected_n, 768) or len(labels) != expected_n
            or expected_n < 2 or not np.isfinite(features).all()
            or not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all()
            or np.any(np.linalg.norm(features.astype(np.float64), axis=1) <= 0)):
        raise ValueError("Invalid raw feature bank: expected finite, nonzero 768-D rows and labels")


def write_new_npz(path, **arrays):
    """Atomically publish raw features without clobbering another extraction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".geometry-", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_values(features, labels, reference, *, device):
    from eval.utils.geometry_metrics import (
        cosine_distance_centroids, l2_norm_stats, mmd_rbf_components, neighbor_overlap,
    )

    selected = bank_indices(labels)
    bank = features[selected]
    mean, std, cv = l2_norm_stats(bank)
    full_mean, full_std, full_cv = l2_norm_stats(features)
    values = dict(
        uniformity_t2=full_uniformity(features, device=device),
        uniformity_t2_subset=full_uniformity(features[subset_indices(labels)], device=device),
        l2_norm_mean=mean, l2_norm_std=std, l2_norm_cv=cv,
        full_l2_norm_mean=full_mean, full_l2_norm_std=full_std, full_l2_norm_cv=full_cv,
        cosine_dist_centroid=cosine_distance_centroids(bank, reference),
        neighbor_overlap_k20=neighbor_overlap(bank, reference, k=20),
        neighbor_overlap_k50=neighbor_overlap(bank, reference, k=50),
        **mmd_rbf_components(bank, reference),
    )
    if not all(math.isfinite(values[key]) for key in FIELDS):
        raise ValueError("Non-finite geometry statistic")
    return values, selected


def load_reference(imagenet_dir, transform, num_workers):
    from datasets import DatasetDict, load_from_disk
    from stable_cp.data import HFDatasetWrapper
    from torch.utils.data import DataLoader

    source = load_from_disk(str(imagenet_dir))
    if isinstance(source, DatasetDict):
        if "validation" not in source:
            raise ValueError("ImageNet DatasetDict must contain the validation split")
        source = source["validation"]
    split = getattr(source, "split", None)
    if split is not None and str(split) not in ("validation", "val"):
        raise ValueError(f"Expected ImageNet validation data, received split={split}")
    if len(source) < 5000 or not {"image", "label"}.issubset(source.column_names):
        raise ValueError("Expected an ImageNet validation Dataset with >=5000 labeled images")
    indices = np.sort(np.random.RandomState(42).choice(len(source), 5000, replace=False))
    dataset = HFDatasetWrapper(source.select(indices.tolist()), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    metadata = dict(source=str(Path(imagenet_dir).resolve()), split="validation",
                    source_n_samples=len(source), n_samples=5000, sampling_seed=42,
                    source_fingerprint=getattr(source, "_fingerprint", None),
                    selected_indices_sha256=_array_hash(indices, "<i8"),
                    eval_transform=repr(transform))
    return loader, indices, metadata


def run(cache_dir, imagenet_dir, outdir, *, num_workers=8):
    if num_workers < 0:
        raise ValueError("Worker count must be nonnegative")
    if Path(outdir).exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to mix runs: {outdir}")
    if not Path(imagenet_dir).is_dir():
        raise FileNotFoundError(f"Missing local ImageNet validation dataset: {imagenet_dir}")
    for _, subpath, _ in DATASET_META.values():
        path = Path(cache_dir) / "stable_datasets/processed" / subpath
        if not path.is_dir():
            raise FileNotFoundError(f"Missing processed target cache: {path}")

    import lightning as pl
    import torch
    from continued_pretraining import _create_shared_eval_data, get_dataset_config, load_backbone
    from stable_cp.data import create_transforms
    from stable_cp.evaluation.zero_shot_eval import extract_features

    gpu = check_gpu()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    pl.seed_everything(42, workers=True)
    args = SimpleNamespace(backbone=MODEL_ID, pool_strategy=POOL, batch_size=64,
                           num_workers=num_workers, cache_dir=str(cache_dir))
    backbone, device = load_backbone(args, img_size=224, pretrained=True)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this feature extraction job")
    native = official_normalization("SigLIP", backbone.pretrained_cfg)
    backbone.requires_grad_(False)
    backbone.eval()
    weights_sha256 = _weights_hash(backbone)
    backbone.to(device)
    common = dict(protocol=PROTOCOL, schema_version=1, status="complete", encoder="SigLIP",
                  backbone=MODEL_ID, pool_strategy=POOL, initialization="public_pretrained",
                  no_cp=True, no_ft=True, no_lp=True, feature_precision="float32",
                  feature_batch_size=64, normalization_mode="official_mean_std_only",
                  normalization=native, weights_sha256=weights_sha256, gpu=gpu,
                  **_provenance())
    common["code_sha256"].update({str(path.relative_to(ROOT)): file_hash(path)
                                   for path in (Path(__file__), EXPECTED,
                                                ROOT / "eval/utils/geometry_metrics.py")})
    reference_cfg = dict(input_size=224, normalization=native)
    _, reference_tf = create_transforms(reference_cfg, n_views=1, strong_aug=False)
    loader, reference_indices, reference_meta = load_reference(imagenet_dir, reference_tf, num_workers)
    print("STAGE extract_imagenet_reference n=5000 normalization=official", flush=True)
    reference, reference_y = extract_features(backbone, loader, device, pool_strategy=POOL)
    check_features(reference, reference_y, 5000)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    ref_path = outdir / "features/SigLIP__imagenet.npz"
    write_new_npz(ref_path, bank_X=reference, bank_y=reference_y, bank_indices=reference_indices)
    reference_meta.update(common)
    reference_meta.update(dataset="imagenet", feature_file=str(ref_path.relative_to(outdir)),
                          feature_file_sha256=file_hash(ref_path))
    write_new_json(outdir / "imagenet.json", reference_meta)
    del loader

    for task in build_plan():
        dataset, seed, n = task["dataset"], task["seed"], task["n_samples"]
        print(f"GEOMETRY dataset={dataset} seed={seed} n={n} no_cp=True no_ft=True no_lp=True", flush=True)
        pl.seed_everything(seed, workers=True)
        args.dataset, args.seed, args.n_samples = dataset, seed, n
        cfg = dict(get_dataset_config(dataset), normalization=native)
        eval_tf, test_loader, lp_loader, clean_loader, indices = _create_shared_eval_data(
            args, cfg, Path(cache_dir))
        base_train = clean_loader.dataset.dataset
        if len(indices) != n or len(base_train) != n:
            raise ValueError(f"MAX train split mismatch for {dataset}")
        print("STAGE extract_clean_MAX_train", flush=True)
        features, labels = extract_features(backbone, clean_loader, device, pool_strategy=POOL)
        _check_features(features, labels, n, cfg["num_classes"], dataset, train=True)
        check_features(features, labels, n)
        print("STAGE compute_uniformity_mmd_overlap_norms", flush=True)
        values, selected = geometry_values(features, labels, reference, device=device)
        # Reject silent changes in model, split, pooling, or input preprocessing.
        check_uniformity(dataset, seed, values["uniformity_t2"])
        feature_path = outdir / "features" / f"SigLIP__{dataset}__seed{seed}.npz"
        write_new_npz(feature_path, bank_X=features[selected], bank_y=labels[selected],
                      bank_indices=selected, full_norms=np.linalg.norm(features, axis=1))
        record = dict(common, **task, **values, budget="MAX", n_bank=len(selected),
                      eval_transform=repr(eval_tf), reference_feature_sha256=file_hash(ref_path),
                      train_indices_sha256=_array_hash(indices, "<i8"),
                      train_labels_sha256=_array_hash(labels, "<i8"),
                      train_features_sha256=_array_hash(features, "<f4"),
                      train_fingerprint=getattr(getattr(base_train, "hf_dataset", None), "_fingerprint", None),
                      bank_selection="up to 5000 stratified train samples, sorted indices, seed 42",
                      uniformity_selection="all distinct pairs in clean MAX train",
                      uniformity_expected=load_expected_uniformity()[(dataset, seed)],
                      uniformity_difference=values["uniformity_t2"] - load_expected_uniformity()[(dataset, seed)],
                      feature_file=str(feature_path.relative_to(outdir)),
                      feature_file_sha256=file_hash(feature_path))
        write_new_json(outdir / "results" / f"{dataset}__seed{seed}.json", record)
        print(json.dumps(dict(dataset=dataset, seed=seed, **values)), flush=True)
        del features, labels, test_loader, lp_loader, clean_loader, base_train
    export_results(outdir)
    print(f"DONE geometry={outdir / 'geometry.csv'}", flush=True)


def _validate_file(outdir, record):
    relative = Path(record["feature_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Feature file must be inside the result directory")
    path = Path(outdir) / relative
    if file_hash(path) != record["feature_file_sha256"]:
        raise ValueError(f"Feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        check_features(arrays["bank_X"], arrays["bank_y"], record.get("n_bank", record["n_samples"]))
        if record["dataset"] != "imagenet":
            norms = arrays["full_norms"]
            if norms.shape != (record["n_samples"],) or not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError(f"Invalid raw lengths: {path}")


def export_results(outdir):
    """Export only complete, internally matched extractions; no partial averages."""
    outdir = Path(outdir)
    reference = json.loads((outdir / "imagenet.json").read_text())
    official_normalization("SigLIP", reference.get("normalization", {}))
    shared_identity = dict(protocol=PROTOCOL, status="complete", encoder="SigLIP",
                           backbone=MODEL_ID, pool_strategy=POOL, initialization="public_pretrained",
                           no_cp=True, no_ft=True, no_lp=True, feature_precision="float32")
    for key, value in dict(shared_identity, dataset="imagenet", n_samples=5000).items():
        if reference.get(key) != value:
            raise ValueError(f"Invalid reference identity: {key}")
    _validate_file(outdir, reference)
    rows = []
    for task in build_plan():
        record = json.loads((outdir / "results" / f"{task['dataset']}__seed{task['seed']}.json").read_text())
        expected = dict(task, **shared_identity, n_bank=min(task["n_samples"], 5000),
                        weights_sha256=reference["weights_sha256"],
                        reference_feature_sha256=reference["feature_file_sha256"])
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"Result identity mismatch: {task} {key}")
        official_normalization("SigLIP", record.get("normalization", {}))
        if not all(type(record.get(key)) in (int, float) and math.isfinite(record[key])
                   for key in FIELDS):
            raise ValueError(f"Invalid geometry metrics: {task}")
        if (not -8.000001 <= record["uniformity_t2_subset"] <= .000001
                or any(not 0 <= record[key] <= 1 for key in ("neighbor_overlap_k20", "neighbor_overlap_k50"))
                or any(record[key] < 0 for key in ("l2_norm_std", "l2_norm_cv", "full_l2_norm_std", "full_l2_norm_cv"))
                or any(record[key] <= 0 for key in ("l2_norm_mean", "full_l2_norm_mean", "mmd_gamma"))
                or record["mmd_rbf"] < -1e-6):
            raise ValueError(f"Out-of-range geometry statistic: {task}")
        check_uniformity(task["dataset"], task["seed"], record["uniformity_t2"])
        _validate_file(outdir, record)
        rows.append(record)
    columns = ["encoder", "dataset", "seed", "n_samples", "n_bank", *FIELDS]
    with (outdir / "geometry_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    means = []
    for dataset in DATASET_META:
        group = [row for row in rows if row["dataset"] == dataset]
        means.append(dict(encoder="SigLIP", dataset=dataset, n_seeds=len(group),
                          n_samples=group[0]["n_samples"], n_bank=group[0]["n_bank"],
                          **{key: statistics.mean(row[key] for row in group) for key in FIELDS}))
    with (outdir / "geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    print(f"Validated {len(rows)}/17 target splits, 15/15 datasets, and native ImageNet reference",
          file=sys.stderr)
    return means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cache-dir", type=Path, required=True)
    run_parser.add_argument("--imagenet-dir", type=Path, required=True)
    run_parser.add_argument("--outdir", type=Path, required=True)
    run_parser.add_argument("--num-workers", type=int, default=8)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        for task in build_plan():
            print("PLAN " + " ".join(f"{key}={value}" for key, value in task.items()))
        print("17 target splits + one ImageNet reference; no_cp=True no_ft=True no_lp=True")
    elif args.action == "export":
        rows = export_results(args.outdir)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    else:
        run(args.cache_dir, args.imagenet_dir, args.outdir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
