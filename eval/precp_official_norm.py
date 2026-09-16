"""Audit frozen MAX representations using checkpoint-native RGB mean/std only."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from eval.full_ft.manifest import DATASET_META, ENCODERS, SEEDS


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "precp_official_norm_v1"
ENCODER_ORDER = ("SigLIP", "CLIP", "DINOv3", "MAE")
EXPECTED_NORMALIZATIONS = {
    "SigLIP": {"mean": [.5, .5, .5], "std": [.5, .5, .5]},
    "CLIP": {"mean": [.48145466, .4578275, .40821073],
             "std": [.26862954, .26130258, .27577711]},
    "DINOv3": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "MAE": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
}
METRICS = ("pre_knn_f1", "pre_linear_f1", "pre_knn_acc", "pre_linear_acc")
GEOMETRY = ("uniformity_t2", "uniformity_t2_subset")


def build_plan(encoder):
    if encoder not in ENCODER_ORDER:
        raise ValueError(f"Unsupported encoder: {encoder}")
    model_id, pool = ENCODERS[encoder]
    tasks = []
    for dataset, (_, subpath, n_samples) in DATASET_META.items():
        geometry_only = encoder == "SigLIP" and dataset == "food101"
        for seed in ((42,) if geometry_only else SEEDS):
            tasks.append(dict(encoder=encoder, dataset=dataset, seed=seed, budget="MAX",
                              model_id=model_id, pool=pool, n_samples=n_samples,
                              processed_subpath=subpath, geometry_only=geometry_only))
    return tasks


def official_normalization(encoder, pretrained_cfg):
    expected = EXPECTED_NORMALIZATIONS[encoder]
    try:
        native = {key: [float(v) for v in pretrained_cfg[key]] for key in ("mean", "std")}
        valid = all(len(native[key]) == 3 and all(
            math.isfinite(a) and math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
            for a, b in zip(native[key], expected[key])) for key in native)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Unexpected {encoder} pretrained normalization: {pretrained_cfg}")
    return native


def full_uniformity(features, *, device, block_size=512):
    """Log mean exp(-2 * squared distance) over all distinct normalized pairs."""
    import numpy as np
    import torch

    features = np.asarray(features)
    if (features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1
            or not np.isfinite(features).all() or block_size < 1):
        raise ValueError("Uniformity requires finite features, at least two rows, and positive blocks")
    with torch.no_grad():
        f = torch.as_tensor(features, dtype=torch.float32, device=device)
        norms = f.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError("Uniformity cannot normalize zero or non-finite feature norms")
        f = f / norms
        n = len(f)
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            # Ordered distinct pairs have the same mean as the upper triangle.
            kernel = f[start:end] @ f.T
            kernel.mul_(-2).add_(2).clamp_(0, 4).mul_(-2).exp_()
            diagonal = torch.arange(end - start, device=device)
            kernel[diagonal, start + diagonal] = 0
            total += kernel.sum(dtype=torch.float64)
        value = math.log(total.item() / (n * (n - 1)))
    if not math.isfinite(value) or not -8.000001 <= value <= .000001:
        raise ValueError(f"Invalid uniformity: {value}")
    return value


def subset_indices(labels):
    """Historical 5000-stratified then 3000-uniform sampling, within this train split."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if len(indices) > 5000:
        indices, _ = train_test_split(indices, train_size=5000, stratify=labels, random_state=42)
        indices = np.sort(indices)
    if len(indices) > 3000:
        indices = indices[np.random.RandomState(42).choice(len(indices), 3000, replace=False)]
    return indices


def write_new_json(path, record):
    """Publish a complete JSON file atomically, never replacing an existing result."""
    payload = json.dumps(record, indent=2, allow_nan=False, default=str) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".audit-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def result_path(outdir, task):
    prefix = "geometry_seed" if task["geometry_only"] else "seed"
    return Path(outdir) / task["encoder"] / task["dataset"] / f"{prefix}{task['seed']}.json"


def validate_result(row, task):
    expected = dict(protocol=PROTOCOL, status="complete", encoder=task["encoder"],
                    dataset=task["dataset"], seed=task["seed"], n_samples=task["n_samples"],
                    backbone=task["model_id"], pool_strategy=task["pool"], budget="MAX",
                    geometry_only=task["geometry_only"], no_cp=True, no_ft=True,
                    initialization="public_pretrained")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Result identity mismatch: {key}")
    official_normalization(task["encoder"], row.get("normalization", {}))
    fields = GEOMETRY if task["geometry_only"] else GEOMETRY + METRICS
    for key in fields:
        value = row.get(key)
        lower, upper = (-8.000001, .000001) if key in GEOMETRY else (0, 1)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper):
            raise ValueError(f"Invalid or missing metric: {key}={value}")


def check_gpu():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This audit requires one allocated CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "V100" not in name:
        raise RuntimeError(f"Expected the requested V100, received {name}")
    return name


def _array_hash(array, dtype):
    import numpy as np

    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def _weights_hash(model):
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _provenance():
    versions = {}
    for package in ("torch", "timm", "lightning", "stable-pretraining", "stable-datasets",
                    "scikit-learn", "torchmetrics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    files = [Path(__file__), ROOT / "continued_pretraining.py", ROOT / "eval/full_ft/manifest.py",
             *sorted((ROOT / "stable_cp").rglob("*.py")),
             *sorted((ROOT / "run/slurm/pre-cp-official").glob("*.sh"))]
    return dict(versions=versions, git_commit=commit.stdout.strip() if commit.returncode == 0 else "unknown",
                code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in files},
                slurm={key: os.environ.get(key) for key in
                       ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")})


def _check_features(features, labels, n, n_classes, name, *, train=False):
    import numpy as np

    if features.shape != (n, 768) or len(labels) != n or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature/label shape or non-finite features: {name}")
    if (not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or np.any(labels < 0) or np.any(labels >= n_classes)):
        raise ValueError(f"Invalid class indices: {name}")
    if train and len(np.unique(labels)) != n_classes:
        raise ValueError(f"Training split is missing classes: {name}")


def run_task(task, *, cache_dir, outdir, num_workers=8, uniformity_block_size=512):
    path = result_path(outdir, task)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    if num_workers < 0 or uniformity_block_size < 1:
        raise ValueError("Invalid worker count or uniformity block size")
    import lightning as pl
    import numpy as np
    import torch
    from continued_pretraining import _create_shared_eval_data, get_dataset_config, load_backbone
    from stable_cp.evaluation.zero_shot_eval import (
        extract_features, knn_evaluate, linear_probe_pytorch_evaluate,
    )

    gpu = check_gpu()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args = SimpleNamespace(dataset=task["dataset"], backbone=task["model_id"],
                           n_samples=task["n_samples"], batch_size=64, num_workers=num_workers,
                           seed=task["seed"], cache_dir=str(cache_dir), pool_strategy=task["pool"])
    pl.seed_everything(args.seed, workers=True)
    cfg = get_dataset_config(args.dataset)
    original_normalization = cfg["normalization"]
    backbone, device = load_backbone(args, img_size=cfg["input_size"], pretrained=True)
    if device.type != "cuda":
        raise RuntimeError("CUDA was not selected for feature extraction")
    native = official_normalization(task["encoder"], getattr(backbone, "pretrained_cfg", {}))
    cfg = {**cfg, "normalization": native}
    backbone.requires_grad_(False)
    backbone.eval()
    weights_sha256 = _weights_hash(backbone)
    backbone.to(device)
    print(json.dumps(dict(encoder=task["encoder"], dataset=args.dataset, seed=args.seed,
                          normalization=native, pool=args.pool_strategy)), flush=True)
    eval_tf, test_loader, lp_loader, knn_loader, indices = _create_shared_eval_data(
        args, cfg, Path(cache_dir))
    base_train = lp_loader.dataset.dataset
    n_train, n_test, n_classes = len(indices), len(test_loader.dataset), cfg["num_classes"]
    if n_train != args.n_samples or len(base_train) != n_train:
        raise ValueError(f"MAX split mismatch: requested={args.n_samples}, subset={n_train}, full={len(base_train)}")

    metrics = {}
    if not task["geometry_only"]:
        print("STAGE extract_lp_train", flush=True)
        train, train_y = extract_features(backbone, lp_loader, device, pool_strategy=args.pool_strategy, verbose=True)
        _check_features(train, train_y, n_train, n_classes, "lp_train", train=True)
        print("STAGE extract_test", flush=True)
        test, test_y = extract_features(backbone, test_loader, device, pool_strategy=args.pool_strategy, verbose=True)
        _check_features(test, test_y, n_test, n_classes, "test")
    print("STAGE extract_clean_train", flush=True)
    clean, clean_y = extract_features(backbone, knn_loader, device, pool_strategy=args.pool_strategy, verbose=True)
    _check_features(clean, clean_y, n_train, n_classes, "clean_train", train=True)
    if not task["geometry_only"]:
        if not np.array_equal(train_y, clean_y):
            raise ValueError("LP and kNN train labels use different ordering")
        print("STAGE knn", flush=True)
        knn = knn_evaluate(clean, clean_y, test, test_y, k=20)
        print("STAGE pytorch_lp", flush=True)
        lp = linear_probe_pytorch_evaluate(train, train_y, test, test_y, device=device,
                                           lr=1e-3, min_epochs=150, min_steps=10000,
                                           batch_size=512, verbose=True)
        metrics = dict(pre_knn_f1=float(knn["knn_f1"]), pre_knn_acc=float(knn["knn_acc"]),
                       pre_linear_f1=float(lp["linear_pytorch_f1"]),
                       pre_linear_acc=float(lp["linear_pytorch_acc"]))
    print(f"STAGE uniformity_all_pairs n={n_train} t=2", flush=True)
    uniformity = full_uniformity(clean, device=device, block_size=uniformity_block_size)
    selected = subset_indices(clean_y)
    subset = full_uniformity(clean[selected], device=device, block_size=uniformity_block_size)
    record = dict(
        protocol=PROTOCOL, schema_version=1, status="complete", stage="pre_cp_evaluation",
        encoder=task["encoder"], dataset=args.dataset, backbone=args.backbone, seed=args.seed,
        initialization="public_pretrained", budget="MAX", n_samples=n_train, n_test=n_test,
        no_cp=True, no_ft=True, geometry_only=task["geometry_only"], pool_strategy=args.pool_strategy,
        feature_batch_size=64, feature_precision="float32", input_size=cfg["input_size"],
        normalization_mode="official_mean_std_only", normalization=native,
        dataset_normalization=original_normalization, splits=cfg["splits"],
        lp_train_transform=repr(base_train.transform), eval_transform=repr(eval_tf),
        pretrained_config=getattr(backbone, "pretrained_cfg", {}), weights_sha256=weights_sha256,
        train_indices_sha256=_array_hash(indices, "<i8"), train_labels_sha256=_array_hash(clean_y, "<i8"),
        train_fingerprint=getattr(getattr(base_train, "hf_dataset", None), "_fingerprint", None),
        uniformity_t2=uniformity, uniformity_t2_subset=subset,
        geometry=dict(source="clean_MAX_train", t=2., l2_normalized=True,
                      n_samples=n_train, ordered_pair_count=n_train * (n_train - 1),
                      self_pairs=False, block_size=uniformity_block_size,
                      subset_n_samples=len(selected), subset_indices_sha256=_array_hash(selected, "<i8"),
                      subset_selection="5000 stratified then 3000 random, both seed 42, within this MAX train split"),
        gpu=gpu, **_provenance(), **metrics)
    if not task["geometry_only"]:
        record.update(knn_k=20, lp_method="pytorch", lp_lr=1e-3, lp_min_epochs=150,
                      lp_min_steps=10000, lp_batch_size=512, test_labels_sha256=_array_hash(test_y, "<i8"))
    validate_result(record, task)
    write_new_json(path, record)
    print(json.dumps(dict(encoder=task["encoder"], dataset=args.dataset, seed=args.seed,
                          uniformity_t2=uniformity, uniformity_t2_subset=subset, **metrics)), flush=True)
    print(f"Results saved to {path}", flush=True)
    return path


def summarize(outdir):
    writer = csv.writer(sys.stdout)
    writer.writerow(["encoder", "dataset", "seed", "status", *METRICS, *GEOMETRY, "note"])
    ok, total = 0, 0
    for encoder in ENCODER_ORDER:
        for task in build_plan(encoder):
            total += 1
            path = result_path(outdir, task)
            try:
                row = json.loads(path.read_text())
                validate_result(row, task)
                state = "GEOMETRY_ONLY" if task["geometry_only"] else "OK"
                writer.writerow([encoder, task["dataset"], task["seed"], state,
                                 *[f"{row[k]:.8f}" if k in row else "" for k in METRICS + GEOMETRY], ""])
                ok += 1
            except (OSError, ValueError, TypeError) as exc:
                writer.writerow([encoder, task["dataset"], task["seed"], "CHECK",
                                 *[""] * (len(METRICS) + len(GEOMETRY)), str(exc)])
    print(f"Validated records: {ok}/{total}", file=sys.stderr)
    return 0 if ok == total else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="Print the fixed per-encoder plan without loading torch")
    plan.add_argument("--encoder", choices=ENCODER_ORDER, required=True)
    plan.add_argument("--tsv", action="store_true")
    run = sub.add_parser("run", help="Evaluate one MAX target/seed using public pretrained weights")
    run.add_argument("--encoder", choices=ENCODER_ORDER, required=True)
    run.add_argument("--dataset", choices=tuple(DATASET_META), required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--uniformity-block-size", type=int, default=512)
    summary = sub.add_parser("summarize", help="Print validated per-seed metrics, including missing records")
    summary.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "summarize":
        return summarize(args.outdir)
    tasks = build_plan(args.encoder)
    if args.action == "plan":
        for task in tasks:
            if args.tsv:
                print("\t".join(str(task[k]) for k in
                                ("dataset", "processed_subpath", "seed", "n_samples", "geometry_only")))
            else:
                mode = "uniformity" if task["geometry_only"] else "knn,pytorch_lp,uniformity"
                print(f"PLAN encoder={args.encoder} dataset={task['dataset']} seed={task['seed']} "
                      f"budget=MAX n={task['n_samples']} normalization=official evaluators={mode}")
        return 0
    selected = [t for t in tasks if t["dataset"] == args.dataset and t["seed"] == args.seed]
    if not selected:
        parser.error("SigLIP Food-101 only requires a seed-42 geometry pass")
    run_task(selected[0], cache_dir=args.cache_dir, outdir=args.outdir, num_workers=args.num_workers,
             uniformity_block_size=args.uniformity_block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
