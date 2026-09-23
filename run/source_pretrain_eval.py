"""Fixed held-out target geometry for source pretraining from random initialization."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import pdist

from stable_cp.evaluation.geometry import _unit_features, select_geometry_indices

PROTOCOL = "source_pretrain_test_geometry_v1"
METRICS = ("uniformity_t2", "mean_pairwise_cos", "rankme_l2_uncentered")
PAIR_CONFIG_KEYS = (
    "protocol",
    "seed",
    "total_steps",
    "architecture",
    "split_seed",
    "loss",
    "batch_size",
    "accumulation",
    "normalization",
    "optimizer",
    "scheduler",
    "warmup_steps",
    "projector",
    "precision",
    "pretrained",
    "trainable_blocks",
    "global_crops",
    "global_size",
    "global_scale",
    "local_crops",
    "local_size",
    "local_scale",
    "photometric",
    "drop_path_rate",
    "activation_checkpointing",
    "readout",
    "sampling",
    "online_evaluation",
    "online_validation",
)
PAIR_GEOMETRY_KEYS = (
    "protocol",
    "dataset",
    "split",
    "split_seed",
    "sampling_seed",
    "readout",
    "normalization",
    "n_test",
    "n_geometry",
    "feature_dim",
    "indices_sha256",
    "source_indices_sha256",
    "dataset_config",
)


def _probe_loss(predictions, labels):
    import torch.nn.functional as F

    # Target-domain images have no labels for the ImageNet-only online probe.
    return F.cross_entropy(predictions, labels, ignore_index=-100, reduction="sum") / (
        labels != -100
    ).sum().clamp_min(1)


def online_callbacks(module, config):
    """Use native probes on detached backbone features and ImageNet labels only."""
    import stable_pretraining as spt
    import torch
    from torchmetrics.classification import MulticlassAccuracy

    settings = config["online_evaluation"]
    n_classes = settings["num_classes"]
    return [
        spt.callbacks.OnlineProbe(
            module,
            name="imagenet_lp",
            input="probe_embedding",
            target="probe_label",
            probe=torch.nn.Linear(module.backbone.num_features, n_classes),
            loss=_probe_loss,
            optimizer=settings["lp_optimizer"],
            metrics={
                "top1": MulticlassAccuracy(n_classes, average="micro", ignore_index=-100),
                "top5": MulticlassAccuracy(n_classes, average="micro", top_k=5, ignore_index=-100),
            },
        ),
        spt.callbacks.OnlineKNN(
            name="imagenet_knn",
            input="knn_embedding",
            target="knn_label",
            queue_length=settings["knn_queue_length"],
            input_dim=module.backbone.num_features,
            target_dim=(),
            num_classes=n_classes,
            k=settings["knn_k"],
            temperature=settings["knn_temperature"],
            distance_metric=settings["knn_distance"],
            metrics={
                "top1": MulticlassAccuracy(n_classes, average="micro"),
                "top5": MulticlassAccuracy(n_classes, average="micro", top_k=5),
            },
        ),
    ]


def evaluate_online(module, loader, config, directory, checkpoint):
    """Restore native probe and queue state in a fresh validation-only Trainer."""
    import lightning as pl
    from lightning.pytorch.loggers import CSVLogger

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config["precision"],
        callbacks=online_callbacks(module, config),
        logger=CSVLogger(str(directory), name="online_final"),
        enable_checkpointing=False,
    )
    metrics = trainer.validate(module, dataloaders=loader, ckpt_path=str(checkpoint))[0]
    _atomic_json(
        directory / "online_metrics.json",
        {
            "global_step": config["total_steps"],
            "n_validation": len(loader.dataset),
            "metrics": metrics,
        },
    )
    print(f"ONLINE FINAL {json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


def geometry_metrics(features):
    """Use unit rows and all unordered pairs, without a self-similarity diagonal."""
    unit = _unit_features(features)
    squared_distances = pdist(unit, metric="sqeuclidean")
    uniformity = float(np.log(np.exp(-2 * squared_distances).mean()))
    n = len(unit)
    total = unit.sum(axis=0)
    mean_cosine = float((total @ total - n) / (n * (n - 1)))
    singular = np.linalg.svd(unit, compute_uv=False)
    probabilities = singular[singular > 0] / singular.sum()
    rankme = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "uniformity_t2": uniformity,
        "mean_pairwise_cos": mean_cosine,
        "rankme_l2_uncentered": rankme,
    }


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(handle.name).replace(path)


def _indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def evaluate_targets(
    backbone, target_cache_dirs, output_dir, *, device, num_workers=8, batch_size=64
):
    """Persist raw final-norm CLS banks and their test-only geometry descriptors."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from stable_cp.data.datasets import get_dataset, get_dataset_config
    from stable_cp.data.heldout import IndexedSplit
    from run.source_pretrain_data import SPLIT_SEED, TARGETS, make_transforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    _, clean = make_transforms()
    results = {}
    was_training = backbone.training
    backbone.eval()
    try:
        for name in TARGETS:
            dataset = get_dataset(
                name, "test", transform=clean, cache_dir=target_cache_dirs[name], seed=SPLIT_SEED
            )
            indices = select_geometry_indices(len(dataset))
            loader = DataLoader(
                Subset(dataset, indices.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            batches = []
            with torch.inference_mode():
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=device.type == "cuda")
                    cls = backbone.forward_features(images)[:, 0]
                    batches.append(cls.to(device="cpu", dtype=torch.float32).numpy())
            features = np.concatenate(batches)
            raw = dataset.hf_dataset
            source_indices = (
                raw.source_indices[indices] if isinstance(raw, IndexedSplit) else indices
            )
            config = get_dataset_config(name)
            metadata = {
                "protocol": PROTOCOL,
                "dataset": name,
                "split": "test",
                "split_seed": SPLIT_SEED,
                "sampling_seed": 42,
                "sampling": "min(5000, n_test), sorted RandomState(42) choice without replacement",
                "source_split": raw.source_split if isinstance(raw, IndexedSplit) else "test",
                "partition": raw.partition if isinstance(raw, IndexedSplit) else "official_test",
                "readout": "forward_features_final_norm_cls",
                "feature_dtype": "float32",
                "stored_features": "raw, no projector or L2 normalization",
                "metric_features": "row L2 normalized, uncentered",
                "normalization": {
                    "color": "RGB",
                    "resize": [224, 224],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "dataset_config": {
                    key: config[key]
                    for key in (
                        "config_name",
                        "num_classes",
                        "input_size",
                        "dataset_kwargs",
                        "manual_split",
                    )
                    if key in config
                },
                "n_test": len(dataset),
                "n_geometry": len(features),
                "feature_dim": features.shape[1],
                "indices_sha256": _indices_hash(indices),
                "source_indices_sha256": _indices_hash(source_indices),
                "features_file": f"{name}.npz",
                **geometry_metrics(features),
            }
            archive_path = output_dir / metadata["features_file"]
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False
            ) as handle:
                np.savez_compressed(
                    handle,
                    features=features,
                    indices=indices,
                    source_indices=source_indices,
                    metadata=json.dumps(metadata, allow_nan=False),
                )
            Path(handle.name).replace(archive_path)
            results[name] = metadata
            print(
                f"GEOMETRY {output_dir.name}/{name}: n={len(features)}/{len(dataset)} "
                f"uniformity_t2={metadata['uniformity_t2']:.8f}",
                flush=True,
            )
    finally:
        backbone.train(was_training)
    _atomic_json(output_dir / "geometry.json", results)
    return results


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _matching(left, right, keys):
    return all(key in left and key in right and left[key] == right[key] for key in keys)


def report_results(output_root: Path):
    """Report completed runs and matched-seed mixed-minus-ImageNet differences."""
    from run.source_pretrain_data import TARGETS

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, pairs, runs = [], [], {}
    for condition in ("imagenet", "mixed"):
        for seed in (42,):
            directory = output_root / condition / f"seed{seed}"
            required = (
                directory / "config.json",
                directory / "training.json",
                directory / "checkpoints/final.ckpt",
                directory / "final/geometry.json",
            )
            if not all(path.is_file() for path in required):
                continue
            config = json.loads(required[0].read_text())
            training = json.loads(required[1].read_text())
            if not training.get("complete") or training.get("global_step") != config["total_steps"]:
                continue
            if config["seed"] != seed or config["condition"] != condition:
                raise ValueError(f"Run configuration does not match its directory: {directory}")
            final = json.loads(required[3].read_text())
            if not all(name in final for name in TARGETS):
                continue
            initial_path = directory / "initial/geometry.json"
            initial = json.loads(initial_path.read_text()) if initial_path.is_file() else {}
            for name in TARGETS:
                last, first = final[name], initial.get(name)
                if last["protocol"] != PROTOCOL or last["split"] != "test":
                    raise ValueError(f"Incompatible target geometry: {directory}/{name}")
                if first is not None and not _matching(first, last, PAIR_GEOMETRY_KEYS):
                    raise ValueError(f"Initial/final target samples differ: {directory}/{name}")
                row = {
                    "condition": condition,
                    "seed": seed,
                    "dataset": name,
                    "n_test": last["n_test"],
                    "n_geometry": last["n_geometry"],
                    "global_step": training["global_step"],
                    "seen_images": json.dumps(training.get("seen_images", {}), sort_keys=True),
                    "initial_uniformity_t2": first["uniformity_t2"] if first else "",
                    **{f"final_{key}": last[key] for key in METRICS},
                }
                rows.append(row)
            runs[(condition, seed)] = (config, initial, final)
    for seed in (42,):
        if not all((condition, seed) in runs for condition in ("imagenet", "mixed")):
            continue
        a_config, a_initial, a_final = runs[("imagenet", seed)]
        b_config, b_initial, b_final = runs[("mixed", seed)]
        if (
            not _matching(a_config, b_config, PAIR_CONFIG_KEYS)
            or a_config.get("imagenet_source") != b_config.get("imagenet_source")
        ):
            print(f"UNPAIRED seed{seed}: source-pretraining configurations differ", flush=True)
            continue
        for name in TARGETS:
            a, b = a_final[name], b_final[name]
            if not _matching(a, b, PAIR_GEOMETRY_KEYS):
                print(f"UNPAIRED seed{seed}/{name}: target geometry protocols differ", flush=True)
                continue
            pairs.append(
                {
                    "seed": seed,
                    "dataset": name,
                    "n_test": a["n_test"],
                    "n_geometry": a["n_geometry"],
                    "imagenet_initial_uniformity_t2": a_initial.get(name, {}).get(
                        "uniformity_t2", ""
                    ),
                    "mixed_initial_uniformity_t2": b_initial.get(name, {}).get("uniformity_t2", ""),
                    **{f"imagenet_final_{key}": a[key] for key in METRICS},
                    **{f"mixed_final_{key}": b[key] for key in METRICS},
                    "delta_uniformity_t2_b_minus_a": b["uniformity_t2"] - a["uniformity_t2"],
                }
            )
    _write_csv(
        output_root / "geometry_results.csv",
        rows,
        (
            "condition",
            "seed",
            "dataset",
            "n_test",
            "n_geometry",
            "global_step",
            "seen_images",
            "initial_uniformity_t2",
            *(f"final_{key}" for key in METRICS),
        ),
    )
    _write_csv(
        output_root / "paired_geometry.csv",
        pairs,
        (
            "seed",
            "dataset",
            "n_test",
            "n_geometry",
            "imagenet_initial_uniformity_t2",
            "mixed_initial_uniformity_t2",
            *(f"imagenet_final_{key}" for key in METRICS),
            *(f"mixed_final_{key}" for key in METRICS),
            "delta_uniformity_t2_b_minus_a",
        ),
    )
    print(
        f"Complete runs: {len(runs)}/2; target results: {len(rows)}/6; paired results: {len(pairs)}/3\n"
        f"Reports: {output_root / 'geometry_results.csv'}, {output_root / 'paired_geometry.csv'}",
        flush=True,
    )
    return {"complete_runs": len(runs), "geometry_rows": len(rows), "paired_rows": len(pairs)}
