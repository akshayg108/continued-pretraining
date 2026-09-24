#!/usr/bin/env python3
"""Full-budget CP on eleven small training splits, with three seeds per job."""

import argparse
from contextlib import ExitStack
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from precp import (
    ENCODERS as PRE_ENCODERS,
    GEOMETRY_METRICS,
    METRICS as PRE_METRICS,
    REPO,
    SEEDS,
    completed_result as pre_result,
    encoder_pool_strategy,
    result_path as pre_path,
)
from stable_cp.utils.backbone import feature_readout
from stable_cp.utils.lp_protocol import LP_DIRECTORY, LP_PROTOCOL, lp_config

DATASETS = {
    "breastmnist": 546,
    "dermamnist": 7007,
    "dtd": 1880,
    "fgvc_aircraft": 3334,
    "cars196": 8144,
    "cub200": 5994,
    "flowers102": 1020,
    "oxford_pet": 3680,
    "aid": 8000,
    "jena_flowers30": 1183,
    "flavia": 1525,
}
ENCODERS = {
    "DINOv3-B": "DINOv3",
    "CLIP": "CLIP",
    "SigLiP-2": "SigLIP-2",
    "DINOv3-L": "DINOv3-L",
    "MAE": "MAE-Mean",
}
METHODS = {"LeJEPA-CP": "lejepa", "SimCLR-CP": "simclr", "DIET-CP": "diet", "MAE-CP": "mae"}
TASKS = tuple(product(ENCODERS, DATASETS, METHODS))
PROTOCOL = "cp_full_small_v1"
POST_METRICS = tuple(key.replace("pre_", "post_") for key in PRE_METRICS)


def gpu_for(task):
    encoder, _, method = task
    return "a100" if encoder == "DINOv3-L" or method == "LeJEPA-CP" else "v100"


def encoder_readout(encoder):
    alias = ENCODERS[encoder]
    return feature_readout(PRE_ENCODERS[alias], encoder_pool_strategy(alias))


def recipe(task):
    encoder, _, method = task
    config = {
        "cp_method": METHODS[method],
        "epochs": 150,
        "freeze_epochs": 15,
        "warmup_epochs": 15,
        "num_trained_blocks": 2,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "knn_k": 20,
        "batch_size": 256,
        "accumulate_grad_batches": 1,
        "eval_batch_size": 32,
        "eval_num_workers": 2,
        "pool_strategy": encoder_pool_strategy(ENCODERS[encoder]),
    }
    if method in {"LeJEPA-CP", "SimCLR-CP"}:
        config.update(proj_dim=128, hidden_dim=2048)
        if encoder == "DINOv3-L":
            config.update(batch_size=128, accumulate_grad_batches=2)
    if method == "LeJEPA-CP":
        config.update(
            n_views=8,
            lamb=0.05,
            num_slices=1024,
            t_max=3.0,
            n_points=17,
            multivariate_test="slicing",
            univariate_test="epps_pulley",
            reduction="mean",
        )
    elif method == "SimCLR-CP":
        config.update(temperature=0.5)
    elif method == "DIET-CP":
        config.update(
            batch_size=32,
            label_smoothing=0.3,
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            mixup_cutmix_prob=0.0,
            mixup_cutmix_switch_prob=0.5,
        )
    else:
        config.update(decoder_dim=512, decoder_depth=4, mask_ratio=0.75)
    return config


def seed_dir(root, task, seed):
    encoder, dataset, method = task
    return root / "outputs/results" / dataset / encoder / method / "Full" / str(seed)


def evaluation_dir(root, task, seed):
    return seed_dir(root, task, seed) / LP_DIRECTORY


def command_for(root, task, seed, cache_dir, reference_dir, workers):
    encoder, dataset, _ = task
    output = evaluation_dir(root, task, seed)
    command = [
        sys.executable,
        "-u",
        str(REPO / "continued_pretraining.py"),
        "--full-train",
        "--skip-baseline",
        "--resume",
        "--dataset",
        dataset,
        "--backbone",
        PRE_ENCODERS[ENCODERS[encoder]],
        "--seed",
        str(seed),
        "--num-workers",
        str(workers),
        "--cache-dir",
        str(cache_dir),
        "--checkpoint-path",
        str(seed_dir(root, task, seed) / "cp.ckpt"),
        "--results-json",
        str(output / "result.json"),
        "--post-geometry-reference-data",
        str(reference_dir),
        "--post-geometry-dir",
        str(output),
    ]
    for key, value in recipe(task).items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    lp = lp_config()
    for key in ("epochs", "batch_size", "lr", "forward_batch_size"):
        if lp[key] is not None:
            command.extend(("--lp-" + key.replace("_", "-"), str(lp[key])))
    return command


def baseline(root, task, seed):
    encoder, dataset, _ = task
    path = pre_path(root, ENCODERS[encoder], dataset, seed)
    row = pre_result(path, ENCODERS[encoder], dataset, seed)
    if row is None or row["n_train_actual"] != DATASETS[dataset]:
        raise ValueError(f"Missing or incorrect full-training baseline: {path}")
    pool = encoder_pool_strategy(ENCODERS[encoder])
    # Older baseline JSONs record readout only as cp_config.pool_strategy.
    if (
        row["cp_config"].get("pool_strategy") != pool
        or row.get("feature_readout", pool) != encoder_readout(encoder)
    ):
        raise ValueError(f"Baseline feature readout does not match CP: {path}")
    return row, path, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_reference(root, encoders):
    import numpy as np
    from precp_reference import validate_prepared

    prepared = root / "data/imagenet_reference_5000"
    manifest = json.loads((prepared / "metadata.json").read_text())
    indices = manifest["indices"]
    source = manifest["source_metadata"]
    expected = np.sort(
        np.random.RandomState(42).choice(source["n_source_images"], 5000, replace=False)
    )
    if (
        manifest.get("protocol") != "precp_geometry_5000_v1"
        or manifest.get("n_reference") != 5000
        or manifest.get("sampling_seed") != 42
        or not np.array_equal(indices, expected)
    ):
        raise ValueError(f"Incorrect fixed ImageNet selection: {prepared}")
    validate_prepared(prepared, manifest)
    for encoder in encoders:
        path = root / "outputs/precp_full/reference" / f"{ENCODERS[encoder]}.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            if (
                not np.array_equal(archive["indices"], indices)
                or metadata.get("backbone") != PRE_ENCODERS[ENCODERS[encoder]]
                or metadata.get("pool_strategy") != encoder_pool_strategy(ENCODERS[encoder])
                or metadata.get("feature_readout", metadata.get("pool_strategy"))
                != encoder_readout(encoder)
                or metadata.get("source_fingerprint") != source.get("source_fingerprint")
            ):
                raise ValueError(f"Pre/post reference images do not match: {path}")
    return prepared


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_config(task, seed, digest):
    encoder, dataset, method = task
    return dict(
        protocol=PROTOCOL,
        encoder=encoder,
        dataset=dataset,
        method=method,
        seed=seed,
        full_train=True,
        sft=False,
        gpu=gpu_for(task),
        backbone=PRE_ENCODERS[ENCODERS[encoder]],
        baseline_sha256=digest,
        n_train=DATASETS[dataset],
        recipe=recipe(task),
    )


def run_config(task, seed, digest):
    return dict(
        checkpoint_config(task, seed, digest),
        lp_protocol=LP_PROTOCOL,
        pre_lp=lp_config(),
        post_lp=lp_config(),
    )


def validate_checkpoint_config(root, task, seed):
    path = seed_dir(root, task, seed) / "config.json"
    if not path.is_file():
        raise ValueError(f"Existing checkpoint artifacts have no run configuration: {path.parent}")
    actual = json.loads(path.read_text())
    expected = checkpoint_config(task, seed, None)
    # Only the evaluation baseline changed; all checkpoint training provenance must match.
    actual = {key: value for key, value in actual.items() if key != "baseline_sha256"}
    expected.pop("baseline_sha256")
    if actual != expected:
        raise ValueError(f"Existing checkpoint has a different CP recipe or identity: {path}")


def prepare_run(root, task, seed, digest):
    checkpoint = seed_dir(root, task, seed)
    checkpoint.mkdir(parents=True, exist_ok=True)
    if (checkpoint / "config.json").exists():
        validate_checkpoint_config(root, task, seed)
    elif any(checkpoint.iterdir()):
        raise ValueError(f"Existing checkpoint artifacts have no run configuration: {checkpoint}")
    else:
        write_json(checkpoint / "config.json", checkpoint_config(task, seed, digest))

    path = evaluation_dir(root, task, seed)
    path.mkdir(exist_ok=True)
    config = run_config(task, seed, digest)
    if (path / "config.json").exists():
        if json.loads((path / "config.json").read_text()) != config:
            raise ValueError(f"Existing LP evaluation has a different recipe or baseline: {path}")
    elif any(path.iterdir()):
        raise ValueError(f"Existing LP artifacts have no run configuration: {path}")
    else:
        write_json(path / "config.json", config)
    return path


def completed_result(root, task, seed, pre, digest, *, allow_unattached=False):
    path = evaluation_dir(root, task, seed)
    result = path / "result.json"
    if not result.is_file():
        return None
    validate_checkpoint_config(root, task, seed)
    row = json.loads(result.read_text())
    encoder, dataset, method = task
    expected = dict(
        dataset=dataset,
        backbone=PRE_ENCODERS[ENCODERS[encoder]],
        method=METHODS[method],
        seed=seed,
        full_train=True,
        no_cp=False,
        epochs=150,
        n_samples=DATASETS[dataset],
        n_train_actual=DATASETS[dataset],
        n_test=pre["n_test"],
        num_classes=pre["num_classes"],
        normalization_mode="pretrained",
        normalization=pre["normalization"],
        feature_readout=encoder_readout(encoder),
        post_lp=lp_config(),
    )
    config = row.get("cp_config", {})
    valid = all(row.get(key) == value for key, value in expected.items())
    valid = valid and pre.get("pre_lp") == lp_config()
    attached = "baseline_sha256" in row
    if attached:
        valid = valid and row["baseline_sha256"] == digest and row.get("pre_lp") == lp_config()
    elif row.get("pre_lp") is not None:
        valid = valid and row["pre_lp"] == lp_config()
    valid = valid and all(config.get(key) == value for key, value in recipe(task).items())
    valid = valid and not config.get("pre_cp_sft", False) and not config.get("post_cp_sft", False)
    valid = valid and all(
        isinstance(row.get(key), (int, float))
        and math.isfinite(row[key])
        and -1e-6 <= row[key] <= 1 + 1e-6
        for key in POST_METRICS
    )
    geometry = row.get("post_geometry", {})
    valid = valid and all(
        geometry.get(key) == value
        for key, value in {
            "protocol": "precp_geometry_5000_v1",
            "n_geometry": min(DATASETS[dataset], 5000),
            "n_reference": 5000,
            "phase": "post",
            "reference_encoder": "post_cp",
        }.items()
    )
    valid = valid and all(
        isinstance(geometry.get(key), (int, float)) and math.isfinite(geometry[key])
        for key in GEOMETRY_METRICS
    )
    if encoder == "MAE":
        valid = valid and geometry.get("feature_readout") == encoder_readout(encoder)
    valid = valid and (seed_dir(root, task, seed) / "cp.ckpt").is_file()
    valid = valid and all(
        (path / name).is_file() for name in ("post_reference.npz", "post_features.npz")
    )
    config_path = path / "config.json"
    valid = valid and config_path.is_file()
    valid = valid and json.loads(config_path.read_text()) == run_config(task, seed, digest)
    if not valid:
        raise ValueError(f"Result does not match this CP recipe: {result}")
    # Interrupted attachment is pending, never a completed run to skip.
    return row if attached or allow_unattached else None


def attach_baseline(row, pre, source, digest):
    if pre.get("pre_lp") != lp_config() or row.get("post_lp") != lp_config():
        raise ValueError("Cannot mix legacy or different pre/post linear-probe protocols")
    if row.get("pre_lp") is not None and row["pre_lp"] != lp_config():
        raise ValueError("Existing result has a different pre-CP linear-probe protocol")
    if "baseline_sha256" in row and (
        row["baseline_sha256"] != digest or row.get("pre_lp") != lp_config()
    ):
        raise ValueError("Existing result belongs to a different baseline or LP protocol")
    row.update({key: pre[key] for key in PRE_METRICS})
    row.update(
        protocol=PROTOCOL,
        baseline_file=str(source),
        baseline_sha256=digest,
        lp_protocol=LP_PROTOCOL,
        pre_lp=pre["pre_lp"],
        pre_geometry=pre["geometry"],
    )
    row["delta"] = {key[4:]: row[key.replace("pre_", "post_")] - pre[key] for key in PRE_METRICS}
    row["delta_geometry"] = {
        key: row["post_geometry"][key] - pre["geometry"][key] for key in GEOMETRY_METRICS
    }
    return row


def run_task(args):
    task = TASKS[args.task_id]
    encoder, dataset, method = task
    if args.dry_run:
        print(f"STAGE {dataset} and ImageNet reference to node-local storage")
        for seed in SEEDS:
            print(
                shlex.join(
                    command_for(
                        args.root,
                        task,
                        seed,
                        args.root / "data",
                        args.root / "data/imagenet_reference_5000",
                        args.num_workers,
                    )
                )
            )
        return
    import torch
    from filelock import FileLock
    from data_cache import staged_dataset, staged_directory

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit through Slurm.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; requested={gpu_for(task)}", flush=True)
    directory = seed_dir(args.root, task, SEEDS[0]).parent
    directory.mkdir(parents=True, exist_ok=True)
    failures = []
    with FileLock(str(directory / ".run.lock"), timeout=0), ExitStack() as stack:
        pending = []
        for seed in SEEDS:
            pre, source, digest = baseline(args.root, task, seed)
            path = prepare_run(args.root, task, seed, digest)
            row = completed_result(args.root, task, seed, pre, digest)
            if row:
                write_json(path / "result.json", attach_baseline(row, pre, source, digest))
                print(f"SKIP {encoder}/{dataset}/{method}/{seed}", flush=True)
            else:
                pending.append((seed, pre, source, digest))
        if not pending:
            return
        reference = validate_reference(args.root, (encoder,))
        local_data = stack.enter_context(staged_dataset(args.root / "data", dataset))
        local_reference = stack.enter_context(staged_directory(reference))
        for seed, pre, source, digest in pending:
            label = f"{encoder}/{dataset}/{method}/{seed}"
            path = evaluation_dir(args.root, task, seed)
            command = command_for(
                args.root, task, seed, local_data, local_reference, args.num_workers
            )
            log = path / "run.log"
            print(f"RUN {label} log={log}", flush=True)
            with log.open("a") as stream:
                stream.write(f"\nCOMMAND {shlex.join(command)}\n")
                stream.flush()
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            row = (
                completed_result(args.root, task, seed, pre, digest, allow_unattached=True)
                if result.returncode == 0
                else None
            )
            if row:
                write_json(path / "result.json", attach_baseline(row, pre, source, digest))
                print(f"DONE {label}", flush=True)
            else:
                failures.append(label)
                print(f"FAILED {label} exit={result.returncode} log={log}", flush=True)
    if failures:
        raise SystemExit("Failed CP evaluations: " + ", ".join(failures))


def report(root, encoders=None):
    encoders = tuple(ENCODERS) if encoders is None else tuple(encoders)
    records, summaries = [], []
    metrics = (
        *PRE_METRICS,
        *POST_METRICS,
        *(f"{stage}_{key}" for stage in ("pre", "post", "delta") for key in GEOMETRY_METRICS),
        *(key.replace("pre_", "delta_") for key in PRE_METRICS),
    )
    for task in TASKS:
        encoder, dataset, method = task
        if encoder not in encoders:
            continue
        found = []
        for seed in SEEDS:
            result = evaluation_dir(root, task, seed) / "result.json"
            row = None
            if result.is_file():
                pre, source, digest = baseline(root, task, seed)
                row = completed_result(root, task, seed, pre, digest)
            record = dict(
                encoder=encoder,
                dataset=dataset,
                method=method,
                seed=seed,
                status="OK" if row else "MISSING",
                source=str(result),
            )
            if row:
                row = attach_baseline(row, pre, source, digest)
                record.update({key: row[key] for key in (*PRE_METRICS, *POST_METRICS)})
                for stage in ("pre", "post", "delta"):
                    record.update(
                        {
                            f"{stage}_{key}": row[f"{stage}_geometry"][key]
                            for key in GEOMETRY_METRICS
                        }
                    )
                record.update({f"delta_{key}": value for key, value in row["delta"].items()})
                found.append(record)
            records.append(record)
        summary = dict(
            encoder=encoder,
            dataset=dataset,
            method=method,
            n_train=DATASETS[dataset],
            seeds=len(found),
        )
        for key in metrics:
            values = [row[key] for row in found]
            summary[f"{key}_mean"] = statistics.mean(values) if values else ""
            summary[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summaries.append(summary)
    destination = root / "outputs/results" / LP_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "" if encoders == tuple(ENCODERS) else "." + "_".join(encoders)
    for name, rows in (
        (f"cp_full_results{suffix}.csv", records),
        (f"cp_full_summary{suffix}.csv", summaries),
    ):
        path = destination / name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(path)
    print(
        f"Completed: {sum(row['status'] == 'OK' for row in records)}/{len(records)}; means use completed seeds; SD is sample SD."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "array"):
        command = sub.add_parser(name)
        command.add_argument("--gpu", choices=("v100", "a100"), required=name == "array")
        command.add_argument("--encoder", nargs="+", choices=tuple(ENCODERS))
    for name in ("check", "run", "report"):
        command = sub.add_parser(name)
        command.add_argument(
            "--root", type=Path, default=Path(os.environ.get("CP_ROOT", REPO.parent))
        )
        if name == "run":
            command.add_argument("--task-id", type=int, required=True, choices=range(len(TASKS)))
            command.add_argument("--num-workers", type=int, default=8)
            command.add_argument("--dry-run", action="store_true")
        else:
            command.add_argument("--encoder", nargs="+", choices=tuple(ENCODERS))
    args = parser.parse_args()
    encoders = tuple(
        encoder for encoder in ENCODERS
        if getattr(args, "encoder", None) is None or encoder in args.encoder
    )
    if args.command in {"list", "array"}:
        selected = [
            (i, task)
            for i, task in enumerate(TASKS)
            if task[0] in encoders and (args.gpu is None or gpu_for(task) == args.gpu)
        ]
        if args.command == "array":
            print(",".join(str(i) for i, _ in selected))
        else:
            for i, task in selected:
                print(f"{i:3d} {gpu_for(task):5s} {' / '.join(task)} seeds=42,43,44")
            print(f"{len(selected)} jobs; {len(selected) * 3} CP runs; Full; no FT.")
        return
    args.root = args.root.expanduser().resolve()
    if args.command == "run":
        run_task(args)
    elif args.command == "report":
        report(args.root, encoders)
    else:
        for encoder, dataset, seed in product(encoders, DATASETS, SEEDS):
            baseline(args.root, (encoder, dataset, "LeJEPA-CP"), seed)
        validate_reference(args.root, encoders)
        count = len(encoders) * len(DATASETS) * len(SEEDS)
        print(f"READY: {count} full-training baselines and the same 5000 ImageNet reference images.")


if __name__ == "__main__":
    main()
