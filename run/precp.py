#!/usr/bin/env python3
"""Frozen evaluation: 23 datasets, five encoder models, six readouts, three seeds."""

import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from stable_cp.utils.backbone import default_pool_strategy

ENCODERS = {
    "DINOv3": "vit_base_patch16_dinov3.lvd1689m",
    "CLIP": "vit_base_patch16_clip_224.openai",
    "SigLIP-2": "vit_base_patch16_siglip_224.v2_webli",
    "MAE-CLS": "vit_base_patch16_224.mae",
    "MAE-Mean": "vit_base_patch16_224.mae",
    "DINOv3-L": "vit_large_patch16_dinov3.lvd1689m",
}
SEEDS = (42, 43, 44)
GROUPS = (
    (
        "breastmnist",
        "dermamnist",
        "octmnist",
        "organamnist",
        "pathmnist",
        "bloodmnist",
        "tissuemnist",
    ),
    *(
        (name,)
        for name in (
            "galaxy10",
            "eurosat",
            "plant_village",
            "dtd",
            "food101",
            "fgvc_aircraft",
            "cars196",
            "cub200",
            "flowers102",
            "oxford_pet",
            "aid",
            "resisc45",
            "stanford_dogs",
            "jena_flowers30",
            "flavia",
            "ip102",
        )
    ),
)
METRICS = ("pre_knn_f1", "pre_linear_f1", "pre_knn_acc", "pre_linear_acc")
GEOMETRY_METRICS = (
    "uniformity_t2",
    "mean_pairwise_cos",
    "rankme_l2_uncentered",
    "mmd_rbf",
    "neighbor_overlap_k50",
)


def encoder_pool_strategy(encoder):
    if encoder == "MAE-CLS":
        return "cls"
    if encoder == "MAE-Mean":
        return "mean"
    return default_pool_strategy(ENCODERS[encoder])


def result_path(root, encoder, dataset, seed):
    return root / "outputs/precp_full/results" / encoder / dataset / f"seed{seed}.json"


def completed_result(path, encoder, dataset, seed):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    expected = {
        "dataset": dataset,
        "backbone": ENCODERS[encoder],
        "seed": seed,
        "full_train": True,
        "no_cp": True,
        "normalization_mode": "pretrained",
    }
    if ENCODERS[encoder].endswith(".mae"):
        from stable_cp.utils.backbone import feature_readout

        expected["feature_readout"] = feature_readout(
            ENCODERS[encoder], encoder_pool_strategy(encoder)
        )
    valid = all(row.get(key) == value for key, value in expected.items())
    valid = valid and row.get("n_train_actual", 0) == row.get("n_samples", -1) > 0
    valid = valid and row.get("cp_config", {}).get("knn_k") == 20
    if ENCODERS[encoder].endswith(".mae"):
        valid = valid and row.get("cp_config", {}).get("pool_strategy") == encoder_pool_strategy(
            encoder
        )
    valid = valid and all(
        isinstance(row.get(key), (int, float))
        and math.isfinite(row[key])
        and -1e-6 <= row[key] <= 1 + 1e-6
        for key in METRICS
    )
    geometry = row.get("geometry", {})
    valid = valid and geometry.get("protocol") == "precp_geometry_5000_v1"
    valid = valid and geometry.get("n_geometry") == min(row.get("n_train_actual", 0), 5000)
    valid = valid and geometry.get("n_reference") == 5000
    valid = valid and all(
        isinstance(geometry.get(key), (int, float)) and math.isfinite(geometry[key])
        for key in GEOMETRY_METRICS
    )
    if not valid:
        raise ValueError(f"Result does not match the full pre-CP grid: {path}")
    return row


def run_group(args):
    if not args.dry_run:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; submit this command through Slurm.")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    failures = []
    for dataset in GROUPS[args.task_id]:
        pending = []
        for encoder in args.encoder:
            for seed in SEEDS:
                if completed_result(result_path(args.root, encoder, dataset, seed), encoder, dataset, seed):
                    print(f"SKIP {encoder}/{dataset}/seed{seed}", flush=True)
                else:
                    pending.append((encoder, seed))
        if not pending:
            continue
        cache_dir = args.root / "data"
        staging = nullcontext(cache_dir)
        if args.stage_data:
            if args.dry_run:
                print(f"STAGE {dataset}: shared cache -> node-local storage (dry run)")
            else:
                from data_cache import staged_dataset

                staging = staged_dataset(cache_dir, dataset)
        with staging as local:
            failures.extend(run_dataset(args, dataset, pending, local))
    if failures:
        raise SystemExit("Failed evaluations: " + ", ".join(failures))


def run_dataset(args, dataset, pending, cache_dir):
    failures = []
    for encoder, seed in pending:
        label = f"{encoder}/{dataset}/seed{seed}"
        output = result_path(args.root, encoder, dataset, seed)
        command = [
            sys.executable,
            "-u",
            str(REPO / "continued_pretraining.py"),
            "--no-cp",
            "--full-train",
            "--dataset",
            dataset,
            "--backbone",
            ENCODERS[encoder],
            "--seed",
            str(seed),
            "--batch-size",
            "32",
            "--num-workers",
            str(args.num_workers),
            "--knn-k",
            "20",
            "--cache-dir",
            str(cache_dir),
            "--results-json",
            str(output),
            "--geometry-reference",
            str(args.root / "outputs/precp_full/reference" / f"{encoder}.npz"),
            "--geometry-features",
            str(args.root / "outputs/precp_full/features" / encoder / dataset / f"seed{seed}.npz"),
        ]
        if ENCODERS[encoder].endswith(".mae"):
            command.extend(("--pool-strategy", encoder_pool_strategy(encoder)))
        if args.dry_run:
            print(shlex.join(command))
            continue
        log = args.root / "outputs/precp_full/logs" / encoder / dataset / f"seed{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"RUN {label} log={log}", flush=True)
        with log.open("a") as stream:
            stream.write(f"\nCOMMAND {shlex.join(command)}\n")
            stream.flush()
            result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0 and completed_result(output, encoder, dataset, seed):
            print(f"DONE {label}", flush=True)
        else:
            failures.append(label)
            print(f"FAILED {label} exit={result.returncode} log={log}", flush=True)
    return failures


def report(root, encoders=None):
    encoders = tuple(encoders) if encoders is not None else tuple(ENCODERS)
    rows, summary = [], []
    for group in GROUPS:
        for dataset in group:
            for encoder in encoders:
                found = []
                for seed in SEEDS:
                    path = result_path(root, encoder, dataset, seed)
                    row = completed_result(path, encoder, dataset, seed)
                    record = {
                        "encoder": encoder,
                        "dataset": dataset,
                        "seed": seed,
                        "status": "OK" if row else "MISSING",
                        "source": str(path),
                    }
                    if row:
                        record.update({key: row[key] for key in METRICS})
                        record.update({key: row[key] for key in ("n_train_actual", "n_test")})
                        record.update(
                            {
                                key: row["geometry"][key]
                                for key in (*GEOMETRY_METRICS, "n_geometry", "n_reference")
                            }
                        )
                        found.append(row)
                    rows.append(record)
                cell = {"encoder": encoder, "dataset": dataset, "n_seeds": len(found)}
                for key in (*METRICS, *GEOMETRY_METRICS):
                    scores = [row[key] if key in METRICS else row["geometry"][key] for row in found]
                    cell[f"{key}_mean"] = statistics.mean(scores) if scores else ""
                    cell[f"{key}_sd"] = statistics.stdev(scores) if len(scores) > 1 else ""
                summary.append(cell)

    directory = root / "outputs/precp_full"
    directory.mkdir(parents=True, exist_ok=True)
    fields = [
        "encoder",
        "dataset",
        "seed",
        "status",
        "n_train_actual",
        "n_test",
        "n_geometry",
        "n_reference",
        *METRICS,
        *GEOMETRY_METRICS,
        "source",
    ]
    suffix = "" if encoders == tuple(ENCODERS) else "." + "_".join(encoders)
    for name, records, columns in (
        (f"results{suffix}.csv", rows, fields),
        (f"summary{suffix}.csv", summary, list(summary[0])),
    ):
        path = directory / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        print(path)
    n_ok = sum(row["status"] == "OK" for row in rows)
    print(f"Completed: {n_ok}/{len(rows)}; means use completed seeds; SD is sample SD.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show Slurm task IDs and dataset groups.")
    run = commands.add_parser("run", help="Run one dataset group, skipping completed results.")
    run.add_argument("--task-id", type=int, choices=range(len(GROUPS)), required=True)
    run.add_argument("--num-workers", type=int, default=2)
    run.add_argument("--stage-data", action="store_true", help="Prepare shared caches and copy data to node-local storage.")
    run.add_argument("--dry-run", action="store_true")
    reports = commands.add_parser(
        "report", help="Export per-seed results and means with sample SD."
    )
    for subparser in (listing, run, reports):
        subparser.add_argument(
            "--encoder",
            nargs="+",
            choices=tuple(ENCODERS),
            help="Select readout variants (default: all).",
        )
    for subparser in (run, reports):
        subparser.add_argument(
            "--root", type=Path, default=Path(os.environ.get("CP_ROOT", REPO.parent))
        )
    args = parser.parse_args()
    args.encoder = tuple(
        encoder for encoder in ENCODERS if args.encoder is None or encoder in args.encoder
    )
    if args.command == "list":
        for task_id, group in enumerate(GROUPS):
            print(
                f"{task_id:2d}  {','.join(group)}  evaluations={len(group) * len(args.encoder) * len(SEEDS)}"
            )
        n_datasets = sum(len(group) for group in GROUPS)
        n_models = len({ENCODERS[encoder] for encoder in args.encoder})
        n_evaluations = n_datasets * len(args.encoder) * len(SEEDS)
        print(
            f"{len(GROUPS)} jobs; {n_datasets} datasets x {len(args.encoder)} readouts "
            f"({n_models} encoder models) x {len(SEEDS)} seeds = {n_evaluations} "
            "pre-CP evaluations; 0 CP; 0 FT."
        )
        print(
            "One additional reference job; all five geometry metrics use min(train size, 5000) target images."
        )
    elif args.command == "run":
        args.root = args.root.expanduser().resolve()
        run_group(args)
    else:
        report(args.root.expanduser().resolve(), args.encoder)


if __name__ == "__main__":
    main()
