#!/usr/bin/env python3
"""Full-training-set frozen evaluation: 23 datasets, five encoders, three seeds."""

import argparse
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
ENCODERS = {
    "DINOv3": "vit_base_patch16_dinov3.lvd1689m",
    "CLIP": "vit_base_patch16_clip_224.openai",
    "SigLIP-2": "vit_base_patch16_siglip_224.v2_webli",
    "MAE": "vit_base_patch16_224.mae",
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
    valid = all(row.get(key) == value for key, value in expected.items())
    valid = valid and row.get("n_train_actual", 0) == row.get("n_samples", -1) > 0
    valid = valid and row.get("cp_config", {}).get("knn_k") == 20
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
        for encoder, backbone in ENCODERS.items():
            for seed in SEEDS:
                label = f"{encoder}/{dataset}/seed{seed}"
                output = result_path(args.root, encoder, dataset, seed)
                if completed_result(output, encoder, dataset, seed) is not None:
                    print(f"SKIP {label}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-u",
                    str(REPO / "continued_pretraining.py"),
                    "--no-cp",
                    "--full-train",
                    "--dataset",
                    dataset,
                    "--backbone",
                    backbone,
                    "--seed",
                    str(seed),
                    "--batch-size",
                    "32",
                    "--num-workers",
                    str(args.num_workers),
                    "--knn-k",
                    "20",
                    "--cache-dir",
                    str(args.root / "data"),
                    "--results-json",
                    str(output),
                    "--geometry-reference",
                    str(args.root / "outputs/precp_full/reference" / f"{encoder}.npz"),
                    "--geometry-features",
                    str(
                        args.root
                        / "outputs/precp_full/features"
                        / encoder
                        / dataset
                        / f"seed{seed}.npz"
                    ),
                ]
                if args.dry_run:
                    print(shlex.join(command))
                    continue
                log = args.root / "outputs/precp_full/logs" / encoder / dataset / f"seed{seed}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f"RUN {label} log={log}", flush=True)
                with log.open("a") as stream:
                    stream.write(f"\nCOMMAND {shlex.join(command)}\n")
                    stream.flush()
                    result = subprocess.run(
                        command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT
                    )
                if result.returncode == 0 and completed_result(output, encoder, dataset, seed):
                    print(f"DONE {label}", flush=True)
                else:
                    failures.append(label)
                    print(f"FAILED {label} exit={result.returncode} log={log}", flush=True)
    if failures:
        raise SystemExit("Failed evaluations: " + ", ".join(failures))


def report(root):
    rows, summary = [], []
    for group in GROUPS:
        for dataset in group:
            for encoder in ENCODERS:
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
    for name, records, columns in (
        ("results.csv", rows, fields),
        ("summary.csv", summary, list(summary[0])),
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
    commands.add_parser("list", help="Show Slurm task IDs and dataset groups.")
    run = commands.add_parser("run", help="Run one dataset group, skipping completed results.")
    run.add_argument("--task-id", type=int, choices=range(len(GROUPS)), required=True)
    run.add_argument("--num-workers", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    reports = commands.add_parser(
        "report", help="Export per-seed results and means with sample SD."
    )
    for subparser in (run, reports):
        subparser.add_argument(
            "--root", type=Path, default=Path(os.environ.get("CP_ROOT", REPO.parent))
        )
    args = parser.parse_args()
    if args.command == "list":
        for task_id, group in enumerate(GROUPS):
            print(
                f"{task_id:2d}  {','.join(group)}  evaluations={len(group) * len(ENCODERS) * len(SEEDS)}"
            )
        print("17 jobs; 23 datasets x 5 encoders x 3 seeds = 345 pre-CP evaluations; 0 CP; 0 FT.")
        print(
            "One additional reference job; all five geometry metrics use min(train size, 5000) target images."
        )
    elif args.command == "run":
        args.root = args.root.expanduser().resolve()
        run_group(args)
    else:
        report(args.root.expanduser().resolve())


if __name__ == "__main__":
    main()
