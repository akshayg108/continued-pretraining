#!/usr/bin/env python3
"""Rerun frozen linear probes, without CP training, kNN, geometry, or FT."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import cp_full
import precp
from stable_cp.utils.backbone import feature_readout
from stable_cp.utils.lp_protocol import LP_DIRECTORY, lp_config

PRE_TASKS = tuple(dataset for group in precp.GROUPS for dataset in group)
PRE_ENCODERS = ("DINOv3", "CLIP", "SigLIP-2", "MAE-Mean", "DINOv3-L")
REPO = Path(__file__).resolve().parents[1]


def evaluations(phase, task_id):
    if phase == "pre":
        return [
            (encoder, PRE_TASKS[task_id], None, seed)
            for encoder in PRE_ENCODERS
            for seed in precp.SEEDS
        ]
    return [(*cp_full.TASKS[task_id], seed) for seed in precp.SEEDS]


def expected_steps(task):
    recipe = cp_full.recipe(task)
    return recipe["epochs"] * math.ceil(
        cp_full.DATASETS[task[1]] / (recipe["batch_size"] * recipe["accumulate_grad_batches"])
    )


def checkpoint_status(path, task):
    """Read CP progress; loading a partial checkpoint never resumes training."""
    if not path.is_file():
        return "missing"
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    epochs = cp_full.recipe(task)["epochs"]
    progress = saved.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {})
    processed = progress.get("current", {}).get("processed", -1)
    if saved.get("epoch", -1) < epochs - 1 or processed < epochs:
        return "incomplete"
    if saved.get("global_step") != expected_steps(task):
        raise ValueError(
            f"Completed checkpoint has {saved.get('global_step')} updates; "
            f"expected {expected_steps(task)}: {path}"
        )
    return "complete"


def available_tasks(root, phase, gpu=None):
    if phase == "pre":
        return list(range(len(PRE_TASKS))) if gpu in (None, "v100") else []
    return [
        index
        for index, task in enumerate(cp_full.TASKS)
        if (gpu is None or cp_full.gpu_for(task) == gpu)
        and any(
            checkpoint_status(cp_full.seed_dir(root, task, seed) / "cp.ckpt", task) == "complete"
            for seed in precp.SEEDS
        )
    ]


def output_path(root, phase, encoder, dataset, method, seed):
    if phase == "pre":
        return (
            root
            / "outputs/precp_full"
            / LP_DIRECTORY
            / "lp_results"
            / encoder
            / dataset
            / f"seed{seed}.json"
        )
    return cp_full.seed_dir(root, (encoder, dataset, method), seed) / LP_DIRECTORY / "lp.json"


def identity(args, encoder, dataset, method, seed):
    alias = encoder if args.phase == "pre" else cp_full.ENCODERS[encoder]
    backbone = precp.ENCODERS[alias]
    pool = precp.encoder_pool_strategy(alias)
    row = dict(
        evaluation="lp_only",
        phase=args.phase,
        encoder=encoder,
        dataset=dataset,
        seed=seed,
        backbone=backbone,
        pool_strategy=pool,
        feature_readout=feature_readout(backbone, pool),
    )
    row[f"{args.phase}_lp"] = lp_config(
        args.epochs, args.batch_size, args.lr, args.forward_batch_size
    )
    if args.phase == "post":
        row["method"] = method
        row["checkpoint"] = str(
            cp_full.seed_dir(args.root, (encoder, dataset, method), seed) / "cp.ckpt"
        )
    return row


def completed_result(path, expected):
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except (ValueError, OSError):
        return False
    return all(row.get(key) == value for key, value in expected.items()) and all(
        isinstance(row.get(f"{expected['phase']}_linear_{metric}"), (int, float))
        for metric in ("acc", "f1", "auroc")
    )


def stage_dataset(cache_dir, dataset):
    from data_cache import staged_dataset

    return staged_dataset(cache_dir, dataset)


def evaluation_command(args, run, cache_dir):
    encoder, _, _, seed = run
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "evaluate",
        "--phase",
        args.phase,
        "--root",
        str(args.root),
        "--task-id",
        str(args.task_id),
        "--encoder",
        encoder,
        "--seed",
        str(seed),
        "--cache-dir",
        str(cache_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.forward_batch_size is not None:
        command.extend(("--forward-batch-size", str(args.forward_batch_size)))
    return command


def run_task(args):
    pending = []
    for run in evaluations(args.phase, args.task_id):
        encoder, dataset, method, seed = run
        label = "/".join(str(value) for value in run if value is not None)
        if args.phase == "post":
            task = (encoder, dataset, method)
            checkpoint = cp_full.seed_dir(args.root, task, seed) / "cp.ckpt"
            status = checkpoint_status(checkpoint, task)
            if status != "complete":
                print(f"SKIP {label} checkpoint={status}", flush=True)
                continue
        output = output_path(args.root, args.phase, *run)
        if completed_result(output, identity(args, *run)):
            print(f"SKIP {label} LP already complete", flush=True)
        else:
            pending.append(run)
    if not pending:
        return
    if args.dry_run:
        for run in pending:
            print(shlex.join(evaluation_command(args, run, args.root / "data")))
        return
    failures = []
    with stage_dataset(args.root / "data", pending[0][1]) as local:
        for run in pending:
            output = output_path(args.root, args.phase, *run)
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix(".log")
            label = "/".join(str(value) for value in run if value is not None)
            command = evaluation_command(args, run, local)
            print(f"RUN {label} log={log}", flush=True)
            with log.open("a") as stream:
                stream.write(f"\nCOMMAND {shlex.join(command)}\n")
                stream.flush()
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0 and completed_result(output, identity(args, *run)):
                print(f"DONE {label}", flush=True)
            else:
                print(f"FAILED {label} exit={result.returncode} log={log}", flush=True)
                failures.append(label)
    if failures:
        raise SystemExit("Failed LP evaluations: " + ", ".join(failures))


def restore_encoder(model, saved, method):
    """Restore only the original timm encoder, never a CP head or optimizer."""
    if method == "MAE-CP":
        import torch

        prefix = "backbone.vit."
        if hasattr(getattr(model, "head", None), "in_features"):
            model.head = torch.nn.Identity()
    else:
        prefix = "backbone."
    state = {
        key[len(prefix) :]: value
        for key, value in saved["state_dict"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state, strict=True)


def evaluate(args):
    import random
    import numpy as np
    import timm
    import torch
    from stable_cp.data.datasets import get_dataset, get_dataset_config
    from stable_cp.data.loaders import create_lp_transforms
    from stable_cp.evaluation.linear_probe import linear_probe_online_evaluate

    run = next(
        run
        for run in evaluations(args.phase, args.task_id)
        if run[0] == args.encoder and run[3] == args.seed
    )
    row = identity(args, *run)
    _, dataset, method, seed = run
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit LP reruns through Slurm.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = get_dataset_config(dataset)
    model = timm.create_model(
        row["backbone"], pretrained=args.phase == "pre", img_size=config["input_size"]
    )
    if args.phase == "post":
        checkpoint = Path(row["checkpoint"])
        if checkpoint_status(checkpoint, run[:3]) != "complete":
            raise ValueError(f"LP requires a completed CP checkpoint: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        restore_encoder(model, saved, method)
        del saved
    config = dict(
        config, normalization={key: list(model.pretrained_cfg[key]) for key in ("mean", "std")}
    )
    train_transform, test_transform = create_lp_transforms(config)
    splits = config.get("splits", ("train", "validation", "test"))
    train = get_dataset(
        dataset, split=splits[0], transform=train_transform, cache_dir=args.cache_dir, seed=seed
    )
    test = get_dataset(
        dataset, split=splits[-1], transform=test_transform, cache_dir=args.cache_dir, seed=seed
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(
        train, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(test, **loader_kwargs)
    model.to("cuda")
    print(
        f"LP ONLY {row['encoder']}/{dataset}/seed{seed}: "
        f"train={len(train)} test={len(test)} epochs={args.epochs} "
        f"batch={args.batch_size} forward_batch={args.forward_batch_size or args.batch_size}",
        flush=True,
    )
    scores = linear_probe_online_evaluate(
        model,
        train_loader,
        test_loader,
        torch.device("cuda"),
        pool_strategy=row["pool_strategy"],
        epochs=args.epochs,
        lr=args.lr,
        forward_batch_size=args.forward_batch_size,
        num_classes=config["num_classes"],
        seed=seed,
    )
    row.update(
        n_train_actual=len(train),
        n_test=len(test),
        normalization=config["normalization"],
        num_classes=config["num_classes"],
    )
    row.update(
        {
            f"{args.phase}_linear_{metric}": scores[f"linear_pytorch_{metric}"]
            for metric in ("acc", "f1", "auroc")
        }
    )
    output = output_path(args.root, args.phase, *run)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps(row, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "array", "run", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--phase", choices=("pre", "post"), required=True)
        command.add_argument("--root", type=Path, default=os.environ.get("CP_ROOT", REPO.parent))
        if name == "array":
            command.add_argument("--gpu", choices=("v100", "a100"))
        if name in ("run", "evaluate"):
            command.add_argument("--task-id", type=int, required=True)
            command.add_argument("--epochs", type=int, default=150)
            command.add_argument("--batch-size", type=int, default=512)
            command.add_argument("--lr", type=float, default=1e-3)
            command.add_argument("--forward-batch-size", type=int)
            command.add_argument("--num-workers", type=int, default=2)
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
        if name == "evaluate":
            command.add_argument("--encoder", required=True)
            command.add_argument("--seed", type=int, choices=precp.SEEDS, required=True)
            command.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if args.command == "list":
        tasks = PRE_TASKS if args.phase == "pre" else cp_full.TASKS
        for index, task in enumerate(tasks):
            label = task if isinstance(task, str) else " / ".join(task)
            print(f"{index:3d} {label}")
    elif args.command == "array":
        print(",".join(map(str, available_tasks(args.root, args.phase, args.gpu))))
    else:
        count = len(PRE_TASKS if args.phase == "pre" else cp_full.TASKS)
        if not 0 <= args.task_id < count:
            parser.error(f"--task-id must be between 0 and {count - 1}")
        if args.command == "run":
            run_task(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
