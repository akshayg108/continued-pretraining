#!/usr/bin/env python3
"""Rerun fixed-recipe full FT from CP weights; write metrics, never FT weights."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval.full_ft.manifest import DATASET_META, ENCODERS, SEEDS

PROTOCOL = "full_ft_v1"
IDENTITY_FIELDS = ("phase", "scope", "encoder", "method", "dataset", "budget",
                   "n_samples", "model_id", "pool")
METRICS = ("sft_acc", "sft_f1", "sft_auroc")
CODE_FILES = (
    "eval/full_ft/run.py", "eval/full_ft/checkpoint.py",
    "stable_cp/evaluation/sft_eval.py", "stable_cp/evaluation/zero_shot_eval.py",
    "stable_cp/data/datasets.py", "stable_cp/data/loaders.py",
    "continued_pretraining.py",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def implementation_sha256():
    return digest_json({p: file_sha256(REPO_ROOT / p) for p in CODE_FILES})


def validate_task(task):
    try:
        encoder, phase, method = task["encoder"], task["phase"], task["method"]
        dataset, n = task["dataset"], task["n_samples"]
        model, pool = ENCODERS[encoder]
        _display, subpath, maximum = DATASET_META[dataset]
        if (phase not in {"pre", "post"} or task["model_id"] != model
                or task["pool"] != pool or task["processed_subpath"] != subpath
                or type(n) is not int or not 32 <= n <= maximum):
            raise ValueError("invalid model, phase, dataset path, or sample count")
        if task["scope"] != ("siglip" if encoder == "SigLIP" else "main"):
            raise ValueError("encoder scope mismatch")
        if (phase == "pre" and method != "PRE") or (phase == "post" and method not in
                ({"LeJEPA", "SimCLR", "DIET"} if encoder == "SigLIP" else
                 {"LeJEPA", "SimCLR", "DIET", "MAE"})):
            raise ValueError("unsupported phase/objective")
        budget = task["budget"]
        if budget not in {"100", "500", "1000", "10000", "25000", "MAX"}:
            raise ValueError("unsupported budget")
        if budget == "MAX" and n != maximum:
            raise ValueError("MAX sample count mismatch")
        if budget not in {"100", "MAX"} and n != int(budget):
            raise ValueError("budget/sample count mismatch")
        if budget == "100" and n != max(100, {"food101": 101, "cars196": 196,
                                              "cub200": 200, "flowers102": 102}.get(dataset, 100)):
            raise ValueError("nominal 100 budget mismatch")
        paths = task["checkpoints"]
        if set(paths) != {str(s) for s in SEEDS}:
            raise ValueError("all three seed slots must be declared")
        for seed in SEEDS:
            candidates = paths[str(seed)]
            if not isinstance(candidates, list) or (phase == "pre" and candidates):
                raise ValueError("invalid checkpoint candidate list")
            suffix = f"{dataset}_{model}_n{n}_s{seed}.ckpt"
            if any(not isinstance(p, str) or Path(p).name != suffix
                   or any(t in {"sft_pre", "sft_post"} for t in Path(p).parts)
                   for p in candidates):
                raise ValueError("checkpoint model/dataset/seed mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed FT task: {exc}") from exc
    return task


def load_tasks(path):
    document = json.loads(Path(path).read_text())
    tasks = document.get("tasks", [])
    if document.get("schema_version") != 1 or not isinstance(tasks, list) or not tasks:
        raise ValueError("Expected a nonempty schema_version=1 manifest")
    seen = set()
    for index, task in enumerate(tasks):
        validate_task(task)
        identity = tuple(task[f] for f in IDENTITY_FIELDS)
        if task.get("task_id") != index or identity in seen:
            raise ValueError("Duplicate task identity or non-contiguous task IDs")
        seen.add(identity)
    return tasks


def result_path(outdir, task, seed):
    return (Path(outdir) / task["phase"] / task["encoder"] / task["method"] /
            task["dataset"] / f"{task['budget']}_n{task['n_samples']}" / f"seed{seed}.json")


def atomic_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(row, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def seed_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"FT seed is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_checkpoint(task, seed):
    if task["phase"] == "pre":
        return None
    for candidate in task["checkpoints"][str(seed)]:
        path = Path(candidate).expanduser()
        if path.exists():
            if not path.is_file() or not os.access(path, os.R_OK):
                raise ValueError(f"Checkpoint exists but is not a readable file: {path}")
            return path
    return None


def validate_result(row):
    if row.get("schema_version") != 1 or row.get("status") != "success":
        raise ValueError("Not a completed full-FT result")
    if row.get("sft_protocol") != PROTOCOL or row.get("seed") not in SEEDS:
        raise ValueError("FT protocol/seed mismatch")
    for metric in METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid metric: {metric}")
    for key in ("sft_total_params", "sft_trainable_params", "n_train_actual", "n_test"):
        if type(row.get(key)) is not int or row[key] <= 0:
            raise ValueError(f"Invalid positive count: {key}")
    if row["sft_total_params"] != row["sft_trainable_params"]:
        raise ValueError("Partially frozen FT result")
    for key in ("train_indices_sha256", "test_labels_sha256", "implementation_sha256",
                "task_sha256"):
        value = row.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing/invalid provenance: {key}")
    identity = {k: row[k] for k in IDENTITY_FIELDS}
    if row["task_sha256"] != digest_json(identity):
        raise ValueError("Result identity hash mismatch")
    return row


def software_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "torchmetrics", "timm", "lightning",
                    "stable-pretraining", "stable-datasets", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def train_one(task, seed, checkpoint, *, cache_dir, device, num_workers=8):
    import torch
    import lightning as pl
    from continued_pretraining import (_create_shared_eval_data, _create_sft_data,
                                      get_dataset_config, load_backbone)
    from stable_cp.evaluation.sft_eval import sft_evaluate, SFT_BATCH_SIZE
    from eval.full_ft.checkpoint import discard_native_head, load_backbone_state

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a CPU fallback")
    pl.seed_everything(seed, workers=True)
    args = SimpleNamespace(dataset=task["dataset"], backbone=task["model_id"],
                           n_samples=task["n_samples"], seed=seed, num_workers=num_workers,
                           batch_size=SFT_BATCH_SIZE, cache_dir=str(cache_dir))
    config = get_dataset_config(task["dataset"])
    backbone, _ = load_backbone(args, img_size=config["input_size"],
                                pretrained=task["phase"] == "pre")
    discard_native_head(backbone)
    audit = load_backbone_state(backbone, checkpoint) if checkpoint is not None else {}
    eval_tf, test, eval_train, knn_train, indices = _create_shared_eval_data(args, config, cache_dir)
    del eval_train, knn_train
    data = _create_sft_data(args, config, cache_dir, eval_tf, indices)
    labels_source = test.dataset.hf_dataset if hasattr(test.dataset, "hf_dataset") else test.dataset
    labels = labels_source["label"]
    label_digest = digest_json([int(label) for label in labels])
    if len(indices) < SFT_BATCH_SIZE:
        raise ValueError("FT train subset has no complete batch")
    metrics = sft_evaluate(backbone, data, test, device,
                           num_classes=config["num_classes"], embed_dim=backbone.num_features,
                           n_samples=len(indices), pool_strategy=task["pool"], seed=seed,
                           ckpt_path=None, prefix="sft")
    metrics.update(n_train_actual=len(indices), n_test=len(test.dataset),
                   train_indices_sha256=digest_json([int(i) for i in indices]),
                   test_labels_sha256=label_digest, checkpoint_load=audit)
    return metrics


def run_task(task, *, outdir, cache_dir, device="cuda", seeds=SEEDS, num_workers=8, dry_run=False):
    validate_task(task)
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(SEEDS):
        raise ValueError("seeds must be a nonempty, unique subset of 42 43 44")
    identity = {k: task[k] for k in IDENTITY_FIELDS}
    code_hash = implementation_sha256()
    versions = software_versions()
    errors = 0
    for seed in seeds:
        path = result_path(outdir, task, seed)
        base = dict(identity, seed=seed, schema_version=1, sft_protocol=PROTOCOL,
                    task_sha256=digest_json(identity), implementation_sha256=code_hash,
                    software=versions)
        if dry_run:
            checkpoint = resolve_checkpoint(task, seed)
            print(json.dumps(dict(base, output=str(path), checkpoint=str(checkpoint) if checkpoint else None,
                                  action="pretrained" if task["phase"] == "pre" else
                                  ("full_ft" if checkpoint else "skip_missing_checkpoint"))))
            continue
        try:
            with seed_lock(path):
                existing = json.loads(path.read_text()) if path.exists() else None
                checkpoint = resolve_checkpoint(task, seed)
                if checkpoint is not None:
                    base["checkpoint_path"] = str(checkpoint)
                    base["checkpoint_sha256"] = file_sha256(checkpoint)
                else:
                    base["checkpoint_path"] = None
                    base["checkpoint_sha256"] = None
                if existing is not None and existing.get("status") == "success":
                    validate_result(existing)
                    for field in (*IDENTITY_FIELDS, "seed", "task_sha256", "implementation_sha256", "software"):
                        if existing[field] != base[field]:
                            raise ValueError(f"Completed result has changed {field}; use a new output directory: {path}")
                    # A completed seed can outlive its input checkpoint; retain its metrics.
                    if checkpoint is not None and existing.get("checkpoint_sha256") != base["checkpoint_sha256"]:
                        raise ValueError(f"CP weights changed after completed FT: {path}")
                    print(f"RESUME success: {path}", flush=True)
                    continue
                if task["phase"] == "post" and checkpoint is None:
                    atomic_json(path, dict(base, status="skipped_missing_checkpoint",
                                           checkpoint_candidates=task["checkpoints"][str(seed)]))
                    print(f"SKIP missing CP checkpoint: {task['encoder']} {task['method']} "
                          f"{task['dataset']} {task['budget']} seed={seed}", flush=True)
                    continue
                started = time.monotonic()
                try:
                    metrics = train_one(task, seed, checkpoint, cache_dir=cache_dir,
                                        device=device, num_workers=num_workers)
                    row = dict(base, **metrics, status="success", elapsed_seconds=time.monotonic() - started)
                    validate_result(row)
                    if checkpoint is not None and file_sha256(checkpoint) != base["checkpoint_sha256"]:
                        raise ValueError("CP checkpoint changed during FT")
                    atomic_json(path, row)
                    print(f"SUCCESS seed={seed}: F1={row['sft_f1']:.6f} -> {path}", flush=True)
                except Exception as exc:
                    atomic_json(path, dict(base, status="failed", error=f"{type(exc).__name__}: {exc}"))
                    raise
        except Exception as exc:
            errors += 1
            print(f"ERROR seed={seed}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            gc.collect()
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
    print(f"TASK {task['task_id']} finished: {len(seeds)} seeds checked, {errors} errors", flush=True)
    return int(errors > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.manifest)
    if not 0 <= args.task_id < len(tasks):
        parser.error("task-id is outside the manifest")
    return run_task(tasks[args.task_id], outdir=args.outdir, cache_dir=args.cache_dir.expanduser(),
                    device=args.device, seeds=args.seeds, num_workers=args.num_workers,
                    dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
