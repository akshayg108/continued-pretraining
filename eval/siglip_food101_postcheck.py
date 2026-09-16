"""Re-evaluate existing Food-101 CP weights, never train or replace checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from eval.full_ft.run import file_sha256
from eval.siglip_mainrule.protocol import DEFAULT_OUTPUT_BASE, METHODS, cp_recipe
from eval.siglip_mainrule.run import validate_cp_result

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "siglip_food101_postcheck_official_norm_v1"
MODEL = "vit_base_patch16_siglip_224.v2_webli"
NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
METRICS = ("post_knn_f1", "post_linear_f1", "post_knn_acc", "post_linear_acc")


def source_task(method):
    if method not in METHODS:
        raise ValueError(f"Unsupported CP objective: {method}")
    return dict(dataset="food101", n_samples=75750, model_id=MODEL,
                method=method, cp_recipe=cp_recipe(method, 75750))


def input_hashes(task):
    return {key + "_sha256": file_sha256(task[key])
            for key in ("checkpoint", "cp_result", "receipt") if task.get(key)}


def validate_inputs(task):
    """Bind complete result records to exact weights, including low-scoring runs."""
    seed = task["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Expected a nonnegative integer seed")
    ckpt = Path(task["checkpoint"])
    expected_name = f"food101_{MODEL}_n75750_s{seed}.ckpt"
    if ckpt.name != expected_name or ckpt.parent.name != "cp":
        raise ValueError("Checkpoint identity mismatch")
    if not ckpt.is_file() or ckpt.stat().st_size == 0:
        raise ValueError(f"Missing or empty CP checkpoint: {ckpt}")
    template = source_task(task["method"])
    row = json.loads(Path(task["cp_result"]).read_text())
    validate_cp_result(row, template, seed)
    hashes = input_hashes(task)
    for key, value in hashes.items():
        if key in task and task[key] != value:
            raise ValueError(f"Input changed after planning: {key}")
    if task.get("receipt"):
        receipt = json.loads(Path(task["receipt"]).read_text())
        expected = dict(protocol="siglip_mainrule_v1", status="cp_complete", seed=seed,
                        cp_recipe=template["cp_recipe"],
                        checkpoint_sha256=hashes["checkpoint_sha256"],
                        cp_result_sha256=hashes["cp_result_sha256"])
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"Mainrule receipt mismatch: {key}")
    return row, hashes


def build_manifest(output_base=DEFAULT_OUTPUT_BASE, *, source="mainrule",
                   recheck_job_id="17950124", methods=None, seeds=None):
    base = Path(output_base).expanduser().resolve()
    if source not in {"mainrule", "recheck"}:
        raise ValueError("Source must be mainrule or recheck")
    if not str(recheck_job_id).isdigit():
        raise ValueError("Recheck job ID must be numeric")
    methods = list(methods) if methods is not None else list(METHODS if source == "mainrule"
                                                          else ("LeJEPA",))
    seeds = list(seeds) if seeds is not None else ([42, 43, 44] if source == "mainrule"
                                                  else [43, 44, 45, 46])
    if (not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods)
            or (source == "recheck" and methods != ["LeJEPA"])):
        raise ValueError("Recheck contains LeJEPA only; methods must be supported and unique")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    root = (base / "siglip_mainrule_v1" if source == "mainrule" else
            base / "siglip_lejepa_recheck_v1" / str(recheck_job_id))
    source_id = "mainrule" if source == "mainrule" else f"recheck-{recheck_job_id}"
    tasks, missing = [], []
    for method in methods:
        for seed in seeds:
            directory = root / "checkpoints" / method / "food101"
            if source == "recheck":
                directory /= f"seed{seed}"
            checkpoint = directory / "cp" / f"food101_{MODEL}_n75750_s{seed}.ckpt"
            result = root / "cp_results" / method / "food101" / f"seed{seed}.json"
            receipt = (root / "provenance" / method / "food101" / f"seed{seed}.json"
                       if source == "mainrule" else None)
            task = dict(input_source=source_id, method=method, seed=seed,
                        checkpoint=str(checkpoint), cp_result=str(result),
                        receipt=str(receipt) if receipt is not None else None)
            absent = [str(p) for p in (checkpoint, result, receipt) if p is not None and not p.is_file()]
            if absent:
                missing.append(dict(task, missing_paths=absent))
                continue
            _, hashes = validate_inputs(task)
            tasks.append(dict(task, task_id=len(tasks), **hashes))
    return dict(schema_version=1, protocol=PROTOCOL, input_source=source_id,
                output_base=str(base), requested_methods=methods, requested_seeds=seeds,
                tasks=tasks, missing=missing)


def write_new_json(path, record):
    path = Path(path)
    payload = json.dumps(record, indent=2, allow_nan=False, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    if document.get("protocol") != PROTOCOL or document.get("schema_version") != 1:
        raise ValueError("Wrong checkpoint-audit manifest protocol")
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Manifest must contain a task list")
    seen = set()
    for index, task in enumerate(tasks):
        identity = (task["input_source"], task["method"], task["seed"])
        if (task["task_id"] != index or identity in seen or
                task["input_source"] != document["input_source"]):
            raise ValueError("Invalid task ordering, source, or duplicate identity")
        source_task(task["method"])
        seen.add(identity)
        for key in ("checkpoint_sha256", "cp_result_sha256"):
            if not isinstance(task.get(key), str) or len(task[key]) != 64:
                raise ValueError(f"Missing input hash: {key}")
    return document


def run_task(task, *, cache_dir, outdir, num_workers=8):
    path = (Path(outdir) / task["input_source"] / task["method"] /
            "food101" / f"seed{task['seed']}.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num-workers must be a nonnegative integer")
    old, hashes = validate_inputs(task)
    import lightning as pl
    import numpy as np
    import torch
    from continued_pretraining import _create_shared_eval_data, get_dataset_config, load_backbone
    from eval.full_ft.checkpoint import discard_native_head, load_backbone_state
    from stable_cp.evaluation.zero_shot_eval import (
        extract_features, knn_evaluate, linear_probe_pytorch_evaluate,
    )

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "V100" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("This audit requires one allocated V100")
    args = SimpleNamespace(dataset="food101", backbone=MODEL, n_samples=75750,
                           batch_size=64, num_workers=num_workers, seed=task["seed"],
                           cache_dir=str(cache_dir), pool_strategy="map")
    pl.seed_everything(args.seed, workers=True)
    cfg = get_dataset_config(args.dataset)
    dataset_normalization = cfg["normalization"]
    backbone, device = load_backbone(args, img_size=cfg["input_size"], pretrained=False)
    if device.type != "cuda":
        raise RuntimeError("CUDA device was not selected")
    native = {k: list(getattr(backbone, "pretrained_cfg", {}).get(k, ())) for k in ("mean", "std")}
    if native != NORMALIZATION:
        raise ValueError(f"Unexpected SigLIP-2 pretrained normalization: {native}")
    # Reuse only the strict CP weight loader, not the full-FT training path.
    discard_native_head(backbone)
    checkpoint_load = load_backbone_state(backbone, task["checkpoint"])
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    cfg = {**cfg, "normalization": native}
    eval_tf, test_loader, lp_loader, knn_loader, indices = _create_shared_eval_data(
        args, cfg, Path(cache_dir))
    if len(indices) != 75750 or len(test_loader.dataset) != 25250:
        raise ValueError("Unexpected Food-101 split sizes")
    train, train_y = extract_features(backbone, lp_loader, device, pool_strategy="map", verbose=True)
    test, test_y = extract_features(backbone, test_loader, device, pool_strategy="map", verbose=True)
    clean, clean_y = extract_features(backbone, knn_loader, device, pool_strategy="map", verbose=True)
    for name, features, labels, n in (("lp_train", train, train_y, 75750),
                                      ("test", test, test_y, 25250),
                                      ("knn_train", clean, clean_y, 75750)):
        if features.shape != (n, 768) or len(labels) != n:
            raise ValueError(f"Unexpected feature/label shape: {name}")
        if not np.isfinite(features).all() or len(np.unique(labels)) != 101:
            raise ValueError(f"Non-finite features or missing classes: {name}")
    if not np.array_equal(train_y, clean_y):
        raise ValueError("LP and kNN train label ordering differs")
    print("Running k-NN evaluation...", flush=True)
    knn = knn_evaluate(clean, clean_y, test, test_y, k=20)
    print("Running linear probe evaluation (PyTorch)...", flush=True)
    lp = linear_probe_pytorch_evaluate(train, train_y, test, test_y, device=device,
                                       lr=1e-3, min_epochs=150, min_steps=10000,
                                       batch_size=512, verbose=True)
    metrics = dict(post_knn_f1=float(knn["knn_f1"]), post_knn_acc=float(knn["knn_acc"]),
                   post_linear_f1=float(lp["linear_pytorch_f1"]),
                   post_linear_acc=float(lp["linear_pytorch_acc"]))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in metrics.values()):
        raise ValueError("Invalid frozen evaluation metric")
    if input_hashes(task) != hashes:
        raise ValueError("Inputs changed during evaluation")
    versions = {}
    for package in ("torch", "timm", "lightning", "stable-pretraining", "stable-datasets",
                    "scikit-learn", "torchmetrics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    files = [ROOT / "continued_pretraining.py", Path(__file__), ROOT / "eval/full_ft/checkpoint.py",
             *sorted((ROOT / "stable_cp").rglob("*.py"))]
    result = dict(protocol=PROTOCOL, status="complete", stage="post_cp_evaluation",
                  dataset="food101", backbone=MODEL, method=task["method"], seed=args.seed,
                  n_samples=75750, n_test=25250, initialization="cp_checkpoint",
                  cp_training_performed=False, no_ft=True, pool_strategy="map",
                  input_source=task["input_source"], checkpoint_path=task["checkpoint"],
                  cp_result_path=task["cp_result"], checkpoint_load=checkpoint_load,
                  **hashes, original_post_metrics={k: old[k] for k in METRICS},
                  normalization_mode="official", normalization=native,
                  dataset_normalization=dataset_normalization,
                  checkpoint_retrained=False, feature_precision="float32", feature_batch_size=64,
                  knn_k=20, lp_method="pytorch", lp_lr=1e-3, lp_min_epochs=150,
                  lp_min_steps=10000, lp_batch_size=512, splits=cfg["splits"],
                  lp_train_transform=repr(lp_loader.dataset.dataset.transform),
                  eval_transform=repr(eval_tf), pretrained_config=backbone.pretrained_cfg,
                  train_indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0), versions=versions,
                  git_commit=commit.stdout.strip() if commit.returncode == 0 else "unknown",
                  code_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in files}, **metrics)
    write_new_json(path, result)
    print(json.dumps(dict(method=task["method"], seed=args.seed, **metrics)), flush=True)
    print(f"Results saved to {path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="Check saved CP artifacts without importing torch")
    plan.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--source", choices=("mainrule", "recheck"), default="mainrule")
    plan.add_argument("--recheck-job-id", default="17950124")
    plan.add_argument("--methods", nargs="+", choices=METHODS)
    plan.add_argument("--seeds", type=int, nargs="+")
    run = commands.add_parser("run", help="Evaluate one existing CP checkpoint on a V100")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "plan":
        if args.manifest.exists():
            parser.error("Refusing to overwrite a manifest")
        document = build_manifest(args.output_base, source=args.source,
                                  recheck_job_id=args.recheck_job_id, methods=args.methods,
                                  seeds=args.seeds)
        for task in document["missing"]:
            print(f"MISSING {task['input_source']} {task['method']} seed={task['seed']}: "
                  + ", ".join(task["missing_paths"]), flush=True)
        for task in document["tasks"]:
            print(f"READY task={task['task_id']} {task['input_source']} {task['method']} "
                  f"seed={task['seed']} checkpoint={task['checkpoint']}", flush=True)
        write_new_json(args.manifest, document)
        print(f"MANIFEST {args.manifest}: {len(document['tasks'])} frozen evaluations, 0 CP fits, 0 FT fits")
    else:
        document = load_manifest(args.manifest)
        if not 0 <= args.task_id < len(document["tasks"]):
            parser.error("task-id out of range")
        task = document["tasks"][args.task_id]
        print(f"POST_CP_ONLY {task['input_source']} method={task['method']} seed={task['seed']} "
              "normalization=official cp_training=false ft=false", flush=True)
        if args.dry_run:
            print(json.dumps(task, sort_keys=True))
        else:
            run_task(task, cache_dir=args.cache_dir, outdir=args.outdir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
