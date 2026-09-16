"""All-dataset SigLIP CP with native mean/std, fresh weights, and no FT."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

from eval.full_ft import run as artifacts
from eval.full_ft.manifest import DATASET_META, ENCODERS, SEEDS
from eval.siglip_mainrule import protocol as original
from eval.siglip_mainrule import run as legacy


PROTOCOL = "siglip_native_cp_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_DATASETS = {"octmnist", "pathmnist", "food101"}
NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
METRICS = ("post_knn_f1", "post_linear_f1", "post_knn_acc", "post_linear_acc")
GPU_RESOURCES = {
    "v100": dict(gres="gpu:v100:1", constraint="", model="V100", min_memory_gb=0),
    "a100": dict(gres="gpu:a100:1", constraint="", model="A100", min_memory_gb=0),
    "a100_80gb": dict(gres="gpu:a100:1", constraint="80g", model="A100", min_memory_gb=75),
}


def implementation_sha256():
    return artifacts.digest_json({
        "training": legacy.implementation_sha256(),
        "runner": artifacts.file_sha256(__file__),
    })


def _selection(values, available, name):
    values = list(available) if values is None else list(values)
    if not values or len(values) != len(set(values)) or set(values) - set(available):
        raise ValueError(f"Invalid {name} selection: {values}")
    return [v for v in available if v in values]


def gpu_profile(method, blocks):
    if method not in original.METHODS or blocks not in (2, 4, 6, -1):
        raise ValueError(f"Unknown GPU routing configuration: {method}, blocks={blocks}")
    if blocks == 2:
        return "v100"
    return "a100_80gb" if method == "LeJEPA" and blocks == -1 else "a100"


def build_manifest(output_base=original.DEFAULT_OUTPUT_BASE, *, datasets=None, methods=None):
    datasets = _selection(datasets, sorted(DATASET_META, key=lambda d: -DATASET_META[d][2]), "dataset")
    methods = _selection(methods, original.METHODS, "method")
    model, pool = ENCODERS["SigLIP"]
    tasks = []
    for dataset in datasets:
        display, subpath, n = DATASET_META[dataset]
        for method in methods:
            recipe = original.cp_recipe(method, n)
            profile = gpu_profile(method, recipe["num_trained_blocks"])
            groups = [[seed] for seed in SEEDS] if dataset in LARGE_DATASETS else [list(SEEDS)]
            for seeds in groups:
                tasks.append(dict(task_id=len(tasks), dataset=dataset, display=display,
                                  processed_subpath=subpath, n_samples=n, budget="MAX",
                                  encoder="SigLIP", model_id=model, pool=pool, method=method,
                                  seeds=seeds, cp_recipe=dict(recipe), gpu_profile=profile))
    return dict(schema_version=1, protocol=PROTOCOL,
                output_root=str(Path(output_base).expanduser().resolve() / PROTOCOL),
                normalization=json.loads(json.dumps(NORMALIZATION)),
                normalization_mode="pretrained", transform_contract="official_mean_std_only",
                implementation_sha256=implementation_sha256(),
                datasets=datasets, methods=methods, tasks=tasks,
                selected_task_ids=list(range(len(tasks))))


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    try:
        root = Path(document["output_root"])
        if not root.is_absolute() or root.name != PROTOCOL:
            raise ValueError("Native CP outputs must use their isolated namespace")
        expected = build_manifest(root.parent, datasets=document["datasets"], methods=document["methods"])
        selected = document["selected_task_ids"]
        if (not isinstance(selected, list)
                or any(type(i) is not int or not 0 <= i < len(expected["tasks"]) for i in selected)
                or selected != sorted(set(selected))):
            raise ValueError("Invalid native CP task selection")
        expected["selected_task_ids"] = selected
        if artifacts.digest_json(document) != artifacts.digest_json(expected):
            raise ValueError("Native CP manifest changed its tasks, recipe, normalization, paths, or code")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed native CP manifest") from exc
    return document


def submission_groups(document, concurrency):
    """Use separate resource arrays without multiplying the requested concurrency cap."""
    if type(concurrency) is not int or not 1 <= concurrency <= 12:
        raise ValueError("concurrency must be 1..12")
    groups = []
    for profile in GPU_RESOURCES:
        ids = [i for i in document["selected_task_ids"]
               if document["tasks"][i]["gpu_profile"] == profile]
        if ids:
            groups.append(dict(gpu_profile=profile, task_ids=ids, concurrency=0, serial=False))
    if concurrency < len(groups):
        for group in groups:
            group.update(concurrency=min(concurrency, len(group["task_ids"])), serial=True)
    else:
        remaining = min(concurrency, sum(len(g["task_ids"]) for g in groups))
        while remaining:
            for group in groups:
                if remaining and group["concurrency"] < len(group["task_ids"]):
                    group["concurrency"] += 1
                    remaining -= 1
    return groups


def result_paths(document, task, seed):
    suffix = Path(task["method"]) / task["dataset"] / f"seed{seed}.json"
    root = Path(document["output_root"])
    return dict(cp_result=root / "cp_results" / suffix, receipt=root / "provenance" / suffix)


def _identity(document, task, seed):
    # A fit's identity is independent of filtering or grouping seeds into jobs.
    fit = {key: task[key] for key in ("dataset", "n_samples", "budget", "encoder", "model_id",
                                    "pool", "method", "cp_recipe")}
    return dict(schema_version=1, protocol=PROTOCOL, fit_sha256=artifacts.digest_json(fit),
                seed=seed, cp_recipe=task["cp_recipe"], normalization=document["normalization"],
                gpu_profile=task["gpu_profile"],
                implementation_sha256=document["implementation_sha256"],
                initialization="public_pretrained")


def validate_result(row, task, seed):
    if not isinstance(row, dict):
        raise ValueError("Malformed native CP result")
    legacy.validate_cp_result(row, task, seed)
    if row.get("normalization_mode") != "pretrained" or row.get("normalization") != NORMALIZATION:
        raise ValueError("CP result normalization does not match native SigLIP mean/std")
    expected = dict(task["cp_recipe"], pool_strategy="map", skip_baseline=True,
                    skip_final_eval=False, resume=False, pre_cp_sft=False, post_cp_sft=False)
    config = row.get("cp_config")
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("CP result has an incompatible training/evaluation configuration")
    return row


def completed_result(document, task, seed):
    paths = result_paths(document, task, seed)
    if not paths["receipt"].exists():
        if paths["cp_result"].exists():
            raise ValueError(f"CP result has no provenance receipt: {paths['cp_result']}")
        return None
    record = json.loads(paths["receipt"].read_text())
    expected = _identity(document, task, seed)
    if (not isinstance(record, dict)
            or any(artifacts.digest_json(record.get(k)) != artifacts.digest_json(v)
                   for k, v in expected.items())):
        raise ValueError(f"CP receipt identity mismatch: {paths['receipt']}")
    if record.get("status") in {"cp_pending", "cp_failed"}:
        if paths["cp_result"].exists():
            raise ValueError(f"Uncommitted CP result requires inspection: {paths['cp_result']}")
        return None
    if record.get("status") != "cp_complete":
        raise ValueError(f"Unrecognized CP receipt state: {paths['receipt']}")
    if (not paths["cp_result"].is_file()
            or record.get("cp_result_sha256") != artifacts.file_sha256(paths["cp_result"])):
        raise ValueError(f"Completed CP result SHA256 mismatch: {paths['cp_result']}")
    validate_result(json.loads(paths["cp_result"].read_text()), task, seed)
    checkpoint = Path(record.get("checkpoint_path", ""))
    attempt_root = (Path(document["output_root"]) / "attempts" / task["method"]
                    / task["dataset"] / f"seed{seed}").resolve()
    if (attempt_root not in checkpoint.resolve().parents or not checkpoint.is_file()
            or checkpoint.stat().st_size == 0
            or record.get("checkpoint_sha256") != artifacts.file_sha256(checkpoint)):
        raise ValueError(f"Completed checkpoint path/SHA256 mismatch: {checkpoint}")
    return paths["cp_result"]


def write_plan(output_base, manifest, *, datasets=None, methods=None):
    manifest = Path(manifest)
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    document = build_manifest(output_base, datasets=datasets, methods=methods)
    selected, fits = [], 0
    for task in document["tasks"]:
        missing = [seed for seed in task["seeds"] if completed_result(document, task, seed) is None]
        if missing:
            selected.append(task["task_id"])
            fits += len(missing)
        print(f"{'RUN' if missing else 'SKIP'} task={task['task_id']} "
              f"{task['method']} {task['dataset']} gpu={task['gpu_profile']} "
              f"pending_seeds={missing}", flush=True)
    document["selected_task_ids"] = selected
    artifacts.atomic_json(manifest, document)
    print(f"MANIFEST {manifest}: {len(selected)} jobs, {fits} CP fits, 0 FT fits", flush=True)
    return selected


def fresh_command(document, task, seed, attempt, *, cache_dir, num_workers):
    filename = f"{task['dataset']}_{task['model_id'].replace('/', '_')}_n{task['n_samples']}_s{seed}.ckpt"
    checkpoint = attempt / "checkpoints" / "cp" / filename
    working = dict(task, phase="post", checkpoints={str(seed): [str(checkpoint)]})
    command = legacy.cp_command({"output_root": str(attempt)}, working, seed,
                                cache_dir=cache_dir, num_workers=num_workers, resume=False)
    command.extend(["--normalization-mode", "pretrained"])
    command[command.index("--project") + 1] = PROTOCOL
    command[command.index("--run-name") + 1] = (
        f"{PROTOCOL}_{task['method']}_{task['dataset']}_s{seed}_{attempt.name}")
    return command, dict(checkpoint=checkpoint, cp_result=Path(command[command.index("--results-json") + 1]))


def require_gpu(profile):
    import torch

    if profile not in GPU_RESOURCES:
        raise ValueError(f"Unknown GPU profile: {profile}")
    required = GPU_RESOURCES[profile]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"CUDA required: expose exactly one {profile} GPU per job")
    props = torch.cuda.get_device_properties(0)
    if (required["model"] not in props.name.upper() or "MIG" in props.name.upper()
            or props.total_memory < required["min_memory_gb"] * 1024**3):
        raise RuntimeError(f"GPU profile {profile} required, got {props}")
    return props.name


def _run_seed(manifest, document, task, seed, *, cache_dir, num_workers, gpu):
    paths = result_paths(document, task, seed)
    with artifacts.seed_lock(paths["receipt"]):
        if completed_result(document, task, seed) is not None:
            print(f"SKIP {task['method']} {task['dataset']} seed={seed}: verified native CP", flush=True)
            return
        attempt = (Path(document["output_root"]) / "attempts" / task["method"]
                   / task["dataset"] / f"seed{seed}" / uuid.uuid4().hex)
        attempt.mkdir(parents=True, exist_ok=False)
        command, working = fresh_command(document, task, seed, attempt,
                                         cache_dir=cache_dir, num_workers=num_workers)
        record = dict(_identity(document, task, seed), software=artifacts.software_versions(),
                      gpu=gpu, attempt_dir=str(attempt), manifest=str(Path(manifest).resolve()),
                      command=command)
        artifacts.atomic_json(paths["receipt"], dict(record, status="cp_pending"))
        artifacts.atomic_json(attempt / "provenance.json", dict(record, status="cp_pending"))
        print(f"CP_ONLY task={task['task_id']} {task['method']} {task['dataset']} seed={seed} "
              f"normalization=pretrained initialization=public_pretrained", flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            hashes = legacy._completed_cp(working, task, seed)
            row = validate_result(json.loads(working["cp_result"].read_text()), task, seed)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            failed = dict(record, status="cp_failed", error=str(exc))
            artifacts.atomic_json(paths["receipt"], failed)
            artifacts.atomic_json(attempt / "provenance.json", failed)
            raise
        artifacts.atomic_json(paths["cp_result"], row)
        hashes["attempt_cp_result_sha256"] = hashes["cp_result_sha256"]
        hashes["cp_result_sha256"] = artifacts.file_sha256(paths["cp_result"])
        complete = dict(record, status="cp_complete", checkpoint_path=str(working["checkpoint"]), **hashes)
        artifacts.atomic_json(paths["receipt"], complete)
        artifacts.atomic_json(attempt / "provenance.json", complete)
        print(f"SUCCESS {task['method']} {task['dataset']} seed={seed} -> {paths['cp_result']}", flush=True)


def run_task(manifest, task_id, *, cache_dir, num_workers=8, dry_run=False):
    document = load_manifest(manifest)
    if type(task_id) is not int or not 0 <= task_id < len(document["tasks"]):
        raise ValueError("task-id is out of range")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num-workers must be a nonnegative integer")
    task = document["tasks"][task_id]
    cache_dir = Path(cache_dir).expanduser().resolve()
    errors, gpu = 0, None
    for seed in task["seeds"]:
        try:
            if completed_result(document, task, seed) is not None:
                print(f"SKIP {task['method']} {task['dataset']} seed={seed}: verified native CP", flush=True)
                continue
            if dry_run:
                attempt = (Path(document["output_root"]) / "attempts" / task["method"]
                           / task["dataset"] / f"seed{seed}" / "dry-run")
                command, _ = fresh_command(document, task, seed, attempt,
                                           cache_dir=cache_dir, num_workers=num_workers)
                print(json.dumps(dict(task_id=task_id, seed=seed, stage="cp", command=command)))
                continue
            if gpu is None:
                gpu = require_gpu(task["gpu_profile"])
            _run_seed(manifest, document, task, seed, cache_dir=cache_dir, num_workers=num_workers, gpu=gpu)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors += 1
            print(f"FAIL {task['method']} {task['dataset']} seed={seed}: {exc}", file=sys.stderr, flush=True)
    print(f"TASK {task_id}: {len(task['seeds']) - errors}/{len(task['seeds'])} seeds "
          f"{'planned' if dry_run else 'complete'}, 0 FT fits", flush=True)
    return int(errors > 0)


def collect_results(output_base=original.DEFAULT_OUTPUT_BASE, *, datasets=None, methods=None):
    document = build_manifest(output_base, datasets=datasets, methods=methods)
    writer = csv.writer(sys.stdout)
    writer.writerow(["method", "dataset", "seed", "status", *METRICS, "source", "note"])
    groups, count, failures = {}, 0, 0
    for task in document["tasks"]:
        key = task["method"], task["dataset"]
        groups.setdefault(key, [])
        for seed in task["seeds"]:
            paths = result_paths(document, task, seed)
            values, status, note = [""] * len(METRICS), "MISSING", ""
            try:
                path = completed_result(document, task, seed)
                if path is not None:
                    row = json.loads(path.read_text())
                    values = [f"{row[m]:.8f}" for m in METRICS]
                    status = "VERIFIED"
                    groups[key].append(row)
                    count += 1
                elif paths["receipt"].exists():
                    receipt = json.loads(paths["receipt"].read_text())
                    status = receipt["status"].upper()
                    note = receipt.get("error", "")
            except (OSError, ValueError, RuntimeError) as exc:
                status, note = "CHECK", str(exc)
                failures += 1
            writer.writerow([*key, seed, status, *values, paths["cp_result"], note])
    for key, rows in groups.items():
        if rows:
            writer.writerow([*key, "MEAN", f"n={len(rows)}/3",
                             *[f"{statistics.mean(r[m] for r in rows):.8f}" for m in METRICS], "", ""])
            if len(rows) > 1:
                writer.writerow([*key, "SD", f"n={len(rows)}/3",
                                 *[f"{statistics.stdev(r[m] for r in rows):.8f}" for m in METRICS],
                                 "", "sample SD"])
    total = sum(len(t["seeds"]) for t in document["tasks"])
    print(f"Verified native CP results: {count}/{total}. Means use available verified seeds only; "
          "missing/failed seeds are not zero scores.", file=sys.stderr)
    return int(failures > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "collect"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--output-base", type=Path, default=original.DEFAULT_OUTPUT_BASE)
        cmd.add_argument("--datasets", nargs="+", choices=sorted(DATASET_META))
        cmd.add_argument("--methods", nargs="+", choices=original.METHODS)
        if action == "plan":
            cmd.add_argument("--manifest", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            write_plan(args.output_base, args.manifest, datasets=args.datasets, methods=args.methods)
            return 0
        if args.action == "collect":
            return collect_results(args.output_base, datasets=args.datasets, methods=args.methods)
        return run_task(args.manifest, args.task_id, cache_dir=args.cache_dir,
                        num_workers=args.num_workers, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL native CP: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
