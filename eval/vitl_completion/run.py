"""Run three sequential ViT-L CP seeds with pre/post kNN and LP, without FT."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

from eval.full_ft.manifest import SEEDS
from eval.full_ft.run import atomic_json, digest_json, file_sha256, software_versions
from eval.vitl_completion.protocol import PROTOCOL, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def require_a100_80gb():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: this protocol runs only on one A100 80GB")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one A100 80GB GPU per task")
    props = torch.cuda.get_device_properties(0)
    # A full 80GB card exposes about 79 GiB. Reject 40GB cards and MIG slices.
    if ("A100" not in props.name.upper() or "MIG" in props.name.upper()
            or props.total_memory < 75 * 1024**3):
        raise RuntimeError(f"A100 80GB required, got {props.name} "
                           f"({props.total_memory / 1024**3:.1f} GiB)")
    print(f"GPU {props.name}: {props.total_memory / 1024**3:.1f} GiB", flush=True)


def task_identity(task):
    # Selection changes array indices, never the experiment or its artifact paths.
    return {key: value for key, value in task.items() if key != "task_id"}


def seed_paths(document, task, seed):
    root = Path(document["output_root"])
    suffix = Path(task["method"]) / task["dataset"] / f"seed{seed}.json"
    return dict(checkpoint=Path(task["checkpoints"][str(seed)]),
                cp_result=root / "cp_results" / suffix,
                receipt=root / "provenance" / suffix)


def cp_command(document, task, seed, *, cache_dir, num_workers=8, resume=False):
    paths = seed_paths(document, task, seed)
    command = [sys.executable, str(REPO_ROOT / "continued_pretraining.py"),
               "--cp-method", task["method"].lower(), "--dataset", task["dataset"],
               "--backbone", task["model_id"], "--n-samples", str(task["n_samples"]),
               "--pool-strategy", task["pool"], "--seed", str(seed),
               "--num-workers", str(num_workers), "--cache-dir", str(cache_dir),
               "--checkpoint-dir", str(paths["checkpoint"].parent.parent),
               "--results-json", str(paths["cp_result"]), "--project", PROTOCOL,
               "--run-name", f"{PROTOCOL}_{task['method']}_{task['dataset']}_"
                             f"blk{task['cp_recipe']['num_trained_blocks']}_s{seed}"]
    for key, value in task["cp_recipe"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if resume:
        command.append("--resume")
    return command


def implementation_sha256():
    files = {"continued_pretraining.py", "eval/full_ft/run.py", "eval/full_ft/manifest.py",
             "eval/vitl_completion/protocol.py", "eval/vitl_completion/run.py"}
    files.update(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "stable_cp").rglob("*.py"))
    return digest_json({p: file_sha256(REPO_ROOT / p) for p in sorted(files)})


@contextmanager
def seed_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"CP seed is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_cp_result(row, task, seed):
    expected = dict(dataset=task["dataset"], n_samples=task["n_samples"],
                    backbone=task["model_id"], method=task["method"].lower(),
                    seed=seed, epochs=task["cp_recipe"]["epochs"],
                    random_init=False, no_cp=False)
    if not isinstance(row, dict):
        raise ValueError("CP result must be an object")
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ValueError(f"CP result identity mismatch: {key}")
    for phase in ("pre", "post"):
        for metric in ("knn_f1", "linear_f1", "knn_acc", "linear_acc"):
            key = f"{phase}_{metric}"
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid or missing CP metric: {key}")
    return row


def completed_artifacts(paths, task, seed):
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing/empty CP checkpoint: {checkpoint}")
    validate_cp_result(json.loads(paths["cp_result"].read_text()), task, seed)
    return dict(checkpoint_sha256=file_sha256(checkpoint),
                cp_result_sha256=file_sha256(paths["cp_result"]))


def run_seed(document, task, seed, *, cache_dir, num_workers, code_hash, versions):
    paths = seed_paths(document, task, seed)
    expected = dict(schema_version=1, protocol=PROTOCOL,
                    task_sha256=digest_json(task_identity(task)), seed=seed,
                    cp_recipe=task["cp_recipe"], implementation_sha256=code_hash,
                    software=versions)
    with seed_lock(paths["receipt"]):
        existing = None
        if paths["receipt"].exists():
            existing = json.loads(paths["receipt"].read_text())
            if not isinstance(existing, dict):
                raise ValueError("Malformed CP receipt")
            for key, value in expected.items():
                if digest_json(existing.get(key)) != digest_json(value):
                    raise ValueError(f"CP receipt changed {key}; refusing incompatible resume")
            if existing.get("status") not in {"cp_pending", "cp_complete"}:
                raise ValueError("Unrecognized CP receipt state")
        elif paths["checkpoint"].exists() or paths["cp_result"].exists():
            raise ValueError("Existing artifacts have no protocol receipt; refusing to overwrite")
        else:
            atomic_json(paths["receipt"], dict(expected, status="cp_pending"))

        if existing and existing["status"] == "cp_complete":
            hashes = completed_artifacts(paths, task, seed)
            if any(existing.get(key) != value for key, value in hashes.items()):
                raise ValueError("Completed CP artifacts changed after receipt was frozen")
            print(f"SKIP complete: {task['method']} {task['dataset']} seed={seed}", flush=True)
            return

        checkpoint = paths["checkpoint"]
        if checkpoint.exists() and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
            raise ValueError(f"Invalid partial CP checkpoint: {checkpoint}")
        command = cp_command(document, task, seed, cache_dir=cache_dir,
                             num_workers=num_workers, resume=checkpoint.exists())
        print(f"CP {task['method']} {task['dataset']} seed={seed} "
              f"blocks={task['cp_recipe']['num_trained_blocks']}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        hashes = completed_artifacts(paths, task, seed)
        atomic_json(paths["receipt"], dict(expected, status="cp_complete", **hashes))
        print(f"SUCCESS {task['method']} {task['dataset']} seed={seed}", flush=True)


def run_task(manifest, task_id, *, cache_dir, num_workers=8, dry_run=False):
    document = load_manifest(Path(manifest).resolve())
    if type(task_id) is not int or not 0 <= task_id < len(document["tasks"]):
        raise ValueError(f"task-id must be in 0..{len(document['tasks']) - 1}")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num-workers must be a nonnegative integer")
    task = document["tasks"][task_id]
    cache_dir = Path(cache_dir).expanduser().resolve()
    if dry_run:
        for seed in SEEDS:
            command = cp_command(document, task, seed, cache_dir=cache_dir, num_workers=num_workers)
            print(json.dumps(dict(task_id=task_id, seed=seed, stage="cp", command=command)))
        return 0
    require_a100_80gb()
    code_hash, versions = implementation_sha256(), software_versions()
    errors = 0
    for seed in SEEDS:
        try:
            run_seed(document, task, seed, cache_dir=cache_dir, num_workers=num_workers,
                     code_hash=code_hash, versions=versions)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors += 1
            print(f"FAIL {task['method']} {task['dataset']} seed={seed}: {exc}",
                  file=sys.stderr, flush=True)
    print(f"TASK {task_id}: {len(SEEDS) - errors}/{len(SEEDS)} CP seeds complete (no FT)", flush=True)
    return int(errors > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run_task(args.manifest, args.task_id, cache_dir=args.cache_dir,
                    num_workers=args.num_workers, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
