"""Run three sequential CP -> full-FT seeds under an isolated protocol."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from eval.full_ft import run as ft
from eval.full_ft.manifest import SEEDS
from eval.siglip_mainrule.protocol import PROTOCOL, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SigLIP CP + full-FT runs")


def seed_paths(document, task, seed):
    root = Path(document["output_root"])
    suffix = Path(task["method"]) / task["dataset"] / f"seed{seed}.json"
    return dict(checkpoint=Path(task["checkpoints"][str(seed)][0]),
                cp_result=root / "cp_results" / suffix,
                receipt=root / "provenance" / suffix,
                ft_result=ft.result_path(root / "full_ft", task, seed))


def cp_command(document, task, seed, *, cache_dir, num_workers=8, resume=False):
    paths = seed_paths(document, task, seed)
    command = [sys.executable, str(REPO_ROOT / "continued_pretraining.py"),
               "--cp-method", task["method"].lower(), "--dataset", task["dataset"],
               "--backbone", task["model_id"], "--n-samples", str(task["n_samples"]),
               "--pool-strategy", task["pool"], "--seed", str(seed),
               "--num-workers", str(num_workers), "--cache-dir", str(cache_dir),
               "--checkpoint-dir", str(paths["checkpoint"].parent.parent),
               "--results-json", str(paths["cp_result"]), "--skip-baseline",
               "--project", PROTOCOL,
               "--run-name", f"{PROTOCOL}_{task['method']}_{task['dataset']}_"
                             f"blk{task['cp_recipe']['num_trained_blocks']}_s{seed}"]
    for key, value in task["cp_recipe"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if resume:
        command.append("--resume")
    return command


def ft_command(manifest, document, task, seed, *, cache_dir, num_workers=8):
    return [sys.executable, str(REPO_ROOT / "eval/full_ft/run.py"),
            "--manifest", str(manifest), "--task-id", str(task["task_id"]),
            "--outdir", str(Path(document["output_root"]) / "full_ft"),
            "--cache-dir", str(cache_dir), "--num-workers", str(num_workers),
            "--device", "cuda", "--seeds", str(seed)]


def implementation_sha256():
    files = {"continued_pretraining.py", "eval/full_ft/run.py", "eval/full_ft/manifest.py",
             "eval/full_ft/checkpoint.py", "eval/siglip_mainrule/protocol.py",
             "eval/siglip_mainrule/run.py"}
    files.update(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "stable_cp").rglob("*.py"))
    return ft.digest_json({p: ft.file_sha256(REPO_ROOT / p) for p in sorted(files)})


def validate_cp_result(row, task, seed):
    expected = dict(dataset=task["dataset"], n_samples=task["n_samples"],
                    backbone=task["model_id"], method=task["method"].lower(),
                    seed=seed, epochs=task["cp_recipe"]["epochs"],
                    random_init=False, no_cp=False)
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"CP result identity mismatch: {key}")
    for key in ("post_knn_f1", "post_linear_f1", "post_knn_acc", "post_linear_acc"):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid or missing CP metric: {key}")
    return row


def _completed_cp(paths, task, seed):
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing/empty CP checkpoint: {checkpoint}")
    row = json.loads(paths["cp_result"].read_text())
    validate_cp_result(row, task, seed)
    return dict(checkpoint_sha256=ft.file_sha256(checkpoint),
                cp_result_sha256=ft.file_sha256(paths["cp_result"]))


def _run_seed(manifest, document, task, seed, *, cache_dir, num_workers, code_hash, versions):
    paths = seed_paths(document, task, seed)
    expected = dict(schema_version=1, protocol=PROTOCOL, task_sha256=ft.digest_json(task),
                    seed=seed, cp_recipe=task["cp_recipe"], implementation_sha256=code_hash,
                    software=versions)
    with ft.seed_lock(paths["receipt"]):
        existing = None
        if paths["receipt"].exists():
            existing = json.loads(paths["receipt"].read_text())
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ValueError(f"CP receipt changed {key}; refusing incompatible resume")
            if existing.get("status") not in {"cp_pending", "cp_complete"}:
                raise ValueError("Unrecognized CP receipt state")
        elif any(paths[k].exists() for k in ("checkpoint", "cp_result", "ft_result")):
            raise ValueError("Existing artifacts have no protocol receipt; refusing unbound weights/results")
        else:
            ft.atomic_json(paths["receipt"], dict(expected, status="cp_pending"))

        if existing and existing["status"] == "cp_complete":
            hashes = _completed_cp(paths, task, seed)
            if any(existing.get(k) != v for k, v in hashes.items()):
                raise ValueError("Completed CP artifacts changed after receipt was frozen")
            print(f"RESUME CP complete: {task['method']} {task['dataset']} seed={seed}", flush=True)
        else:
            # Only a receipt-bound checkpoint from this pass may restore CP optimizer state.
            checkpoint = paths["checkpoint"]
            if checkpoint.exists() and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
                raise ValueError(f"Invalid partial CP checkpoint: {checkpoint}")
            command = cp_command(document, task, seed, cache_dir=cache_dir,
                                 num_workers=num_workers, resume=checkpoint.exists())
            print(f"CP {task['method']} {task['dataset']} seed={seed} "
                  f"blocks={task['cp_recipe']['num_trained_blocks']}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            hashes = _completed_cp(paths, task, seed)
            ft.atomic_json(paths["receipt"], dict(expected, status="cp_complete", **hashes))

        print(f"FULL FT {task['method']} {task['dataset']} seed={seed}", flush=True)
        subprocess.run(ft_command(manifest, document, task, seed, cache_dir=cache_dir,
                                  num_workers=num_workers), cwd=REPO_ROOT, check=True)
        # The reusable FT runner allows missing old checkpoints; this new CP pass must not.
        result = ft.validate_result(json.loads(paths["ft_result"].read_text()))
        if (result["seed"] != seed or any(result[k] != task[k] for k in ft.IDENTITY_FIELDS)
                or result.get("checkpoint_sha256") != hashes["checkpoint_sha256"]):
            raise ValueError("FT result does not belong to this completed CP checkpoint")
        print(f"SUCCESS {task['method']} {task['dataset']} seed={seed}: "
              f"all {result['sft_total_params']} FT parameters trainable", flush=True)


def run_task(manifest, task_id, *, cache_dir, num_workers=8, dry_run=False):
    manifest = Path(manifest).resolve()
    document = load_manifest(manifest)
    if type(task_id) is not int or not 0 <= task_id < len(document["tasks"]):
        raise ValueError("task-id must be in 0..28")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num-workers must be a nonnegative integer")
    task = document["tasks"][task_id]
    cache_dir = Path(cache_dir).expanduser().resolve()
    if dry_run:
        for seed in SEEDS:
            cp = cp_command(document, task, seed, cache_dir=cache_dir, num_workers=num_workers)
            full_ft = ft_command(manifest, document, task, seed,
                                 cache_dir=cache_dir, num_workers=num_workers)
            for stage, command in (("cp", cp), ("full_ft", full_ft)):
                print(json.dumps(dict(task_id=task_id, seed=seed, stage=stage, command=command)))
        return 0
    require_cuda()
    code_hash, versions = implementation_sha256(), ft.software_versions()
    errors = 0
    for seed in SEEDS:
        try:
            _run_seed(manifest, document, task, seed, cache_dir=cache_dir,
                      num_workers=num_workers, code_hash=code_hash, versions=versions)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors += 1
            print(f"FAIL {task['method']} {task['dataset']} seed={seed}: {exc}",
                  file=sys.stderr, flush=True)
    print(f"TASK {task_id}: {3 - errors}/3 CP + full-FT seed pairs complete", flush=True)
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
