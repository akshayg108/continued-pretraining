"""Restart unfinished large-dataset SigLIP LeJEPA seeds without fine-tuning."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid

from eval.full_ft import run as ft
from eval.full_ft.manifest import SEEDS
from eval.siglip_mainrule import protocol as original
from eval.siglip_mainrule import run as legacy


PROTOCOL = "siglip_cp_only_v1"
DATASETS = ("octmnist", "pathmnist", "food101")
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_manifest(output_base=original.DEFAULT_OUTPUT_BASE):
    base = Path(output_base).expanduser().resolve()
    source = original.build_manifest(base)
    by_dataset = {t["dataset"]: t for t in source["tasks"] if t["method"] == "LeJEPA"}
    entries = [dict(task_id=i, source_task=by_dataset[dataset], seed=seed)
               for i, (dataset, seed) in enumerate(
                   (dataset, seed) for dataset in DATASETS for seed in SEEDS)]
    return dict(schema_version=1, protocol=PROTOCOL,
                output_root=str(base / PROTOCOL), source_output_root=source["output_root"],
                training_implementation_sha256=legacy.implementation_sha256(),
                tasks=entries, selected_task_ids=list(range(len(entries))))


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    try:
        root = Path(document["output_root"])
        if not root.is_absolute() or root.name != PROTOCOL:
            raise ValueError("CP-only outputs must use their isolated namespace")
        expected = build_manifest(root.parent)
        selected = document["selected_task_ids"]
        if (not isinstance(selected, list)
                or any(type(i) is not int or not 0 <= i < 9 for i in selected)
                or selected != sorted(set(selected))):
            raise ValueError("Invalid CP-only task selection")
        expected["selected_task_ids"] = selected
        if ft.digest_json(document) != ft.digest_json(expected):
            raise ValueError("CP-only manifest changed its tasks, recipe, paths, or training code")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed CP-only manifest") from exc
    return document


def result_paths(document, entry):
    root = Path(document["output_root"])
    suffix = Path("LeJEPA") / entry["source_task"]["dataset"] / f"seed{entry['seed']}.json"
    return dict(cp_result=root / "cp_results" / suffix,
                receipt=root / "provenance" / suffix)


def _verified_result(document, entry, paths, *, source):
    if not paths["receipt"].exists():
        if paths["cp_result"].exists():
            raise ValueError(f"CP result has no provenance receipt: {paths['cp_result']}")
        return None
    record = json.loads(paths["receipt"].read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Malformed CP receipt: {paths['receipt']}")
    if record.get("status") == "cp_pending":
        return None
    if record.get("status") != "cp_complete":
        raise ValueError(f"Unrecognized CP receipt state: {paths['receipt']}")
    task, seed = entry["source_task"], entry["seed"]
    expected = dict(schema_version=1, protocol=original.PROTOCOL if source else PROTOCOL,
                    task_sha256=ft.digest_json(task if source else entry),
                    seed=seed, cp_recipe=task["cp_recipe"])
    code_key = "implementation_sha256" if source else "training_implementation_sha256"
    expected[code_key] = document["training_implementation_sha256"]
    if not source:
        expected.update(implementation_sha256=ft.file_sha256(__file__),
                        initialization="public_pretrained")
    if any(ft.digest_json(record.get(k)) != ft.digest_json(v) for k, v in expected.items()):
        raise ValueError(f"Completed CP receipt identity mismatch: {paths['receipt']}")
    row = json.loads(paths["cp_result"].read_text())
    if not isinstance(row, dict):
        raise ValueError(f"Malformed CP result: {paths['cp_result']}")
    legacy.validate_cp_result(row, task, seed)
    if record.get("cp_result_sha256") != ft.file_sha256(paths["cp_result"]):
        raise ValueError(f"Completed CP result SHA256 mismatch: {paths['cp_result']}")
    # Completed metrics suffice here. Old weights and FT artifacts are never reused.
    return paths["cp_result"]


def completed_result(document, entry):
    source_paths = legacy.seed_paths(
        {"output_root": document["source_output_root"]}, entry["source_task"], entry["seed"])
    for paths, is_source in ((source_paths, True), (result_paths(document, entry), False)):
        result = _verified_result(document, entry, paths, source=is_source)
        if result is not None:
            return result
    return None


def write_plan(output_base, manifest):
    manifest = Path(manifest)
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    document = build_manifest(output_base)
    selected = []
    for entry in document["tasks"]:
        result = completed_result(document, entry)
        name = f"LeJEPA {entry['source_task']['dataset']} seed={entry['seed']}"
        if result is None:
            selected.append(entry["task_id"])
            print(f"RESTART task={entry['task_id']} {name}: public weights, no FT", flush=True)
        else:
            print(f"SKIP {name}: verified CP -> {result}", flush=True)
    document["selected_task_ids"] = selected
    ft.atomic_json(manifest, document)
    print(f"MANIFEST {manifest}: {len(selected)} jobs, one CP seed per job, 0 FT fits", flush=True)
    return selected


def fresh_command(document, entry, attempt, *, cache_dir, num_workers):
    task, seed = copy.deepcopy(entry["source_task"]), entry["seed"]
    filename = Path(task["checkpoints"][str(seed)][0]).name
    task["checkpoints"][str(seed)] = [str(attempt / "checkpoints" / "cp" / filename)]
    working = {"output_root": str(attempt)}
    command = legacy.cp_command(working, task, seed, cache_dir=cache_dir,
                                num_workers=num_workers, resume=False)
    command[command.index("--project") + 1] = PROTOCOL
    command[command.index("--run-name") + 1] = (
        f"{PROTOCOL}_LeJEPA_{task['dataset']}_s{seed}_{attempt.name}")
    return command, legacy.seed_paths(working, task, seed)


def require_gpu():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA required: expose one A100 80GB per CP seed")
    props = torch.cuda.get_device_properties(0)
    if ("A100" not in props.name.upper() or "MIG" in props.name.upper()
            or props.total_memory < 75 * 1024**3):
        raise RuntimeError(f"A100 80GB required, got {props.name}")


def run_task(manifest, task_id, *, cache_dir, num_workers=8, dry_run=False):
    document = load_manifest(manifest)
    if type(task_id) is not int or not 0 <= task_id < len(document["tasks"]):
        raise ValueError("task-id must be in 0..8")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num-workers must be a nonnegative integer")
    entry = document["tasks"][task_id]
    task, seed = entry["source_task"], entry["seed"]
    result = completed_result(document, entry)
    if result is not None:
        print(f"SKIP LeJEPA {task['dataset']} seed={seed}: verified CP -> {result}", flush=True)
        return 0
    cache_dir = Path(cache_dir).expanduser().resolve()
    attempts = (Path(document["output_root"]) / "attempts" / "LeJEPA"
                / task["dataset"] / f"seed{seed}")
    if dry_run:
        command, _ = fresh_command(document, entry, attempts / "dry-run",
                                   cache_dir=cache_dir, num_workers=num_workers)
        print(json.dumps(dict(task_id=task_id, seed=seed, stage="cp", command=command)))
        return 0
    require_gpu()
    paths = result_paths(document, entry)
    source_paths = legacy.seed_paths({"output_root": document["source_output_root"]}, task, seed)
    # Keep the existing runner's lock boundary so a live original job cannot race this seed.
    with ft.seed_lock(source_paths["receipt"]), ft.seed_lock(paths["receipt"]):
        result = completed_result(document, entry)
        if result is not None:
            print(f"SKIP LeJEPA {task['dataset']} seed={seed}: verified CP -> {result}", flush=True)
            return 0
        attempt = attempts / uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        command, working_paths = fresh_command(document, entry, attempt,
                                               cache_dir=cache_dir, num_workers=num_workers)
        record = dict(schema_version=1, protocol=PROTOCOL, task_sha256=ft.digest_json(entry),
                      seed=seed, cp_recipe=task["cp_recipe"], initialization="public_pretrained",
                      training_implementation_sha256=document["training_implementation_sha256"],
                      implementation_sha256=ft.file_sha256(__file__), software=ft.software_versions(),
                      attempt_dir=str(attempt), manifest=str(Path(manifest).resolve()), command=command)
        ft.atomic_json(paths["receipt"], dict(record, status="cp_pending"))
        print(f"CP LeJEPA {task['dataset']} seed={seed}: restart from public weights, no FT", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        hashes = legacy._completed_cp(working_paths, task, seed)
        row = json.loads(working_paths["cp_result"].read_text())
        ft.atomic_json(paths["cp_result"], row)
        hashes["attempt_cp_result_sha256"] = hashes["cp_result_sha256"]
        hashes["cp_result_sha256"] = ft.file_sha256(paths["cp_result"])
        ft.atomic_json(paths["receipt"], dict(record, status="cp_complete",
                       checkpoint_path=str(working_paths["checkpoint"]), **hashes))
        print(f"SUCCESS LeJEPA {task['dataset']} seed={seed}: CP only -> {paths['cp_result']}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="Select unfinished seeds without loading CUDA")
    plan.add_argument("--output-base", type=Path, default=original.DEFAULT_OUTPUT_BASE)
    plan.add_argument("--manifest", type=Path, required=True)
    run = commands.add_parser("run", help="Run exactly one CP seed, never FT or checkpoint resume")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            write_plan(args.output_base, args.manifest)
            return 0
        return run_task(args.manifest, args.task_id, cache_dir=args.cache_dir,
                        num_workers=args.num_workers, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL CP-only: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
