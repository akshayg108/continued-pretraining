"""Retry held-out LeJEPA on A100 without changing the frozen training recipe.

The original manifest remains a record of the V100 plan. This additive
entrypoint records the hardware override separately and retains the existing
metric-roundoff recovery, baseline checks, seed locks, and fresh initialization.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from eval import heldout_metric_roundoff as numerics
from eval.heldout_cp import protocol as p
from eval.heldout_cp import runtime


POLICY = "heldout_lejepa_a100_retry_v1"


def require_lejepa(task):
    if task["method"] != "LeJEPA":
        raise ValueError("A100 recovery is restricted to LeJEPA; leave other methods unchanged")


@contextmanager
def a100_allocation(doc, task, seed):
    require_lejepa(task)
    original_check, original_write = runtime.check_environment, p.atomic_json
    target = (task["encoder"], task["dataset"], task["method"], seed)
    audit = dict(
        policy=POLICY,
        planned_gpu=task["gpu"],
        requested_gpu="a100",
        wrapper_sha256=p.file_sha256(Path(__file__)),
        base_implementation_sha256=doc["implementation_sha256"],
    )

    def check_environment():
        import torch
        from stable_datasets import images

        # Keep the original environment and precision checks; only the GPU changes.
        for name in (
            "MedMNIST", "AID", "RESISC45", "StanfordDogs", "JenaFlowers30", "Flavia", "IP102"
        ):
            if not hasattr(images, name):
                raise RuntimeError(
                    f"Missing stable-datasets reader {name}; install cc01e36 or newer"
                )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("This experiment requires exactly one allocated CUDA GPU")
        gpu = torch.cuda.get_device_name(0)
        if "A100" not in gpu.upper():
            raise RuntimeError(f"Expected an A100 allocation, received {gpu}")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        memory = int(torch.cuda.get_device_properties(0).total_memory)
        audit.update(actual_gpu=gpu, gpu_total_memory_bytes=memory)
        print(
            f"RESOURCE_OVERRIDE policy={POLICY} planned=v100 requested=a100 "
            f"actual={gpu} memory_gib={memory / 2**30:.2f} "
            f"wrapper_sha256={audit['wrapper_sha256']}",
            flush=True,
        )
        return gpu

    def audited_write(path, row):
        key = tuple(row.get(field) for field in ("encoder", "dataset", "method", "seed"))
        if key == target:
            if "actual_gpu" not in audit or row.get("gpu") != audit["actual_gpu"]:
                raise ValueError("A100 result does not match the verified allocation")
            row = dict(row, resource_override=dict(audit))
        return original_write(path, row)

    runtime.check_environment, p.atomic_json = check_environment, audited_write
    try:
        yield
    finally:
        runtime.check_environment, p.atomic_json = original_check, original_write


def preflight(doc):
    """Require already frozen, intact preparations before any recovery submission."""
    tasks = [task for task in doc["tasks"] if task["method"] == "LeJEPA"]
    for dataset in dict.fromkeys(task["dataset"] for task in tasks):
        if not p.predictions_path(doc, dataset).is_file():
            raise ValueError(f"Missing frozen predictions for {dataset}; do not start CP")
        p.freeze_predictions(doc, dataset)
        print(f"PREPARED {dataset}: both encoders and all three seeds verified", flush=True)
    complete = 0
    for task in tasks:
        for seed in task["seeds"]:
            baseline = p.validate_pre(doc, task["encoder"], task["dataset"], seed)
            audit = baseline.get("evaluation_numerics")
            if audit is not None and (
                audit.get("wrapper_sha256") != p.file_sha256(Path(numerics.__file__))
                or audit.get("base_implementation_sha256") != doc["implementation_sha256"]
                or audit.get("policy") != numerics.POLICY
            ):
                raise ValueError("Metric recovery implementation changed since preparation")
            path = p.result_path(doc, task, seed)
            if path.exists():
                p.validate_result(doc, task, seed, json.loads(path.read_text()))
                complete += 1
    total = sum(len(task["seeds"]) for task in tasks)
    print(
        f"READY {len(tasks)} LeJEPA jobs, {total} CP fits; "
        f"verified={complete}, pending={total - complete}, gpu=a100, no_ft=True",
        flush=True,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--task-id", type=int, required=True)
    fit.add_argument("--seed", type=int, choices=p.SEEDS, required=True)
    fit.add_argument("--cache-dir", type=Path, default=Path("/scratch/gs4133/zhd/CP/data"))
    fit.add_argument("--num-workers", type=int, default=8)
    fit.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    doc = p.load_manifest(args.manifest)
    if args.command == "preflight":
        preflight(doc)
        return
    if not 0 <= args.task_id < len(doc["tasks"]):
        parser.error("task-id must be 0..47")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    task = doc["tasks"][args.task_id]
    require_lejepa(task)
    if args.dry_run:
        print(
            f"CP_ONLY task={args.task_id} {task['encoder']} LeJEPA {task['dataset']} "
            f"seed={args.seed} n=1000 blocks=2 batch=256 views=8 gpu=a100 "
            "initialization=public_pretrained no_ft=True",
            flush=True,
        )
        return
    with a100_allocation(doc, task, args.seed):
        numerics.main(argv)


if __name__ == "__main__":
    main()
