"""Plan, prepare, run, and collect the fixed eight-target CP experiment."""

import argparse
from pathlib import Path

from eval.heldout_cp import protocol as p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output-base", type=Path, default=p.DEFAULT_OUTPUT_BASE)
    plan.add_argument("--manifest", type=Path, required=True)
    for command in ("prepare", "run", "fit", "collect"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path, required=True)
        if command == "collect":
            item.add_argument("--outdir", type=Path, required=True)
            continue
        item.add_argument(
            "--cache-dir", type=Path, default=Path("/scratch/gs4133/zhd/CP/data")
        )
        item.add_argument("--num-workers", type=int, default=8)
        item.add_argument("--dry-run", action="store_true")
        item.add_argument(
            "--dataset-id" if command == "prepare" else "--task-id",
            type=int,
            required=True,
        )
        if command == "fit":
            item.add_argument("--seed", type=int, required=True, choices=p.SEEDS)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.manifest.exists():
            parser.error("Manifest exists; choose a new filename")
        doc = p.build_manifest(args.output_base)
        p.atomic_json(args.manifest, doc)
        print(
            f"MANIFEST {args.manifest.resolve()}: 8 preparation jobs, 48 CP jobs, 144 CP fits, 0 FT fits"
        )
        for task in doc["tasks"]:
            print(
                f"TASK {task['task_id']} {task['encoder']} {task['method']} {task['dataset']} "
                "seeds=42,43,44 n=1000 blocks=2 gpu=v100"
            )
        return
    doc = p.load_manifest(args.manifest)
    if args.command == "collect":
        from eval.heldout_cp.collect import collect

        collect(doc, args.outdir)
        return
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.command == "prepare":
        if not 0 <= args.dataset_id < len(p.DATASETS):
            parser.error("dataset-id must be 0..7")
        if args.dry_run:
            print(
                f"PREPARE {p.DATASETS[args.dataset_id]} encoders=DINOv3,CLIP seeds=42,43,44 "
                "n=1000 geometry_max=3000 gpu=v100 no_cp=True no_ft=True"
            )
        else:
            from eval.heldout_cp.prepare import prepare_dataset

            prepare_dataset(
                doc,
                args.dataset_id,
                cache_dir=args.cache_dir.resolve(),
                num_workers=args.num_workers,
            )
        return
    if not 0 <= args.task_id < len(doc["tasks"]):
        parser.error("task-id must be 0..47")
    task = doc["tasks"][args.task_id]
    if args.dry_run:
        for seed in [args.seed] if args.command == "fit" else task["seeds"]:
            print(
                f"CP_ONLY task={args.task_id} {task['encoder']} {task['method']} "
                f"{task['dataset']} seed={seed} n=1000 blocks=2 gpu=v100 "
                f"normalization={task['normalization']} initialization=public_pretrained no_ft=True"
            )
        return
    from eval.heldout_cp.run import fit_seed, run_task

    if args.command == "fit":
        fit_seed(
            doc,
            task,
            args.seed,
            cache_dir=args.cache_dir.resolve(),
            num_workers=args.num_workers,
        )
    else:
        run_task(
            args.manifest.resolve(),
            doc,
            task,
            cache_dir=args.cache_dir.resolve(),
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
