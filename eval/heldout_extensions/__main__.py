"""Plan, prepare, run, or collect the two additional held-out encoders."""

import argparse
from pathlib import Path

from eval.heldout_extensions import protocol as p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare", "run", "fit", "collect"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path, required=True)
        if command == "plan":
            item.add_argument("--source-manifest", type=Path, required=True)
            item.add_argument("--output-base", type=Path, default=p.DEFAULT_OUTPUT_BASE)
        elif command == "collect":
            item.add_argument("--outdir", type=Path, required=True)
        else:
            item.add_argument("--cache-dir", type=Path, default=Path("/scratch/gs4133/zhd/CP/data"))
            item.add_argument("--num-workers", type=int, default=8)
            item.add_argument("--dry-run", action="store_true")
            item.add_argument("--preparation-id" if command == "prepare" else "--task-id",
                              type=int, required=True)
            if command == "fit":
                item.add_argument("--seed", type=int, choices=p.SEEDS, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.manifest.exists():
            parser.error("Manifest exists; choose a new filename")
        doc = p.build_manifest(args.output_base, args.source_manifest)
        p.atomic_json(args.manifest, doc)
        print(f"MANIFEST {args.manifest.resolve()}: 16 preparation jobs, 48 CP jobs, 144 CP fits, 0 FT fits")
        for task in doc["tasks"]:
            print(f"TASK {task['task_id']} {task['encoder']} {task['method']} {task['dataset']} "
                  f"seeds=42,43,44 n=1000 blocks=2 gpu={task['gpu']} pool={task['pool']}")
        return
    doc = p.load_manifest(args.manifest)
    if args.command == "collect":
        from eval.heldout_extensions.collect import collect
        collect(doc, args.outdir)
        return
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    prepare = args.command == "prepare"
    index = args.preparation_id if prepare else args.task_id
    grid = doc["preparations" if prepare else "tasks"]
    if not 0 <= index < len(grid):
        parser.error(f"{'preparation-id' if prepare else 'task-id'} must be 0..{len(grid)-1}")
    task = grid[index]
    if args.dry_run:
        if prepare:
            print(f"PREPARE {task['encoder']} {task['dataset']} gpu={task['gpu']} "
                  "seeds=42,43,44 n=1000 geometry_max=3000 no_cp=True no_ft=True")
        else:
            for seed in (args.seed,) if args.command == "fit" else task["seeds"]:
                print(f"CP_ONLY task={index} {task['encoder']} {task['method']} {task['dataset']} "
                      f"seed={seed} gpu={task['gpu']} pool={task['pool']} recipe={task['recipe']} "
                      f"normalization={task['normalization']} no_ft=True")
        return
    options = dict(cache_dir=args.cache_dir.resolve(), num_workers=args.num_workers)
    if prepare:
        from eval.heldout_extensions.prepare import prepare
        prepare(doc, task, **options)
    else:
        from eval.heldout_extensions.run import fit_seed, run_task
        if args.command == "fit":
            fit_seed(doc, task, args.seed, **options)
        else:
            run_task(args.manifest.resolve(), doc, task, **options)


if __name__ == "__main__":
    main()
