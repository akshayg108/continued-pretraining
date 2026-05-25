#!/usr/bin/env python3
"""
Post-migration cleanup. Run AFTER migrate_to_layout_a.py.

Does three things on the cluster:

  Task 1: rename `*_mae_results.csv` → `*_mae_cp_results.csv`
          (only under cp/MAE/pretrained/, to match the other 3 methods)

  Task 2: rename DIET/pretrained/FGVC_Aircraft files containing `n3400`
          → `n3334` (logs json + ckpts) — old data used a different split
          count; new sh scripts use 3334

  Task 3: unify size folder labels (logs/ + ckpts/):
              1000/ → small/        (merge if small/ exists)
              1k/   → small/        (merge if small/ exists)
              10000/ → 10k/         (rename, no merge expected)
              25000/ → 25k/         (rename)
              max/   → all/         (rename)
          Applied across cp/{SimCLR,DIET,LeJEPA,MAE}/pretrained/<DS>/<MODEL>/
          where <MODEL> ∈ {CLIP, DINOv3, MAE}.

Usage (on cluster, from any dir):
    # dry-run (default)
    python3 cleanup_post_migration.py

    # actually apply
    python3 cleanup_post_migration.py --apply

    # apply + remove emptied source dirs (recommended)
    python3 cleanup_post_migration.py --apply --cleanup-empty
"""

import argparse
import os
import shutil
import sys

LOG_BASE_DEFAULT = "/scratch/gs4133/zhd/CP/outputs/logs"
CKPT_BASE_DEFAULT = "/scratch/gs4133/zhd/CP/outputs/ckpts"

METHODS = ["SimCLR", "DIET", "LeJEPA", "MAE"]
ALL_DATASETS = [
    "BreastMNIST", "Cars196", "CUB200", "DermaMNIST", "DTD",
    "EuroSAT", "FGVC_Aircraft", "Flowers102", "Food101", "Galaxy10",
    "OctMNIST", "OrganAMNIST", "OxfordPet", "PathMNIST", "PlantVillage",
]
ALL_BACKBONES = ["CLIP", "DINOv3", "MAE"]

# Size relabel rules. Order matters for log output only.
# (src_label, dst_label, is_merge)  — merge=True means src may need to merge into existing dst
SIZE_RENAMES = [
    ("1000",  "small", True),   # EuroSAT/PlantVillage CLIP/MAE migrated from Layout B
    ("1k",    "small", True),   # CLIP/MAE on other sized datasets (from old _run_1000.sh)
    ("10000", "10k",   False),  # EuroSAT/PlantVillage DINOv3 (Layout-B origin)
    ("25000", "25k",   False),  # PlantVillage DINOv3
    ("max",   "all",   False),  # EuroSAT/PlantVillage all backbones
]


# ============================================================
# Generic helpers
# ============================================================
class Plan:
    def __init__(self):
        self.actions = []  # list of (kind, src, dst, note)
        self.collisions = []

    def add(self, kind, src, dst, note=""):
        self.actions.append((kind, src, dst, note))

    def collide(self, src, dst):
        self.collisions.append((src, dst))

    def print(self, base_for_relpath=None):
        for kind, src, dst, note in self.actions:
            if base_for_relpath:
                src_disp = ".../" + os.path.relpath(src, base_for_relpath)
                dst_disp = ".../" + os.path.relpath(dst, base_for_relpath) if dst else ""
            else:
                src_disp, dst_disp = src, dst
            extra = f"  ({note})" if note else ""
            if dst:
                print(f"  [{kind:13}] {src_disp}{extra}")
                print(f"  {' '*15} -> {dst_disp}")
            else:
                print(f"  [{kind:13}] {src_disp}{extra}")
        for s, d in self.collisions:
            print(f"  [COLLISION]    {s}\n                  vs target {d}  (SKIPPED)")


def move_file(src, dst, apply):
    """Move single file, ensuring parent dir exists. Caller checked no collision."""
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def rename_dir_or_merge(src_dir, dst_dir, apply, plan):
    """If dst doesn't exist: simple rename. Else: recursive file-level merge.
    On file-level collision: skip (record in plan). After merging, remove empty src dirs."""
    if not os.path.isdir(src_dir):
        return
    if not os.path.exists(dst_dir):
        plan.add("RENAME-DIR", src_dir, dst_dir)
        if apply:
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            os.rename(src_dir, dst_dir)
        return
    # Merge: walk every file under src_dir
    plan.add("MERGE-DIR", src_dir, dst_dir, "target exists, file-level merge")
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(dst_dir, rel, fname) if rel != "." else os.path.join(dst_dir, fname)
            if os.path.exists(dst):
                plan.collide(src, dst)
                continue
            plan.add("MV-FILE", src, dst)
            if apply:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)


def cleanup_empty(start_dirs, plan, apply):
    """Recursively rmdir empty directories (deepest first)."""
    removed = 0
    # Collect every directory under each start (deepest first)
    all_dirs = []
    for sd in start_dirs:
        if not os.path.isdir(sd):
            continue
        for root, dirs, _ in os.walk(sd, topdown=False):
            all_dirs.append(root)
    for d in all_dirs:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                if apply:
                    os.rmdir(d)
                removed += 1
        except OSError:
            pass
    return removed


# ============================================================
# Task 1: rename `*_mae_results.csv` → `*_mae_cp_results.csv`
# ============================================================
def plan_task1(logs_base, plan):
    mae_root = os.path.join(logs_base, "cp", "MAE", "pretrained")
    if not os.path.isdir(mae_root):
        return
    for root, _, files in os.walk(mae_root):
        for f in files:
            if f.endswith("_mae_results.csv"):
                src = os.path.join(root, f)
                dst = os.path.join(root, f.replace("_mae_results.csv", "_mae_cp_results.csv"))
                if os.path.exists(dst):
                    plan.collide(src, dst)
                    continue
                plan.add("CSV-RENAME", src, dst)


# ============================================================
# Task 2: rename DIET FGVC_Aircraft n3400 → n3334
# ============================================================
def plan_task2(logs_base, ckpts_base, plan):
    for base_name, base in [("logs", logs_base), ("ckpts", ckpts_base)]:
        root = os.path.join(base, "cp", "DIET", "pretrained", "FGVC_Aircraft")
        if not os.path.isdir(root):
            continue
        for r, _, files in os.walk(root):
            for f in files:
                if "_n3400_" not in f:
                    continue
                src = os.path.join(r, f)
                dst = os.path.join(r, f.replace("_n3400_", "_n3334_"))
                if os.path.exists(dst):
                    plan.collide(src, dst)
                    continue
                plan.add(f"N3400-{base_name.upper()}", src, dst)


# ============================================================
# Task 3: unify size folder labels
# ============================================================
def collect_size_renames(logs_base, ckpts_base):
    """Yield (src_dir, dst_dir) pairs for Task 3."""
    for base in (logs_base, ckpts_base):
        for method in METHODS:
            for ds in ALL_DATASETS:
                ds_dir = os.path.join(base, "cp", method, "pretrained", ds)
                if not os.path.isdir(ds_dir):
                    continue
                for backbone in ALL_BACKBONES:
                    bk_dir = os.path.join(ds_dir, backbone)
                    if not os.path.isdir(bk_dir):
                        continue
                    for src_lbl, dst_lbl, _ in SIZE_RENAMES:
                        src = os.path.join(bk_dir, src_lbl)
                        dst = os.path.join(bk_dir, dst_lbl)
                        if os.path.isdir(src):
                            yield (src, dst)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-base", default=LOG_BASE_DEFAULT)
    p.add_argument("--ckpts-base", default=CKPT_BASE_DEFAULT)
    p.add_argument("--apply", action="store_true", help="Actually do it. Default = dry-run.")
    p.add_argument("--cleanup-empty", action="store_true", help="rmdir any directories emptied by the cleanup")
    p.add_argument("--only", choices=["1", "2", "3"], help="Run only this task")
    args = p.parse_args()

    print(f"logs-base : {args.logs_base}")
    print(f"ckpts-base: {args.ckpts_base}")
    print(f"mode      : {'APPLY (real)' if args.apply else 'DRY-RUN (no changes)'}")
    print("=" * 76)

    plan = Plan()

    # ----- Task 1 -----
    if args.only in (None, "1"):
        print("--- Task 1: rename *_mae_results.csv → *_mae_cp_results.csv ---")
        before = len(plan.actions)
        plan_task1(args.logs_base, plan)
        print(f"  {len(plan.actions) - before} CSV files to rename")
        for kind, src, dst, _ in plan.actions[before:]:
            print(f"    {src}\n     -> {dst}")
            if args.apply:
                os.rename(src, dst)

    # ----- Task 2 -----
    if args.only in (None, "2"):
        print()
        print("--- Task 2: DIET FGVC_Aircraft n3400 → n3334 ---")
        before = len(plan.actions)
        plan_task2(args.logs_base, args.ckpts_base, plan)
        print(f"  {len(plan.actions) - before} files to rename")
        for kind, src, dst, _ in plan.actions[before:]:
            print(f"    {src}\n     -> {dst}")
            if args.apply:
                os.rename(src, dst)

    # ----- Task 3 -----
    if args.only in (None, "3"):
        print()
        print("--- Task 3: unify size folder labels ---")
        renames = list(collect_size_renames(args.logs_base, args.ckpts_base))
        print(f"  {len(renames)} size dirs to rename/merge")
        for src, dst in renames:
            rename_dir_or_merge(src, dst, apply=args.apply, plan=plan)

    # ----- Summary -----
    print()
    print("=" * 76)
    print(f"Total actions planned: {len(plan.actions)}")
    if plan.collisions:
        print(f"Collisions (SKIPPED, manual resolution needed): {len(plan.collisions)}")
        for s, d in plan.collisions[:20]:
            print(f"  {s}\n     vs {d}")

    if args.cleanup_empty:
        starts = [os.path.join(args.logs_base, "cp"), os.path.join(args.ckpts_base, "cp")]
        n = cleanup_empty(starts, plan, apply=args.apply)
        print(f"Cleanup: {'removed' if args.apply else 'would remove'} {n} empty directories")

    if not args.apply:
        print()
        print("(dry-run — re-run with --apply to actually do it, +--cleanup-empty to rmdir emptied source dirs)")


if __name__ == "__main__":
    main()
