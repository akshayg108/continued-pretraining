#!/usr/bin/env python3
"""
Migrate existing logs/ckpts on the cluster from Layout B to Layout A
for the 7 affected datasets across cp/{SimCLR,DIET,LeJEPA,MAE}/pretrained/.

Layout B (current, mixed by backbone):
  logs/.../pretrained/Cars196/CLIP_cars196_n1000_seed42.json
  logs/.../pretrained/EuroSAT/max/CLIP_eurosat_n16200_seed42.json
  ckpts/.../pretrained/Cars196/cp/cars196_<timm>_n1000_s42.ckpt
  ckpts/.../pretrained/EuroSAT/max/cp/eurosat_<timm>_n16200_s42.ckpt

Layout A (target, separated by backbone — matches DermaMNIST):
  logs/.../pretrained/Cars196/CLIP/CLIP_cars196_n1000_seed42.json
  logs/.../pretrained/EuroSAT/CLIP/max/CLIP_eurosat_n16200_seed42.json
  ckpts/.../pretrained/Cars196/CLIP/cp/cars196_<timm>_n1000_s42.ckpt
  ckpts/.../pretrained/EuroSAT/CLIP/max/cp/eurosat_<timm>_n16200_s42.ckpt

Affected: cp/{SimCLR,DIET,LeJEPA,MAE}/pretrained/{Cars196,CUB200,DTD,Flowers102,
OxfordPet,EuroSAT,PlantVillage}/  (pre-cp-only already on Layout A — skipped)

Usage (on cluster):
  # 1) dry-run — preview every move, no filesystem changes
  python3 migrate_to_layout_a.py
  # 2) actually apply
  python3 migrate_to_layout_a.py --apply
  # 3) custom output roots (defaults shown)
  python3 migrate_to_layout_a.py \\
      --logs-base /scratch/gs4133/zhd/CP/outputs/logs \\
      --ckpts-base /scratch/gs4133/zhd/CP/outputs/ckpts \\
      --apply
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

METHODS = ["SimCLR", "DIET", "LeJEPA", "MAE"]
NONSIZED5 = ["Cars196", "CUB200", "DTD", "Flowers102", "OxfordPet"]
SIZED2 = ["EuroSAT", "PlantVillage"]

# Maps the TIMM backbone string (substring match) to the model tag used as
# the subdir in Layout A.
TIMM_TO_MODEL = [
    ("vit_base_patch16_dinov3.lvd1689m", "DINOv3"),
    ("vit_base_patch16_clip_224.openai", "CLIP"),
    ("vit_base_patch16_224.mae",         "MAE"),
]

# In log filenames, the first underscore-separated token is the model tag.
LOG_MODEL_PREFIXES = {"DINOv3", "CLIP", "MAE"}


def model_from_log_filename(fname):
    """Extract MODEL prefix from a log filename like 'CLIP_cars196_n1000_seed42.json'
    or 'CLIP_lejepa_cp_results.csv'."""
    head = fname.split("_", 1)[0]
    return head if head in LOG_MODEL_PREFIXES else None


def model_from_ckpt_filename(fname):
    """Extract MODEL from a ckpt filename by matching TIMM substring."""
    for timm, model in TIMM_TO_MODEL:
        if timm in fname:
            return model
    return None


def plan_log_moves(logs_base):
    """Yield (src, dst) pairs for log files (json + csv)."""
    for method in METHODS:
        base = os.path.join(logs_base, "cp", method, "pretrained")
        if not os.path.isdir(base):
            continue
        # Non-sized 5: <base>/<DS>/<MODEL>_*.{json,csv}  →  <base>/<DS>/<MODEL>/...
        for ds in NONSIZED5:
            d = os.path.join(base, ds)
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                src = os.path.join(d, entry)
                if not os.path.isfile(src):
                    continue
                model = model_from_log_filename(entry)
                if not model:
                    continue  # not a model-prefixed file, leave alone
                dst = os.path.join(d, model, entry)
                if src != dst:
                    yield ("log", src, dst)
        # Sized 2: <base>/<DS>/<size>/<MODEL>_*.{json,csv}  →  <base>/<DS>/<MODEL>/<size>/...
        for ds in SIZED2:
            d = os.path.join(base, ds)
            if not os.path.isdir(d):
                continue
            for size_dir in os.listdir(d):
                size_path = os.path.join(d, size_dir)
                if not os.path.isdir(size_path):
                    continue
                # If this is already a MODEL dir (already migrated), skip
                if size_dir in LOG_MODEL_PREFIXES:
                    continue
                for entry in os.listdir(size_path):
                    src = os.path.join(size_path, entry)
                    if not os.path.isfile(src):
                        continue
                    model = model_from_log_filename(entry)
                    if not model:
                        continue
                    dst = os.path.join(d, model, size_dir, entry)
                    if src != dst:
                        yield ("log", src, dst)


def plan_ckpt_moves(ckpts_base):
    """Yield (src, dst) pairs for ckpt files."""
    for method in METHODS:
        base = os.path.join(ckpts_base, "cp", method, "pretrained")
        if not os.path.isdir(base):
            continue
        # Non-sized 5: <base>/<DS>/cp/<file>.ckpt  →  <base>/<DS>/<MODEL>/cp/<file>.ckpt
        for ds in NONSIZED5:
            cp_dir = os.path.join(base, ds, "cp")
            if not os.path.isdir(cp_dir):
                continue
            for entry in os.listdir(cp_dir):
                src = os.path.join(cp_dir, entry)
                if not os.path.isfile(src) or not entry.endswith(".ckpt"):
                    continue
                model = model_from_ckpt_filename(entry)
                if not model:
                    continue
                dst = os.path.join(base, ds, model, "cp", entry)
                if src != dst:
                    yield ("ckpt", src, dst)
        # Sized 2: <base>/<DS>/<size>/cp/<file>.ckpt  →  <base>/<DS>/<MODEL>/<size>/cp/<file>.ckpt
        for ds in SIZED2:
            ds_dir = os.path.join(base, ds)
            if not os.path.isdir(ds_dir):
                continue
            for size_dir in os.listdir(ds_dir):
                size_path = os.path.join(ds_dir, size_dir)
                if not os.path.isdir(size_path):
                    continue
                if size_dir in LOG_MODEL_PREFIXES:
                    continue  # already migrated
                cp_dir = os.path.join(size_path, "cp")
                if not os.path.isdir(cp_dir):
                    continue
                for entry in os.listdir(cp_dir):
                    src = os.path.join(cp_dir, entry)
                    if not os.path.isfile(src) or not entry.endswith(".ckpt"):
                        continue
                    model = model_from_ckpt_filename(entry)
                    if not model:
                        continue
                    dst = os.path.join(ds_dir, model, size_dir, "cp", entry)
                    if src != dst:
                        yield ("ckpt", src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-base", default="/scratch/gs4133/zhd/CP/outputs/logs")
    p.add_argument("--ckpts-base", default="/scratch/gs4133/zhd/CP/outputs/ckpts")
    p.add_argument("--apply", action="store_true", help="Actually move files. Default is dry-run.")
    p.add_argument("--cleanup-empty", action="store_true",
                   help="After moving, rmdir any now-empty source directories (cp/, size/, etc.).")
    args = p.parse_args()

    print(f"logs-base : {args.logs_base}")
    print(f"ckpts-base: {args.ckpts_base}")
    print(f"mode      : {'APPLY (real moves)' if args.apply else 'DRY-RUN (no changes)'}")
    print("=" * 70)

    all_moves = list(plan_log_moves(args.logs_base)) + list(plan_ckpt_moves(args.ckpts_base))
    by_kind = defaultdict(int)
    collisions = []
    moved = 0
    src_dirs_to_check = set()

    for kind, src, dst in all_moves:
        by_kind[kind] += 1
        if os.path.exists(dst):
            collisions.append((src, dst))
            print(f"  [COLLISION] {kind:5} {src}\n              -> {dst}  (target exists, SKIP)")
            continue
        rel = os.path.relpath(src, args.logs_base if kind == "log" else args.ckpts_base)
        print(f"  [{'MOVE' if args.apply else 'DRY '}] {kind:5} .../{rel}")
        print(f"           -> .../{os.path.relpath(dst, args.logs_base if kind == 'log' else args.ckpts_base)}")
        if args.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
            src_dirs_to_check.add(os.path.dirname(src))

    print("=" * 70)
    print(f"Plan: {len(all_moves)} total moves ({by_kind.get('log', 0)} logs, {by_kind.get('ckpt', 0)} ckpts)")
    if collisions:
        print(f"  Collisions (skipped): {len(collisions)} — target file already exists")
    if args.apply:
        print(f"Applied: {moved} files moved")
        if args.cleanup_empty:
            removed = 0
            # Try removing source dirs that may now be empty (deepest-first)
            for d in sorted(src_dirs_to_check, key=lambda p: -p.count("/")):
                try:
                    while d and os.path.isdir(d) and not os.listdir(d):
                        os.rmdir(d)
                        removed += 1
                        d = os.path.dirname(d)
                except OSError:
                    pass
            print(f"Cleanup: removed {removed} now-empty directories")
    else:
        print("(Re-run with --apply to actually move. Add --cleanup-empty to also rmdir empty source dirs.)")

    if not all_moves:
        print("\nNothing to migrate — looks like migration is already complete (or no Layout B files exist).")


if __name__ == "__main__":
    main()
