#!/usr/bin/env python
"""
audit_checkpoints.py — audit EVERY CP checkpoint on disk: did its backbone actually train?

Independent of the geometry sweep. For each *.ckpt under --root we read the Lightning
checkpoint's `epoch` / `global_step` (free, via mmap) and compare the last transformer
block's weights to the timm pretrained reference (one tensor read). A checkpoint is flagged
UNTRAINED when its backbone never updated:

  - epoch <= FROZEN_EPOCHS-1            -> died at/before the unfreeze (only frozen warm-up ran)
  - backbone == pretrained (blk unchanged, epoch>frozen) -> unfreeze never took effect

CPU only. Fast: mmap loads metadata lazily and we touch a single weight tensor per file.
Resumable (skips paths already in --out) and shardable (--shard i/N) for array jobs.

Output CSV columns:
  ckpt, method, encoder, dataset, size, seed, epoch, global_step, blk_maxdiff, verdict

Run (single CPU node, ~20-40 min for ~2.5k ckpts):
  python eval/audit_checkpoints.py --root /scratch/gs4133/zhd/CP/outputs/ckpts \
         --out eval/outputs/ckpt_audit.csv

Parallel (SLURM array of N):
  python eval/audit_checkpoints.py --root <...> --out eval/outputs/ckpt_audit_${SLURM_ARRAY_TASK_ID}.csv \
         --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}
"""
import argparse
import csv
import glob
import os
import re
import sys

import torch
import timm

FROZEN_EPOCHS = 15           # recipe: epochs 0..14 frozen; unfreeze last 2 blocks at epoch 15
PROBE = "blocks.11.attn.qkv.weight"   # last block — most likely to be unfrozen; falls back to blocks.10
FALLBACK = "blocks.10.attn.qkv.weight"
CHANGE_TOL = 1e-5

_PRE = {}


def pretrained_sd(timm_id):
    if timm_id not in _PRE:
        _PRE[timm_id] = timm.create_model(timm_id, pretrained=True, num_classes=0).eval().state_dict()
    return _PRE[timm_id]


def parse_meta(path):
    """method/encoder/dataset/size/seed/timm_id from the path + filename."""
    fn = os.path.basename(path)
    m = re.search(r"_(vit_base_patch16_[A-Za-z0-9_.]+?)_n(\d+)_s(\d+)\.ckpt$", fn)
    timm_id = m.group(1) if m else None
    size = m.group(2) if m else ""
    seed = m.group(3) if m else ""
    parts = path.split(os.sep)
    method = encoder = dataset = ""
    try:                                    # .../ckpts/cp/<METHOD>/<variant>/<Dataset>/<ENC>/<sizetag>/cp/<file>
        ci = parts.index("cp")              # first 'cp' (right after 'ckpts')
        method, dataset, encoder = parts[ci + 1], parts[ci + 3], parts[ci + 4]
    except (ValueError, IndexError):
        pass
    return method, encoder, dataset, size, seed, timm_id


def load_ckpt(path):
    try:                                    # fast path: mmap loads metadata lazily
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:                       # torch<2.1 (no mmap kw) or non-zip/legacy ckpt
        return torch.load(path, map_location="cpu")


def block_diff(sd, pre, key):
    cands = [k for k in sd if k.endswith(key) and sd[k].shape == pre[key].shape]
    if not cands:
        return None
    return (sd[cands[0]].float() - pre[key].float()).abs().max().item()


def audit_one(path):
    method, encoder, dataset, size, seed, timm_id = parse_meta(path)
    base = [path, method, encoder, dataset, size, seed]
    try:
        ck = load_ckpt(path)
    except Exception as e:
        return base + ["", "", "", f"LOAD_ERROR:{type(e).__name__}"]

    epoch = ck.get("epoch", "") if isinstance(ck, dict) else ""
    gstep = ck.get("global_step", "") if isinstance(ck, dict) else ""
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck

    # FROM-SCRATCH = the 'random' variant dir (NOT a substring match — the cluster FS root
    # is /scratch/..., which would false-positive every path).
    is_scratch = (os.sep + "random" + os.sep) in path or (encoder or "").upper() == "RANDOM"
    diff = ""
    if timm_id and not is_scratch:
        try:
            pre = pretrained_sd(timm_id)
            d = block_diff(sd, pre, PROBE)
            if d is None:
                d = block_diff(sd, pre, FALLBACK)
            diff = "" if d is None else f"{d:.4g}"
        except Exception as e:
            diff = f"ERR:{type(e).__name__}"

    # verdict
    if isinstance(epoch, int) and epoch <= FROZEN_EPOCHS - 1:
        verdict = f"UNTRAINED:died@epoch{epoch}(<=frozen)"
    elif is_scratch:
        verdict = "SCRATCH"
    elif diff not in ("", None) and not diff.startswith("ERR") and float(diff) < CHANGE_TOL:
        verdict = "UNTRAINED:backbone==pretrained"
    elif diff.startswith("ERR") or diff == "":
        verdict = "UNKNOWN(no-backbone-key)"
    else:
        verdict = "OK:trained"
    return base + [epoch, gstep, diff, verdict]


FIELDS = ["ckpt", "method", "encoder", "dataset", "size", "seed",
          "epoch", "global_step", "blk_maxdiff", "verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root dir to glob *.ckpt under (e.g. .../outputs/ckpts)")
    ap.add_argument("--out", default="eval/outputs/ckpt_audit.csv")
    ap.add_argument("--shard", default=None, help="i/N to process only shard i of N (0-indexed)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*.ckpt"), recursive=True))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        paths = [p for j, p in enumerate(paths) if j % n == i]
    print(f"found {len(paths)} ckpt(s) under {args.root}" + (f"  (shard {args.shard})" if args.shard else ""))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["ckpt"])
        print(f"resuming: {len(done)} already audited")

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    counts = {}
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(FIELDS)
        for idx, p in enumerate(paths):
            if p in done:
                continue
            row = audit_one(p)
            w.writerow(row)
            f.flush()
            v = row[-1].split(":")[0].split("(")[0]
            counts[v] = counts.get(v, 0) + 1
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(paths)} ...", flush=True)

    print("\n=== summary (this run) ===")
    for k in sorted(counts):
        print(f"  {k:35} {counts[k]}")
    print(f"\nwrote -> {args.out}")
    print("Inspect the bad ones:")
    print(f"  grep UNTRAINED {args.out} | sort -t, -k7 -n   # by epoch")


if __name__ == "__main__":
    main()
