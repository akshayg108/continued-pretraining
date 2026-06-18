#!/usr/bin/env python
"""
prebuild_datasets.py — NEW. Build every target dataset's processed cache ONCE, single-threaded.

Why: the sweep calls load_target_dataset PER ckpt. When the cluster killed sweep jobs mid-
extraction, HuggingFace left incomplete builds + stale locks in processed/, so later jobs HUNG
forever trying to (re)extract the same dataset. Building each dataset once, in a single process
with no concurrency, produces a clean complete cache; afterwards every per-ckpt load is a fast
read and the sweep never extracts at runtime.

Run on a LOGIN / CPU node (no GPU), AFTER clearing stale locks (see the recovery steps):
  python eval/prebuild_datasets.py \
      --download-dir  /scratch/gs4133/zhd/CP/data/stable_datasets/downloads \
      --processed-dir /scratch/gs4133/zhd/CP/data/stable_datasets/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_metrics import DS_REGISTRY, load_target_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="/scratch/gs4133/zhd/CP/data/stable_datasets/downloads")
    ap.add_argument("--processed-dir", default="/scratch/gs4133/zhd/CP/data/stable_datasets/processed")
    args = ap.parse_args()

    names = [n for n in DS_REGISTRY if n != "imagenet"]
    print(f"pre-building {len(names)} datasets (single process, no concurrency)...")
    ok, fail = [], []
    for n in names:
        try:
            loader = load_target_dataset(n, args.download_dir, args.processed_dir)
            print(f"  OK   {n}  ({len(loader.dataset)} samples)")
            ok.append(n)
        except Exception as e:
            print(f"  FAIL {n}: {type(e).__name__}: {e}")
            fail.append(n)
    print(f"\ndone: {len(ok)} ok, {len(fail)} failed" + (f" -> {', '.join(fail)}" if fail else ""))
    if fail:
        print("Re-run for the failed ones; if a dataset keeps failing, delete its dir under "
              "processed/ and try again.")


if __name__ == "__main__":
    main()
