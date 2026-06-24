#!/usr/bin/env python3
"""
Download ONLY ImageNet-1k validation set (~6.7GB) from HuggingFace.

Uses data_files to avoid downloading the 150GB train split.

Prerequisites:
  1. pip install datasets huggingface_hub
  2. Accept license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  3. huggingface-cli login

Usage:
  python3 download_imagenet_val.py --save-dir /path/to/save
"""

import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./imagenet_val")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ImageNet-1k VALIDATION ONLY (~6.7GB)...")

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        data_files={"validation": "data/validation-*.parquet"},
        verification_mode="no_checks",
    )

    print(f"Downloaded {len(ds)} samples")
    print(f"Saving to {save_dir}...")
    ds.save_to_disk(str(save_dir))
    print(f"Done! Upload {save_dir} to Google Drive.")


if __name__ == "__main__":
    main()
