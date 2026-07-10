#!/usr/bin/env python
"""
postcp_sweep.py — Scaled Exp A: post-CP intrinsic geometry over ALL cp/ checkpoints.

Auto-discovers every post-CP backbone checkpoint under --ckpt-root (only dirs literally named
`cp`, so `sft_post` is ignored), parses (method, variant, encoder, dataset, size, seed) from the
path + filename, loads the post-CP backbone, extracts [cls]/mean features on the target dataset,
and records:
  - l2_norm_cv        (Exp A: how far CP pushed features off the sphere)
  - uniformity_t2     (free F3 intrinsic metric: how spread the cloud is; no ImageNet needed)
  - neighbor_overlap  (only if --imagenet-dir given; EXPENSIVE — ImageNet re-embedded per ckpt)

Cheap per checkpoint (one forward over <=5000 imgs, no ImageNet for CV+uniformity).
Resumable (skips ckpts already in the output CSV) and SLURM-friendly (--shard i/N).

Path layout assumed (relative to --ckpt-root):
  <METHOD>/<variant>/<Dataset>/<Backbone?>/[<bucket>/]cp/<dataset>_<timm_id>_n<size>_s<seed>.ckpt
(METHOD in DIET/LeJEPA/MAE/SimCLR; variant pretrained|random; random has no <Backbone> level.)

Run (Colab/cluster, in the CP env):
  python eval/postcp_sweep.py --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/postcp_sweep.csv
  # restrict + parallelise:
  python eval/postcp_sweep.py --ckpt-root ... --seeds 42 --shard 0/8 --out .../sweep_0.csv
"""
import argparse
import csv
import re
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent))  # eval/ root
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / 'utils'))  # eval/utils
from postcp_features import load_cp_backbone
from geometry_metrics import (
    load_target_dataset, extract_features, l2_norm_stats,
    wang_isola_uniformity, neighbor_overlap, load_imagenet_val,
)
from load_results import DATASET_KEY

METHODS = {"DIET", "LeJEPA", "MAE", "SimCLR"}
FIELDS = ["method", "variant", "encoder", "dataset", "size", "seed",
          "n_samples", "l2_norm_cv", "uniformity_t2", "neighbor_overlap_k50", "ckpt"]


def encoder_from_name(fname):
    """Robustly read encoder + timm_id + pool from the checkpoint filename."""
    if "clip" in fname:
        return "CLIP", "vit_base_patch16_clip_224.openai", "cls"
    if "dinov3" in fname:
        return "DINOv3", "vit_base_patch16_dinov3.lvd1689m", "cls"
    if ".mae" in fname:
        return "MAE", "vit_base_patch16_224.mae", "mean"
    return "RANDOM", "vit_base_patch16_224", "cls"


def parse_ckpt(path, ckpt_root):
    p = Path(path)
    try:
        rel = p.relative_to(ckpt_root)
    except ValueError:
        return None
    if len(rel.parts) < 3 or rel.parts[0] not in METHODS:
        return None
    method, variant, dataset_folder = rel.parts[0], rel.parts[1], rel.parts[2]
    m = re.search(r"_n(\d+)_s(\d+)$", p.stem)
    if not m:
        return None
    enc, timm_id, pool = encoder_from_name(p.stem)
    return dict(method=method, variant=variant, encoder=enc, timm_id=timm_id, pool=pool,
                dataset=DATASET_KEY.get(dataset_folder, dataset_folder.lower()),
                size=int(m.group(1)), seed=int(m.group(2)), ckpt=str(p))


def discover(ckpt_root):
    """All *.ckpt whose immediate parent dir is named 'cp' (excludes sft_post)."""
    out = []
    for f in Path(ckpt_root).rglob("*.ckpt"):
        if f.parent.name == "cp":
            cfg = parse_ckpt(f, ckpt_root)
            if cfg:
                out.append(cfg)
    return sorted(out, key=lambda c: c["ckpt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, help="dir containing <METHOD>/ subdirs (…/ckpts/cp)")
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imagenet-dir", default=None, help="if set, also compute overlap (EXPENSIVE)")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N: process only checkpoint indices ≡ i (mod N)")
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfgs = discover(args.ckpt_root)
    if args.methods:  cfgs = [c for c in cfgs if c["method"] in args.methods]
    if args.encoders: cfgs = [c for c in cfgs if c["encoder"] in args.encoders]
    if args.datasets: cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    if args.seeds:    cfgs = [c for c in cfgs if c["seed"] in args.seeds]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cfgs = [c for k, c in enumerate(cfgs) if k % n == i]
    print(f"{len(cfgs)} checkpoints to process (device={device})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        import pandas as pd
        done = set(pd.read_csv(out)["ckpt"].astype(str))
        print(f"  resuming: {len(done)} already done")
    new_file = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    imn_loader = load_imagenet_val(args.imagenet_dir, args.imagenet_samples) if args.imagenet_dir else None

    for k, c in enumerate(cfgs):
        if c["ckpt"] in done:
            continue
        try:
            model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
            loader = load_target_dataset(c["dataset"], args.download_dir, args.processed_dir)
            feat, _ = extract_features(model, loader, device, c["pool"])
            _, _, cv = l2_norm_stats(feat)
            unif = wang_isola_uniformity(feat, t=2.0, l2_normalize=True)
            overlap = ""
            if imn_loader is not None:
                feat_i, _ = extract_features(model, imn_loader, device, c["pool"])
                overlap = round(neighbor_overlap(feat, feat_i, k=50), 4)
            w.writerow({"method": c["method"], "variant": c["variant"], "encoder": c["encoder"],
                        "dataset": c["dataset"], "size": c["size"], "seed": c["seed"],
                        "n_samples": len(feat), "l2_norm_cv": round(cv, 5),
                        "uniformity_t2": round(unif, 4), "neighbor_overlap_k50": overlap,
                        "ckpt": c["ckpt"]})
            f.flush()
            print(f"[{k+1}/{len(cfgs)}] {c['method']}+{c['encoder']}+{c['dataset']}+n{c['size']}+s{c['seed']}"
                  f"  cv={cv:.4f} unif={unif:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL {c['ckpt']}: {e}")
    f.close()
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
