#!/usr/bin/env python
"""
postcp_class_sweep.py — P-D: does CP-induced WITHIN-CLASS spreading mediate the ΔFT side of
the reversal?

Mechanism candidate (complements spread/collision, does not replace them): FT must re-collapse
class manifolds to build label boundaries. Invariance/instance CP spreads instances without
label knowledge; where that spreading lands WITHIN classes (embedded/FG data), the FT
initialization gets worse (ΔFT<0 side of the reversal); where it lands BETWEEN classes (OOD
data), frozen metrics gain at little FT cost. Prediction (falsifiable):
  (i)  Δwithin_spread (post − pre) correlates NEGATIVELY with ΔFT on sphere encoders at MAX,
  (ii) partialling Δwithin_spread out of (pre-CP geometry → ΔFT) weakens the reversal, i.e.
       Δwithin mediates it,
  (iii) DIET-CP shows the largest Δwithin among angular methods (its per-instance geometry
        pushes same-class samples apart hardest) — consistent with its larger FT drops.

Scope: MAX-size checkpoints only (4 methods x {DINOv3, CLIP, MAE} x 15 datasets x 3 seeds
~ 540 ckpts; shardable). Uses the production pooled readout (cls / mean) — same features the
behavioral metrics see. Labels used only for measurement.

GPU. Run (cluster):
  python eval/adjudicate/postcp_class_sweep.py --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/postcp_class_max.csv [--shard i/N]
"""
import argparse
import csv
import re
import sys
from pathlib import Path as _P

import numpy as np
import torch

sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from geometry_metrics import load_target_dataset, extract_features  # noqa: E402
from postcp_features import load_cp_backbone  # noqa: E402
from geometry_class import class_manifold_stats  # noqa: E402

ROOT = _P(__file__).resolve().parent.parent.parent
MAX_N = {"food101": 75750, "octmnist": 97477, "plant_village": 43596, "organamnist": 34561,
         "galaxy10": 14188, "fgvc_aircraft": 3334, "cars196": 8144, "breastmnist": 546,
         "cub200": 5994, "dermamnist": 7007, "dtd": 1880, "eurosat": 16200,
         "flowers102": 1020, "oxford_pet": 3680, "pathmnist": 89996}
DS_FOLDER = {"OctMNIST": "octmnist", "OrganAMNIST": "organamnist", "OxfordPet": "oxford_pet",
             "PlantVillage": "plant_village", "FGVC_Aircraft": "fgvc_aircraft"}
FIELDS = ["method", "encoder", "dataset", "size", "seed",
          "n_classes", "within_spread", "between_spread", "wb_ratio", "nc1_ratio",
          "center_margin", "cdnv", "ckpt"]


def encoder_from_name(fname):
    if "clip" in fname:
        return "CLIP", "vit_base_patch16_clip_224.openai", "cls"
    if "dinov3" in fname:
        return "DINOv3", "vit_base_patch16_dinov3.lvd1689m", "cls"
    if ".mae" in fname:
        return "MAE", "vit_base_patch16_224.mae", "mean"
    return None, None, None


def discover(ckpt_root):
    out = []
    for f in _P(ckpt_root).rglob("*.ckpt"):
        if f.parent.name != "cp":
            continue
        try:
            rel = f.relative_to(ckpt_root)
        except ValueError:
            continue
        if len(rel.parts) < 3 or rel.parts[1] != "pretrained":
            continue
        method = rel.parts[0]
        ds_key = DS_FOLDER.get(rel.parts[2], rel.parts[2].lower())
        m = re.search(r"_n(\d+)_s(\d+)$", f.stem)
        if not m or ds_key not in MAX_N or int(m.group(1)) != MAX_N[ds_key]:
            continue
        enc, timm_id, pool = encoder_from_name(f.stem)
        if enc is None:
            continue
        out.append(dict(method=method, encoder=enc, timm_id=timm_id, pool=pool,
                        dataset=ds_key, size=int(m.group(1)), seed=int(m.group(2)),
                        ckpt=str(f)))
    return sorted(out, key=lambda c: c["ckpt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp")
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/postcp_class_max.csv"))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfgs = discover(args.ckpt_root)
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in args.methods]
    if args.encoders:
        cfgs = [c for c in cfgs if c["encoder"] in args.encoders]
    if args.datasets:
        cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cfgs = [c for k, c in enumerate(cfgs) if k % n == i]
    print(f"{len(cfgs)} MAX ckpts to process (device={device})")

    out = _P(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        import pandas as pd
        done = set(pd.read_csv(out)["ckpt"].astype(str))
    new_file = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    loaders = {}
    for k, c in enumerate(cfgs):
        if c["ckpt"] in done:
            continue
        try:
            if c["dataset"] not in loaders:
                loaders[c["dataset"]] = load_target_dataset(
                    c["dataset"], args.download_dir, args.processed_dir)
            model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
            feat, labels = extract_features(model, loaders[c["dataset"]], device, c["pool"])
            row = class_manifold_stats(feat, labels)
            w.writerow({**{f: c[f] for f in ["method", "encoder", "dataset", "size", "seed",
                                             "ckpt"]}, **row})
            fh.flush()
            print(f"[{k+1}/{len(cfgs)}] {c['method']}+{c['encoder']}+{c['dataset']}+s{c['seed']}"
                  f"  within={row['within_spread']:.4f} wb={row['wb_ratio']:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL {c['ckpt']}: {e}")
    fh.close()
    print(f"done -> {args.out}")
    print("Pre-CP reference: the same stats come from eval/geometry_class.py; "
          "Δ = this file − geometry_class_15.csv, joined on (encoder, dataset).")


if __name__ == "__main__":
    main()
