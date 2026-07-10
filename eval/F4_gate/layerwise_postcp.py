#!/usr/bin/env python
"""
layerwise_postcp.py — Exp I (post-CP side): per-layer internal kNN + uniformity for MAX
checkpoints, so layerwise_law.py can form the per-layer delta against layerwise_pre.csv.

Scope (eval/DESIGN_spectrum_transport.md, Design 2, user-amended to include DIET):
  main grid : methods != MAE (i.e. LeJEPA/SimCLR/DIET) x {DINOv3, CLIP, MAE} x 15 x MAX x seeds
  SigLIP grid: {LeJEPA, SimCLR} x SigLIP x 15 x MAX x seeds (root layout
               <siglip-root>/cp/<method>/<DsFolder>/SigLIP/cp/*.ckpt; absent root -> skipped)

Same layer readout + internal-kNN protocol as layerwise_geometry.py (imports LayerTap /
internal_knn from it — the split seed MUST match for the delta to be valid).

GPU. Cluster (one array task per dataset):
  python eval/adjudicate/layerwise_postcp.py --datasets <ds> \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --siglip-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/layerwise_postcp_shards/<ds>.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path as _P

import numpy as np
import torch

sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'utils'))
from geometry_metrics import load_target_dataset, wang_isola_uniformity  # noqa: E402
from postcp_features import load_cp_backbone  # noqa: E402
from layerwise_geometry import LayerTap, internal_knn, LAYER_READOUT  # noqa: E402
from postcp_class_sweep import discover, MAX_N, DS_FOLDER  # noqa: E402

ROOT = _P(__file__).resolve().parent.parent.parent
SIGLIP_TIMM = "vit_base_patch16_siglip_224.v2_webli"
UNFREEZE_EPOCH = 15   # CP recipe: backbone frozen for epochs 0-14; ckpts that died before
                      # the unfreeze are bit-identical to the pretrained encoder (audit
                      # 2026-06-30 found 6 such SigLIP MAX ckpts) -> degenerate delta rows.
FIELDS = ["method", "encoder", "dataset", "size", "seed", "layer",
          "uniformity_t2", "knn_internal", "ckpt"]


def died_before_unfreeze(ckpt_path):
    """True if the Lightning epoch field shows the backbone never unfroze."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1
    except Exception:
        return False
    return 0 <= epoch < UNFREEZE_EPOCH


def discover_siglip(siglip_root):
    """MAX ckpts under the cp-siglip layout: cp/<method>/<DsFolder>/SigLIP/cp/*.ckpt."""
    root = _P(siglip_root)
    if not root.exists():
        print(f"WARN: SigLIP root {root} absent — skipping SigLIP grid.")
        return []
    out = []
    for f in root.rglob("*.ckpt"):
        parts = f.relative_to(root).parts
        if f.parent.name != "cp" or "SigLIP" not in parts or len(parts) < 5:
            continue
        method = parts[list(parts).index("SigLIP") - 2]
        ds_key = DS_FOLDER.get(parts[list(parts).index("SigLIP") - 1],
                               parts[list(parts).index("SigLIP") - 1].lower())
        m = re.search(r"_n(\d+)_s(\d+)$", f.stem)
        if not m or ds_key not in MAX_N or int(m.group(1)) != MAX_N[ds_key]:
            continue
        out.append(dict(method=method, encoder="SigLIP", timm_id=SIGLIP_TIMM, pool="map",
                        dataset=ds_key, size=int(m.group(1)), seed=int(m.group(2)),
                        ckpt=str(f)))
    return sorted(out, key=lambda c: c["ckpt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp")
    ap.add_argument("--siglip-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip")
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/layerwise_postcp.csv"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # main grid minus the MAE-CP *method* (the MAE *encoder* stays, per the design)
    cfgs = [c for c in discover(args.ckpt_root) if "MAE" not in c["method"].upper()]
    cfgs += discover_siglip(args.siglip_root)
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in args.methods]
    if args.datasets:
        cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cfgs = [c for k, c in enumerate(cfgs) if k % n == i]
    by_m = {}
    for c in cfgs:
        by_m[c["method"], c["encoder"]] = by_m.get((c["method"], c["encoder"]), 0) + 1
    print(f"{len(cfgs)} MAX ckpts to process (device={device}); per (method, encoder): {by_m}")

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
            if died_before_unfreeze(c["ckpt"]):
                print(f"  SKIP untrained (died before unfreeze): {c['ckpt']}")
                continue
            if c["dataset"] not in loaders:
                loaders[c["dataset"]] = load_target_dataset(
                    c["dataset"], args.download_dir, args.processed_dir)
            model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
            tap = LayerTap(model, LAYER_READOUT[c["encoder"]])
            feats, labels = tap.collect(loaders[c["dataset"]], device)
            tap.close()
            rows = []
            for layer in sorted(feats):
                rows.append(dict(method=c["method"], encoder=c["encoder"],
                                 dataset=c["dataset"], size=c["size"], seed=c["seed"],
                                 layer=layer,
                                 uniformity_t2=round(wang_isola_uniformity(feats[layer]), 5),
                                 knn_internal=round(internal_knn(feats[layer], labels), 5),
                                 ckpt=c["ckpt"]))
            w.writerows(rows)      # all 12 layers at once -> no partial ckpts on crash
            fh.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{k+1}/{len(cfgs)}] {c['method']}+{c['encoder']}+{c['dataset']}+s{c['seed']}"
                  f": L12 knn={rows[-1]['knn_internal']} unif={rows[-1]['uniformity_t2']}")
        except Exception as e:
            print(f"  FAIL {c['ckpt']}: {type(e).__name__}: {e}")

    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
