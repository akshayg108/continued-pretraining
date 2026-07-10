#!/usr/bin/env python
"""
nd2_spectral_sweep.py — ND2 (GPU pass): spectral geometry over ALL post-CP checkpoints.

Same discovery / resume / shard machinery as eval/F2_forces/postcp_sweep.py (imported from
it), but records the spectral vocabulary instead of the sphere metrics:
  rankme, alpha(-ReQ) + fit R2, coherence_mu (+99% variant), uniformity_t2 (to co-locate
  the F2 spread force and the spectral axes on the same rows).

Purpose (papers/new_direction/NEW_DIRECTION.md §H-4 / F2): Li et al. 2025 find LLM
pretraining passes through entropy-seeking EXPANSION then compression-seeking ANISOTROPIC
consolidation, measured with exactly RankMe + alpha-ReQ. This pass measures the same two
scalars along our CP size axis; eval/new_direction/nd2_verdict.py (CPU, local) joins
dknn + the pre-CP baseline (nd1_precp_spectral.csv) and adjudicates. Main cp root only
(DINOv3 / CLIP / MAE backbones); the SigLIP grid is not needed for the F2 question.

Cluster (shard-parallel, mirrors postcp_sweep):
  python eval/new_direction/nd2_spectral_sweep.py \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --download-dir <raw> --processed-dir <arrow> \
      --shard ${i}/8 --out eval/outputs/nd2_spectral_shards/shard_${i}.csv
"""
import argparse
import csv
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                      # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))     # eval/utils
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "F2_forces")) # sweep machinery
from postcp_sweep import discover                                            # noqa: E402
from postcp_features import load_cp_backbone                                  # noqa: E402
from geometry_metrics import (load_target_dataset, extract_features,          # noqa: E402
                              wang_isola_uniformity)
from spectral_metrics import spectral_row                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = ["method", "variant", "encoder", "dataset", "size", "seed", "n_samples",
          "rankme", "alpha", "alpha_r2", "coherence_mu", "coherence_mu99",
          "uniformity_t2", "ckpt"]
KEY = ("method", "variant", "encoder", "dataset", "size", "seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/nd2_spectral_sweep.csv"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=["pretrained"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N: only checkpoint indices == i (mod N)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfgs = discover(args.ckpt_root)
    cfgs = [c for c in cfgs if c["variant"] in args.variants and c["encoder"] != "RANDOM"]
    if args.datasets:
        cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in args.methods]
    if args.seeds:
        cfgs = [c for c in cfgs if c["seed"] in args.seeds]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        cfgs = [c for j, c in enumerate(cfgs) if j % n == i]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {tuple(str(r[k]) for k in KEY) for r in csv.DictReader(f)}
    cfgs = [c for c in cfgs if tuple(str(c[k]) for k in KEY) not in done]
    print(f"{len(cfgs)} checkpoints to process ({len(done)} already done)  device={device}")

    loaders = {}   # dataset features are re-extracted per ckpt; loaders cached per dataset
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for idx, c in enumerate(cfgs):
            print(f"[{idx + 1}/{len(cfgs)}] {c['method']}/{c['encoder']}/{c['dataset']}"
                  f"/n{c['size']}/s{c['seed']}")
            # per-ckpt guard, same semantics as postcp_sweep: a corrupt/diverged ckpt is
            # logged and skipped (retried on next resume), never wedges the whole shard.
            try:
                if c["dataset"] not in loaders:
                    loaders[c["dataset"]] = load_target_dataset(
                        c["dataset"], args.download_dir, args.processed_dir)
                model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
                feat, _ = extract_features(model, loaders[c["dataset"]], device, c["pool"])
                row = {k: c[k] for k in KEY}
                row.update(ckpt=c["ckpt"], n_samples=len(feat),
                           uniformity_t2=wang_isola_uniformity(feat))
                row.update(spectral_row(feat))
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                del model
            except Exception as e:
                print(f"  FAIL {c['ckpt']}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
