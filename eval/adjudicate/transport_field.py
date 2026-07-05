#!/usr/bin/env python
"""
transport_field.py — Exp J: measure the CP transport itself. Every sample passes through the
pre and the post encoder in the SAME order (load_target_dataset is shuffle=False with a fixed
stratified subset), so each sample has a displacement vector d_i = z^post_i - z^pre_i on the
unit sphere. Exact variance decomposition (eval/DESIGN_spectrum_transport.md, Design 3):

  E||d||^2 = ||mu_d||^2                        (global translation)
           + sum_c (n_c/N) ||mu_c - mu_d||^2   (between-class motion)
           + E||d_i - mu_c(i)||^2              (within-class scramble)

plus the toward-ImageNet projection <mu_d, v_E>, v_E = normalized centroid of the PRE
encoder's ImageNet-val cloud — the vector version of collision.

Scope: ALL 4 methods (incl. MAE-CP, user amendment) x {DINOv3, CLIP, MAE} x 15 x MAX x seeds
(the Exp-H checkpoint set). Labels used only for measurement.

GPU. Cluster (one array task per dataset; stages dataset + ImageNet-val):
  python eval/adjudicate/transport_field.py --datasets <ds> \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --imagenet-dir <imagenet_val> --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/transport_field_shards/<ds>.csv
"""
import argparse
import csv
import sys
from pathlib import Path as _P

import numpy as np
import torch
from sklearn.preprocessing import normalize

sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from geometry_metrics import (ENCODERS, load_target_dataset, load_imagenet_val,  # noqa: E402
                              extract_features)
from postcp_features import load_cp_backbone  # noqa: E402
from postcp_class_sweep import discover  # noqa: E402

ROOT = _P(__file__).resolve().parent.parent.parent
FIELDS = ["method", "encoder", "dataset", "size", "seed", "n",
          "total_energy", "trans_energy", "between_energy", "within_energy",
          "resid_identity", "mu_norm", "toward_imagenet", "cos_mu_imagenet", "ckpt"]


def decompose(pre_feats, post_feats, labels, v_e):
    a, b = normalize(pre_feats), normalize(post_feats)
    d = b - a
    mu = d.mean(0)
    total = float((d ** 2).sum(1).mean())
    trans = float((mu ** 2).sum())
    between = within = 0.0
    for c in np.unique(labels):
        dc = d[labels == c]
        mc = dc.mean(0)
        wgt = len(dc) / len(d)
        between += wgt * float(((mc - mu) ** 2).sum())
        within += wgt * float(((dc - mc) ** 2).sum(1).mean())
    mu_norm = float(np.linalg.norm(mu))
    toward = float(mu @ v_e)
    return dict(n=len(d),
                total_energy=round(total, 6), trans_energy=round(trans, 6),
                between_energy=round(between, 6), within_energy=round(within, 6),
                resid_identity=round(total - trans - between - within, 10),
                mu_norm=round(mu_norm, 6), toward_imagenet=round(toward, 6),
                cos_mu_imagenet=round(toward / (mu_norm + 1e-12), 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp")
    ap.add_argument("--imagenet-dir", type=str, required=True)
    ap.add_argument("--imagenet-samples", type=int, default=5000)
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/transport_field_max.csv"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfgs = discover(args.ckpt_root)          # ALL methods incl. MAE-CP
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in args.methods]
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

    import timm
    imn_loader = load_imagenet_val(args.imagenet_dir, args.imagenet_samples)
    loaders, pre_cache = {}, {}   # pre_cache[(enc, ds)] = (pre_feats, labels, v_e)
    for k, c in enumerate(cfgs):
        if c["ckpt"] in done:
            continue
        try:
            key = (c["encoder"], c["dataset"])
            if c["dataset"] not in loaders:
                loaders[c["dataset"]] = load_target_dataset(
                    c["dataset"], args.download_dir, args.processed_dir)
            if key not in pre_cache:
                cfg = ENCODERS[c["encoder"]]
                pre_model = timm.create_model(cfg["timm_id"], pretrained=True,
                                              num_classes=0).eval().to(device)
                pre_feats, labels = extract_features(pre_model, loaders[c["dataset"]],
                                                     device, cfg["pool"])
                f_imn, _ = extract_features(pre_model, imn_loader, device, cfg["pool"])
                v_e = normalize(normalize(f_imn).mean(0, keepdims=True))[0]
                pre_cache[key] = (pre_feats, labels, v_e)
                del pre_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  cached pre features + ImageNet direction for {key}")
            pre_feats, labels, v_e = pre_cache[key]
            model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
            post_feats, _ = extract_features(model, loaders[c["dataset"]], device, c["pool"])
            row = dict(method=c["method"], encoder=c["encoder"], dataset=c["dataset"],
                       size=c["size"], seed=c["seed"], ckpt=c["ckpt"])
            row.update(decompose(pre_feats, post_feats, labels, v_e))
            w.writerow(row)
            fh.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{k+1}/{len(cfgs)}] {c['method']}+{c['encoder']}+{c['dataset']}+s{c['seed']}: "
                  f"total={row['total_energy']} trans={row['trans_energy']} "
                  f"within={row['within_energy']} toward={row['toward_imagenet']} "
                  f"resid={row['resid_identity']}")
        except Exception as e:
            print(f"  FAIL {c['ckpt']}: {type(e).__name__}: {e}")

    print(f"\nwrote -> {out}")
    print("Next (login node): python eval/adjudicate/transport_law.py")


if __name__ == "__main__":
    main()
