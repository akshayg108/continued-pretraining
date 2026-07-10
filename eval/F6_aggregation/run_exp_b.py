#!/usr/bin/env python
"""
run_exp_b.py — Exp B: Selective-Aggregation LP over the post-CP checkpoints.

For each post-CP encoder checkpoint (method x encoder x dataset x MAX x seed) it loads the post-CP
backbone, extracts patch tokens once, and on the SAME tokens computes:
  - baseline_lp_f1 : the production avgpool/cls LP  (mean for MAE encoder, cls for DINOv3/CLIP)
                     == the `post_linear_f1` the original runs reported.
  - sa_lp_f1       : Selective-Aggregation LP — depth-1 ABMILP + BatchNorm1d(affine=False) + Adam.
                     This is the recipe the ablation confirmed (BatchNorm, NOT L2; LARS not needed).

Reading: sa_lp >> baseline  => MAE-CP degradation is an AGGREGATION FAILURE (recoverable by a learned
pool); sa_lp ~= baseline (both low) => INFORMATION LOSS.

Scope (default): {MAE, LeJEPA} x {DINOv3, CLIP, MAE} x 7 severity-spanning datasets x MAX x seeds.
Resumable (skips ckpts already in the output CSV); SLURM-friendly (--shard i/N, or one --datasets per
array task so the per-dataset processed cache can be staged node-local).

Run (cluster, CP env):
  python eval/run_exp_b.py --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --cache-dir /scratch/gs4133/zhd/CP/data --datasets cars196 \
      --out eval/outputs/exp_b/cars196.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/ root: make `import postcp_features` work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))  # eval/utils: make `import postcp_features` work
from postcp_features import load_cp_backbone  # noqa: E402  (proven post-CP loader)

from stable_cp.data import get_dataset_config  # noqa: E402
from stable_cp.data.loaders import create_eval_loaders, create_transforms  # noqa: E402
from stable_cp.evaluation.zero_shot_eval import linear_probe_pytorch_evaluate  # noqa: E402
from stable_cp.evaluation.abmilp import ABMILPHead  # noqa: E402

METHODS = {"MAE", "LeJEPA"}                       # CP-method folder names for Exp B
ENCODERS_OK = {"DINOv3", "CLIP", "MAE"}
# dataset key -> MAX n_samples (hardcoded; Exp B uses the MAX size only)
MAX_N = {
    "food101": 75750, "octmnist": 97477, "plant_village": 43596, "organamnist": 34561,
    "galaxy10": 14188, "fgvc_aircraft": 3334, "cars196": 8144, "breastmnist": 546,
    "cub200": 5994, "dermamnist": 7007, "dtd": 1880, "eurosat": 16200,
    "flowers102": 1020, "oxford_pet": 3680, "pathmnist": 89996,
}
FIELDS = ["method", "encoder", "dataset", "size", "seed", "pool",
          "baseline_lp_f1", "sa_lp_f1", "sa_minus_baseline", "ckpt"]


def encoder_from_name(fname):
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
    if variant != "pretrained":
        return None
    m = re.search(r"_n(\d+)_s(\d+)$", p.stem)
    if not m:
        return None
    enc, timm_id, pool = encoder_from_name(p.stem)
    ds_key = {"OctMNIST": "octmnist", "OrganAMNIST": "organamnist", "OxfordPet": "oxford_pet",
              "PlantVillage": "plant_village", "FGVC_Aircraft": "fgvc_aircraft"}.get(
        dataset_folder, dataset_folder.lower())
    return dict(method=method, encoder=enc, timm_id=timm_id, pool=pool, dataset=ds_key,
                size=int(m.group(1)), seed=int(m.group(2)), ckpt=str(p))


def discover(ckpt_root):
    """All post-CP ckpts (parent dir 'cp'), keeping only the hardcoded MAX size per dataset."""
    out = []
    for f in Path(ckpt_root).rglob("*.ckpt"):
        if f.parent.name == "cp":
            cfg = parse_ckpt(f, ckpt_root)
            if cfg and cfg["encoder"] in ENCODERS_OK and cfg["dataset"] in MAX_N \
                    and cfg["size"] == MAX_N[cfg["dataset"]]:
                out.append(cfg)
    return sorted(out, key=lambda c: c["ckpt"])


@torch.no_grad()
def extract_tokens(model, loader, device):
    """Full token sequence (N, 1+L, D), float32 (fits in RAM with --mem 256G)."""
    toks, labs = [], []
    model.eval()
    for batch in loader:
        if isinstance(batch, dict):
            x, y = batch["image"], batch["label"]
        else:
            x, y = batch[0], batch[1]
        feat = model.forward_features(x.to(device))
        assert feat.dim() == 3, f"expected (B,1+L,D) tokens, got {feat.shape}"
        toks.append(feat.cpu().numpy())
        labs.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.concatenate(toks), np.concatenate(labs).ravel()


def sa_lp_bn(train_tokens, train_labels, test_tokens, test_labels, device,
             lr=1e-3, batch_size=512, min_epochs=150, min_steps=10000):
    """Selective-Aggregation LP — depth-1 ABMILP + BatchNorm1d(affine=False) + Adam (confirmed recipe)."""
    num_patches = train_tokens.shape[1] - 1
    dim = train_tokens.shape[-1]
    num_classes = int(len(np.unique(train_labels)))
    sa = ABMILPHead(dim=dim, self_attention_apply_to="none", depth=1, cond="none",
                    content="patch", num_patches=num_patches).to(device)
    bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6).to(device)
    clf = nn.Linear(dim, num_classes).to(device)

    tr = torch.from_numpy(train_tokens)            # float16, CPU
    trl = torch.from_numpy(train_labels).long()
    te = torch.from_numpy(test_tokens)
    opt = torch.optim.Adam(list(sa.parameters()) + list(clf.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(tr)
    spe = max(n // batch_size, 1)
    epochs = max(min_epochs, (min_steps + spe - 1) // spe)

    sa.train(); bn.train(); clf.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(spe):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            if len(idx) < 2:
                continue
            x = tr[idx].to(device, non_blocking=True).float()
            y = trl[idx].to(device, non_blocking=True)
            loss = crit(clf(bn(sa(x))), y)
            opt.zero_grad(); loss.backward(); opt.step()

    sa.eval(); bn.eval(); clf.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), batch_size):
            x = te[i:i + batch_size].to(device, non_blocking=True).float()
            preds.append(clf(bn(sa(x))).argmax(dim=1).cpu())
    pred = torch.cat(preds)
    return MulticlassF1Score(num_classes=num_classes, average="macro")(
        pred, torch.from_numpy(test_labels).long()).item()


def main():
    ap = argparse.ArgumentParser(description="Exp B: SA-LP vs baseline LP over post-CP checkpoints.")
    ap.add_argument("--ckpt-root", default="/scratch/gs4133/zhd/CP/outputs/ckpts/cp")
    ap.add_argument("--cache-dir", default="/scratch/gs4133/zhd/CP/data")
    ap.add_argument("--out", default="eval/outputs/exp_b/sa_lp.csv")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N: process only config indices ≡ i (mod N)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfgs = discover(args.ckpt_root)
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in args.methods]
    if args.encoders:
        cfgs = [c for c in cfgs if c["encoder"] in args.encoders]
    if args.datasets:
        cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    if args.seeds:
        cfgs = [c for c in cfgs if c["seed"] in args.seeds]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cfgs = [c for k, c in enumerate(cfgs) if k % n == i]
    print(f"{len(cfgs)} configs to process (device={device})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        import pandas as pd
        done = set(pd.read_csv(out)["ckpt"].astype(str))
        print(f"  resuming: {len(done)} already done")
    new_file = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for k, c in enumerate(cfgs):
        if c["ckpt"] in done:
            continue
        try:
            ds_cfg = get_dataset_config(c["dataset"])
            la = SimpleNamespace(dataset=c["dataset"], batch_size=args.batch_size,
                                 num_workers=args.num_workers, seed=c["seed"], n_samples=c["size"])
            train_tf, eval_tf = create_transforms(ds_cfg, n_views=1, strong_aug=False)
            test_loader, eval_train_loader, _ = create_eval_loaders(
                la, ds_cfg, eval_train_transform=train_tf, val_transform=eval_tf,
                data_dir=args.cache_dir)

            model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
            tr_tok, tr_lab = extract_tokens(model, eval_train_loader, device)
            te_tok, te_lab = extract_tokens(model, test_loader, device)

            # baseline LP on the production pooling (cls / mean), same tokens
            if c["pool"] == "mean":
                base_tr = tr_tok[:, 1:, :].mean(axis=1)
                base_te = te_tok[:, 1:, :].mean(axis=1)
            else:  # cls
                base_tr = tr_tok[:, 0, :]
                base_te = te_tok[:, 0, :]
            base_f1 = linear_probe_pytorch_evaluate(
                base_tr, tr_lab, base_te, te_lab, device=device, lr=args.lr,
                verbose=False)["linear_pytorch_f1"]

            sa_f1 = sa_lp_bn(tr_tok, tr_lab, te_tok, te_lab, device,
                             lr=args.lr, batch_size=args.batch_size)

            w.writerow({"method": c["method"], "encoder": c["encoder"], "dataset": c["dataset"],
                        "size": c["size"], "seed": c["seed"], "pool": c["pool"],
                        "baseline_lp_f1": round(base_f1, 4), "sa_lp_f1": round(sa_f1, 4),
                        "sa_minus_baseline": round(sa_f1 - base_f1, 4), "ckpt": c["ckpt"]})
            fh.flush()
            print(f"[{k + 1}/{len(cfgs)}] {c['method']}+{c['encoder']}+{c['dataset']}+s{c['seed']}"
                  f"  base={base_f1:.4f} sa={sa_f1:.4f} Δ={sa_f1 - base_f1:+.4f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL {c['ckpt']}: {e}")
    fh.close()
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
