#!/usr/bin/env python3
"""
ablate_sa_lp.py — localize WHY Selective-Aggregation LP underperforms cls/avgpool LP.

This is a one-off systematic-debugging harness (NOT a production fix). It loads ONE post-CP
encoder checkpoint (default: MAE-CP on a MAE encoder — the matched, non-degrading case the
Beyond-[CLS] paper says SA *must* win on), extracts patch tokens once, then trains 4 probes on
the SAME cached tokens to isolate the cause:

  v0  avgpool-LP            mean(patch) -> L2 -> Linear, Adam            == production `post_linear_f1`
  v1  SA-LP (L2, Adam)      ABMILP      -> L2 -> Linear, Adam            == production `sa_lp_f1` (reproduces the bug)
  v2  SA-LP (+BN, Adam)     ABMILP      -> BatchNorm1d(affine=False) -> Linear, Adam
  v3  SA-LP (+BN, LARS)     ABMILP      -> BatchNorm1d(affine=False) -> Linear, LARS + cosine + warmup

Reading the deltas:  v1 vs v0 = the bug;  v2 vs v1 = effect of BatchNorm (reference) vs L2 (ours);
v3 vs v2 = effect of the reference optimizer/schedule (LARS + cosine + 10-epoch warmup) vs Adam.

Hypothesis (to confirm/refute): v1 < v0  AND  v2 >= v0  (and/or v3 >= v0) -> the failure is the
training RECIPE (L2 + constant-LR Adam), not information loss; the fix is to align the SA probe to
the reference (BatchNorm + LARS/cosine).  If v2 and v3 ALSO stay below v0, that is evidence the
degradation is real (information loss), not a recipe artifact.

Run on the cluster (GPU). The backbone is `vit_base_patch16_224.mae`; pool for the avgpool
baseline is `mean` (the pooling MAE-CP trains with).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score

from stable_cp.data import get_dataset_config
from stable_cp.data.loaders import create_eval_loaders, create_transforms
from stable_cp.utils.backbone import load_backbone
from stable_cp.evaluation.zero_shot_eval import (
    load_backbone_from_checkpoint,
    extract_all_tokens,
    linear_probe_pytorch_evaluate,
    selective_aggregation_lp_evaluate,
)
from stable_cp.evaluation.abmilp import ABMILPHead

BACKBONE = "vit_base_patch16_224.mae"


# --------------------------------------------------------------------------------------
# Vendored LARS (MAE/MoCo-v3 reference implementation) — used only by variant v3.
# --------------------------------------------------------------------------------------
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0, weight_decay=0.0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                      momentum=momentum, trust_coefficient=trust_coefficient))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if p.ndim > 1:  # adapt only weight matrices (not bias / 1-D params)
                    dp = dp.add(p, alpha=g["weight_decay"])
                    pn, un = torch.norm(p), torch.norm(dp)
                    one = torch.ones_like(pn)
                    q = torch.where(pn > 0.0,
                                    torch.where(un > 0.0, g["trust_coefficient"] * pn / un, one),
                                    one)
                    dp = dp.mul(q)
                st = self.state[p]
                if "mu" not in st:
                    st["mu"] = torch.zeros_like(p)
                st["mu"].mul_(g["momentum"]).add_(dp)
                p.add_(st["mu"], alpha=-g["lr"])


def _num_epochs(n, batch_size, min_epochs=150, min_steps=10000):
    """Match production: selective_aggregation_lp_evaluate epoch count."""
    steps_per_epoch = max(n // batch_size, 1)
    return max(min_epochs, (min_steps + steps_per_epoch - 1) // steps_per_epoch)


def train_sa_probe(train_tokens, train_labels, test_tokens, test_labels, device,
                   norm="bn", optim="adam", lr=1e-3, blr=0.1, batch_size=512,
                   warmup_epochs=10, verbose=True):
    """SA probe with configurable normalization (bn|l2) and optimizer (adam|lars+cosine+warmup)."""
    num_patches = train_tokens.shape[1] - 1
    dim = train_tokens.shape[-1]
    num_classes = int(len(np.unique(train_labels)))

    sa_head = ABMILPHead(dim=dim, self_attention_apply_to="none", depth=1,
                         cond="none", content="patch", num_patches=num_patches).to(device)
    bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6).to(device) if norm == "bn" else None
    clf = nn.Linear(dim, num_classes).to(device)

    train_tokens_t = torch.from_numpy(train_tokens).float()
    train_labels_t = torch.from_numpy(train_labels).long()
    test_tokens_t = torch.from_numpy(test_tokens).float()

    params = list(sa_head.parameters()) + list(clf.parameters())
    n = len(train_tokens_t)
    num_epochs = _num_epochs(n, batch_size)
    steps_per_epoch = max(n // batch_size, 1)

    if optim == "lars":
        eff_lr = blr * batch_size / 256.0
        optimizer = LARS(params, lr=eff_lr, weight_decay=0.0)
    else:
        eff_lr = lr
        optimizer = torch.optim.Adam(params, lr=eff_lr)
    criterion = nn.CrossEntropyLoss()

    def lr_for(epoch):
        if optim != "lars":
            return eff_lr  # constant (matches production Adam)
        if epoch < warmup_epochs:
            return eff_lr * (epoch + 1) / max(warmup_epochs, 1)
        prog = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * eff_lr * (1.0 + math.cos(math.pi * prog))

    def pool(x):
        agg = sa_head(x)
        agg = bn(agg) if bn is not None else F.normalize(agg, dim=-1)
        return clf(agg)

    sa_head.train(); clf.train()
    if bn is not None:
        bn.train()
    for epoch in range(num_epochs):
        for grp in optimizer.param_groups:
            grp["lr"] = lr_for(epoch)
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            if len(idx) < 2:  # BatchNorm needs >=2 samples
                continue
            x = train_tokens_t[idx].to(device, non_blocking=True)
            y = train_labels_t[idx].to(device, non_blocking=True)
            logits = pool(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % max(num_epochs // 5, 1) == 0:
            print(f"      [{norm}/{optim}] epoch {epoch + 1}/{num_epochs} lr={lr_for(epoch):.2e} loss={loss.item():.4f}")

    sa_head.eval(); clf.eval()
    if bn is not None:
        bn.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_tokens_t), batch_size):
            x = test_tokens_t[i:i + batch_size].to(device, non_blocking=True)
            preds.append(pool(x).argmax(dim=1).cpu())
    pred = torch.cat(preds)
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro")(
        pred, torch.from_numpy(test_labels).long()).item()
    return f1


def resolve_ckpt(ckpt_dir: Path, n_samples: int, seed: int) -> Path:
    """Find the post-CP encoder ckpt for (n_samples, seed) under a config dir (prefer the 'all' bucket)."""
    cands = [f for f in ckpt_dir.rglob(f"*_n{n_samples}_s{seed}.ckpt")]
    if not cands:
        raise FileNotFoundError(f"no *_n{n_samples}_s{seed}.ckpt under {ckpt_dir}")
    cands.sort(key=lambda p: (0 if "all" in p.parts else 1, str(p)))  # prefer .../all/cp/...
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="Ablate SA-LP recipe on a MAE-CP+MAE checkpoint.")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help="config dir, e.g. .../ckpts/cp/MAE/pretrained/Cars196/MAE (auto-finds MAX ckpt)")
    ap.add_argument("--checkpoint", type=str, default=None, help="explicit .ckpt path (overrides --ckpt-dir)")
    ap.add_argument("--dataset", type=str, default="cars196")
    ap.add_argument("--n-samples", type=int, default=8144)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=str, default="/scratch/gs4133/zhd/CP/data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam lr for v0/v1/v2 (matches production)")
    ap.add_argument("--blr", type=float, default=0.1, help="LARS base lr for v3 (eff = blr*bs/256)")
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.ckpt_dir:
        ckpt = resolve_ckpt(Path(args.ckpt_dir), args.n_samples, args.seed)
    else:
        raise SystemExit("provide --checkpoint or --ckpt-dir")
    print(f"checkpoint: {ckpt}")
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build the same eval loaders the post-CP pipeline uses ---
    ds_cfg = get_dataset_config(args.dataset)
    loader_args = SimpleNamespace(dataset=args.dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, seed=args.seed,
                                  n_samples=args.n_samples)
    train_tf, eval_tf = create_transforms(ds_cfg, n_views=1, strong_aug=False)
    test_loader, eval_train_loader, _ = create_eval_loaders(
        loader_args, ds_cfg, eval_train_transform=train_tf, val_transform=eval_tf,
        data_dir=args.cache_dir)

    # --- load the post-CP MAE encoder ---
    backbone, _ = load_backbone(BACKBONE)
    load_backbone_from_checkpoint(backbone, str(ckpt), strict=False)
    backbone = backbone.to(device).eval()

    # --- extract patch tokens once (N, 1+L, D) ---
    print("extracting tokens (train/test)...")
    train_tokens, train_labels = extract_all_tokens(backbone, eval_train_loader, device)
    test_tokens, test_labels = extract_all_tokens(backbone, test_loader, device)
    print(f"tokens: train={train_tokens.shape} test={test_tokens.shape}")

    results = {}

    # v0 — avgpool-LP (== production post_linear_f1: mean(patch) -> L2 -> Adam linear)
    print("\n[v0] avgpool-LP ...")
    mean_train = train_tokens[:, 1:, :].mean(axis=1)
    mean_test = test_tokens[:, 1:, :].mean(axis=1)
    results["v0_avgpool_LP"] = linear_probe_pytorch_evaluate(
        mean_train, train_labels, mean_test, test_labels, device=device,
        lr=args.lr, verbose=False)["linear_pytorch_f1"]

    # v1 — SA-LP current (L2 + Adam) == production code (reproduces the bug)
    print("[v1] SA-LP (L2, Adam) = production ...")
    results["v1_SA_L2_Adam"] = selective_aggregation_lp_evaluate(
        train_tokens, train_labels, test_tokens, test_labels, device=device,
        lr=args.lr, verbose=False)["sa_lp_f1"]

    # v2 — SA-LP + BatchNorm (Adam)
    print("[v2] SA-LP (+BN, Adam) ...")
    results["v2_SA_BN_Adam"] = train_sa_probe(
        train_tokens, train_labels, test_tokens, test_labels, device,
        norm="bn", optim="adam", lr=args.lr, batch_size=args.batch_size)

    # v3 — SA-LP + BatchNorm + LARS + cosine + warmup (reference recipe)
    print("[v3] SA-LP (+BN, LARS+cosine+warmup) ...")
    results["v3_SA_BN_LARS"] = train_sa_probe(
        train_tokens, train_labels, test_tokens, test_labels, device,
        norm="bn", optim="lars", blr=args.blr, batch_size=args.batch_size)

    # --- report ---
    base = results["v0_avgpool_LP"]
    print("\n" + "=" * 64)
    print(f"SA-LP recipe ablation — {args.dataset}  MAE-CP+MAE  n={args.n_samples} s={args.seed}")
    print("=" * 64)
    print(f"{'variant':22} {'macro-F1':>9} {'Δ vs avgpool':>13}")
    print("-" * 48)
    for k, v in results.items():
        d = "" if k == "v0_avgpool_LP" else f"{v - base:+.4f}"
        print(f"{k:22} {v:9.4f} {d:>13}")
    print("=" * 64)
    print("read: v1<v0 reproduces the bug; v2>=v0 => BatchNorm fixes it (recipe, not info-loss);")
    print("      v3>=v2 => optimizer/schedule helps further; v2&v3<v0 => degradation is real (info-loss).")


if __name__ == "__main__":
    main()
