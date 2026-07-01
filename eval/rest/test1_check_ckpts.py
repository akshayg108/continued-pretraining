#!/usr/bin/env python
"""Test 1 — are the 60 re-trained checkpoints CORRECT?

For each checkpoint in rerun_geometry.csv: read the Lightning `epoch` and compare the last
transformer block's weights to the timm pretrained reference. PASS = reached epoch >= --min-epoch
AND the backbone actually trained (last-block max|Δ| vs pretrained > 1e-5, i.e. NOT the
admin-cancelled pretrained-weights state we found before). CPU only.

Run:  python eval/rest/test1_check_ckpts.py [--csv eval/outputs/rerun_geometry.csv] [--min-epoch 149]
"""
import argparse
import csv
import os
import sys

import torch
import timm

_PRE = {}


def pretrained_sd(timm_id):
    if timm_id not in _PRE:
        _PRE[timm_id] = timm.create_model(timm_id, pretrained=True, num_classes=0).eval().state_dict()
    return _PRE[timm_id]


def enc_timm(fn):
    if "dinov3" in fn:
        return "DINOv3", "vit_base_patch16_dinov3.lvd1689m"
    if "clip" in fn:
        return "CLIP", "vit_base_patch16_clip_224.openai"
    if "224.mae" in fn:
        return "MAE", "vit_base_patch16_224.mae"
    return "?", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/outputs/rerun_geometry.csv")
    ap.add_argument("--min-epoch", type=int, default=149)
    args = ap.parse_args()

    rows = [r for r in csv.reader(open(args.csv)) if r and r[0].strip() not in ("", "ckpt")]
    print(f"checking {len(rows)} checkpoints  (PASS = epoch>={args.min_epoch} AND backbone trained)\n")
    print(f"{'epoch':>6} {'blk11|d|':>9}  verdict   checkpoint")
    npass = nfail = 0
    for r in rows:
        ck = r[0].strip()
        fn = os.path.basename(ck)
        tag, timm_id = enc_timm(fn)
        try:
            c = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)
        except Exception as e:
            print(f"{'?':>6} {'?':>9}  LOAD_ERR  {fn} ({type(e).__name__})")
            nfail += 1
            continue
        ep = c.get("epoch", "?") if isinstance(c, dict) else "?"
        sd = c.get("state_dict", c) if isinstance(c, dict) else c
        d = float("nan")
        if timm_id:
            pre = pretrained_sd(timm_id)
            probe = "blocks.11.attn.qkv.weight"
            cand = [k for k in sd if k.endswith(probe) and sd[k].shape == pre[probe].shape]
            if cand:
                d = (sd[cand[0]].float() - pre[probe].float()).abs().max().item()
        ok = isinstance(ep, int) and ep >= args.min_epoch and (d == d) and d > 1e-5
        print(f"{str(ep):>6} {d:9.4g}  {'PASS  ' if ok else 'FAIL  '}  {fn}")
        npass += int(ok)
        nfail += int(not ok)

    print(f"\nPASS {npass} / {len(rows)}    FAIL {nfail}")
    if nfail:
        print("FAIL = either epoch < min-epoch (didn't finish) or backbone == pretrained (didn't train).")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
