#!/usr/bin/env python
"""
nd4_projector_spectra.py — ND4 (GPU pass): pre- vs post-head spectra of the CP checkpoints.

Jing et al. 2022 (papers/new_direction/Jing2022_DimensionalCollapse.pdf, Fig. 7b):
with a projector, dimensional collapse is confined to the post-projector embedding space
and the encoder-output spectrum does not collapse; without one, collapse propagates into
the representation space. ND4 measures both sides on OUR checkpoints: per MAX checkpoint
(method in LeJEPA/SimCLR/DIET, all encoders incl. the SigLIP grid), extract
  backbone features  (encoder output, standard eval protocol)          -> *_bb columns
  head outputs       (the method's own trained head applied to them)   -> *_head columns
and record rankme / alpha-ReQ / coherence for each, plus uniformity_t2 on the backbone.

Head application mirrors the training forwards exactly:
  SimCLR  projector(emb)                       (simclr_cp_forward.py:42)
  LeJEPA  projector(emb)                       (lejepa_forward.py:51)
  DIET    diet_head(F.normalize(emb))          (diet_forward.py:58 — directional logits)
Heads are REBUILT from checkpoint weight shapes and loaded with strict=True — a shape or
key mismatch is a hard error, not a silent skip. MAE-CP is excluded (reconstruction
decoder, not an embedding head). Checkpoints that died before the epoch-15 unfreeze are
skipped via layerwise_postcp.died_before_unfreeze (audit 2026-06-30: 6 SigLIP MAX ckpts).
Log lines: "SKIP untrained" = dead ckpt; "HEAD SKIP" = no head weights in the state_dict;
"CKPT FAIL" = unreadable file; "CELL FAIL" = extraction/spectral failure (e.g. diverged).

eval/new_direction/nd4_verdict.py (CPU, local) joins the pre-CP baseline
(nd1_precp_spectral.csv) and adjudicates the buffer/gate readouts.

Cluster (one array task per dataset):
  python eval/new_direction/nd4_projector_spectra.py --datasets <ds> \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --siglip-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/nd4_projector_shards/<ds>.csv
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                      # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))     # eval/utils
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "F2_forces")) # sweep machinery
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "F4_gate"))   # dead-ckpt guard
from postcp_sweep import discover                                            # noqa: E402
from layerwise_postcp import died_before_unfreeze                             # noqa: E402
from postcp_features import load_cp_backbone                                  # noqa: E402
from geometry_metrics import (load_target_dataset, extract_features,          # noqa: E402
                              wang_isola_uniformity)
from load_results import DATASET_KEY                                          # noqa: E402
from spectral_metrics import spectral_row                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
METHODS = ["LeJEPA", "SimCLR", "DIET"]
SIGLIP_TIMM = "vit_base_patch16_siglip_224.v2_webli"
HEAD_ATTR = {"LeJEPA": "projector", "SimCLR": "projector", "DIET": "diet_head"}
FIELDS = ["method", "encoder", "dataset", "size", "seed", "n_samples",
          "emb_dim", "head_dim",
          "rankme_bb", "alpha_bb", "alpha_r2_bb", "coherence_mu_bb", "uniformity_t2_bb",
          "rankme_head", "alpha_head", "alpha_r2_head", "coherence_mu_head", "ckpt"]
KEY = ("method", "encoder", "dataset", "size", "seed")


def extract_head_sd(ckpt_path, attr):
    """Sub-state-dict of the head module `attr` (prefix auto-derived, cf. postcp_features)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    marker = attr + "."
    votes = Counter()
    for k in sd:
        i = k.find(marker)
        if i == 0 or (i > 0 and k[i - 1] == "."):
            votes[k[:i]] += 1
    if not votes:
        raise KeyError(f"no '{marker}*' keys in {ckpt_path}")
    pre = votes.most_common(1)[0][0] + marker
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}


def rebuild_head(method, sub_sd):
    """Reconstruct the training head from weight shapes; strict load verifies arch."""
    if method == "DIET":
        w = sub_sd["weight"]
        head = nn.Linear(w.shape[1], w.shape[0], bias=False)
    elif method == "SimCLR":
        # build_simclr_projector (stable_cp/methods/simclr/simclr_cp.py)
        import stable_pretraining as spt
        e, h = sub_sd["0.weight"].shape[1], sub_sd["0.weight"].shape[0]
        p = sub_sd["3.weight"].shape[0]
        head = nn.Sequential(
            nn.Linear(e, h, bias=False), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
            nn.Linear(h, p, bias=False), spt.utils.BatchNorm1dNoBias(p))
    elif method == "LeJEPA":
        # build_lejepa_projector = torchvision MLP(e, [h, h, p], norm_layer=BatchNorm1d)
        from torchvision.ops import MLP
        lin = sorted((int(k.split(".")[0]) for k, v in sub_sd.items()
                      if k.endswith("weight") and v.dim() == 2))
        dims = [sub_sd[f"{i}.weight"].shape[0] for i in lin]
        head = MLP(sub_sd[f"{lin[0]}.weight"].shape[1], dims, norm_layer=nn.BatchNorm1d)
    else:
        raise ValueError(method)
    head.load_state_dict(sub_sd, strict=True)
    return head.eval()


@torch.no_grad()
def apply_head(head, feats, method, device, batch=1024):
    head = head.to(device)
    out = []
    for i in range(0, len(feats), batch):
        x = torch.from_numpy(feats[i:i + batch]).float().to(device)
        if method == "DIET":
            x = F.normalize(x, p=2, dim=1)   # diet_forward.py:58
        out.append(head(x).cpu().numpy())
    head.cpu()
    return np.vstack(out)


def discover_siglip_all(siglip_root):
    """All ckpts under the cp-siglip layout cp/<method>/<DsFolder>/SigLIP/cp/*.ckpt
    (cf. layerwise_postcp.discover_siglip, without its MAX-only filter)."""
    root = Path(siglip_root)
    if not root.exists():
        print(f"WARN: SigLIP root {root} absent — skipping SigLIP grid.")
        return []
    out = []
    for f in root.rglob("*.ckpt"):
        parts = f.relative_to(root).parts
        if f.parent.name != "cp" or "SigLIP" not in parts or len(parts) < 5:
            continue
        method = parts[list(parts).index("SigLIP") - 2]
        ds_folder = parts[list(parts).index("SigLIP") - 1]
        ds_key = DATASET_KEY.get(ds_folder, ds_folder.lower())
        m = re.search(r"_n(\d+)_s(\d+)$", f.stem)
        if not m:
            continue
        out.append(dict(method=method, variant="pretrained", encoder="SigLIP",
                        timm_id=SIGLIP_TIMM, pool="map", dataset=ds_key,
                        size=int(m.group(1)), seed=int(m.group(2)), ckpt=str(f)))
    return sorted(out, key=lambda c: c["ckpt"])


def max_only(cfgs):
    """Keep, per (method, encoder, dataset), only the largest-size checkpoints (all seeds)."""
    mx = {}
    for c in cfgs:
        k = (c["method"], c["encoder"], c["dataset"])
        mx[k] = max(mx.get(k, 0), c["size"])
    return [c for c in cfgs if c["size"] == mx[(c["method"], c["encoder"], c["dataset"])]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--siglip-root", default=None)
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/nd4_projector_spectra.csv"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfgs = [c for c in discover(args.ckpt_root)
            if c["variant"] == "pretrained" and c["method"] in METHODS
            and c["encoder"] != "RANDOM"]
    if args.siglip_root:
        cfgs += [c for c in discover_siglip_all(args.siglip_root) if c["method"] in METHODS]
    if args.datasets:
        cfgs = [c for c in cfgs if c["dataset"] in args.datasets]
    cfgs = max_only(cfgs)
    if args.seeds:
        cfgs = [c for c in cfgs if c["seed"] in args.seeds]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {tuple(str(r[k]) for k in KEY) for r in csv.DictReader(f)}
    cfgs = [c for c in cfgs if tuple(str(c[k]) for k in KEY) not in done]
    print(f"{len(cfgs)} MAX checkpoints to process ({len(done)} done)  device={device}")

    loaders = {}
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for idx, c in enumerate(cfgs):
            print(f"[{idx + 1}/{len(cfgs)}] {c['method']}/{c['encoder']}/{c['dataset']}"
                  f"/n{c['size']}/s{c['seed']}")
            # audit 2026-06-30: 6 SigLIP MAX ckpts died before the epoch-15 unfreeze —
            # backbone bit-identical to pretrained, head trained during freeze; a silent
            # row here would bias bb_shift/buffer_gap (same guard as layerwise_postcp).
            if died_before_unfreeze(c["ckpt"]):
                print("  SKIP untrained (died before unfreeze): " + c["ckpt"])
                continue
            try:
                sub_sd = extract_head_sd(c["ckpt"], HEAD_ATTR[c["method"]])
            except KeyError as e:
                print(f"  HEAD SKIP ({e}) — checkpoint saved without head weights?")
                continue
            except Exception as e:
                print(f"  CKPT FAIL ({type(e).__name__}: {e}) — unreadable checkpoint, "
                      f"skipping cell")
                continue
            # strict-load mismatch stays a hard error (arch drift must not be skipped)
            head = rebuild_head(c["method"], sub_sd)
            model = None
            try:
                if c["dataset"] not in loaders:
                    loaders[c["dataset"]] = load_target_dataset(
                        c["dataset"], args.download_dir, args.processed_dir)
                model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
                feat, _ = extract_features(model, loaders[c["dataset"]], device, c["pool"])
                if not np.isfinite(feat).all():
                    print("  CELL FAIL (non-finite backbone features — diverged "
                          "checkpoint?) — skipping")
                    continue
                post = apply_head(head, feat, c["method"], device)
                bb = spectral_row(feat)
                hd = spectral_row(post)
            except Exception as e:
                print(f"  CELL FAIL ({type(e).__name__}: {e}) — skipping cell")
                continue
            finally:
                del head
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            row = {k: c[k] for k in KEY}
            row.update(ckpt=c["ckpt"], n_samples=len(feat),
                       emb_dim=feat.shape[1], head_dim=post.shape[1],
                       uniformity_t2_bb=wang_isola_uniformity(feat),
                       **{f"{m}_bb": bb[m] for m in
                          ("rankme", "alpha", "alpha_r2", "coherence_mu")},
                       **{f"{m}_head": hd[m] for m in
                          ("rankme", "alpha", "alpha_r2", "coherence_mu")})
            w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(f"  bb: rank={bb['rankme']:.1f}  head: rank={hd['rankme']:.1f} "
                  f"(head_dim={post.shape[1]})")
    print(f"Done -> {out_path}\nNext (local): python eval/new_direction/nd4_verdict.py")


if __name__ == "__main__":
    main()
