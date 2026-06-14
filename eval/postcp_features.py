#!/usr/bin/env python
"""
postcp_features.py — load a POST-CP checkpoint and extract features.

Shared by postcp_normcv.py (Exp A) and postcp_growth.py (Exp C). Runs in the same
Colab env as geometry_metrics.py (needs torch/timm/stable_datasets + the post-CP .ckpt).

The CP runs save PyTorch-Lightning checkpoints wrapping a timm ViT-B backbone. We don't
hard-code the state_dict layout: `load_cp_backbone` AUTO-DETECTS the backbone key prefix by
maximising overlap with a fresh timm model's keys, loads with strict=False, and prints match
stats so you can verify. If <50% of keys match it warns and dumps the first checkpoint keys.
"""
import torch
import timm

# Reuse the working loaders + metric fns from the geometry recompute.
from geometry_metrics import (
    ENCODERS, load_target_dataset, extract_features,
    l2_norm_stats, wang_isola_uniformity, neighbor_overlap,
    cosine_distance_centroids, mmd_rbf_components,
)

def _derive_prefix(sd_keys, target):
    """Auto-derive the wrapper prefix by ANCHORING on distinctive timm keys.

    A fixed candidate list misses arbitrary nesting — e.g. MAE-CP wraps the ViT in
    `MaskedEncoder` (`spt.Module(backbone=MaskedEncoder(vit))`), so the real ViT keys can sit
    under `backbone.backbone.` / `backbone.encoder.` etc. We instead look at where distinctive
    timm keys actually appear in the checkpoint and take that prefix verbatim.
    """
    from collections import Counter
    anchors = [t for t in ("cls_token", "pos_embed", "patch_embed.proj.weight",
                            "blocks.0.norm1.weight", "norm.weight") if t in target]
    if not anchors:
        anchors = list(target)[:5]
    votes = Counter()
    for t in anchors:
        for k in sd_keys:
            if k == t or k.endswith("." + t):
                votes[k[: len(k) - len(t)]] += 1   # prefix incl. trailing '.', or '' if k == t
    return votes.most_common(1)[0][0] if votes else ""


def load_cp_backbone(ckpt_path, timm_id, device):
    """Build the timm arch and load the post-CP backbone weights (prefix auto-derived from keys)."""
    model = timm.create_model(timm_id, pretrained=False, num_classes=0)
    target = set(model.state_dict().keys())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    pre = _derive_prefix(sd.keys(), target)
    # keep only keys that, after stripping the prefix, name a real timm param
    # (this also drops the MAE decoder / mask token / loss buffers automatically)
    stripped = {k[len(pre):]: v for k, v in sd.items()
                if k.startswith(pre) and k[len(pre):] in target}
    hits = len(target & set(stripped.keys()))
    print(f"  backbone prefix='{pre}'  matched {hits}/{len(target)} timm keys")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"  load_state_dict(strict=False): {len(missing)} missing, {len(unexpected)} unexpected")
    if hits < 0.8 * len(target):
        print(f"  WARNING: only {hits}/{len(target)} matched — backbone likely mis-loaded. Sample ckpt keys:")
        for k in list(sd.keys())[:15]:
            print("    ", k)
    return model.eval().to(device)


def extract_postcp(ckpt_path, encoder, dataset_key, download_dir, processed_dir, device):
    """Return (features, labels) for `dataset_key` from the post-CP `ckpt_path`.

    encoder in {"DINOv3","CLIP","MAE"} -> timm_id + pool from ENCODERS.
    """
    cfg = ENCODERS[encoder]
    model = load_cp_backbone(ckpt_path, cfg["timm_id"], device)
    loader = load_target_dataset(dataset_key, download_dir, processed_dir)
    return extract_features(model, loader, device, pool_strategy=cfg["pool"])
