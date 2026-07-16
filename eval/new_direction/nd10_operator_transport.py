#!/usr/bin/env python
"""
nd10_operator_transport.py — ND10+ND11 (GPU pass): pre->post OPERATOR change on MAX
checkpoints — the accessibility decomposition (spectral flow vs basis rotation) and the
local cosine-kNN graph quantities, on MATCHED samples.

Theory targets (theory-unification 2026-07-15, adopted #2 and #3):
  ND10  decompose the pre->post change of the ridge-weighted accessibility A(kappa) into
        dA_spec (eigenvalue flow on the frozen pre basis) + dA_rot (basis rotation at the
        post spectrum) — the operator-level version of "rank thermostat + placement tide";
        exact partition, kappa anchored on the pre spectrum (kappa_rel=1.0 -> mean lam_pre,
        matching nd9's acc_r1 column).
  ND11  the local carrier candidates kNN cannot see through global spectra: neighbour
        label purity and graph placement (label energy in the K lowest normalized-Laplacian
        modes) of the cosine-kNN graph, pre and post.

Per MAX checkpoint (same population, dead-ckpt guard and per-cell error handling as ND7):
extract POST features, extract (cached per encoder x dataset) PRE features from the
pretrained encoder with the SAME loader, assert label-sequence identity (row-matched
samples), then record the decomposition + graph columns. Algorithms in
nd9_task_operator.py (TDD-tested). Adjudication: nd10_verdict.py (pre-registered).

Cluster (one array task per dataset):
  python eval/new_direction/nd10_operator_transport.py --datasets <ds> \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --siglip-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/nd10_operator_shards/<ds>.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))                      # new_direction/
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))     # eval/utils
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "F2_forces")) # sweep machinery
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "F4_gate"))   # dead-ckpt guard
from postcp_sweep import discover                                             # noqa: E402
from layerwise_postcp import died_before_unfreeze                             # noqa: E402
from postcp_features import load_cp_backbone                                  # noqa: E402
from geometry_metrics import load_target_dataset, extract_features            # noqa: E402
from nd4_projector_spectra import discover_siglip_all, max_only, METHODS      # noqa: E402
from nd9_task_operator import (spectrum_transplant_decomposition,             # noqa: E402
                               graph_label_metrics)

ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_K, GRAPH_NMAX = 10, 2000
DECOMP_COLS = ["dA_total", "dA_spec", "dA_rot", "A_pre", "A_post", "dcC_K", "affinity_topK"]
FIELDS = (["method", "encoder", "dataset", "size", "seed", "n_samples", "n_classes"]
          + DECOMP_COLS
          + ["knn_purity_pre", "knn_purity_post", "graph_cC_K_pre", "graph_cC_K_post",
             "graph_n_used", "ckpt"])
KEY = ("method", "encoder", "dataset", "size", "seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--siglip-root", default=None)
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/nd10_operator.csv"))
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

    import timm
    loaders, pre_cache = {}, {}   # pre_cache[(encoder, dataset)] = (feat, labels, graph_row)

    def pre_features(c):
        key = (c["encoder"], c["dataset"])
        if key not in pre_cache:
            model = timm.create_model(c["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            feat, labels = extract_features(model, loaders[c["dataset"]], device, c["pool"])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            g = graph_label_metrics(feat, labels, k=GRAPH_K, n_max=GRAPH_NMAX)
            pre_cache[key] = (feat, labels, g)
        return pre_cache[key]

    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    mismatch_streak = 0   # systematic loader-order regressions must abort, not burn 12 h
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        for idx, c in enumerate(cfgs):
            print(f"[{idx + 1}/{len(cfgs)}] {c['method']}/{c['encoder']}/{c['dataset']}"
                  f"/n{c['size']}/s{c['seed']}")
            if died_before_unfreeze(c["ckpt"]):
                print("  SKIP untrained (died before unfreeze): " + c["ckpt"])
                continue
            model = None
            try:
                if c["dataset"] not in loaders:
                    loaders[c["dataset"]] = load_target_dataset(
                        c["dataset"], args.download_dir, args.processed_dir)
                feat0, labels0, g0 = pre_features(c)
                model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
                feat1, labels1 = extract_features(model, loaders[c["dataset"]], device,
                                                  c["pool"])
                if not np.isfinite(feat1).all():
                    print("  CELL FAIL (non-finite features — diverged ckpt?) — skipping")
                    continue
                # matched-sample guard: the decomposition is meaningless if rows drift
                assert np.array_equal(labels0, labels1), "pre/post label sequences differ"
                dec = spectrum_transplant_decomposition(feat0, feat1, labels0)
                g1 = graph_label_metrics(feat1, labels1, k=GRAPH_K, n_max=GRAPH_NMAX)
                row = {k: c[k] for k in KEY}
                row.update(ckpt=c["ckpt"], n_samples=len(feat1),
                           n_classes=len(np.unique(labels1)),
                           knn_purity_pre=g0["knn_purity"], knn_purity_post=g1["knn_purity"],
                           graph_cC_K_pre=g0["graph_cC_K"], graph_cC_K_post=g1["graph_cC_K"],
                           graph_n_used=g1["n_used"])
                row.update({k: dec[k] for k in DECOMP_COLS})
            except Exception as e:
                if "label sequences differ" in str(e):
                    mismatch_streak += 1
                    if mismatch_streak >= 3:
                        raise RuntimeError(
                            "3 consecutive pre/post label mismatches — loader ordering "
                            "regressed; aborting the shard instead of skipping cells") from e
                print(f"  CELL FAIL ({type(e).__name__}: {e}) — skipping cell")
                continue
            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            mismatch_streak = 0
            print(f"  dA={row['dA_total']:+.4f} (spec {row['dA_spec']:+.4f} / "
                  f"rot {row['dA_rot']:+.4f})  purity {row['knn_purity_pre']:.3f}->"
                  f"{row['knn_purity_post']:.3f}  gcC_K {row['graph_cC_K_pre']:.3f}->"
                  f"{row['graph_cC_K_post']:.3f}")
    print(f"Done -> {out_path}\nNext (local): python eval/new_direction/nd10_verdict.py")


if __name__ == "__main__":
    main()
