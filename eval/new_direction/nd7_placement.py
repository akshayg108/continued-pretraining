#!/usr/bin/env python
"""
nd7_placement.py — ND7 (GPU pass): label-placement C(rho) + rankme on POST-CP MAX
checkpoints. The missing measurement of the story's spine: "CP moves class information
up the encoder's difference ranking".

Per MAX checkpoint (LeJEPA/SimCLR/DIET x DINOv3/CLIP/MAE + LeJEPA/SimCLR x SigLIP, all
seeds — same population, dead-ckpt guard and per-cell error handling as ND4), extract
backbone features on the target dataset (standard <=5000 protocol) and record:
  - C(rho) summaries (raw + class-centered label powers), same code path as ND6, so the
    delta vs eval/outputs/nd6_alignment.csv (pre-CP) is apples-to-apples;
  - rankme (for the ND7-2 mediation test against the ND2/ND4 shape channel).

Adjudication in nd7_verdict.py (local; pre-registered readouts in its docstring).

Cluster (one array task per dataset):
  python eval/new_direction/nd7_placement.py --datasets <ds> \
      --ckpt-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp \
      --siglip-root /scratch/gs4133/zhd/CP/outputs/ckpts/cp-siglip \
      --download-dir <raw> --processed-dir <arrow> \
      --out eval/outputs/nd7_placement_shards/<ds>.csv
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
from postcp_sweep import discover                                            # noqa: E402
from layerwise_postcp import died_before_unfreeze                             # noqa: E402
from postcp_features import load_cp_backbone                                  # noqa: E402
from geometry_metrics import load_target_dataset, extract_features            # noqa: E402
from nd4_projector_spectra import discover_siglip_all, max_only, METHODS      # noqa: E402
from nd6_alignment import mode_label_powers, c_rho_summary, C_RHOS            # noqa: E402
from spectral_metrics import rankme                                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = (["method", "encoder", "dataset", "size", "seed", "n_samples", "n_classes",
           "rankme"]
          + [f"C{r}" for r in C_RHOS] + ["C_K", "aucC_log"]
          + [f"cC{r}" for r in C_RHOS] + ["cC_K", "caucC_log", "ckpt"])
KEY = ("method", "encoder", "dataset", "size", "seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--siglip-root", default=None)
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--out", default=str(ROOT / "eval/outputs/nd7_placement.csv"))
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
            if died_before_unfreeze(c["ckpt"]):
                print("  SKIP untrained (died before unfreeze): " + c["ckpt"])
                continue
            model = None
            try:
                if c["dataset"] not in loaders:
                    loaders[c["dataset"]] = load_target_dataset(
                        c["dataset"], args.download_dir, args.processed_dir)
                model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
                feat, labels = extract_features(model, loaders[c["dataset"]], device,
                                                c["pool"])
                if not np.isfinite(feat).all():
                    print("  CELL FAIL (non-finite features — diverged ckpt?) — skipping")
                    continue
                lam, p_raw, p_cen, n_cls = mode_label_powers(feat, labels)
                row = {k: c[k] for k in KEY}
                row.update(ckpt=c["ckpt"], n_samples=len(feat), n_classes=n_cls,
                           rankme=rankme(feat))
                row.update(c_rho_summary(p_raw, n_cls))
                row.update(c_rho_summary(p_cen, n_cls, "c"))
            except Exception as e:
                print(f"  CELL FAIL ({type(e).__name__}: {e}) — skipping cell")
                continue
            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(f"  rankme={row['rankme']:.1f}  cC_K={row['cC_K']:.3f} "
                  f"caucC={row['caucC_log']:.3f}")
    print(f"Done -> {out_path}\nNext (local): python eval/new_direction/nd7_verdict.py")


if __name__ == "__main__":
    main()
