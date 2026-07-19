#!/usr/bin/env python
"""
int1_features_dump.py — INT1 (GPU pass): dump frozen PRE-CP bank + query features so
that ALL Intervention-1 surgery and evaluation run locally on CPU (INT1_PREREG.md).

Per (encoder, dataset) cell: bank = the ND12 vote-bank protocol (standard <= 5000
seed-42 train cloud; galaxy10 = evaluator manual-split train), query = test split
<= 2000 (load_vote_loaders from nd10_operator_transport — protocol identity by
reuse). Saves eval/outputs/int1_features/<enc>__<ds>.npz with float32 arrays
bank_X, bank_y, query_X, query_y. Resume: existing .npz cells are skipped ONLY if they pass
validate_npz (Codex round-2: bare existence is not acceptance — a torn or truncated
file must be redumped, not trusted). --verify-only prints the full-grid manifest
without touching any model.

Cluster (one array task per dataset):
  python eval/new_direction/int1_features_dump.py --datasets <ds> --device cuda \
      --download-dir <raw> --processed-dir <arrow> \
      --outdir eval/outputs/int1_features
After the array finishes:
  python eval/new_direction/int1_features_dump.py --verify-only
"""
import argparse
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

# NOTE: load_vote_loaders is imported inside main() — the nd10 -> geometry_metrics
# chain needs torch + stable_datasets (cluster env); validate_npz/manifest must stay
# importable anywhere (they are unit-tested locally and used by --verify-only)

ROOT = Path(__file__).resolve().parent.parent.parent

NPZ_KEYS = ("bank_X", "bank_y", "query_X", "query_y")

# mirrors geometry_metrics.ENCODERS / TARGET_DATASETS so --verify-only works in any
# env (no torch); the dump path asserts agreement against the live module
ENCODER_NAMES = ["DINOv3", "MAE", "CLIP", "SigLIP"]
DATASET_NAMES = [
    "breastmnist", "dermamnist", "octmnist", "organamnist", "pathmnist", "galaxy10",
    "eurosat", "plant_village", "dtd",
    "food101", "fgvc_aircraft",
    "cars196", "cub200", "flowers102", "oxford_pet",
]


def validate_npz(path):
    """Structural acceptance for one dumped cell: loadable, all four arrays present,
    2-D non-empty features with matching dims and label lengths, all finite."""
    try:
        z = np.load(path)
        arrs = {k: z[k] for k in z.files}
    except Exception as e:
        return False, f"unreadable ({e.__class__.__name__})"
    missing = [k for k in NPZ_KEYS if k not in arrs]
    if missing:
        return False, f"missing key(s): {missing}"
    bX, by, qX, qy = (arrs[k] for k in NPZ_KEYS)
    if bX.ndim != 2 or qX.ndim != 2:
        return False, "features not 2-D"
    if bX.shape[0] == 0 or qX.shape[0] == 0:
        return False, "empty bank or query"
    if bX.shape[1] != qX.shape[1]:
        return False, f"bank/query dim mismatch {bX.shape[1]} vs {qX.shape[1]}"
    if len(by) != bX.shape[0] or len(qy) != qX.shape[0]:
        return False, "label length mismatch"
    if not (np.isfinite(bX).all() and np.isfinite(qX).all()):
        return False, "non-finite features"
    return True, "ok"


def manifest(outdir, encoders, datasets):
    """Validate the full requested grid; return (ok_cells, bad) where bad maps
    cell -> reason (missing or failed acceptance)."""
    ok, bad = [], {}
    for enc in encoders:
        for ds in datasets:
            p = Path(outdir) / f"{enc}__{ds}.npz"
            if not p.exists():
                bad[f"{enc}__{ds}"] = "missing"
                continue
            good, msg = validate_npz(p)
            (ok.append(f"{enc}__{ds}") if good else bad.__setitem__(f"{enc}__{ds}", msg))
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--outdir", type=str, default=str(ROOT / "eval/outputs/int1_features"))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="validate the full grid and exit; no model is loaded")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    if args.verify_only:                       # torch-free path, runs in any env
        encoders = args.encoders or ENCODER_NAMES
        datasets = args.datasets or DATASET_NAMES
        ok, bad = manifest(outdir, encoders, datasets)
        print(f"MANIFEST {len(ok)}/{len(ok) + len(bad)} cells accepted under {outdir}")
        for cell, why in sorted(bad.items()):
            print(f"  BAD {cell}: {why}")
        raise SystemExit(0 if not bad else 1)

    import timm
    import torch
    from nd10_operator_transport import load_vote_loaders
    from geometry_metrics import (ENCODERS, TARGET_DATASETS, load_target_dataset,
                                  extract_features)
    assert list(ENCODERS.keys()) == ENCODER_NAMES and TARGET_DATASETS == DATASET_NAMES, \
        "local ENCODER_NAMES/DATASET_NAMES fell out of sync with geometry_metrics"
    encoders = args.encoders or list(ENCODERS.keys())
    datasets = args.datasets or TARGET_DATASETS
    outdir.mkdir(parents=True, exist_ok=True)

    # Codex round-2: a dump that silently lands on CPU wastes the GPU allocation and
    # runs 20-50x slower — fail fast unless CPU was requested explicitly.
    if args.device and args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "--device cuda requested but CUDA unavailable"
    elif args.device is None:
        assert torch.cuda.is_available(), \
            "no CUDA and no explicit --device: pass --device cpu to allow a CPU dump"
    device = torch.device(args.device or "cuda")

    # fail-fast: build all loaders before touching any model (round-3 discipline)
    loaders = {}
    for ds in datasets:
        bank_loader, query_loader = load_vote_loaders(ds, args.download_dir,
                                                      args.processed_dir)
        train_loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
        loaders[ds] = (bank_loader if bank_loader is not None else train_loader,
                       query_loader)
        print(f"  loaders OK: {ds} (bank {'manual-split' if bank_loader else 'G1 cloud'})")

    for enc in encoders:
        todo = []
        for d in datasets:
            p = outdir / f"{enc}__{d}.npz"
            if p.exists():
                good, msg = validate_npz(p)
                if good:
                    continue                             # accepted resume cell
                print(f"  REJECT {p.name} ({msg}) — redumping")
                p.unlink()
            todo.append(d)
        if not todo:
            continue
        cfg = ENCODERS[enc]
        print(f"\n===== {enc} ({cfg['timm_id']}) — {len(todo)} cells")
        model = timm.create_model(cfg["timm_id"], pretrained=True,
                                  num_classes=0).eval().to(device)
        for ds in todo:
            bank_loader, query_loader = loaders[ds]
            bX, by = extract_features(model, bank_loader, device, cfg["pool"])
            qX, qy = extract_features(model, query_loader, device, cfg["pool"])
            path = outdir / f"{enc}__{ds}.npz"
            tmp = outdir / f".{enc}__{ds}.tmp.npz"       # torn-write guard (review):
            np.savez_compressed(tmp, bank_X=bX.astype(np.float32), bank_y=by,
                                query_X=qX.astype(np.float32), query_y=qy)
            tmp.replace(path)                            # atomic publish
            print(f"  {ds:>14}: bank {bX.shape}, query {qX.shape} -> {path.name}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ok, bad = manifest(outdir, encoders, datasets)
    print(f"\nMANIFEST {len(ok)}/{len(ok) + len(bad)} cells accepted for this task's grid")
    for cell, why in sorted(bad.items()):
        print(f"  BAD {cell}: {why}")
    if bad:
        raise SystemExit(1)
    print("Done. rsync eval/outputs/int1_features/ back, then run locally:\n"
          "  python eval/new_direction/int1_run.py && python eval/new_direction/int1_verdict.py")


if __name__ == "__main__":
    main()
