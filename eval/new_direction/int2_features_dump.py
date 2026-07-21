#!/usr/bin/env python
"""
int2_features_dump.py — INT2 (GPU pass): dump frozen POST-CP bank + query features
for every checkpoint in the audited ND12 manifest (INT2_PREREG.md v1.0), so all
reversion surgery and evaluation run locally on CPU.

Manifest-driven: keys and ckpt paths come from eval/outputs/nd12_operator.csv
(484 rows; NO re-discover()). Bank/query protocol identical to INT1/ND12
(load_vote_loaders). Saves {enc}__{ds}__{method}__{seed}.npz with float32
bank_X, bank_y, query_X, query_y. Resume: existing .npz accepted only through
validate_npz (INT1 machinery); --verify-only prints the manifest census without
touching torch.

Cluster (one array task per dataset):
  python eval/new_direction/int2_features_dump.py --datasets <ds> --device cuda \
      --download-dir <raw> --processed-dir <arrow> --outdir eval/outputs/int2_features
After the array finishes (ON THE CLUSTER — this freezes int2_features.sha256;
an existing sidecar is never overwritten):
  python eval/new_direction/int2_features_dump.py --verify-only
Then rsync the feature dir TOGETHER WITH int2_features.sha256. Locally, either
run int2_run.py directly (it verifies the sidecar read-only) or:
  python eval/new_direction/int2_features_dump.py --verify-sidecar-only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_features_dump import validate_npz, ENCODER_NAMES, DATASET_NAMES  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "eval/outputs/nd12_operator.csv"
MANIFEST_COLS = ("method", "encoder", "dataset", "seed", "size", "ckpt",
                 "knn_f1_hat_post", "vote_margin_post", "vote_pos_frac_post",
                 "n_test", "n_samples")
FROZEN_MANIFEST_SHA256 = "352247ca2aefefd6c74ef3ffe80f3aaa55c50923ac02c54336cfbf1549e4069f"
FROZEN_SEED_COUNTS = {42: 165, 43: 160, 44: 159}


FROZEN_CENSUS = {"SimCLR": 180, "LeJEPA": 170, "DIET": 134}      # total 484


def load_manifest(path=MANIFEST, expect_census=True):
    """Read the frozen ND12 manifest and attach cell_id = enc__ds__method__seed.
    By default the 484-key contract is LOCKED (Codex round-6): method census,
    total row count and seed set must match the frozen values exactly."""
    m = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLS if c not in m.columns]
    assert not missing, f"manifest {path} lacks columns: {missing}"
    if expect_census:
        got = m.method.value_counts().to_dict()
        assert got == FROZEN_CENSUS and len(m) == 484, \
            f"manifest census mismatch: {got} (n={len(m)}) != {FROZEN_CENSUS} (484)"
        sc = m.seed.astype(int).value_counts().to_dict()
        assert sc == FROZEN_SEED_COUNTS, f"seed census mismatch: {sc}"
        assert m.ckpt.is_unique, "duplicate ckpt paths in manifest"
        import hashlib as _hl
        h = _hl.sha256(Path(path).read_bytes()).hexdigest()
        assert h == FROZEN_MANIFEST_SHA256, \
            f"manifest hash {h[:12]}... != frozen {FROZEN_MANIFEST_SHA256[:12]}..."
    m = m.copy()
    m["cell_id"] = (m.encoder + "__" + m.dataset + "__" + m.method
                    + "__" + m.seed.astype(int).astype(str))
    assert m.cell_id.is_unique, "manifest keys are not unique"
    return m


NPZ_KEYS_INT2 = ("bank_X", "bank_y", "query_X", "query_y")


def validate_npz_int2(path, n_test=None, n_samples=None):
    """INT2-strict acceptance (round-7): float32 2-D 768-dim features, finite
    integer-like labels, EXACT field set, non-empty."""
    try:
        z = np.load(path)
        arrs = {k: z[k] for k in z.files}
    except Exception as e:
        return False, f"unreadable ({e.__class__.__name__})"
    if set(arrs) != set(NPZ_KEYS_INT2):
        return False, f"extra/missing fields: {sorted(set(arrs) ^ set(NPZ_KEYS_INT2))}"
    bX, by, qX, qy = (arrs[k] for k in NPZ_KEYS_INT2)
    for X, lab in ((bX, "bank"), (qX, "query")):
        if X.ndim != 2 or X.shape[0] == 0:
            return False, f"{lab} features not 2-D non-empty"
        if X.shape[1] != 768:
            return False, f"{lab} dim {X.shape[1]} != 768"
        if X.dtype != np.float32:
            return False, f"{lab} dtype {X.dtype} != float32"
        if not np.isfinite(X).all():
            return False, f"non-finite {lab} features"
    if bX.shape[0] < 2:
        return False, "bank has fewer than 2 samples"
    for y, X, lab in ((by, bX, "bank"), (qy, qX, "query")):
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != X.shape[0]:
            return False, f"{lab} label shape/length mismatch"
        yf = y.astype(np.float64)
        if not np.isfinite(yf).all():
            return False, f"non-finite {lab} labels"
        if not np.allclose(yf, np.round(yf)):
            return False, f"{lab} labels not integer class ids"
    if n_test is not None and np.isfinite(n_test) and len(qy) != int(n_test):
        return False, f"query n {len(qy)} != manifest n_test {int(n_test)}"
    if n_samples is not None and np.isfinite(n_samples) \
            and bX.shape[0] != int(n_samples):
        return False, f"bank n {bX.shape[0]} != manifest n_samples {int(n_samples)}"
    return True, "ok"


def manifest_census(outdir, rows, check_foreign=True):
    """check_foreign=False for per-shard array tasks (round-8: a shard must not
    flag sibling shards' legitimate outputs; only the FULL --verify-only pass
    enforces the exact 484-file set)."""
    ok, bad = [], {}
    expected = {f"{cid}.npz" for cid in rows.cell_id}
    if check_foreign and Path(outdir).exists():
        for f in Path(outdir).iterdir():
            if f.name not in expected:
                bad[f.name] = "unexpected file in outdir"
    for _, row in rows.iterrows():
        cid = row.cell_id
        p = Path(outdir) / f"{cid}.npz"
        if not p.exists():
            bad[cid] = "missing"
            continue
        good, msg = validate_npz_int2(p, n_test=row.get("n_test"),
                                      n_samples=row.get("n_samples"))
        (ok.append(cid) if good else bad.__setitem__(cid, msg))
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", type=str, default=str(ROOT / "eval/data/downloads"))
    ap.add_argument("--processed-dir", type=str, default=str(ROOT / "eval/data/processed"))
    ap.add_argument("--outdir", type=str, default=str(ROOT / "eval/outputs/int2_features"))
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--allow-nonfrozen-manifest", action="store_true",
                    help="TEST/SMOKE ONLY: bypass the frozen 484-key census "
                         "lock (loudly). Never use for a real launch.")
    ap.add_argument("--verify-sidecar-only", action="store_true",
                    help="READ-ONLY: verify features against an existing "
                         "sidecar; never writes anything (local use)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if args.allow_nonfrozen_manifest:
        print("WARNING: census lock BYPASSED (--allow-nonfrozen-manifest) — "
              "test/smoke use only")
    man = load_manifest(args.manifest,
                        expect_census=not args.allow_nonfrozen_manifest)
    if args.datasets:
        man = man[man.dataset.isin(set(args.datasets))]
    assert len(man), "manifest selection is empty"

    if args.verify_sidecar_only:               # round-11: read-only local check
        import sys as _sys2
        _sys2.path.insert(0, str(_P0(__file__).resolve().parent))
        from int2_run import verify_post_checksums
        expected = {f"{cid}.npz" for cid in man.cell_id}
        ok, msg = verify_post_checksums(Path(outdir).parent / "int2_features.sha256",
                                        outdir, expected_names=expected)
        print(f"SIDECAR VERIFY: {'PASS' if ok else 'FAIL — ' + msg}")
        raise SystemExit(0 if ok else 1)

    if args.verify_only:                       # torch-free path
        full = args.datasets is None
        ok, bad = manifest_census(outdir, man, check_foreign=full)
        print(f"MANIFEST {len(ok)}/{len(ok) + len(bad)} cells accepted under {outdir}")
        for cid, why in sorted(bad.items()):
            print(f"  BAD {cid}: {why}")
        if full and not bad:                   # round-9/10/11: freeze POST checksums
            import hashlib as _hl
            out = Path(outdir).parent / "int2_features.sha256"
            if out.exists():                   # round-12: verify, never trust blindly
                from int2_run import verify_post_checksums
                expected = {f"{cid}.npz" for cid in man.cell_id}
                okc, msgc = verify_post_checksums(out, outdir,
                                                  expected_names=expected)
                print(f"sidecar already exists — NOT overwritten ({out}); "
                      f"read-only verification: "
                      f"{'PASS' if okc else 'FAIL — ' + msgc}")
                raise SystemExit(0 if okc else 1)
            tmp = Path(outdir).parent / ".int2_features.sha256.tmp"
            with open(tmp, "w") as fh:
                for cid in sorted(man.cell_id):
                    f = Path(outdir) / f"{cid}.npz"
                    fh.write(f"{_hl.sha256(f.read_bytes()).hexdigest()}  "
                             f"{f.name}\n")
            tmp.replace(out)                   # atomic publish
            print(f"POST checksums written -> {out} (rsync this WITH the "
                  "features; the local side only VERIFIES, never regenerates)")
        raise SystemExit(0 if not bad else 1)

    import torch
    from nd10_operator_transport import load_vote_loaders
    from geometry_metrics import ENCODERS, load_target_dataset, extract_features
    from postcp_features import load_cp_backbone
    if args.device and args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "--device cuda requested but CUDA unavailable"
    elif args.device is None:
        assert torch.cuda.is_available(), \
            "no CUDA and no explicit --device: pass --device cpu to allow a CPU dump"
    device = torch.device(args.device or "cuda")
    outdir.mkdir(parents=True, exist_ok=True)

    # fail-fast: loaders for every requested dataset before touching any ckpt
    loaders = {}
    for ds in sorted(man.dataset.unique()):
        bank_loader, query_loader = load_vote_loaders(ds, args.download_dir,
                                                      args.processed_dir)
        train_loader = load_target_dataset(ds, args.download_dir, args.processed_dir)
        loaders[ds] = (bank_loader if bank_loader is not None else train_loader,
                       query_loader)
        print(f"  loaders OK: {ds}")

    # preflight: every ckpt path must exist BEFORE any model is loaded (round-7)
    import os
    missing_ck = [c for c in man.ckpt
                  if not (Path(c).is_file() and os.access(c, os.R_OK))]
    assert not missing_ck, f"{len(missing_ck)} ckpts missing on disk, e.g. {missing_ck[:3]}"
    todo = []
    for _, row in man.iterrows():
        p = outdir / f"{row.cell_id}.npz"
        if p.exists():
            good, msg = validate_npz_int2(p, n_test=row.get("n_test"),
                                          n_samples=row.get("n_samples"))
            if good:
                continue
            print(f"  REJECT {p.name} ({msg}) — redumping")
            p.unlink()
        todo.append(row)
    print(f"{len(todo)} cells to dump (of {len(man)} requested)")

    for i, row in enumerate(todo):
        cfg = ENCODERS[row.encoder]
        assert Path(row.ckpt).exists(), f"ckpt missing on disk: {row.ckpt}"
        model = load_cp_backbone(row.ckpt, cfg["timm_id"], device)
        bank_loader, query_loader = loaders[row.dataset]
        bX, by = extract_features(model, bank_loader, device, cfg["pool"])
        qX, qy = extract_features(model, query_loader, device, cfg["pool"])
        path = outdir / f"{row.cell_id}.npz"
        tmp = outdir / f".{row.cell_id}.tmp.npz"
        np.savez_compressed(tmp, bank_X=bX.astype(np.float32), bank_y=by,
                            query_X=qX.astype(np.float32), query_y=qy)
        tmp.replace(path)                      # atomic publish
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{i + 1}/{len(todo)}] {row.cell_id}: bank {bX.shape}, query {qX.shape}")

    ok, bad = manifest_census(outdir, man, check_foreign=args.datasets is None)
    print(f"\nMANIFEST {len(ok)}/{len(ok) + len(bad)} cells accepted for this task's grid")
    for cid, why in sorted(bad.items()):
        print(f"  BAD {cid}: {why}")
    if bad:
        raise SystemExit(1)
    print("Done. HANDOFF (in order):\n"
          "  1. cluster: python eval/new_direction/int2_features_dump.py --verify-only\n"
          "     (full census; freezes int2_features.sha256 on first run,\n"
          "      read-only-verifies it on later runs)\n"
          "  2. rsync eval/outputs/int2_features/ AND eval/outputs/int2_features.sha256\n"
          "  3. local: python eval/new_direction/int2_features_dump.py --verify-sidecar-only\n"
          "  4. local: python eval/new_direction/int2_run.py && "
          "python eval/new_direction/int2_verdict.py")


if __name__ == "__main__":
    main()
