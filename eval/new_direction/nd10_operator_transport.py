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
                               graph_label_metrics, vote_operator_metrics)

ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_K, GRAPH_NMAX = 10, 2000
DECOMP_COLS = ["dA_total", "dA_spec", "dA_rot", "A_pre", "A_post", "dcC_K", "affinity_topK"]
# GRAPH_VARIANTS: legacy k=10 unweighted/unbalanced (ND10/ND11 protocol) + the ND12
# evaluator-matched variants (k=20 to match the kNN evaluator; class-balanced to match
# macro-F1; m20wb additionally distance-weighted). Codex-audit follow-up 2026-07-16.
GRAPH_VARIANTS = {"": dict(k=GRAPH_K),
                  "_m20b": dict(k=20, class_balanced=True),
                  "_m20wb": dict(k=20, weighted=True, class_balanced=True)}
GRAPH_COLS = (["knn_purity_pre", "knn_purity_post", "purity_macro20_pre",
               "purity_macro20_post"]
              + [f"graph_cC_K{v}_{s}" for v in GRAPH_VARIANTS for s in ("pre", "post")])
# ND12 I1: the true evaluator vote operator (test-to-train, inverse-distance, k=20);
# gate/bridge quantities only — see ND12_PREREG.md role restriction
VOTE_COLS = ["knn_f1_hat_pre", "knn_f1_hat_post", "vote_margin_pre", "vote_margin_post",
             "vote_pos_frac_pre", "vote_pos_frac_post", "n_test"]
TEST_CAP = 2000
FIELDS = (["method", "encoder", "dataset", "size", "seed", "n_samples", "n_classes"]
          + DECOMP_COLS + GRAPH_COLS + VOTE_COLS + ["graph_n_used", "ckpt"])
KEY = ("method", "encoder", "dataset", "size", "seed")


def _capped_loader(hf_ds, cap):
    """Deterministic stratified cap (seed 42) + eval loader, as in load_target_dataset."""
    from torch.utils.data import DataLoader
    from geometry_metrics import StableDatasetWrapper, eval_transform
    if len(hf_ds) > cap:
        labels = np.array(hf_ds["label"]).ravel()
        try:
            from sklearn.model_selection import train_test_split
            sel, _ = train_test_split(np.arange(len(hf_ds)), train_size=cap,
                                      stratify=labels, random_state=42)
        except ValueError:                       # singleton classes: plain seeded draw
            sel = np.random.RandomState(42).choice(len(hf_ds), cap, replace=False)
        hf_ds = hf_ds.select(sorted(np.asarray(sel).tolist()))
    return DataLoader(StableDatasetWrapper(hf_ds, eval_transform()),
                      batch_size=256, num_workers=4, shuffle=False)


def load_vote_loaders(name, download_dir, processed_dir, cap=TEST_CAP):
    """(vote_bank_loader_or_None, query_loader) for the ND12 vote proxy.

    Datasets with a real test asset: bank = None (the standard G1 train cloud is
    reused — bank/query disjoint because they come from different assets); query =
    splits[2] capped at TEST_CAP. Datasets whose class ships only a train asset
    (galaxy10 — round-4 review): REUSE stable_cp.data.datasets._split_single_dataset
    (seed 42, 80/10/10 — the real evaluator's own manual-split protocol) for BOTH
    sides, so bank and query are disjoint by construction and the split matches the
    protocol that produced results.xlsx; the G1 full-cloud loader stays untouched."""
    from geometry_metrics import DS_REGISTRY
    ds_class, config_name, splits, extra_kwargs = DS_REGISTRY[name]
    kwargs = dict(extra_kwargs)
    if config_name is not None:
        kwargs["config_name"] = config_name
    try:
        hf_test = ds_class(split=splits[2], download_dir=str(download_dir),
                           processed_cache_dir=str(processed_dir), **kwargs)
        return None, _capped_loader(hf_test, cap)
    except Exception as e:
        from stable_cp.data.datasets import _split_single_dataset
        print(f"  NOTE: {name} has no loadable '{splits[2]}' asset ({type(e).__name__}) "
              f"-> evaluator manual split (seed 42, 80/10/10) for the vote proxy")
        full = ds_class(split="train", download_dir=str(download_dir),
                        processed_cache_dir=str(processed_dir), **kwargs)
        bank = _split_single_dataset(full, "train", seed=42)
        query = _split_single_dataset(full, "test", seed=42)
        return _capped_loader(bank, 5000), _capped_loader(query, cap)


def graph_rows(feat, labels):
    """All graph variants for one feature cloud -> {suffix: metrics dict}."""
    return {v: graph_label_metrics(feat, labels, n_max=GRAPH_NMAX, **kw)
            for v, kw in GRAPH_VARIANTS.items()}


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
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            rd = csv.DictReader(f)
            # header-drift guard: resuming a shard written by an older FIELDS layout
            # would silently misalign appended rows (Codex-audit era hardening)
            assert rd.fieldnames == FIELDS, \
                f"{out_path} was written with a different column layout — use a fresh --out"
            done = {tuple(str(r[k]) for k in KEY) for r in rd}
    cfgs = [c for c in cfgs if tuple(str(c[k]) for k in KEY) not in done]
    print(f"{len(cfgs)} MAX checkpoints to process ({len(done)} done)  device={device}")

    loaders, test_loaders, pre_cache = {}, {}, {}
    # pre_cache[(encoder, dataset)] = (feat, labels, graph_row, test_feat, test_labels, vote)
    # FAIL-FAST (round-3 review): build BOTH loaders for every dataset in this shard
    # BEFORE the checkpoint loop. A dataset whose test split is unsupported (manual
    # splits etc.) must kill the shard immediately with a clear error — not burn 12h
    # of per-cell silent CELL FAILs.
    vote_bank_loaders = {}
    for ds in sorted({c["dataset"] for c in cfgs}):
        loaders[ds] = load_target_dataset(ds, args.download_dir, args.processed_dir)
        vote_bank_loaders[ds], test_loaders[ds] = load_vote_loaders(
            ds, args.download_dir, args.processed_dir)
        print(f"  loaders OK: {ds} (train batches {len(loaders[ds])}, "
              f"query batches {len(test_loaders[ds])}, "
              f"vote bank: {'manual-split' if vote_bank_loaders[ds] else 'G1 cloud'})")

    import timm

    def pre_features(c):
        key = (c["encoder"], c["dataset"])
        if key not in pre_cache:
            model = timm.create_model(c["timm_id"], pretrained=True,
                                      num_classes=0).eval().to(device)
            feat, labels = extract_features(model, loaders[c["dataset"]], device, c["pool"])
            tfeat, tlabels = extract_features(model, test_loaders[c["dataset"]], device,
                                              c["pool"])
            bl = vote_bank_loaders[c["dataset"]]
            bank = extract_features(model, bl, device, c["pool"]) if bl else (feat, labels)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vote = vote_operator_metrics(bank[0], bank[1], tfeat, tlabels, k=20)
            pre_cache[key] = (feat, labels, graph_rows(feat, labels), tfeat, tlabels, vote)
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
                feat0, labels0, g0, tfeat0, tlabels0, vote0 = pre_features(c)
                model = load_cp_backbone(c["ckpt"], c["timm_id"], device)
                feat1, labels1 = extract_features(model, loaders[c["dataset"]], device,
                                                  c["pool"])
                tfeat1, tlabels1 = extract_features(model, test_loaders[c["dataset"]],
                                                    device, c["pool"])
                bl = vote_bank_loaders[c["dataset"]]
                bank1 = (extract_features(model, bl, device, c["pool"]) if bl
                         else None)                    # manual-split datasets only
                del model
                model = None
                if not np.isfinite(feat1).all() or not np.isfinite(tfeat1).all():
                    print("  CELL FAIL (non-finite features — diverged ckpt?) — skipping")
                    continue
                # matched-sample guard: the decomposition is meaningless if rows drift
                assert np.array_equal(labels0, labels1), "pre/post label sequences differ"
                assert np.array_equal(tlabels0, tlabels1), "pre/post label sequences differ (test)"
                dec = spectrum_transplant_decomposition(feat0, feat1, labels0)
                g1 = graph_rows(feat1, labels1)
                if bank1 is not None:
                    vote1 = vote_operator_metrics(bank1[0], bank1[1], tfeat1, tlabels1, k=20)
                else:
                    vote1 = vote_operator_metrics(feat1, labels1, tfeat1, tlabels1, k=20)
                row = {k: c[k] for k in KEY}
                row.update(ckpt=c["ckpt"], n_samples=len(feat1),
                           n_classes=len(np.unique(labels1)),
                           knn_purity_pre=g0[""]["knn_purity"],
                           knn_purity_post=g1[""]["knn_purity"],
                           purity_macro20_pre=g0["_m20b"]["knn_purity_macro"],
                           purity_macro20_post=g1["_m20b"]["knn_purity_macro"],
                           graph_n_used=g1[""]["n_used"])
                for v in GRAPH_VARIANTS:
                    row[f"graph_cC_K{v}_pre"] = g0[v]["graph_cC_K"]
                    row[f"graph_cC_K{v}_post"] = g1[v]["graph_cC_K"]
                row.update(knn_f1_hat_pre=vote0["knn_f1_hat"],
                           knn_f1_hat_post=vote1["knn_f1_hat"],
                           vote_margin_pre=vote0["vote_margin_mean"],
                           vote_margin_post=vote1["vote_margin_mean"],
                           vote_pos_frac_pre=vote0["vote_margin_pos_frac"],
                           vote_pos_frac_post=vote1["vote_margin_pos_frac"],
                           n_test=vote1["n_test"])
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
