#!/usr/bin/env python3
"""
int1_run.py — INT1 (LOCAL CPU): apply the FROZEN surgery grid (INT1_PREREG.md) to the
dumped features and evaluate both readouts per (cell, transform). Resumable by row.

Grid (frozen, v1.4): identity | rotation seeds {0,1} | power alpha
{0.25,0.5,0.75,1.5,2.0} | power_cp (per-cell alpha calibrated to the realized
post-CP rank; skipped with a note if no nd7 target exists) | demote depth
{64,256,512} (block 16) | shuffle seed 0 | combo (demote 256 then power alpha
{0.5,2.0}) — the interaction arm.

Readouts: kNN = vote_operator_metrics (G2-validated proxy); LP = standardized
deterministic probe (L2 normalize, LogisticRegression(max_iter=1000, C=1.0, lbfgs),
macro-F1 — the zero_shot_eval.linear_probe_evaluate protocol). NOTE (Codex round-2):
the paper's recorded lp_pre/dlp come from the SEPARATE unregularized 10k-step
PyTorch probe (linear_probe_pytorch_evaluate, no seed — nondeterministic);
connectability of this deterministic proxy is checked by the G2-LP gate in
int1_verdict.py. Per row also records post-L2 preservation metrics
(rankme/capture/cC_K on the normalized surgered bank) for the G-P gate.

Run: python eval/new_direction/int1_run.py  [--cells enc__ds ...]
(~60 cells x 15 transforms; LP fits dominate — expect a few hours unattended.)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_surgery import (fit_surgery, apply_surgery, rankme_raw, capture_of,
                          cC_K_of, calibrate_alpha)
from nd9_task_operator import vote_operator_metrics

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
FEAT = OUT / "int1_features"
RESULT = OUT / "int1_results.csv"
FIELDS = ["encoder", "dataset", "transform", "param", "knn_f1", "lp_f1",
          "rankme_raw_bank", "rankme_l2_bank", "capture_l2", "cC_K_l2", "alpha_cp",
          "query_oos_frac", "rankme_target"]

# the 14 mandatory grid keys; power_cp is required per cell iff an nd7 target exists
BASE_KEYS = ([("identity", ""), ("rotation", "0"), ("rotation", "1")]
             + [("power", a) for a in ("0.25", "0.5", "0.75", "1.5", "2.0")]
             + [("demote", d) for d in ("64", "256", "512")] + [("shuffle", "0")]
             + [("combo", c) for c in ("0.5|256", "2.0|256")])


def cell_complete(done, enc, ds, has_target):
    """Resume acceptance for one cell (Codex round-4: exact key-set check — a
    sentinel + row-count heuristic could mark a cell complete when a required
    surgery is missing but a stale extra row pads the count)."""
    have = {(t, p) for (e, d, t, p) in done if (e, d) == (enc, ds)}
    if not all(k in have for k in BASE_KEYS):
        return False
    if has_target and not any(t == "power_cp" for t, _ in have):
        return False
    return True


def lp_f1(bank_X, bank_y, query_X, query_y):
    """Evaluator mirror: zero_shot_eval.linear_probe_evaluate protocol."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import normalize
    # n_jobs dropped: no-op since sklearn 1.8 (deprecation warning flood); the real
    # evaluator passes it but it never affected results — protocol unchanged
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(normalize(bank_X), bank_y)
    pred = clf.predict(normalize(query_X))
    return float(f1_score(query_y, pred, average="macro",
                          labels=np.unique(bank_y), zero_division=0))


def grid(bank_X, alpha_cp):
    """The frozen transform grid as (name, param, map_or_None) tuples."""
    g = [("identity", "", None)]
    for s in (0, 1):
        g.append(("rotation", str(s), fit_surgery(bank_X, "rotation", seed=s)))
    for a in (0.25, 0.5, 0.75, 1.5, 2.0):
        g.append(("power", str(a), fit_surgery(bank_X, "power", alpha=a)))
    if alpha_cp is not None:
        g.append(("power_cp", f"{alpha_cp:.4f}",
                  fit_surgery(bank_X, "power", alpha=alpha_cp)))
    for dep in (64, 256, 512):
        g.append(("demote", str(dep), fit_surgery(bank_X, "demote", block=16, depth=dep)))
    g.append(("shuffle", "0", fit_surgery(bank_X, "shuffle", seed=0)))
    for a in (0.5, 2.0):                                   # v1.4 interaction arm
        g.append(("combo", f"{a}|256",
                  fit_surgery(bank_X, "combo", alpha=a, block=16, depth=256)))
    return g


def cp_rank_targets():
    """Per (encoder, dataset): realized post-CP rank = LeJEPA/SimCLR median rankme
    (nd7 pass, nd1 convention)."""
    nd7 = pd.read_csv(OUT / "nd7_placement.csv")
    nd7 = nd7[nd7.method.isin(["LeJEPA", "SimCLR"])]
    return nd7.groupby(["encoder", "dataset"]).rankme.median().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None, help="enc__ds filters")
    args = ap.parse_args()
    cells = sorted(p.stem for p in FEAT.glob("*.npz"))
    if args.cells:
        cells = [c for c in cells if c in set(args.cells)]
    if not cells:
        raise SystemExit(f"no feature cells found under {FEAT} — rsync the dump first")
    targets = cp_rank_targets()

    done = set()
    if RESULT.exists() and RESULT.stat().st_size > 0:
        with open(RESULT) as f:
            rd = csv.DictReader(f)
            assert rd.fieldnames == FIELDS, f"{RESULT} has a different layout"
            done = {(r["encoder"], r["dataset"], r["transform"], r["param"]) for r in rd}
    write_header = (not RESULT.exists()) or RESULT.stat().st_size == 0
    with open(RESULT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        for ci, cell in enumerate(cells):
            enc, ds = cell.split("__")
            if cell_complete(done, enc, ds,
                             has_target=targets.get((enc, ds)) is not None):
                continue                                   # cell complete: skip SVD/calibration
            z = np.load(FEAT / f"{cell}.npz")
            bX = z["bank_X"].astype(np.float64)
            by, qy = z["bank_y"], z["query_y"]
            qX = z["query_X"].astype(np.float64)
            # v1.4: out-of-span query fraction — DIAGNOSTIC only (the surgery maps
            # now act as identity on the bank row-space complement, so the out-of-
            # span part passes through; a large value means dose dilution, not loss)
            _, _, Vt_ = np.linalg.svd(bX, full_matrices=False)
            qn = np.linalg.norm(qX, axis=1)
            oos = float(np.median(np.linalg.norm(qX - qX @ Vt_.T @ Vt_, axis=1)
                                  / np.maximum(qn, 1e-12)))
            tgt = targets.get((enc, ds))
            alpha_cp = None
            if tgt is not None:
                alpha_cp = calibrate_alpha(bX, target_rankme=float(tgt))
                if alpha_cp > 3.99 or alpha_cp < 0.06:
                    print(f"  WARN: alpha_cp clamped at bound for {cell} ({alpha_cp:.4f})")
            else:
                print(f"  NOTE: no nd7 rank target for {cell} — power_cp skipped")
            for name, param, M in grid(bX, alpha_cp):
                if (enc, ds, name, param) in done:
                    continue
                sb = bX if M is None else apply_surgery(bX, M)
                sq = qX if M is None else apply_surgery(qX, M)
                from sklearn.preprocessing import normalize
                sb_l2 = normalize(sb)
                row = dict(encoder=enc, dataset=ds, transform=name, param=param,
                           knn_f1=vote_operator_metrics(sb, by, sq, qy, k=20)["knn_f1_hat"],
                           lp_f1=lp_f1(sb, by, sq, qy),
                           rankme_raw_bank=rankme_raw(sb),
                           rankme_l2_bank=rankme_raw(sb_l2),
                           capture_l2=capture_of(sb_l2, by),
                           cC_K_l2=cC_K_of(sb_l2, by),
                           alpha_cp="" if alpha_cp is None else f"{alpha_cp:.4f}",
                           query_oos_frac=oos,
                           rankme_target="" if (name != "power_cp" or tgt is None)
                                         else f"{float(tgt):.4f}")
                w.writerow(row)
                f.flush()
            print(f"[{ci + 1}/{len(cells)}] {cell} done")
    print(f"-> {RESULT}\nNext: python eval/new_direction/int1_verdict.py")


if __name__ == "__main__":
    main()
