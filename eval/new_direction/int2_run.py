#!/usr/bin/env python3
"""
int2_run.py — INT2 (LOCAL CPU): apply the FROZEN reversion grid (INT2_PREREG.md
v1.0) to every dumped post-CP cell and evaluate both readouts. Resumable by row.

Grid (8 rows per cell): identity | rotation seeds {0,1} | four RankMe-calibrated
power arms — half / full / wrong-direction / overshoot (targets from the cell's
INT1 pre bank RankMe; fast spectrum-side calibration, acceptance at verdict) |
spectrum transplant (pre bank singular values in rank order).

Readouts: kNN = vote_operator_metrics (G-POST-anchored); LP = standardized
deterministic probe ([proxy] unconditionally this round — INT2_PREREG §3).

Run: python eval/new_direction/int2_run.py  [--cells <enc__ds__method__seed> ...]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P0
_sys.path.insert(0, str(_P0(__file__).resolve().parent))
_sys.path.insert(0, str(_P0(__file__).resolve().parent.parent / "utils"))

from int1_surgery import (fit_surgery, apply_surgery, rankme_raw, capture_of,
                          cC_K_of, rankme_from_s, calibrate_alpha_from_s)
import hashlib
from int1_run import lp_f1
from nd9_task_operator import vote_operator_metrics

ROOT = _P0(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
FEAT = OUT / "int2_features"
PRE_FEAT = OUT / "int1_features"
RESULT = OUT / "int2_results.csv"
FIELDS = ["method", "encoder", "dataset", "seed", "arm", "param", "knn_f1",
          "lp_f1", "vote_margin", "vote_pos_frac", "n_query", "rankme_raw_bank",
          "rankme_l2_bank", "capture_l2", "cC_K_l2", "alpha", "rankme_target",
          "feasible", "spec_profile_err", "spec_max_amp", "spec_rank_ratio",
          "query_oos_frac"]
CALIBRATED_ARMS = ("half", "full", "wrong", "over")


def reversion_targets(rankme_pre, rankme_post, s_post=None):
    """Frozen INT2 targets in RankMe space (all defined in log-RankMe):
    full = pre; half = log-midpoint; wrong = away from pre by the full-reversion
    magnitude; over = beyond pre by the same magnitude. v1.1 (Codex round-6):
    with s_post given, wrong/over are CAPPED to the attainable calibration range
    (alpha in [0.06, 3.99], 2% margin) and flagged infeasible when the capped
    dose magnitude falls below half the required magnitude."""
    lp, lo = np.log(rankme_pre), np.log(rankme_post)
    mag = abs(lp - lo)
    out = dict(half=float(np.exp(0.5 * (lp + lo))), full=float(rankme_pre),
               wrong=float(np.exp(lo + (lo - lp))), over=float(np.exp(lp + (lp - lo))),
               wrong_feasible=True, over_feasible=True)
    if s_post is None:
        return out
    s_post = np.asarray(s_post, dtype=np.float64)
    hi_att = rankme_from_s(s_post ** 0.06) * 0.98      # flattest attainable
    lo_att = rankme_from_s(s_post ** 3.99) * 1.02      # sharpest attainable
    for arm, req in (("wrong", mag), ("over", 2.0 * mag)):
        capped = float(np.clip(out[arm], lo_att, hi_att))
        got = abs(np.log(capped) - lo)
        out[arm] = capped
        out[f"{arm}_feasible"] = bool(got >= 0.5 * req and req > 0)
    return out


def int2_cell_complete(have):
    """Resume acceptance: have = set of (arm, param) rows already written.
    EXACT grid (Codex round-6): exactly 8 rows — one row per non-rotation arm
    (duplicate params for the same arm are rejected) plus both rotation seeds."""
    if len(have) != 8:
        return False
    counts = {}
    for a, _ in have:
        counts[a] = counts.get(a, 0) + 1
    if counts.get("rotation") != 2 or ("rotation", "0") not in have \
            or ("rotation", "1") not in have:
        return False
    if not all(counts.get(a) == 1 for a in
               ("identity", "half", "full", "wrong", "over", "transplant")):
        return False
    for a, pm in have:                        # round-8: no pseudo-grids
        if a in ("identity", "transplant") and pm != "":
            return False
        if a in CALIBRATED_ARMS:
            try:
                if not np.isfinite(float(pm)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def check_pre_post_consistency(pre_path, post_path):
    """Provenance gate (Codex round-6): the post cell must share the pre cell's
    protocol — identical bank/query label sequences and feature dimension."""
    a, b = np.load(pre_path), np.load(post_path)
    if a["bank_X"].shape[1] != b["bank_X"].shape[1]:
        return False, f"feature dim mismatch {a['bank_X'].shape[1]} vs {b['bank_X'].shape[1]}"
    for k in ("bank_y", "query_y"):
        if len(a[k]) != len(b[k]) or not np.array_equal(a[k], b[k]):
            return False, f"label sequence mismatch in {k}"
    return True, "ok"


def verify_pre_checksums(checksum_file=OUT / "int1_checksums.sha256", root=ROOT):
    """Round-7 strict: the manifest must yield EXACTLY 60 verified pre .npz files
    AND a verified int1_results.csv (Y_pre source for INT2-2)."""
    if not checksum_file.exists():
        return False, f"missing {checksum_file}"
    npz_names, seen_results, bad = set(), False, []
    for line in checksum_file.read_text().splitlines():
        digest, rel = line.split()
        f = (root / rel) if not rel.startswith("/") else _P0(rel)
        if not f.exists():
            bad.append(f"{f.name} (missing)")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if f.name.endswith(".npz") and "int1_features" in str(f):
            npz_names.add(f.name)              # round-8: UNIQUE files, not lines
            if h != digest:
                bad.append(f.name)
        elif f.name == "int1_results.csv":
            seen_results = True
            if h != digest:
                bad.append(f.name)
    from int1_features_dump import ENCODER_NAMES, DATASET_NAMES
    expected_set = {f"{e}__{d}.npz" for e in ENCODER_NAMES for d in DATASET_NAMES}
    if npz_names != expected_set:
        return False, (f"pre npz name set mismatch (round-9 exact-set rule): "
                       f"{len(npz_names)} found, expected the 4x15 set")
    if not seen_results:
        return False, "int1_results.csv not covered by the checksum manifest"
    return (not bad), (f"checksum mismatch: {bad}" if bad else "ok")


def verify_post_checksums(sidecar, feat_dir, expected_names):
    """Round-10 exact-set POST provenance: the sidecar (frozen by the CLUSTER
    full --verify-only, transported together with the features, never
    regenerated locally) must name EXACTLY the expected file set, every hash
    must match, and the feature dir must contain no foreign npz."""
    if not sidecar.exists():
        return False, ("missing sidecar — it is written by the cluster full "
                       "--verify-only and must be rsynced WITH the features")
    entries = {}
    for line in sidecar.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            return False, f"malformed sidecar line: {line[:60]!r}"
        digest, name = parts
        if name in entries:
            return False, f"duplicate sidecar entry: {name}"
        entries[name] = digest
    if set(entries) != set(expected_names):
        return False, (f"sidecar name set mismatch: {len(entries)} entries vs "
                       f"expected {len(expected_names)} (must equal the frozen "
                       "manifest set)")
    on_disk = {f.name for f in _P0(feat_dir).glob("*.npz")}
    foreign = on_disk - set(expected_names)
    if foreign:
        return False, f"foreign npz in feature dir: {sorted(foreign)[:5]}"
    bad = []
    for name, digest in entries.items():
        f = _P0(feat_dir) / name
        if not f.exists():
            bad.append(f"{name} (missing)")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return (not bad), (f"checksum mismatch: {bad[:5]}" if bad else "ok")


def spectrum_profile_stats(s_achieved, s_target, s_before, dim_scale=None):
    """Round-7 transplant acceptance inputs: relative L2 error between the
    NORMALIZED achieved and target spectra (top min-rank), and the maximum
    per-direction amplification applied (achieved / before)."""
    a = np.asarray(s_achieved, dtype=np.float64)
    t = np.asarray(s_target, dtype=np.float64)
    braw = np.asarray(s_before, dtype=np.float64)
    b = np.maximum(braw, 1e-12)
    m = min(len(a), len(t))
    an, tn = a[:m] / max(a[:m].sum(), 1e-30), t[:m] / max(t[:m].sum(), 1e-30)
    err = float(np.linalg.norm(an - tn) / max(np.linalg.norm(tn), 1e-30))
    # round-8: amplification = the MAP scale (target/before) on non-null
    # directions — achieved/before underestimates the applied scale
    def _nrank(v):
        v = np.asarray(v, dtype=np.float64)
        # round-9: SAME mask scale as the surgery (smax * max(n, d) * eps)
        sc = dim_scale if dim_scale is not None else len(v)
        return int((v > v.max() * sc * np.finfo(np.float64).eps).sum())
    nb = _nrank(braw)
    mm = min(len(t), nb)
    amp = float(np.max(np.maximum(t[:mm], 1e-12) / b[:mm])) if mm else 1.0
    return err, amp, _nrank(a), _nrank(t)


def pre_side(enc, ds, cache={}):
    """Pre bank spectrum + RankMe from the INT1 features (reused, checksummed)."""
    key = (enc, ds)
    if key not in cache:
        z = np.load(PRE_FEAT / f"{enc}__{ds}.npz")
        s_pre = np.linalg.svd(z["bank_X"].astype(np.float64), compute_uv=False)
        cache[key] = (s_pre, rankme_from_s(s_pre))
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None)
    args = ap.parse_args()
    cells = sorted(p.stem for p in FEAT.glob("*.npz"))
    if args.cells:
        cells = [c for c in cells if c in set(args.cells)]
    if not cells:
        raise SystemExit(f"no feature cells under {FEAT} — rsync the dump first")
    ok_ck, msg_ck = verify_pre_checksums()
    assert ok_ck, f"pre-feature provenance FAILED: {msg_ck}"
    print("pre-feature checksums verified against int1_checksums.sha256")

    from int2_features_dump import load_manifest
    expected_names = {f"{cid}.npz" for cid in load_manifest().cell_id}
    ok_post, msg_post = verify_post_checksums(OUT / "int2_features.sha256", FEAT,
                                              expected_names=expected_names)
    assert ok_post, f"POST provenance FAILED: {msg_post}"
    print("post-feature checksums verified against the CLUSTER-frozen sidecar "
          "(verify only, never regenerate locally)")
    done = set()
    if RESULT.exists() and RESULT.stat().st_size > 0:
        with open(RESULT) as f:
            rd = csv.DictReader(f)
            assert rd.fieldnames == FIELDS, f"{RESULT} has a different layout"
            rows_list = [(r["method"], r["encoder"], r["dataset"], r["seed"],
                          r["arm"], r["param"]) for r in rd]
            assert len(rows_list) == len(set(rows_list)), \
                "duplicate raw rows in existing results — clean before resume"
            done = set(rows_list)
    write_header = (not RESULT.exists()) or RESULT.stat().st_size == 0
    with open(RESULT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
            f.flush()
        for ci, cell in enumerate(cells):
            enc, ds, method, seed = cell.split("__")
            ck = (method, enc, ds, seed)
            # provenance BEFORE any resume skip (round-7)
            ok_pp, msg_pp = check_pre_post_consistency(
                PRE_FEAT / f"{enc}__{ds}.npz", FEAT / f"{cell}.npz")
            assert ok_pp, f"{cell}: pre/post provenance FAILED — {msg_pp}"
            have = {(a, p) for (m, e, d, s_, a, p) in done if (m, e, d, s_) == ck}
            if int2_cell_complete(have):
                continue
            z = np.load(FEAT / f"{cell}.npz")
            bX = z["bank_X"].astype(np.float64)
            qX = z["query_X"].astype(np.float64)
            by, qy = z["bank_y"], z["query_y"]
            s_post = np.linalg.svd(bX, compute_uv=False)
            s_pre, rk_pre = pre_side(enc, ds)
            targets = reversion_targets(rk_pre, rankme_from_s(s_post), s_post=s_post)
            _, _, Vt_ = np.linalg.svd(bX, full_matrices=False)
            qn = np.linalg.norm(qX, axis=1)
            oos = float(np.median(np.linalg.norm(qX - qX @ Vt_.T @ Vt_, axis=1)
                                  / np.maximum(qn, 1e-12)))
            grid = [("identity", "", None, "", "")]
            for s_ in (0, 1):
                grid.append(("rotation", str(s_),
                             fit_surgery(bX, "rotation", seed=s_), "", ""))
            for arm in CALIBRATED_ARMS:
                a = calibrate_alpha_from_s(s_post, target_rankme=targets[arm])
                feas = targets.get(f"{arm}_feasible", True)
                grid.append((arm, f"{a:.4f}", fit_surgery(bX, "power", alpha=a),
                             targets[arm], "1" if feas else "0"))
            m_ = min(len(s_pre), len(s_post))
            s_tr = np.concatenate([s_pre[:m_], s_post[m_:]])
            grid.append(("transplant", "",
                         fit_surgery(bX, "transplant", s_target=s_pre),
                         rankme_from_s(s_tr), ""))
            for arm, param, M, tgt, feas in grid:
                if (arm, param) in have:
                    continue
                sb = bX if M is None else apply_surgery(bX, M)
                sq = qX if M is None else apply_surgery(qX, M)
                from sklearn.preprocessing import normalize
                sb_l2 = normalize(sb)
                vm = vote_operator_metrics(sb, by, sq, qy, k=20)
                if arm == "transplant":
                    s_ach = np.linalg.svd(sb, compute_uv=False)
                    perr, pamp, ar_, tr_ = spectrum_profile_stats(
                        s_ach, s_pre, s_post, dim_scale=max(bX.shape))
                    prr = f"{ar_}/{tr_}"
                else:
                    perr, pamp, prr = "", "", ""
                w.writerow(dict(
                    method=method, encoder=enc, dataset=ds, seed=seed, arm=arm,
                    param=param,
                    knn_f1=vm["knn_f1_hat"],
                    vote_margin=vm["vote_margin_mean"],
                    vote_pos_frac=vm["vote_margin_pos_frac"],
                    n_query=len(qy),
                    spec_profile_err=perr, spec_max_amp=pamp, spec_rank_ratio=prr,
                    lp_f1=lp_f1(sb, by, sq, qy),
                    rankme_raw_bank=rankme_raw(sb),
                    rankme_l2_bank=rankme_raw(sb_l2),
                    capture_l2=capture_of(sb_l2, by),
                    cC_K_l2=cC_K_of(sb_l2, by),
                    alpha=param if arm in CALIBRATED_ARMS else "",
                    rankme_target="" if tgt == "" else f"{float(tgt):.4f}",
                    feasible=feas,
                    query_oos_frac=oos))
                f.flush()
            print(f"[{ci + 1}/{len(cells)}] {cell} done")
    print(f"-> {RESULT}\nNext: python eval/new_direction/int2_verdict.py")


if __name__ == "__main__":
    main()
