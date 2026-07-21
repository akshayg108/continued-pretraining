#!/usr/bin/env python3
"""
int2_verdict.py — INT2 adjudicator (INT2_PREREG.md v1.0, frozen 2026-07-21,
pre-data). Implements one-to-one:

  G0     manifest census (exact key set, 8-row grid per cell, finiteness).
  G-NC   rotation sham, joint per cell, >= 98% of each panel, both seeds.
  G-POST A0 identity kNN vs the ND12 recorded knn_f1_hat_post: rho >= 0.98 AND
         median |diff| < 0.005 on the primary (seed-42) panel. Fail -> stop.
  G-P    per arm: placement drift ratio R = median|d cC_K_l2| / 0.3393 (frozen
         INT1 shuffle scale) > 0.5 OR median |capture drift| > 0.05 -> excluded;
         per-row capture screen (|drift| > 0.05 leaves estimates).
  INT2-1 functional participation (PRIMARY, kNN): A = S x (Y_full - Y_identity),
         95% block CI; + dose consistency (half vs full) + wrong-direction
         artifact clause. YES / no / NO VERDICT.
  INT2-2 net restoration (PRIMARY, kNN): G = |Y_full - Y_pre| - |Y_id - Y_pre|;
         restoration language iff 95% CI < 0.
  INT2-3 overshoot discriminator; INT2-4 strata/contrasts (Bonferroni-3);
  INT2-5 transplant diagnostic. LP mirrors all carry [proxy] unconditionally.

Run (local): python eval/new_direction/int2_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from int1_verdict import block_ci, tri_verdict, calibration_ok            # noqa: E402
from int2_features_dump import load_manifest                              # noqa: E402

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
PLACEMENT_SCALE = 0.3393        # frozen: INT1 shuffle median |d cC_K| (pre features)
CAP_THR = 0.05
NC_TOL = 0.005                  # frozen G-NC per-cell tolerance (module constant so
                                # the synthetic smoke can relax it; production value
                                # is FROZEN in INT2_PREREG §4)
DIR_FLOOR = 0.01                # |d log RankMe| below this: direction-undefined
ARMS = ("identity", "rotation", "half", "full", "wrong", "over", "transplant")
KEY4 = ["method", "encoder", "dataset", "seed"]


def direction_sign(rankme_pre, rankme_post, floor=DIR_FLOOR):
    """S = sign(log pre - log post); 0 (excluded) below the frozen floor."""
    d = float(np.log(rankme_pre) - np.log(rankme_post))
    if abs(d) < floor:
        return 0
    return 1 if d > 0 else -1


EQ_MARGIN = 0.005               # v1.1 frozen equivalence margin for claiming "no"


def int21_decision(mean_A, ci_A, mean_half, mean_full, ci_wrong, eq=EQ_MARGIN):
    """INT2-1 decision (v1.1, Codex round-6 semantics):
      - wrong-direction 'improvement' -> NO VERDICT (surgery artifact);
      - aligned CI > 0 + dose consistency -> YES; aligned but dose-inconsistent
        -> NO VERDICT (mixed evidence);
      - CI < 0 -> no (significant anti-alignment);
      - CI within the pre-registered equivalence margin -> no (bounded null);
      - CI straddling 0 beyond the margin -> NO VERDICT (never a negative)."""
    if ci_wrong[0] > 0:
        return "NO VERDICT"
    if ci_A[0] > 0:
        dose_ok = (np.sign(mean_half) == np.sign(mean_full)
                   and abs(mean_half) <= abs(mean_full) + 1e-12)
        return "YES" if dose_ok else "NO VERDICT"
    if ci_A[1] < 0:
        return "no"
    if -eq <= ci_A[0] and ci_A[1] <= eq:
        return "no"
    return "NO VERDICT"


def parse_rank_pair(v):
    """Round-10 strict parser: exactly two positive integers 'a/t', else None."""
    parts = str(v).split("/")
    if len(parts) != 2:
        return None
    try:
        a, t = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if a <= 0 or t <= 0 or a != int(a) or t != int(t):
        return None
    return int(a), int(t)


def direction_series(r):
    """S per cell over ALL seeds (round-6: S was seed-42-only, silently emptying
    the cross-seed replication). rankme_pre is recovered from the full arm's
    recorded rankme_target; rankme_post from the identity row."""
    idn = r[r.arm == "identity"].set_index(KEY4).rankme_raw_bank
    tgt = pd.to_numeric(r[r.arm == "full"].set_index(KEY4).rankme_target,
                        errors="coerce")
    return pd.Series({k: direction_sign(tgt[k], idn[k])
                      for k in tgt.index if k in idn.index and np.isfinite(tgt[k])})


def infeasible_mask(sub):
    """Round-7: feasible comes back from CSV as 0.0/1.0 — numeric parse, never
    string comparison ('0' != '0.0' let every infeasible dose through)."""
    v = pd.to_numeric(sub.feasible, errors="coerce")
    return (v == 0).values


def gpost_ok(got, ref, tol_cell=0.01):
    """Round-7 G-POST per-cell gate: the WORST per-cell |diff| must stay under
    tol_cell (correlation + median let a single corrupted cell through)."""
    worst = float((np.asarray(got, dtype=float) - np.asarray(ref, dtype=float)
                   ).__abs__().max())
    return worst < tol_cell, worst


def g_values(y_surg, y_post, y_pre):
    """Per-cell INT2-2 values: negative = surgery moved the readout toward pre."""
    return np.abs(np.asarray(y_surg) - np.asarray(y_pre)) \
        - np.abs(np.asarray(y_post) - np.asarray(y_pre))


def deltas4(sub, ident, thr=CAP_THR):
    """Per-ROW deltas vs identity on the 4-part key (method, encoder, dataset,
    seed) with the per-row capture screen (INT1 v1.6 semantics)."""
    s = sub.reset_index(drop=True)
    keys = pd.MultiIndex.from_arrays([s[k] for k in KEY4])
    dk_v = s.knn_f1.values - ident.knn_f1.reindex(keys).values
    dl_v = s.lp_f1.values - ident.lp_f1.reindex(keys).values
    keep = np.isfinite(dk_v) & np.isfinite(dl_v)
    dropped = []
    if thr is not None and "capture_l2" in s.columns:
        drift = np.abs(s.capture_l2.values - ident.capture_l2.reindex(keys).values)
        bad = np.nan_to_num(drift, nan=0.0) > thr
        if "cC_K_l2" in s.columns and "cC_K_l2" in ident.columns:
            pdrift = np.abs(s.cC_K_l2.values - ident.cC_K_l2.reindex(keys).values)
            bad |= np.nan_to_num(pdrift, nan=0.0) > 0.5 * PLACEMENT_SCALE
        arms = s["arm"].astype(str).values if "arm" in s.columns \
            else np.array([""] * len(s))
        dropped = sorted({(m, e, d, sd, a) for (m, e, d, sd), a, b
                          in zip(keys, arms, bad) if b})
        keep &= ~bad
    idx = keys[keep]
    dk = pd.Series(dk_v[keep], index=idx)
    dl = pd.Series(dl_v[keep], index=idx)
    return dk, dl, np.array([i[2] for i in idx]), dropped


def coverage_ok(keys, frac=0.75, total=None, min_datasets=12):
    """v1.0 floor: >= 75% of the stratum's cells AND >= 12 datasets."""
    cells = set(map(tuple, keys))
    if total is not None and len(cells) < frac * total:
        return False
    return len({k[2] for k in cells}) >= min_datasets


def main():
    path = OUT / "int2_results.csv"
    if not path.exists():
        sys.exit(f"MISSING {path} — run int2_run.py first")
    r = pd.read_csv(path, dtype={"param": str, "seed": str, "alpha": str}
                    ).fillna({"param": "", "alpha": ""})
    man = load_manifest()
    man = man.assign(seed=man.seed.astype(int).astype(str))

    # ---- G0 manifest census ----------------------------------------------------------------
    msgs = []
    dup = r.duplicated(KEY4 + ["arm", "param"])
    if dup.any():
        msgs.append(f"duplicate keys: {int(dup.sum())} rows")
    got = set(map(tuple, r[KEY4].drop_duplicates().values))
    want = set(map(tuple, man[["method", "encoder", "dataset", "seed"]].values))
    if got != want:
        msgs.append(f"cell set mismatch: missing {len(want - got)}, "
                    f"unexpected {len(got - want)}")
    from int2_run import int2_cell_complete
    for key, grp in r.groupby(KEY4):
        have = set(map(tuple, grp[["arm", "param"]].values))
        if not int2_cell_complete(have):
            msgs.append(f"{key} incomplete arm grid")
        extra = {a for a, _ in have} - set(ARMS)
        if extra:
            msgs.append(f"{key} unexpected arms {extra}")
    for col in ("knn_f1", "lp_f1", "rankme_raw_bank", "rankme_l2_bank",
                "capture_l2", "cC_K_l2", "query_oos_frac", "vote_margin",
                "vote_pos_frac", "n_query"):
        if not np.isfinite(pd.to_numeric(r[col], errors="coerce")).all():
            msgs.append(f"non-finite values in {col}")
    cal = r[r.arm.isin(("half", "full", "wrong", "over"))]
    if not np.isfinite(pd.to_numeric(cal.rankme_target, errors="coerce")).all():
        msgs.append("non-finite rankme_target on calibrated arms")
    fv = pd.to_numeric(cal.feasible, errors="coerce")
    if not fv.isin((0, 1)).all():
        msgs.append("feasible outside {0,1} on calibrated arms")
    if not (cal.param.astype(str) == cal.alpha.astype(str)).all():
        msgs.append("calibrated rows where param != alpha (non-canonical grid)")
    from int2_run import FIELDS as RUN_FIELDS
    if list(r.columns) != RUN_FIELDS:
        msgs.append(f"result columns differ from the frozen FIELDS contract")
    tp = r[r.arm == "transplant"]
    if len(tp):
        pe = pd.to_numeric(tp.spec_profile_err, errors="coerce")
        pa = pd.to_numeric(tp.spec_max_amp, errors="coerce")
        if not (np.isfinite(pe).all() and (pe >= 0).all()):
            msgs.append("transplant spec_profile_err not finite/non-negative")
        if not (np.isfinite(pa).all() and (pa > 0).all()):
            msgs.append("transplant spec_max_amp not finite/positive (note: a "
                        "purely shrinking map legitimately has max scale < 1)")
        if any(parse_rank_pair(v) is None for v in tp.spec_rank_ratio):
            msgs.append("transplant spec_rank_ratio not two positive integers")
    print(f"G0 census: {'PASS' if not msgs else 'FAIL'}")
    for m in msgs[:20]:
        print(f"  {m}")
    if msgs:
        sys.exit(1)

    ident = r[r.arm == "identity"].set_index(KEY4)
    prim = r[r.seed == "42"]
    print(f"cells: {len(got)} total, primary seed-42 panel: "
          f"{len(prim[KEY4].drop_duplicates())}")

    # ---- G-NC rotation sham ----------------------------------------------------------------
    ok_nc = True
    for s_ in ("0", "1"):
        sub = prim[(prim.arm == "rotation") & (prim.param == s_)]
        dk, dl, _, _ = deltas4(sub, ident, thr=None)
        j = pd.concat([dk.rename("k"), dl.rename("l")], axis=1, join="inner")
        nj = int(((j.k.abs() < NC_TOL) & (j.l.abs() < NC_TOL)).sum())
        need = int(np.ceil(0.98 * len(j)))
        print(f"G-NC rotation seed {s_}: joint < {NC_TOL} on {nj}/{len(j)} (need {need})")
        ok_nc &= nj >= need
    print(f"  -> {'PASS' if ok_nc else 'FAIL — harness bug, STOP'}")
    if not ok_nc:
        sys.exit(1)

    # ---- G-POST reproduction (ALL cells gate + primary reported) ---------------------------
    mi = man.set_index(["method", "encoder", "dataset", "seed"])
    for lab, panel in (("all-484", r), ("seed-42", prim)):
        a0r = panel[panel.arm == "identity"].set_index(KEY4)
        ok_all = True
        for col, refcol, tol in (("knn_f1", "knn_f1_hat_post", 0.01),
                                 ("vote_margin", "vote_margin_post", 0.01),
                                 ("vote_pos_frac", "vote_pos_frac_post", 0.01)):
            assert col in a0r.columns and refcol in mi.columns, \
                f"G-POST required column missing: {col}/{refcol}"
            j = pd.concat([pd.to_numeric(a0r[col], errors="coerce").rename("got"),
                           pd.to_numeric(mi[refcol], errors="coerce").rename("ref")],
                          axis=1, join="inner").dropna()
            if len(j) != len(a0r):             # round-8: exact join count, no NaN slack
                ok_all = False
                print(f"G-POST [{lab}] {col}: join {len(j)} != panel {len(a0r)} — FAIL")
                continue
            okc, worst = gpost_ok(j.got, j.ref, tol_cell=tol)
            rho = spearmanr(j.got, j.ref).correlation if len(j) > 2 else 1.0
            med = float((j.got - j.ref).abs().median())
            ok_all &= okc and rho >= 0.98 and med < 0.005
            print(f"G-POST [{lab}] {col}: rho {rho:+.4f}, med {med:.4f}, "
                  f"worst {worst:.4f} (tol {tol}), n {len(j)}")
        if "n_query" in a0r.columns and "n_test" in mi.columns:
            jn = pd.concat([pd.to_numeric(a0r.n_query, errors="coerce").rename("g"),
                            pd.to_numeric(mi.n_test, errors="coerce").rename("r")],
                           axis=1, join="inner").dropna()
            neq = int((jn.g != jn.r).sum())
            ok_all &= neq == 0
            print(f"G-POST [{lab}] n_test equality: {len(jn) - neq}/{len(jn)}")
        print(f"G-POST [{lab}] -> {'PASS' if ok_all else 'FAIL — STOP'}")
        if not ok_all:
            sys.exit(1)

    # ---- calibration acceptance (incl. transplant achieved-spectrum) -----------------------
    print("\nCalibration acceptance (2% RankMe tolerance; clamped alpha rejected; "
          "wrong/over feasibility from the run):")
    cal_reject = {}
    for arm in ("half", "full", "wrong", "over", "transplant"):
        sub = r[r.arm == arm].rename(columns={"alpha": "alpha_cp"})
        if arm == "transplant":
            tgt = pd.to_numeric(sub.rankme_target, errors="coerce")
            ach = pd.to_numeric(sub.rankme_raw_bank, errors="coerce")
            perr = pd.to_numeric(sub.get("spec_profile_err"), errors="coerce")
            pamp = pd.to_numeric(sub.get("spec_max_amp"), errors="coerce")
            pairs = [parse_rank_pair(v) for v in sub.spec_rank_ratio]
            ar_ = pd.Series([p[0] if p else np.nan for p in pairs],
                            index=sub.index)
            tr_ = pd.Series([p[1] if p else np.nan for p in pairs],
                            index=sub.index)
            sym = ((ar_ - tr_).abs() / tr_.clip(lower=1) <= 0.02)   # symmetric
            mask = (((ach - tgt).abs() / tgt.abs().clip(lower=1e-12) <= 0.02)
                    & (perr >= 0) & (perr <= 0.02) & (pamp > 0)
                    & (pamp <= 1e4) & sym).values
            reasons = [f"{int((~mask).sum())} cells failed RankMe/profile/"
                       "amplification/rank-ratio acceptance (0.02/0.02/1e4/0.98) "
                       "— below rank-ratio the arm is only 'normalized-profile "
                       "matched', not full spectrum restoration"] \
                if (~mask).any() else []
        else:
            mask, reasons = calibration_ok(sub)
            if "feasible" in sub.columns:
                infeas = infeasible_mask(sub)
                if infeas.any():
                    mask = mask & ~infeas
                    reasons.append(f"{int(infeas.sum())} cells dose-infeasible")
        cal_reject[arm] = set(map(tuple, sub[~mask][KEY4].values))
        print(f"  {arm:>10}: {int((~mask).sum())}/{len(sub)} rejected"
              + (f" ({'; '.join(reasons)})" if reasons else ""))

    # ---- G-P --------------------------------------------------------------------------------
    excluded = set()
    print(f"\nG-P (R = median|d cC_K| / {PLACEMENT_SCALE}; capture stop {CAP_THR}; "
          "per-row capture + placement screens active in all estimates):")
    for arm in ("half", "full", "wrong", "over", "transplant"):
        js = prim[prim.arm == arm].set_index(KEY4)
        cckd = (js.cC_K_l2 - ident.cC_K_l2).dropna().abs().median()
        capd = (js.capture_l2 - ident.capture_l2).dropna().abs().median()
        R = cckd / PLACEMENT_SCALE
        viol = bool(R > 0.5) or bool(capd > CAP_THR)
        if viol:
            excluded.add(arm)
        print(f"  {arm:>10}: R = {R:.2f} (capture drift {capd:.4f}) "
              f"-> {'EXCLUDED' if viol else 'ok'}")

    # ---- scaffolding ------------------------------------------------------------------------
    S = direction_series(r)                    # ALL seeds (round-6 fix)
    prim_cells = set(map(tuple, prim[KEY4].drop_duplicates().values))
    n_undef = int((S[[k in prim_cells for k in S.index]] == 0).sum())
    print(f"\ndirection: S=+1 {int((S == 1).sum())}, S=-1 {int((S == -1).sum())}, "
          f"undefined (excluded) {n_undef} on the primary panel")
    int1 = pd.read_csv(OUT / "int1_results.csv", dtype={"param": str}
                       ).fillna({"param": ""})
    y_pre_k = int1[int1["transform"] == "identity"].set_index(
        ["encoder", "dataset"]).knn_f1
    y_pre_l = int1[int1["transform"] == "identity"].set_index(
        ["encoder", "dataset"]).lp_f1
    all_id = r[r.arm == "identity"].set_index(KEY4)      # round-8: all seeds

    def arm_deltas(arm, panel, y="knn", thr=CAP_THR, require_dir=True):
        sub = panel[panel.arm == arm]
        sub = sub[~sub.set_index(KEY4).index.isin(cal_reject.get(arm, set()))]
        dk, dl, _, dropped = deltas4(sub, ident, thr=thr)
        if dropped:
            print(f"    row screen: {len(dropped)} rows left {arm}")
        d = dk if y == "knn" else dl
        if require_dir:
            d = d[[k in S.index and S[k] != 0 for k in d.index]]
        return d

    def a_of(d):
        sv = np.array([S[k] for k in d.index])
        return pd.Series(sv * d.values, index=d.index)

    def run_int21(panel, n_total, y="knn", thr=CAP_THR, label=""):
        bad_arms = [a for a in ("full", "half", "wrong") if a in excluded]
        if bad_arms:
            print(f"  {label}arm(s) {bad_arms} G-P-excluded — NO VERDICT")
            return
        parts = {a: arm_deltas(a, panel, y=y, thr=thr)
                 for a in ("full", "half", "wrong")}
        elig = parts["full"].index.intersection(parts["half"].index)
        elig = elig.intersection(parts["wrong"].index)       # common eligible set
        if not coverage_ok(elig, total=n_total):
            print(f"  {label}insufficient common eligible coverage "
                  f"({len(set(map(tuple, elig)))}/{n_total}) — NO VERDICT")
            return
        A = {a: a_of(parts[a].loc[elig]) for a in parts}
        ds_ = np.array([k[2] for k in elig])
        ciA = block_ci(A["full"].values, ds_)
        ciW = block_ci(A["wrong"].values, ds_)
        v = int21_decision(A["full"].mean(), ciA, A["half"].mean(),
                           A["full"].mean(), ciW)
        print(f"  {label}A_full {A['full'].mean():+.4f} CI[{ciA[0]:+.4f},{ciA[1]:+.4f}]"
              f"  A_half {A['half'].mean():+.4f}  A_wrong {A['wrong'].mean():+.4f} "
              f"CI[{ciW[0]:+.4f},{ciW[1]:+.4f}]  (n={len(elig)})")
        print(f"  {label}-> INT2-1: {v}")
        return A["full"]

    def run_g(panel, n_total, y="knn", thr=CAP_THR, arm="full", label="", level=95.0):
        if arm in excluded:
            print(f"  {label}{arm} arm excluded — NO VERDICT")
            return
        # G uses ALL cells incl. S=0 (prereg: the S filter applies to A-stats only)
        d = arm_deltas(arm, panel, y=y, thr=thr, require_dir=False)
        yp = y_pre_k if y == "knn" else y_pre_l
        keys = [k for k in d.index if (k[1], k[2]) in yp.index and k in all_id.index]
        if not coverage_ok(keys, total=n_total):
            print(f"  {label}insufficient coverage — NO VERDICT")
            return
        col = "knn_f1" if y == "knn" else "lp_f1"
        y0 = all_id[col].loc[keys]
        g = g_values(y0.values + d.loc[keys].values, y0.values,
                     np.array([yp[(k[1], k[2])] for k in keys]))
        ci = block_ci(g, np.array([k[2] for k in keys]), level=level)
        lic = ci[1] < 0
        print(f"  {label}G[{arm}] {g.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] -> "
              f"{'RESTORATION (CI < 0)' if lic else 'no restoration claim'}")

    main_panel = prim[prim.encoder != "SigLIP"]
    n_main = len(main_panel[KEY4].drop_duplicates())

    def panel_scope(panel, y="knn", thr=CAP_THR, quiet=False):
        """Method-level claim scoping (v1.2): methods failing their own eligible
        floor leave the pooled panel; returns (included, pooled_panel)."""
        inc = []
        for meth in ("LeJEPA", "SimCLR", "DIET"):
            pan = panel[panel.method == meth]
            if not len(pan):
                continue
            n_meth = len(pan[KEY4].drop_duplicates())    # round-9: real denominator
            parts = {a: arm_deltas(a, pan, y=y, thr=thr)
                     for a in ("full", "half", "wrong")}
            elig = parts["full"].index.intersection(parts["half"].index
                                                    ).intersection(parts["wrong"].index)
            if coverage_ok(elig, total=n_meth):
                inc.append(meth)
            elif not quiet:
                print(f"  note: {meth} fails its method-level eligible floor "
                      f"({len(set(map(tuple, elig)))}/{n_meth}) — out of "
                      "pooled primary; stratum NO VERDICT")
        return inc, panel[panel.method.isin(inc)]

    def report_panel(panel, tag, y="knn", thr=CAP_THR):
        """Pre-registered INT2-1/2/4 block for one (seed panel, readout)."""
        proxy = " [proxy]" if y == "lp" else ""
        included, pooled = panel_scope(panel, y=y, thr=thr)
        n_pooled = len(pooled[KEY4].drop_duplicates())
        print(f"\n{tag} INT2-1{proxy} (scope = "
              f"{'+'.join(included) if included else 'EMPTY -> NO VERDICT'}):")
        if included:
            run_int21(pooled, n_pooled, y=y, thr=thr, label="  ")
        # round-11: pooled INT2-2 has its OWN G-scope — methods whose screened
        # FULL-arm rows (no direction filter) pass their own 75% floor
        g_inc = []
        for meth in ("LeJEPA", "SimCLR", "DIET"):
            pan = panel[panel.method == meth]
            if not len(pan) or "full" in excluded:
                continue
            d = arm_deltas("full", pan, y=y, thr=thr, require_dir=False)
            if coverage_ok(d.index, total=len(pan[KEY4].drop_duplicates())):
                g_inc.append(meth)
        g_pooled = panel[panel.method.isin(g_inc)]
        print(f"{tag} INT2-2{proxy} (G-scope = "
              f"{'+'.join(g_inc) if g_inc else 'EMPTY -> NO VERDICT'}):")
        if g_inc:
            run_g(g_pooled, len(g_pooled[KEY4].drop_duplicates()), y=y, thr=thr,
                  label="  ")
        strat = {}
        for meth in ("LeJEPA", "SimCLR", "DIET"):  # round-11: A and G independent
            pan = panel[panel.method == meth]
            if "full" in excluded or not len(pan):
                continue
            n_meth = len(pan[KEY4].drop_duplicates())
            A = a_of(arm_deltas("full", pan, y=y, thr=thr))
            if coverage_ok(A.index, total=n_meth):
                strat[meth] = A                # contrast/symmetry are A-statistics
                ci = block_ci(A.values, np.array([k[2] for k in A.index]),
                              level=100.0 - 5.0 / 3.0)
                print(f"  {meth:>7}: A {A.mean():+.4f} "
                      f"CI98.33[{ci[0]:+.4f},{ci[1]:+.4f}]"
                      f"{' *' if ci[0] > 0 or ci[1] < 0 else ''}{proxy}")
            else:
                print(f"  {meth:>7}: A NO VERDICT (coverage)")
            run_g(pan, n_meth, y=y, label=f"    {meth} ",   # G never blocked by A
                  level=100.0 - 5.0 / 3.0)
        if len(strat) == 3:
            uds = sorted({k[2] for A in strat.values() for k in A.index})
            rng = np.random.RandomState(0)
            boots = []
            for _ in range(2000):
                pick = rng.choice(uds, len(uds), replace=True)
                ms = {}
                for meth, A in strat.items():
                    v = np.concatenate([A.values[[k[2] == d_ for k in A.index]]
                                        for d_ in pick])
                    ms[meth] = v.mean() if len(v) else 0.0
                boots.append(ms["DIET"] - 0.5 * (ms["LeJEPA"] + ms["SimCLR"]))
            print(f"  DIET vs mean(SSL): CI[{np.percentile(boots, 2.5):+.4f},"
                  f"{np.percentile(boots, 97.5):+.4f}]{proxy}")
        con = pd.concat(strat.values()) if strat else pd.Series(dtype=float)
        if len(con):
            sv = np.array([S[k] for k in con.index])
            d_pos, d_neg = con.values[sv == 1], con.values[sv == -1]
            if len(d_pos) and len(d_neg):
                uds = sorted({k[2] for k in con.index})
                rng = np.random.RandomState(1)
                boots = []
                for _ in range(2000):
                    pick = rng.choice(uds, len(uds), replace=True)
                    a = np.concatenate([con.values[(sv == 1)
                                        & np.array([k[2] == d_ for k in con.index])]
                                        for d_ in pick])
                    b = np.concatenate([con.values[(sv == -1)
                                        & np.array([k[2] == d_ for k in con.index])]
                                        for d_ in pick])
                    if len(a) and len(b):
                        boots.append(a.mean() - b.mean())
                if boots:
                    print(f"  direction symmetry (S+1 - S-1): "
                          f"{d_pos.mean() - d_neg.mean():+.4f} "
                          f"CI[{np.percentile(boots, 2.5):+.4f},"
                          f"{np.percentile(boots, 97.5):+.4f}]{proxy}")
            mae = con[[k[1] == "MAE" for k in con.index]]
            if len(mae):
                print(f"  MAE-backbone stratum: A {mae.mean():+.4f} (n={len(mae)})")

    # ---- PRIMARY (seed 42, kNN) + all pre-registered mirrors -------------------------------
    report_panel(main_panel, "PRIMARY seed-42 kNN")

    print("\nINT2-3 overshoot discriminator (secondary, kNN):")
    if "over" in excluded or "full" in excluded:
        print("  arm excluded — not computed")
    else:
        do = arm_deltas("over", main_panel)
        df_ = arm_deltas("full", main_panel)
        j = pd.concat([do.rename("o"), df_.rename("f")], axis=1, join="inner").dropna()
        if not coverage_ok(j.index, total=n_main):
            print("  insufficient coverage — NO VERDICT")
        else:
            sv = np.array([S[k] for k in j.index])
            rov = sv * (j.o.values - j.f.values)
            ci = block_ci(rov, np.array([k[2] for k in j.index]))
            tag = ("pre level special" if ci[1] < 0 else
                   "monotone continues past pre" if ci[0] > 0 else "undecided")
            print(f"  R_over {rov.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] -> {tag}")

    sig = prim[prim.encoder == "SigLIP"]
    if len(sig) and "full" not in excluded:
        A = a_of(arm_deltas("full", sig))
        n_sig = len(sig[KEY4].drop_duplicates())
        if coverage_ok(A.index, total=n_sig):
            ci = block_ci(A.values, np.array([k[2] for k in A.index]))
            print(f"\nSigLIP panel (separate, kNN): A {A.mean():+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] (n={len(A)})")
        else:
            print("\nSigLIP panel: NO VERDICT (coverage)")

    print("\nINT2-5 transplant diagnostic (secondary, kNN):")
    if "transplant" not in excluded:
        A = a_of(arm_deltas("transplant", main_panel))
        if coverage_ok(A.index, total=n_main):           # A: S!=0 cells
            ci = block_ci(A.values, np.array([k[2] for k in A.index]))
            print(f"  A_transplant {A.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]")
        else:
            print("  A: NO VERDICT (coverage)")
        run_g(main_panel, n_main, arm="transplant", label="  ")   # G: independent

    print("\nCapture-screen sensitivity (thr 0.03, pre-registered):")
    inc03, pooled03 = panel_scope(main_panel, thr=0.03, quiet=True)   # same thr
    if inc03:
        print(f"  scope@0.03 = {'+'.join(inc03)}")
        run_int21(pooled03, len(pooled03[KEY4].drop_duplicates()), thr=0.03,
                  label="  ")

    # LP mirrors NARROWED to INT2-1/2/4 main panels (prereg v1.2 round-9)
    report_panel(main_panel, "[proxy] LP seed-42", y="lp")

    print("\nCross-seed robustness (pre-registered replication of INT2-1/2/4):")
    for sd in ("43", "44"):
        pan = r[(r.seed == sd) & (r.encoder != "SigLIP")]
        if len(pan):
            report_panel(pan, f"seed-{sd} kNN")
            report_panel(pan, f"seed-{sd} LP", y="lp")
    print("\nDecision map: read against INT2_PREREG.md §6.")


if __name__ == "__main__":
    main()
