#!/usr/bin/env python3
"""
int1_verdict.py — INT1 (CPU adjudicator). Gates and readouts are FROZEN in
eval/new_direction/INT1_PREREG.md (consolidated operative rules v1.5, all
amendments pre-data); this file implements them one-to-one:

  G0     census: EXACT 4 x 15 cell set, unique keys, full mandatory grid per cell,
         ALL numeric metric columns finite (power_cp optional). Fail -> stop.
  G-NC   rotation negative control: JOINT per-cell |delta kNN| AND |delta LP|
         < 0.005 on >= 58/60 cells for both seeds. Fail -> harness bug, stop.
  G-P    contamination gate: transform excluded iff cross-family drift ratio
         R > 0.5 OR median |capture drift| > 0.05. Additionally a PER-CELL capture
         screen (v1.5): cells with |capture drift| > 0.05 leave that transform's
         effect estimates (count reported).
  G2-LP  LP proxy reproduction: identity-cell sklearn LP F1 vs the paper PyTorch
         lp_pre at the 45 MAX levels (is_max only, exact-count asserted). rho >=
         0.9 -> lp_connectable; otherwise every LP conclusion downstream carries
         the [proxy-internal] tag (explicit status, not just a printed note).
  T3     first-stage gate (v1.5): a grade enters the placement trend/decision only
         if its median |cC_K drift| >= 0.05 (it must actually move placement).
         Until INT1-2 passes with this gate the T3 family is described as
         "iso-spectral eigendirection-scale reassignment", not placement causality.
  INT1-1 spectrum: PRIMARY doses alpha {0.25, 2.0} at Bonferroni 97.5% block CIs.
  INT1-2 placement: decided at the deepest non-excluded first-stage-passing grade;
         per-readout demote monotonicity.
  INT1-3 operator specificity: family-pooled contrast at Bonferroni 98.33% CIs
         (3 families, family-wise 5%); per-dose panel descriptive (95%).
  INT1-4 CP reconstruction: rho over non-excluded power_cp cells; reported, no
         threshold.
  INT1-5 interaction: per readout, the 2 combo doses at Bonferroni 97.5% CIs;
         interaction = significant at either dose (per readout).

Run (local): python eval/new_direction/int1_verdict.py
"""
import sys
from pathlib import Path as _P

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(_P(__file__).resolve().parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "utils"))
from load_results import load_long

ROOT = _P(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"

PRIMARY_ALPHAS = ("0.25", "2.0")               # frozen primary doses, Bonferroni 2
COMBO_PARAMS = ("0.5|256", "2.0|256")
EXPECTED_GRID = ([("identity", "")] + [("rotation", s) for s in ("0", "1")]
                 + [("power", a) for a in ("0.25", "0.5", "0.75", "1.5", "2.0")]
                 + [("demote", d) for d in ("64", "256", "512")] + [("shuffle", "0")]
                 + [("combo", c) for c in COMBO_PARAMS])
ENCODER_NAMES = ("DINOv3", "MAE", "CLIP", "SigLIP")
DATASET_NAMES = ("breastmnist", "dermamnist", "octmnist", "organamnist", "pathmnist",
                 "galaxy10", "eurosat", "plant_village", "dtd", "food101",
                 "fgvc_aircraft", "cars196", "cub200", "flowers102", "oxford_pet")
EXPECTED_CELLS = tuple((e, d) for e in ENCODER_NAMES for d in DATASET_NAMES)
METRIC_COLS = ("knn_f1", "lp_f1", "rankme_raw_bank", "rankme_l2_bank",
               "capture_l2", "cC_K_l2", "query_oos_frac")
CAP_THR = 0.05                                 # v1.5 per-cell capture screen
FS_THR = 0.05                                  # v1.5 T3 first-stage floor


def block_ci(values, datasets, stat=np.mean, n=2000, seed=0, level=95.0):
    """Dataset-block bootstrap CI over per-cell values."""
    rng = np.random.RandomState(seed)
    uds = np.unique(datasets)
    stats = []
    for _ in range(n):
        pick = rng.choice(uds, len(uds), replace=True)
        idx = np.concatenate([np.flatnonzero(datasets == d) for d in pick])
        stats.append(stat(values[idx]))
    half = (100.0 - level) / 2.0
    return float(np.percentile(stats, half)), float(np.percentile(stats, 100.0 - half))


def gnc_joint(dk, dl, tol=0.005):
    """Count cells where BOTH readout deltas are below tol (joint per cell — the
    prereg's 'and' is per cell, not two marginal counts)."""
    j = pd.concat([dk.rename("k"), dl.rename("l")], axis=1, join="inner")
    return int(((j.k.abs() < tol) & (j.l.abs() < tol)).sum())


def placement_flags(trend):
    """INT1-2 decision from trend rows (name, param, mean_dk, mean_dl, ci_k, ci_l):
    decided at the last (deepest non-excluded) row; demote monotonicity evaluated
    SEPARATELY per readout (LP must not ride kNN's flag)."""
    dem = [t for t in trend if t[0] == "demote"]
    mono_k = all(abs(dem[i][2]) <= abs(dem[i + 1][2]) + 1e-9 for i in range(len(dem) - 1))
    mono_l = all(abs(dem[i][3]) <= abs(dem[i + 1][3]) + 1e-9 for i in range(len(dem) - 1))
    name, param, mk, ml, ck, cl = trend[-1]
    return dict(decided_at=(name, param), mono_k=mono_k, mono_l=mono_l,
                pk=(ck[0] > 0 or ck[1] < 0) and mono_k,
                pl=(cl[0] > 0 or cl[1] < 0) and mono_l)


def g0_gate(r, cells=None):
    """G0 census (v1.5 strict): unique keys; the EXACT expected cell set (default
    the real 4 x 15 grid) — membership, not just count; full mandatory grid per
    cell; every numeric metric column finite. power_cp is optional per cell."""
    cells = tuple(cells) if cells is not None else EXPECTED_CELLS
    msgs = []
    dup = r.duplicated(["encoder", "dataset", "transform", "param"])
    if dup.any():
        msgs.append("duplicate keys: "
                    f"{r[dup][['encoder', 'dataset', 'transform', 'param']].values.tolist()}")
    got = set(map(tuple, r[["encoder", "dataset"]].drop_duplicates().values))
    want = set(cells)
    if got != want:
        miss, extra = sorted(want - got), sorted(got - want)
        msgs.append(f"cell set mismatch: missing {miss}, unexpected {extra}")
    have = set(map(tuple, r[["encoder", "dataset", "transform", "param"]].values))
    for e, d in sorted(got & want):
        miss = [tp for tp in EXPECTED_GRID if (e, d) + tp not in have]
        if miss:
            msgs.append(f"{e}__{d} missing {miss}")
    for col in METRIC_COLS:
        if col not in r.columns:
            msgs.append(f"metric column absent: {col}")
        elif not np.isfinite(pd.to_numeric(r[col], errors="coerce")).all():
            msgs.append(f"non-finite values in metric column: {col}")
    return (not msgs), msgs


def drop_excluded(r, excluded):
    """Remove rows whose (transform, param) was excluded by G-P."""
    if not len(r):
        return r
    mask = np.array([(t, p) in excluded
                     for t, p in zip(r["transform"], r["param"])])
    return r[~mask]


def screened_deltas(sub, ident, thr=CAP_THR):
    """Per-cell deltas vs identity with the v1.5 capture screen: cells whose
    |capture_l2 drift| exceeds thr leave this transform's effect estimate.
    thr=None disables the screen (G-NC uses the raw negative control)."""
    j = sub.set_index(["encoder", "dataset"])
    dk = (j.knn_f1 - ident.knn_f1).dropna()
    dl = (j.lp_f1 - ident.lp_f1).dropna()
    dropped = []
    if thr is not None and "capture_l2" in j.columns:
        capd = (j.capture_l2 - ident.capture_l2).abs()
        dropped = sorted(capd[capd > thr].index)
        if dropped:
            dk = dk.drop(dropped, errors="ignore")
            dl = dl.drop(dropped, errors="ignore")
    return dk, dl, np.array([i[1] for i in dk.index]), dropped


def first_stage_move(sub, ident):
    """v1.5 T3 first-stage relevance: median |cC_K_l2 drift| of this grade."""
    j = sub.set_index(["encoder", "dataset"])
    return float((j.cC_K_l2 - ident.cC_K_l2).dropna().abs().median())


def main():
    path = OUT / "int1_results.csv"
    if not path.exists():
        sys.exit(f"MISSING {path} — run int1_run.py first")
    r = pd.read_csv(path, dtype={"param": str}).fillna({"param": ""})

    # ---- G0 census -------------------------------------------------------------------------
    ok0, msgs = g0_gate(r)
    print(f"G0 census: {'PASS' if ok0 else 'FAIL'}")
    for m in msgs:
        print(f"  {m}")
    if not ok0:
        sys.exit(1)

    # query_oos_frac is a DIAGNOSTIC (identity-on-complement construction)
    oos = r[r["transform"] == "identity"][["encoder", "dataset", "query_oos_frac"]]
    hi = oos[oos.query_oos_frac > 0.05]
    if len(hi):
        print(f"note: {len(hi)} cells with query_oos_frac > 0.05 (surgery acts only on "
              f"the in-span part — dose diluted, not annihilated): "
              f"{[f'{e}__{d}' for e, d in hi[['encoder', 'dataset']].values]}")
    ident = r[r["transform"] == "identity"].set_index(["encoder", "dataset"])
    print(f"cells: {len(ident)} — transforms per cell: "
          f"{r.groupby(['encoder', 'dataset']).size().min()}"
          f"..{r.groupby(['encoder', 'dataset']).size().max()}")

    def deltas(sub, thr=CAP_THR, label=None):
        dk, dl, ds_, dropped = screened_deltas(sub, ident, thr=thr)
        if dropped:
            print(f"    capture screen: {len(dropped)} cells left "
                  f"{label or 'this readout'}: {[f'{e}__{d}' for e, d in dropped]}")
        return dk, dl, ds_

    # ---- G-NC rotation control (joint per cell; unscreened negative control) ---------------
    ok_nc = True
    for s in ("0", "1"):
        dk, dl, _ = deltas(r[(r["transform"] == "rotation") & (r.param == s)], thr=None)
        nj = gnc_joint(dk, dl, tol=0.005)
        print(f"G-NC rotation seed {s}: joint |dkNN|&|dLP| < 0.005 on {nj}/{len(dk)} "
              f"(max {dk.abs().max():.4f}/{dl.abs().max():.4f})")
        ok_nc &= nj >= 58
    print(f"  -> {'PASS' if ok_nc else 'FAIL — harness bug, STOP'}")
    if not ok_nc:
        sys.exit(1)

    # ---- G-P contamination gate ------------------------------------------------------------
    excluded = set()
    print("\nG-P contamination gate (excluded iff cross-family R > 0.5 OR "
          "median |capture drift| > 0.05):")

    def med_move(fam_params, col, log=False):
        vals = []
        for fam, param in fam_params:
            sub = r[(r["transform"] == fam) & (r.param == param)]
            j = sub.set_index(["encoder", "dataset"])
            d = (np.log(j[col]) - np.log(ident[col])).dropna() if log \
                else (j[col] - ident[col]).dropna()
            vals.append(d.abs().median())
        return float(np.median(vals)) if vals else np.nan

    move_place = med_move([("shuffle", "0")], "cC_K_l2")
    move_spec = med_move([("power", "0.25"), ("power", "2.0")], "rankme_l2_bank", log=True)
    print(f"  intended movements: placement (shuffle) median |d cC_K| = {move_place:.4f}; "
          f"spectrum (alpha 0.25/2.0) median |d log rankme| = {move_spec:.4f}")
    for fam, params in [("power", ["0.25", "0.5", "0.75", "1.5", "2.0"]),
                        ("power_cp", sorted(r[r["transform"] == "power_cp"].param.unique())),
                        ("demote", ["64", "256", "512"]), ("shuffle", ["0"]),
                        ("combo", list(COMBO_PARAMS))]:
        for param in params:
            sub = r[(r["transform"] == fam) & (r.param == param)]
            if not len(sub):
                continue
            j = sub.set_index(["encoder", "dataset"])
            cap_d = (j.capture_l2 - ident.capture_l2).dropna().abs().median()
            if fam == "combo":
                # combo INTENDS both axes — only the capture stop applies
                R, lbl = 0.0, "moves both axes by design"
            elif fam.startswith("power"):
                unint = (j.cC_K_l2 - ident.cC_K_l2).dropna().abs().median()
                R = unint / max(move_place, 1e-12)
                lbl = f"cC_K drift {unint:.4f} / placement move {move_place:.4f}"
            else:
                unint = (np.log(j.rankme_l2_bank) - np.log(ident.rankme_l2_bank)
                         ).dropna().abs().median()
                R = unint / max(move_spec, 1e-12)
                lbl = f"log-rankme drift {unint:.4f} / spectrum move {move_spec:.4f}"
            viol = bool(R > 0.5) or bool(cap_d > CAP_THR)
            if viol:
                excluded.add((fam, param))
            why = (" R>0.5" if R > 0.5 else "") + (" capture>0.05" if cap_d > CAP_THR else "")
            print(f"  {fam:>9}[{param}]: R = {R:.2f} ({lbl}; capture drift {cap_d:.4f}) "
                  f"-> {'EXCLUDED' + why if viol else 'ok'}")

    # ---- G2-LP proxy reproduction (45 MAX levels, is_max only) -----------------------------
    lp_connectable = False
    try:
        df = load_long()
        sub45 = df[df.is_max & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
        lp_levels = sub45.groupby(["Backbone", "dataset_key"]).lp_pre.mean().dropna()
        assert len(lp_levels) == 45, \
            f"G2-LP expects exactly 45 MAX (Backbone, dataset) levels, got {len(lp_levels)}"
        pairs = [(ident.lp_f1.get((e, d)), v) for (e, d), v in lp_levels.items()
                 if (e, d) in ident.index]
        pairs = [(a, b) for a, b in pairs if a is not None and np.isfinite(b)]
        rho_lp = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation
        lp_connectable = bool(rho_lp >= 0.9)
        tag = "connectable (ordering-level)" if lp_connectable else \
            "PROXY-INTERNAL ONLY — paper-LP claims not licensed"
        print(f"\nG2-LP proxy reproduction (n={len(pairs)} MAX-level cells): "
              f"rho(sklearn LP identity F1, paper PyTorch lp_pre @ MAX) = {rho_lp:+.3f} "
              f"-> {tag}")
        print("  (paper LP = unregularized 10k-step Adam probe, "
              "continued_pretraining results_json 'linear_pytorch_f1'; INT1 LP = "
              "deterministic sklearn C=1.0 probe. SigLIP cells have no paper LP.)")
    except Exception as e:
        print(f"\nG2-LP proxy reproduction SKIPPED ({e.__class__.__name__}: {e}) — "
              "LP readouts default to proxy-internal")
    LP_TAG = "" if lp_connectable else " [proxy-internal]"

    # ---- INT1-1 spectrum effect ------------------------------------------------------------
    print("\nINT1-1 spectrum effect (mean delta vs identity, dataset-block bootstrap):")
    any_sig = {"knn": False, "lp": False}
    for a in ("0.25", "0.5", "0.75", "1.5", "2.0"):
        if ("power", a) in excluded:
            print(f"  alpha={a:>4}: G-P-excluded")
            continue
        primary = a in PRIMARY_ALPHAS
        level = 97.5 if primary else 95.0     # Bonferroni over the two primary doses
        dk, dl, ds_ = deltas(r[(r["transform"] == "power") & (r.param == a)],
                             label=f"power[{a}]")
        ck = block_ci(dk.values, ds_, level=level)
        cl = block_ci(dl.values, ds_, level=level)
        sk, sl = (ck[0] > 0 or ck[1] < 0), (cl[0] > 0 or cl[1] < 0)
        if primary:
            any_sig["knn"] |= sk
            any_sig["lp"] |= sl
        print(f"  alpha={a:>4}{' P' if primary else '  '}: "
              f"dkNN {dk.mean():+.4f} CI{level:g}[{ck[0]:+.4f},{ck[1]:+.4f}]"
              f"{' *' if sk else '  '}  dLP {dl.mean():+.4f} "
              f"CI{level:g}[{cl[0]:+.4f},{cl[1]:+.4f}]{' *' if sl else ''}")
    print("  -> functional spectrum effect (PRIMARY doses only, family-wise 5%): "
          f"kNN {'YES' if any_sig['knn'] else 'no'}, "
          f"LP {'YES' if any_sig['lp'] else 'no'}{LP_TAG}"
          "  (non-primary alphas are the descriptive dose-response panel)")

    # ---- INT1-2 placement effect (T3 = iso-spectral eigendirection-scale reassignment) -----
    print("\nINT1-2 T3 effect (iso-spectral eigendirection-scale reassignment; decided at "
          "deepest non-excluded first-stage-passing grade):")
    trend = []
    for name, param in [("demote", "64"), ("demote", "256"), ("demote", "512"),
                        ("shuffle", "0")]:
        if (name, param) in excluded:
            print(f"  {name:>7}[{param:>3}]: G-P-excluded")
            continue
        sub = r[(r["transform"] == name) & (r.param == param)]
        fs = first_stage_move(sub, ident)
        if fs < FS_THR:
            print(f"  {name:>7}[{param:>3}]: first-stage median |d cC_K| = {fs:.4f} "
                  f"< {FS_THR} — no placement movement, dropped from trend/decision")
            continue
        dk, dl, ds_ = deltas(sub, label=f"{name}[{param}]")
        ck, cl = block_ci(dk.values, ds_), block_ci(dl.values, ds_)
        trend.append((name, param, dk.mean(), dl.mean(), ck, cl))
        print(f"  {name:>7}[{param:>3}]: fs {fs:.3f}  dkNN {dk.mean():+.4f} "
              f"CI[{ck[0]:+.4f},{ck[1]:+.4f}]  "
              f"dLP {dl.mean():+.4f} CI[{cl[0]:+.4f},{cl[1]:+.4f}]")
    if trend:
        f = placement_flags(trend)
        print(f"  -> decided at {f['decided_at'][0]}[{f['decided_at'][1]}]; "
              f"|dkNN| demote trend monotone: {f['mono_k']}, |dLP|: {f['mono_l']}; "
              f"functional placement effect: kNN {'YES' if f['pk'] else 'no'}, "
              f"LP {'YES' if f['pl'] else 'no'}{LP_TAG}")
    else:
        print("  -> no T3 grade passes G-P + first-stage — no placement verdict "
              "(pre-declared branch)")

    # ---- INT1-3 operator specificity -------------------------------------------------------
    print("\nINT1-3 operator specificity (paired dkNN - dLP; family-pooled CI at "
          f"Bonferroni 98.33% decides{LP_TAG}; per-dose panel descriptive):")
    for fam, params in [("power", ["0.25", "0.5", "0.75", "1.5", "2.0"]),
                        ("demote", ["64", "256", "512"]), ("shuffle", ["0"])]:
        keep = [p for p in params if (fam, p) not in excluded]
        for p in keep:                                     # per-dose panel
            sub = r[(r["transform"] == fam) & (r.param == p)]
            dk, dl, ds_ = deltas(sub, label=f"{fam}[{p}] contrast")
            diff = (dk - dl).dropna()
            ds_ = np.array([i[1] for i in diff.index])
            ci = block_ci(diff.values, ds_)
            print(f"    {fam}[{p}]: mean {diff.mean():+.4f} CI95[{ci[0]:+.4f},{ci[1]:+.4f}]")
        rows = r[(r["transform"] == fam) & r.param.isin(keep)]
        if not len(rows):
            continue
        dk, dl, _ = deltas(rows, label=f"{fam} pooled contrast")
        diff = (dk - dl).dropna()
        ds_ = np.array([i[1] for i in diff.index])
        ci = block_ci(diff.values, ds_, level=100.0 - 5.0 / 3.0)
        sig = ci[0] > 0 or ci[1] < 0
        print(f"  {fam:>7} POOLED: mean {diff.mean():+.4f} "
              f"CI98.33[{ci[0]:+.4f},{ci[1]:+.4f}]"
              f"{' * operator-specific' + LP_TAG if sig else ''}")

    # ---- INT1-4 CP reconstruction ----------------------------------------------------------
    sub = drop_excluded(r[r["transform"] == "power_cp"], excluded)
    n_dropped = len(r[r["transform"] == "power_cp"]) - len(sub)
    if n_dropped:
        print(f"\nINT1-4: {n_dropped} G-P-excluded power_cp cells dropped")
    if len(sub):
        dk, _, _ = deltas(sub, label="power_cp")
        df = load_long()
        beh = (df[df.Method.str.contains("LeJEPA|SimCLR", na=False) & df.is_max
                  & df.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
               .groupby(["Backbone", "dataset_key"]).dknn.mean())
        real = {(e, d): v for (e, d), v in beh.items()}
        c2_path = OUT / "c2_siglip_score.csv"
        if c2_path.exists():
            for _, row in pd.read_csv(c2_path).iterrows():
                real[("SigLIP", row.dataset)] = row.real_dknn
        else:
            print("  NOTE: c2_siglip_score.csv absent — SigLIP cells skipped in INT1-4")
        pairs = [(dk[key], real[key]) for key in dk.index if key in real]
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        rho = spearmanr(a, b).correlation
        keys = [k for k in dk.index if k in real]
        dsx = np.array([k[1] for k in keys])
        rng = np.random.RandomState(0)
        uds = np.unique(dsx)
        boots = []
        for _ in range(2000):
            pick = rng.choice(uds, len(uds), replace=True)
            idx = np.concatenate([np.flatnonzero(dsx == d_) for d_ in pick])
            if len(np.unique(b[idx])) > 1:
                boots.append(spearmanr(a[idx], b[idx]).correlation)
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        print(f"\nINT1-4 CP reconstruction (n={len(pairs)}): "
              f"rho(surgical dknn @ alpha_cp, realized dknn) = {rho:+.3f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"(reported; no threshold — near 0 means the spectral motion alone does "
              f"not reproduce the realized CP pattern)")

    # ---- INT1-5 interaction ----------------------------------------------------------------
    print("\nINT1-5 interaction (delta combo - delta power - delta demote[256]; per "
          "readout, Bonferroni 97.5% CIs over the 2 combo doses):")
    int5_sig = {"kNN": False, "LP": False}
    for cp in COMBO_PARAMS:
        alpha = cp.split("|")[0]
        needed = [("combo", cp), ("power", alpha), ("demote", "256")]
        if any(t in excluded for t in needed):
            print(f"  combo[{cp}]: component excluded by G-P — contrast not computed")
            continue
        parts = {}
        for t, p in needed:
            dk, dl, _ = deltas(r[(r["transform"] == t) & (r.param == p)],
                               label=f"{t}[{p}] (INT1-5)")
            parts[t] = (dk, dl)
        for lab, i in (("kNN", 0), ("LP", 1)):
            contrast = (parts["combo"][i] - parts["power"][i] - parts["demote"][i]).dropna()
            ds_ = np.array([ix[1] for ix in contrast.index])
            ci = block_ci(contrast.values, ds_, level=97.5)
            sig = ci[0] > 0 or ci[1] < 0
            int5_sig[lab] |= sig
            print(f"  combo[{cp}] {lab}: mean {contrast.mean():+.4f} "
                  f"CI97.5[{ci[0]:+.4f},{ci[1]:+.4f}]{' *' if sig else ''}")
    print(f"  -> interaction: kNN {'YES' if int5_sig['kNN'] else 'no'}, "
          f"LP {'YES' if int5_sig['LP'] else 'no'}{LP_TAG}")

    print(f"\nLP scope for ALL LP conclusions above: "
          f"{'paper-connectable (G2-LP >= 0.9)' if lp_connectable else 'proxy-internal (G2-LP < 0.9 or unavailable)'}")
    print("Decision tree: read the flags above against INT1_PREREG.md §Decision tree.")


if __name__ == "__main__":
    main()
