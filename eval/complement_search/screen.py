#!/usr/bin/env python
"""
screen.py — empirical screen of candidate pre-CP metrics that might COMPLEMENT the main
predictor uniformity_t2 (Wang-Isola uniformity, higher = more concentrated) of dknn@MAX.

Discovery encoders: DINOv3, CLIP (MAE reported alongside). Held-out: SigLIP.
Outcome (main):   mean over {LeJEPA-CP, SimCLR-CP, DIET-CP} at MAX of dknn (and dlp),
                  from eval/outputs/cp_long_refreshed.csv (is_max == True).
Outcome (SigLIP): mean over {LeJEPA-CP, SimCLR-CP} of (knn_f1_mean.1 - knn_f1_mean)
                  from results.xlsx sheet "By Method (SigLIP)" (header row 1).

Candidates: every numeric column of geometry_15.csv (imagenet row dropped) + a derived
log(neighbor_overlap_k50 + 1e-4), every numeric column of geometry_class_15.csv
(label-aware ones flagged), and the feature-derived metrics from feature_metrics.py.

Per (candidate, encoder), n = 15 datasets:
  pooled_rho            Spearman(candidate, dknn)
  pooled_rho_dlp        Spearman(candidate, dlp)
  within_OOD_rho / within_FG_rho   Spearman inside each dataset type (9 OOD / 6 FG)
  type_partial_rho      rank-residual partial Spearman controlling for the OOD/FG dummy
  partial_vs_unif_rho   rank-residual partial Spearman controlling for uniformity_t2
                        (rank both variables, regress each on rank(uniformity) + intercept,
                        Spearman of the residuals — same convention as
                        eval/F7_packing_exploratory/correlate_second_axis.partial_spearman)
  rho_with_unif         Spearman(candidate, uniformity_t2)

The bar (a candidate must pass ALL):
  (a) DINOv3 and CLIP: partial_vs_unif same sign and both |rho| >= 0.35
  (b) SigLIP: pooled_rho AND partial_vs_unif carry the discovery sign
  (c) not the OOD/FG label restated: |type_partial| >= 0.30 on >= 2 encoders with a
      consistent sign, OR within-OOD and within-FG Spearman share a sign on >= 2 encoders
      (the same sign across those encoders)
  (d) |Spearman(candidate, uniformity_t2)| < 0.8 on both discovery encoders

A permutation null (dataset identity of the candidate permuted jointly across the four
encoders, outcomes/uniformity/type fixed) gives the expected number of candidates that
pass the bar by chance given the number of candidates tried.

Usage: python eval/complement_search/screen.py [--n-perm 1000]
Outputs: eval/complement_search/outputs/screen_results.csv   (one row per candidate x encoder)
         eval/complement_search/outputs/bar_verdicts.csv     (one row per candidate)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
OUTDIR = ROOT / "eval/complement_search/outputs"
sys.path.insert(0, str(ROOT / "eval/F5_decision_score"))
from siglip_diet_protocol import DATASETS  # noqa: E402

TYPE = {d["key"]: d["type"] for d in DATASETS}
DISCOVERY = ["DINOv3", "CLIP"]
ENCODERS = ["DINOv3", "CLIP", "MAE", "SigLIP"]
METHODS3 = ["LeJEPA-CP", "SimCLR-CP", "DIET-CP"]
SIGLIP_METHODS = ["LeJEPA-CP", "SimCLR-CP"]
SIGLIP_NAME_MAP = {
    "BreastMNIST": "breastmnist", "DermaMNIST": "dermamnist", "OCTMNIST": "octmnist",
    "OrganAMNIST": "organamnist", "PathMNIST": "pathmnist", "Galaxy10": "galaxy10",
    "EuroSAT": "eurosat", "PlantVillage": "plant_village", "DTD": "dtd", "Food101": "food101",
    "FGVC_Aircraft": "fgvc_aircraft", "Cars196": "cars196", "CUB200": "cub200",
    "Flowers102": "flowers102", "OxfordPet": "oxford_pet"}

# (column, source, label_aware)
CANDIDATES = [
    # geometry_15.csv (label-free); n_samples is a dataset-size covariate, not geometry
    ("n_samples", "geometry_15", False),
    ("l2_norm_mean", "geometry_15", False), ("l2_norm_std", "geometry_15", False),
    ("l2_norm_cv", "geometry_15", False),
    ("uniformity_t2_raw", "geometry_15", False), ("uniformity_at_gamma", "geometry_15", False),
    ("cosine_dist_centroid", "geometry_15", False), ("mmd_rbf", "geometry_15", False),
    ("mmd_m_pp_target", "geometry_15", False), ("mmd_m_qq_imagenet", "geometry_15", False),
    ("mmd_m_pq_cross", "geometry_15", False), ("mmd_gamma", "geometry_15", False),
    ("neighbor_overlap_k20", "geometry_15", False), ("neighbor_overlap_k50", "geometry_15", False),
    ("log_overlap_k50", "geometry_15(derived)", False),
    # geometry_class_15.csv
    ("rankme", "geometry_class_15", False), ("rankme_raw", "geometry_class_15", False),
    ("numerical_rank", "geometry_class_15", False), ("alpha_req", "geometry_class_15", False),
    ("twonn_id", "geometry_class_15", False),
    ("n_classes", "geometry_class_15", True),
    ("within_spread", "geometry_class_15", True), ("between_spread", "geometry_class_15", True),
    ("wb_ratio", "geometry_class_15", True), ("nc1_ratio", "geometry_class_15", True),
    ("center_margin", "geometry_class_15", True), ("cdnv", "geometry_class_15", True),
    ("task_energy_top10", "geometry_class_15", True), ("task_energy_top50", "geometry_class_15", True),
    ("task_energy_in_top10", "geometry_class_15", True),
    ("task_energy_in_top50", "geometry_class_15", True),
    # feature_metrics.py (from int1_features bank_X, <=2000 subsample, L2-normalised)
    ("twonn_id_2k", "features", False), ("participation_ratio", "features", False),
    ("rankme_2k", "features", False), ("alpha_req_top100", "features", False),
    ("pc1_share", "features", False), ("pc5_share", "features", False),
    ("mean_pairwise_cos", "features", False), ("mean_cos_to_centroid", "features", False),
    ("hubness_skew_k10", "features", False),
    ("between_var_share", "features", True), ("fisher_ratio", "features", True),
    ("nc1_papyan", "features", True),
]


# ── data assembly ────────────────────────────────────────────
def outcomes_main():
    c = pd.read_csv(ROOT / "eval/outputs/cp_long_refreshed.csv")
    m = c[(c.is_max == True) & c.Method.isin(METHODS3) & c.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    counts = m.groupby(["Backbone", "dataset_key"]).size()
    assert len(counts) == 45 and counts.eq(3).all(), "expected exactly 3 objectives per (encoder, dataset) at MAX"
    o = m.groupby(["Backbone", "dataset_key"])[["dknn", "dlp"]].mean().reset_index()
    return o.rename(columns={"Backbone": "encoder", "dataset_key": "dataset"})


def outcomes_siglip():
    xl = pd.read_excel(ROOT / "results.xlsx", sheet_name="By Method (SigLIP)", header=1)
    xl = xl[xl.Method.isin(SIGLIP_METHODS)].copy()
    xl["dataset"] = xl.Dataset.map(SIGLIP_NAME_MAP)
    unmapped = xl[xl.dataset.isna()].Dataset.unique()
    assert len(unmapped) == 0, f"unmapped SigLIP dataset names: {unmapped}"
    xl["dknn"] = xl["knn_f1_mean.1"] - xl["knn_f1_mean"]
    xl["dlp"] = xl["linear_f1_mean.1"] - xl["linear_f1_mean"]
    assert xl.groupby("dataset").size().eq(2).all(), "expected 2 methods per SigLIP dataset"
    o = xl.groupby("dataset")[["dknn", "dlp"]].mean().reset_index()
    o.insert(0, "encoder", "SigLIP")
    return o


def wide_table():
    g = pd.read_csv(ROOT / "eval/outputs/geometry_15.csv")
    g = g[g.dataset != "imagenet"].copy()
    g["log_overlap_k50"] = np.log(g.neighbor_overlap_k50 + 1e-4)
    gc = pd.read_csv(ROOT / "eval/outputs/geometry_class_15.csv").drop(columns=["n_samples"])
    fm = pd.read_csv(OUTDIR / "feature_metrics.csv")
    out = pd.concat([outcomes_main(), outcomes_siglip()], ignore_index=True)
    w = (out.merge(g, on=["encoder", "dataset"], how="left")
            .merge(gc, on=["encoder", "dataset"], how="left")
            .merge(fm, on=["encoder", "dataset"], how="left"))
    w["type"] = w.dataset.map(TYPE)
    w["is_fg"] = (w.type == "FG").astype(float)
    assert len(w) == 60 and w.type.notna().all()
    return w.sort_values(["encoder", "dataset"]).reset_index(drop=True)


# ── statistics ───────────────────────────────────────────────
def _finite(*arrs):
    arrs = [np.asarray(a, float) for a in arrs]
    ok = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return [a[ok] for a in arrs]


def sp(x, y):
    x, y = _finite(x, y)
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    return float(r), float(p)


def partial_sp(x, y, z):
    """Rank-residual partial Spearman of x and y controlling for z."""
    x, y, z = _finite(x, y, z)
    if len(x) < 5 or x.std() == 0 or y.std() == 0 or z.std() == 0:
        return np.nan, np.nan
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(len(zr)), zr])
    resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    r, p = spearmanr(resid(xr), resid(yr))
    return float(r), float(p)


def screen_one(w, col, enc):
    s = w[w.encoder == enc]
    x = s[col].values
    ood, fg = s[s.type == "OOD"], s[s.type == "FG"]
    r, p = sp(x, s.dknn)
    pr, pp = partial_sp(x, s.dknn, s.uniformity_t2)
    return {
        "encoder": enc, "n": int(np.isfinite(x).sum()),
        "pooled_rho": r, "pooled_p": p,
        "pooled_rho_dlp": sp(x, s.dlp)[0],
        "within_OOD_rho": sp(ood[col], ood.dknn)[0],
        "within_FG_rho": sp(fg[col], fg.dknn)[0],
        "type_partial_rho": partial_sp(x, s.dknn, s.is_fg)[0],
        "partial_vs_unif_rho": pr, "partial_vs_unif_p": pp,
        "partial_vs_unif_rho_dlp": partial_sp(x, s.dlp, s.uniformity_t2)[0],
        "rho_with_unif": sp(x, s.uniformity_t2)[0],
    }


def screen_all(w, candidates):
    rows = []
    for col, source, label_aware in candidates:
        avail_siglip = bool(np.isfinite(w.loc[w.encoder == "SigLIP", col].astype(float)).sum() >= 10)
        for enc in ENCODERS:
            row = screen_one(w, col, enc)
            row.update(candidate=col, source=source, label_aware=label_aware,
                       available_on_siglip=avail_siglip)
            rows.append(row)
    cols = ["candidate", "source", "encoder", "pooled_rho", "pooled_p", "pooled_rho_dlp",
            "within_OOD_rho", "within_FG_rho", "type_partial_rho", "partial_vs_unif_rho",
            "partial_vs_unif_p", "partial_vs_unif_rho_dlp", "rho_with_unif", "n",
            "label_aware", "available_on_siglip"]
    return pd.DataFrame(rows)[cols]


# ── the bar ──────────────────────────────────────────────────
def _same_sign(a, b):
    return np.isfinite(a) and np.isfinite(b) and a != 0 and b != 0 and np.sign(a) == np.sign(b)


def evaluate_bar(res):
    """res: screen rows of ONE candidate indexed by encoder. Returns dict of criteria."""
    pD, pC = res.loc["DINOv3", "partial_vs_unif_rho"], res.loc["CLIP", "partial_vs_unif_rho"]
    a = _same_sign(pD, pC) and min(abs(pD), abs(pC)) >= 0.35
    ref = np.sign(np.nansum([pD, pC])) if np.isfinite(pD) or np.isfinite(pC) else np.nan
    min_partial = min(abs(pD), abs(pC)) if _same_sign(pD, pC) else 0.0
    # (b) held-out sign replication
    if bool(res.loc["SigLIP", "available_on_siglip"]):
        sP, sPart = res.loc["SigLIP", "pooled_rho"], res.loc["SigLIP", "partial_vs_unif_rho"]
        b = bool(np.isfinite(sP) and np.isfinite(sPart) and np.sign(sP) == ref and np.sign(sPart) == ref)
    else:
        b = False
    # (c) type-partial route
    tp = res["type_partial_rho"]
    strong = tp[(tp.abs() >= 0.30)]
    c_tp = len(strong) >= 2 and (np.sign(strong).nunique() == 1)
    c_tp_sign = float(np.sign(strong).iloc[0]) if c_tp else np.nan
    # (c) within-type route: OOD and FG share a sign, on >= 2 encoders, the same sign across them
    wo, wf = res["within_OOD_rho"], res["within_FG_rho"]
    shared = [float(np.sign(wo[e])) for e in ENCODERS
              if _same_sign(wo[e], wf[e])]
    c_wt = len(shared) >= 2 and len(set(shared)) == 1
    c_wt_sign = shared[0] if c_wt else np.nan
    c = bool(c_tp or c_wt)
    c_sign_matches = bool((c_tp and c_tp_sign == ref) or (c_wt and c_wt_sign == ref))
    # (d) non-redundancy with uniformity on the discovery encoders
    ru = res.loc[DISCOVERY, "rho_with_unif"].abs()
    d = bool(np.isfinite(ru).all() and (ru < 0.8).all())
    return {"a_partial_both": bool(a), "b_siglip_sign": b, "c_not_type": c,
            "c_type_partial_route": bool(c_tp), "c_within_type_route": bool(c_wt),
            "c_sign_matches_discovery": c_sign_matches, "d_not_redundant": d,
            "passes_bar": bool(a and b and c and d), "n_criteria": int(a) + int(b) + int(c) + int(d),
            "discovery_sign": ref, "min_discovery_partial": float(min_partial),
            "partial_DINOv3": pD, "partial_CLIP": pC,
            "partial_MAE": res.loc["MAE", "partial_vs_unif_rho"],
            "partial_SigLIP": res.loc["SigLIP", "partial_vs_unif_rho"],
            "pooled_SigLIP": res.loc["SigLIP", "pooled_rho"],
            "rho_unif_DINOv3": res.loc["DINOv3", "rho_with_unif"],
            "rho_unif_CLIP": res.loc["CLIP", "rho_with_unif"],
            "n_type_partial_ge_0.30": int(len(strong)),
            "n_enc_within_type_consistent": int(len(shared))}


def verdicts(results):
    rows = []
    for cand, res in results.groupby("candidate", sort=False):
        res = res.set_index("encoder")
        v = evaluate_bar(res)
        v.update(candidate=cand, source=res.source.iloc[0], label_aware=bool(res.label_aware.iloc[0]),
                 available_on_siglip=bool(res.available_on_siglip.iloc[0]))
        rows.append(v)
    v = pd.DataFrame(rows)
    v = v.sort_values(["passes_bar", "n_criteria", "min_discovery_partial"],
                      ascending=[False, False, False]).reset_index(drop=True)
    front = ["candidate", "source", "label_aware", "available_on_siglip", "passes_bar", "n_criteria",
             "a_partial_both", "b_siglip_sign", "c_not_type", "d_not_redundant"]
    return v[front + [c for c in v.columns if c not in front]]


# ── permutation null ─────────────────────────────────────────
def permutation_null(w, candidates, n_perm, seed=0):
    """Permute the dataset identity of each candidate jointly across the four encoders
    (outcomes, uniformity and type stay fixed); count candidates passing (a), (a)+(d),
    and the full bar per permutation."""
    rng = np.random.RandomState(seed)
    datasets = sorted(w.dataset.unique())
    blocks = {enc: (w.encoder == enc).values for enc in ENCODERS}
    cols = [c for c, _, _ in candidates]
    base = w[cols].astype(float).values.copy()
    # index arrays so that within each encoder block rows are in dataset order
    order = {enc: np.argsort(w.loc[blocks[enc], "dataset"].values) for enc in ENCODERS}
    counts = {"a": [], "ad": [], "full": []}
    for _ in range(n_perm):
        perm = rng.permutation(len(datasets))
        wp = w.copy()
        vals = base.copy()
        for enc in ENCODERS:
            idx = np.where(blocks[enc])[0][order[enc]]
            vals[idx] = base[idx][perm]
        wp[cols] = vals
        res = screen_all(wp, candidates)
        v = verdicts(res)
        counts["a"].append(int(v.a_partial_both.sum()))
        counts["ad"].append(int((v.a_partial_both & v.d_not_redundant).sum()))
        counts["full"].append(int(v.passes_bar.sum()))
    return {k: np.array(x) for k, x in counts.items()}


# ── main ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    w = wide_table()

    print("Reference: Spearman(uniformity_t2, dknn) per encoder (pooled / OOD / FG)")
    for enc in ENCODERS:
        s = w[w.encoder == enc]
        print(f"  {enc:7s} {sp(s.uniformity_t2, s.dknn)[0]:+.3f} / "
              f"{sp(s[s.type == 'OOD'].uniformity_t2, s[s.type == 'OOD'].dknn)[0]:+.3f} / "
              f"{sp(s[s.type == 'FG'].uniformity_t2, s[s.type == 'FG'].dknn)[0]:+.3f}   n={len(s)}")

    results = screen_all(w, CANDIDATES)
    # reference rows for the main predictor itself (partial vs itself is undefined -> NaN)
    ref_rows = []
    for enc in ENCODERS:
        r = screen_one(w, "uniformity_t2", enc)
        r.update(candidate="uniformity_t2 (reference)", source="geometry_15", label_aware=False,
                 available_on_siglip=True, partial_vs_unif_rho=np.nan, partial_vs_unif_p=np.nan,
                 partial_vs_unif_rho_dlp=np.nan)
        ref_rows.append(r)
    results_out = pd.concat([results, pd.DataFrame(ref_rows)[results.columns]], ignore_index=True)
    results_out.to_csv(OUTDIR / "screen_results.csv", index=False, float_format="%.4f")

    v = verdicts(results)
    v.to_csv(OUTDIR / "bar_verdicts.csv", index=False, float_format="%.4f")

    dup_note = ("rank-identical pairs (Spearman cannot separate them): "
                "log_overlap_k50 ~ neighbor_overlap_k50; fisher_ratio ~ between_var_share")
    print(f"\nCandidates screened: {len(CANDIDATES)} columns "
          f"({len(CANDIDATES) - 2} distinct rankings; {dup_note}); "
          f"{int(v.label_aware.sum())} label-aware; n = 15 datasets per encoder.")
    print(f"Pass the full bar: {int(v.passes_bar.sum())}")
    print("\nPer-candidate summary (partial vs uniformity: D=DINOv3 C=CLIP M=MAE S=SigLIP; "
          "criteria a b c d):")
    for _, r in v.iterrows():
        flag = "".join(k if r[c] else "-" for k, c in
                       zip("abcd", ["a_partial_both", "b_siglip_sign", "c_not_type", "d_not_redundant"]))
        print(f"  {r.candidate:22s} {flag}  partial D {r.partial_DINOv3:+.2f} C {r.partial_CLIP:+.2f} "
              f"M {r.partial_MAE:+.2f} S {r.partial_SigLIP:+.2f} | pooled S {r.pooled_SigLIP:+.2f} | "
              f"rho_unif D {r.rho_unif_DINOv3:+.2f} C {r.rho_unif_CLIP:+.2f} | "
              f"type-partial>=.30: {r['n_type_partial_ge_0.30']}  within-type consistent: "
              f"{r.n_enc_within_type_consistent}" + ("  [label-aware]" if r.label_aware else ""))

    print("\nTop 8 closest to the bar (sorted by criteria passed, then min discovery partial):")
    top = v.head(8)
    hdr = ("| candidate | label-aware | partial D | partial C | partial S | pooled S | "
           "type-partial D/C/M/S | rho_unif D/C | a | b | c | d |")
    print(hdr); print("|" + "---|" * 12)
    for _, r in top.iterrows():
        res = results[results.candidate == r.candidate].set_index("encoder")
        tp = "/".join(f"{res.loc[e, 'type_partial_rho']:+.2f}" for e in ENCODERS)
        print(f"| {r.candidate} | {'yes' if r.label_aware else 'no'} | {r.partial_DINOv3:+.2f} | "
              f"{r.partial_CLIP:+.2f} | {r.partial_SigLIP:+.2f} | {r.pooled_SigLIP:+.2f} | {tp} | "
              f"{r.rho_unif_DINOv3:+.2f}/{r.rho_unif_CLIP:+.2f} | "
              f"{'Y' if r.a_partial_both else 'n'} | {'Y' if r.b_siglip_sign else 'n'} | "
              f"{'Y' if r.c_not_type else 'n'} | {'Y' if r.d_not_redundant else 'n'} |")

    if args.n_perm > 0:
        null = permutation_null(w, CANDIDATES, args.n_perm)
        print(f"\nPermutation null ({args.n_perm} joint dataset permutations of every candidate; "
              f"outcomes/uniformity/type fixed) — candidates passing per permutation:")
        for k, label in [("a", "(a) alone"), ("ad", "(a)+(d)"), ("full", "full bar (a)+(b)+(c)+(d)")]:
            x = null[k]
            print(f"  {label:28s} mean {x.mean():.2f}  95th pct {np.percentile(x, 95):.0f}  "
                  f"P(>=1) = {(x >= 1).mean():.3f}")
        observed = int(v.passes_bar.sum())
        print(f"  observed full-bar passes: {observed}; "
              f"P_null(passes >= {max(observed, 1)}) = {(null['full'] >= max(observed, 1)).mean():.3f}")
    print(f"\nSaved {OUTDIR / 'screen_results.csv'} and {OUTDIR / 'bar_verdicts.csv'}")


if __name__ == "__main__":
    main()
