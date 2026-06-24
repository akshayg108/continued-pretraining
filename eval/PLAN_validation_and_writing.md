# CP Geometry Paper — Validation & Writing Plan

> **For workers:** steps use checkbox (`- [ ]`) syntax. This is a RESEARCH plan: TDD is
> adapted to "define a falsifiable check → run → compare to the predicted sign/threshold →
> save output to `eval/outputs/`". The repo is NOT a git repo, so "commit" = "save the
> output artifact". Heavy feature-extraction tasks run on Colab (GPU + `stable_datasets` +
> post-CP checkpoints); pure-`results.xlsx`/`geometry_15.csv` analyses run anywhere.

**Goal:** Validate the v2 hypothesis set (`hypothesis/HYPOTHESES_v2.md`) on the complete data + three
re-evaluation-only experiments, and draft the ICLR paper around the three findings.

**Architecture:** Phase 0 locks the current-data results (F1 table, F2.1 bivariate law). Phases
1–3 are the three small experiments (post-CP norm-CV, Selective-Aggregation LP, post-CP growth
sweep) — all re-evaluate EXISTING post-CP checkpoints, no retraining. Phase 4 writes the paper.

**Tech stack:** Python (pandas/numpy/scipy/scikit-learn for analysis; torch/timm/`stable_cp`/
`stable_datasets` for feature extraction). Analysis code lives in `eval/`. Predictions and their
falsification criteria come from `hypothesis/HYPOTHESES_v2.md`.

---

## File structure

- `eval/bivariate.py` — **create**. F2.1: position × baseline bivariate rank model, partial
  controls, DTD/FG-heterogeneity resolution. Runs on `geometry_15.csv` + `results.xlsx`.
- `eval/correlate.py` — **reuse as-is** for the F1 table (sections A/C/D already produced).
- `eval/postcp_features.py` — **create**. Load a post-CP checkpoint, extract `[cls]`/mean-pool
  features for a dataset; shared by Phases 1 & 3.
- `eval/postcp_normcv.py` — **create**. Phase 1 (Exp A): post-CP L2-norm CV per config.
- `eval/postcp_growth.py` — **create**. Phase 3 (Exp C): uniformity/overlap vs CP data size.
- Phase 2 (Exp B, SA-LP) uses the existing `stable_cp/evaluation` SA code + the `--aggregation`
  flag (already in `continued_pretraining.py` per `plan.md`); no new `eval/` file, only run + a
  small `eval/sa_lp_compare.py` to tabulate.
- `eval/outputs/` — all result CSVs/PNGs land here.
- Paper draft: `paper/` (sections per finding) — Phase 4.

---

## Phase 0 — Lock current-data results (runs anywhere; no checkpoints needed)

### Task 0.1: Publication F1 table from existing correlate.py
**Files:** Run `eval/correlate.py`; copy console to `eval/outputs/`.
- [ ] **Step 1:** Run `python eval/correlate.py --geometry eval/outputs/geometry_15.csv`.
  Expected: section A/C/D as in `eval/outputs/correlate_console_15.txt` (already saved).
- [ ] **Step 2:** Confirm the F1 falsification checks pass: sphere encoders uniformity→ΔkNN
  ρ≥0.52 (p<0.05) and overlap→ΔkNN<0 & →ΔFT>0; MAE sign-flips. (Confirmed: see console.)
- [ ] **Step 3 (save):** keep `eval/outputs/correlations_15.csv` as the F1 source table.

### Task 0.2: Bivariate law (F2.1) — position × baseline
**Files:** Create `eval/bivariate.py`; Test/output: `eval/outputs/bivariate.csv`.
- [ ] **Step 1: Write `eval/bivariate.py`** (complete code):

```python
#!/usr/bin/env python
"""F2.1: does (angular position, pre-CP baseline) predict ΔkNN better than either alone,
and does position survive controlling for baseline? Resolves DTD + FG-internal heterogeneity."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from load_results import load_long

GEOM = "outputs/geometry_15.csv"; INV = ["LeJEPA-CP", "SimCLR-CP"]

def partial(x, y, ctrl):
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(ctrl).reshape(-1, 1)
    xres = xr - LinearRegression().fit(cr, xr).predict(cr)
    yres = yr - LinearRegression().fit(cr, yr).predict(cr)
    r, p = spearmanr(xres, yres); return float(r), float(p)

def bivariate_r2(pos, base, y):
    """Out-of-sample-ish: rank-linear R^2 of y on {pos} vs {base} vs {pos,base}."""
    R = lambda X: LinearRegression().fit(X, rankdata(y)).score(X, rankdata(y))
    P, B = rankdata(pos).reshape(-1, 1), rankdata(base).reshape(-1, 1)
    return R(P), R(B), R(np.hstack([P, B]))

def main():
    g = pd.read_csv(GEOM); df = load_long()
    inv = df[df.Method.isin(INV) & df.is_max]
    agg = inv.groupby(["Backbone", "dataset_key"]).agg(
        dknn=("dknn", "mean"), knn_pre=("knn_pre", "mean")).reset_index()
    rows = []
    for enc in ["DINOv3", "CLIP", "MAE"]:
        m = g[(g.encoder == enc) & (g.dataset != "imagenet")].merge(
            agg[agg.Backbone == enc], left_on="dataset", right_on="dataset_key")
        for pos in ["uniformity_t2", "neighbor_overlap_k50"]:
            r_pos, _ = spearmanr(m[pos], m.dknn)
            r_base, _ = spearmanr(m.knn_pre, m.dknn)
            pr, pp = partial(m[pos].values, m.dknn.values, m.knn_pre.values)
            r2p, r2b, r2pb = bivariate_r2(m[pos].values, m.knn_pre.values, m.dknn.values)
            rows.append(dict(encoder=enc, position=pos, rho_pos=round(r_pos, 3),
                rho_baseline=round(r_base, 3), partial_pos_ctrl_base=round(pr, 3),
                p_partial=round(pp, 3), R2_pos=round(r2p, 3), R2_base=round(r2b, 3),
                R2_both=round(r2pb, 3)))
    out = pd.DataFrame(rows); out.to_csv("outputs/bivariate.csv", index=False)
    print(out.to_string(index=False))
    # DTD / FG-heterogeneity resolution
    print("\nDTD & FG-cluster pre-CP baseline (DINOv3):")
    d = agg[agg.Backbone == "DINOv3"].set_index("dataset_key")
    for k in ["dtd", "cub200", "cars196", "food101", "flowers102"]:
        print(f"  {k:12s} knn_pre={d.loc[k,'knn_pre']:.3f}  Δknn={d.loc[k,'dknn']:+.3f}")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run** `cd eval && python bivariate.py`. **DONE — result superseded the
  bivariate hope:** `partial(position|baseline)` survives (DINOv3 +0.62 p=.013, CLIP +0.78
  p=.001) → F1 robust, not gap effect; BUT `partial(baseline|position)` is n.s. (DINOv3 +0.04
  p=.88, CLIP +0.19 p=.51) and adjusted R² does not improve → **the position×baseline bivariate
  law does NOT hold**. Honest revision applied to F2 in `HYPOTHESES_v2.md` (exploitation = an
  encoder-level regime, not a per-dataset covariate; DTD/FG-heterogeneity = residual limits).
- [x] **Step 3: Falsification outcome.** The check fired: position survives (F1 ✓) but baseline
  adds nothing beyond position (bivariate ✗) → F2 reframed to "regime sets the sign", second
  axis demoted to future work.
- [x] **Step 4 (save):** `eval/outputs/bivariate.csv` written.

---

## Phase 1 — Exp A: post-CP L2-norm CV (F2.2). Re-evaluation only.

### Task 1.0: Verify post-CP checkpoints exist and confirm load API
**Files:** inspect the CP run outputs (Colab/Drive or cluster).
- [ ] **Step 1:** Locate saved post-CP checkpoints for at least these configs: `MAE-CP+DINOv3+
  food101+MAX`, `MAE-CP+CLIP+food101+MAX`, `LeJEPA-CP+DINOv3+food101+MAX` (control),
  `LeJEPA-CP+MAE+food101+MAX` (constructive control).
- [ ] **Step 2:** Confirm how to load one: a PyTorch-Lightning `.ckpt`; the backbone is a timm
  ViT-B. Record the exact `load_from_checkpoint` / `state_dict` key prefix for the backbone.
- [ ] **Step 3 (GATE):** If checkpoints were NOT saved, Exp A/B/C require re-running CP for the
  selected configs (heavy) — flag to the user and decide scope before proceeding.

### Task 1.1: Compute post-CP norm-CV and test the prediction
**Files:** Create `eval/postcp_features.py` (loader+extractor) and `eval/postcp_normcv.py`.
- [ ] **Step 1: Write `eval/postcp_features.py`** — a thin wrapper that, given a checkpoint
  path + backbone timm id + pool, returns features for a dataset (reuse the extract_features /
  loaders from `geometry_metrics.py`, but build the model from the checkpoint backbone instead
  of `timm.create_model(..., pretrained=True)`). Interface:
  `extract_postcp(ckpt_path, timm_id, pool, dataset_key, download_dir, processed_dir) -> np.ndarray`.
  (Load: `timm.create_model(timm_id, pretrained=False)`; load backbone weights from the ckpt
  state_dict — adapt the key prefix found in Task 1.0; then reuse `geometry_metrics.extract_features`.)
- [ ] **Step 2: Write `eval/postcp_normcv.py`** — for each config, extract post-CP features,
  compute `l2_norm_stats` (reuse from `geometry_metrics`), and also load the matching PRE-CP CV
  from `geometry_15.csv`. Output `eval/outputs/postcp_normcv.csv` with columns
  `config, encoder, method, dataset, pre_cv, post_cv, delta_cv`.
- [ ] **Step 3: Run** on the Task-1.0 configs.
  Expected (prediction P2.2): `delta_cv > 0` and large for **MAE-CP on DINOv3/CLIP** (features
  leave the sphere); `delta_cv ≈ 0` for **LeJEPA-CP** (invariance preserves the sphere).
- [ ] **Step 4: Falsification check.** If MAE-CP does NOT raise CV on DINOv3/CLIP, the
  "pushed off the sphere" mechanism of F2 is wrong → fall back to the aggregation-failure
  reading (Exp B) as the sole mechanism.
- [ ] **Step 5 (save):** `eval/outputs/postcp_normcv.csv` + a bar chart `eval/outputs/normcv.png`
  (pre vs post, grouped by config).

---

## Phase 2 — Exp B: Selective-Aggregation LP (F2.3). Re-evaluation only.

### Task 2.1: Run SA-LP on MAE-CP degraded checkpoints, compare to [cls] LP
**Files:** existing `stable_cp/evaluation` SA code (`--aggregation` flag per `plan.md`);
create `eval/sa_lp_compare.py` to tabulate.
- [ ] **Step 1:** Confirm the SA code path exists: `stable_cp/evaluation/zero_shot_eval.py`
  (`selective_aggregation_lp_evaluate`) + `abmilp.py` (ABMILPHead depth=1). If absent, port the
  minimal SA head from `findings/FINDINGS_v1.md` (the `SelectiveAggregationHead` snippet) + L2-norm + Linear.
- [ ] **Step 2:** For configs `MAE-CP+DINOv3+food101+MAX`, `MAE-CP+CLIP+food101+MAX`,
  `MAE-CP+DINOv3+galaxy10+MAX` (mild), and control `LeJEPA-CP+DINOv3+food101+MAX`, run post-CP
  eval with both standard `[cls]` LP and SA-LP. Record `pre_linear_f1`, `post_linear_f1`,
  `post_sa_lp_f1`.
- [ ] **Step 3: Write `eval/sa_lp_compare.py`** — tabulate the three numbers per config +
  compute `recovery = (post_sa_lp_f1 − post_linear_f1) / (pre_linear_f1 − post_linear_f1)`.
  Output `eval/outputs/sa_lp.csv`.
- [ ] **Step 4: Criterion (P2.3).** `recovery ≈ 1` (SA recovers to ~pre) → aggregation-failure;
  `recovery ≈ 0` (SA still low) → information-loss; in between → both, quantified by `recovery`.
  kNN (always `[cls]`) stays degraded as a control.
- [ ] **Step 5 (save):** `eval/outputs/sa_lp.csv`.

---

## Phase 3 — Exp C: post-CP geometry growth sweep (F3). Re-evaluation only.

### Task 3.1: Trace post-CP uniformity/overlap vs CP data size
**Files:** Create `eval/postcp_growth.py` (reuse `postcp_features.py` + `geometry_metrics`
metric fns). Needs post-CP checkpoints at multiple sizes.
- [ ] **Step 1:** Pick configs with a full size schedule and a known Δ-vs-size shape:
  `LeJEPA-CP+DINOv3+galaxy10` (growth), `LeJEPA-CP+DINOv3+octmnist` (inverted-U, peak n=1000),
  `LeJEPA-CP+DINOv3+organamnist` (zero-crossing). Sizes {100, 1000, 10000, MAX}.
- [ ] **Step 2: Write `eval/postcp_growth.py`** — for each (config, size): extract post-CP
  features, compute `wang_isola_uniformity` (t=2) and `neighbor_overlap` (k=50, needs ImageNet
  features — reuse the cached ImageNet extraction from geometry run). Output
  `eval/outputs/postcp_growth.csv` (config, size, post_uniformity, post_overlap) and overlay
  plots vs the existing Δ-vs-size curves.
- [ ] **Step 3: Criteria.** P3.1 post-uniformity decreases (spreads) with size; P3.2 the ΔkNN
  peak (octmnist@1000) precedes the post-overlap rise; P3.3 organamnist's ΔkNN zero-crossing
  coincides with post-overlap crossing a threshold.
- [ ] **Step 4: Falsification.** If post-overlap stays ≈0 at all sizes (cloud never collides),
  the "expansion-collision" story of F3 is wrong → demote F3 to a Discussion observation.
- [ ] **Step 5 (save):** `eval/outputs/postcp_growth.csv` + `eval/outputs/growth_*.png`.

---

## Phase 4 — Writing (ICLR; Cole-style three findings)

### Task 4.1: Section skeletons
**Files:** Create `paper/{1_intro,2_background,3_setup,4_finding1,5_finding2,6_finding3,7_guidelines,8_limitations}.md`.
- [ ] **Step 1:** Intro — the hook (same DINOv3 +0.35 vs −0.76), the headline law, 4 contribution
  bullets (benchmark; the regime-conditional law; three findings; guideline). Cite Sorkhei 2025
  as the foil, Kumar 2022 for the reversal.
- [ ] **Step 2:** Background — CP formalism; 4 SSL objectives' embedding assumptions; sphere
  background (Wang–Isola/KoLeo/nGPT/DINOv3); kNN/LP = sphere evals, FT = full-space (cite
  `papers/geometry/CORE_PAPERS.md`).
- [ ] **Step 3:** One section per finding, each a **bold falsifiable claim header** + the table/
  figure from `eval/outputs/`. F1: correlations_15 + the regime-flip. F2: bivariate.csv +
  normcv.png + sa_lp.csv. F3: postcp_growth.
- [ ] **Step 4:** Guidelines decision tree; Limitations (scope: no direct CP prior; n=15;
  anchors analogical; ETF not the target).
- [ ] **Step 5:** Related Work straight from `papers/geometry/CORE_PAPERS.md` groups A–F, respecting the
  two refutations (no "JE>recon on LP"; no ETF-as-target).

### Task 4.2: Wording-discipline pass
- [ ] **Step 1:** Grep the draft for overclaims: every geometric "fact" must trace to a theorem
  (Wang–Isola Prop 1, Gretton MMD) or be marked "our empirical claim"; every supervised/language
  anchor marked as mechanism/analogy; "ρ>0.83" must not appear (it's ρ≈0.6–0.75 now).

---

## Self-review (spec coverage)
- F1 → Tasks 0.1 (+ existing correlate.py). ✓
- F2.1 (bivariate) → Task 0.2. ✓  F2.2 (norm-CV) → Phase 1. ✓  F2.3 (SA-LP) → Phase 2. ✓
- F3 (growth) → Phase 3. ✓
- Headline/positioning/refutations → Phase 4 (4.1 step 1/5, 4.2). ✓
- Scope/checkpoint risk → Task 1.0 GATE. ✓
- "Good geometry = isotropic-Gaussian/uniform-sphere, not ETF" → Task 4.1 step 4/5. ✓

---

## Appendix: paper-planning reference (extracted from the retired `plan.md`)

**Target venue:** ICLR (the retired plan said NeurIPS — overridden).

**Candidate titles:**
1. *When Does Continued Pretraining Help? A Geometric Perspective* (recommended; closest to Cole 2022).
2. *The Geometry of Continued Pretraining: Position, Compatibility, and Dynamics.*
3. *Predicting Continued Pretraining Outcomes via Representation Geometry.*

**Section structure (Cole-style):**
1. Introduction · 2. Background & Related Work · 3. Benchmark Setup · 4. Finding 1 (Starting
Position) · 5. Finding 2 (Geometric Compatibility) · 6. Finding 3 (Data-Scale Dynamics) ·
7. Practical Guidelines · 8. Limitations & Open Problems.

**Cole-style narrative requirements:** title is a *decisional question*; each finding is **one
bold, quotable claim** used as a paragraph header; each experiment isolates one variable; no new
method is introduced — the paper analyses when existing methods work.

**Benchmark setup:**
- Methods (4 unsupervised CP objectives): LeJEPA-CP, SimCLR-CP, MAE-CP, DIET-CP. + FROM-SCRATCH
  (random init + LeJEPA) baseline.
- Encoders: DINOv3, CLIP, MAE — all ViT-B/16, 768-d.
- Eval: kNN (k=20, cosine, macro-F1), Linear Probe, Fine-tune.
- Data-size schedule: 100 / 500 / 1000 / 10000 / 25000 / MAX (original 8 datasets);
  new 7 datasets at MAX only (compute-saving "strategy A").

**The 15-dataset benchmark (9 OOD + 6 fine-grained):**

| Type | Dataset | ~Size | #Classes |
|---|---|---:|---:|
| OOD | BreastMNIST | 546 | 2 |
| OOD | DermaMNIST | ~7k | 7 |
| OOD | OCTMNIST | ~97k | 4 |
| OOD | OrganAMNIST | ~34k | 11 |
| OOD | PathMNIST | ~90k | 9 |
| OOD | Galaxy10 | ~17k | 10 |
| OOD | DTD | ~5.6k | 47 |
| OOD | EuroSAT | ~27k | 10 |
| OOD | PlantVillage | ~54k | 38 |
| FG | Food101 | ~75k | 101 |
| FG | FGVC_Aircraft | ~10k | 100 |
| FG | Cars196 | ~16k | 196 |
| FG | CUB200 | ~12k | 200 |
| FG | Flowers102 | ~8k | 102 |
| FG | Oxford-IIIT Pet | ~7.4k | 37 |

**Practical-guidelines decision flow (Section 7) — REVISED to the adjudicated findings (the old
MMD<0.1 thresholds were 8-dataset; use the regime-conditional law instead):**
- Step 1 — is the encoder sphere-native (L2-norm CV < 5%, e.g. DINOv3/CLIP) or off-sphere
  (MAE)? This sets the SIGN of every geometric prediction.
- Step 2 (sphere-native) — compute pre-CP angular position (uniformity / overlap with the
  pretraining set). Isolated/OOD (low uniformity-spread, overlap≈0) → expect kNN/LP ↑, FT ↓.
  Embedded/fine-grained (near-pretraining uniformity, overlap>0) → expect kNN/LP ↓, FT ↑.
- Step 2 (off-sphere MAE) → CP almost always raises kNN/LP (gap effect); avoid MAE-CP on sphere
  encoders (catastrophic, e.g. Food101 MAX ΔkNN ≈ −0.76).
- Step 3 — pick CP data scale by monitoring geometric growth (Finding 3), not only val accuracy.
