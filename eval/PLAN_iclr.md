# CP Geometry Paper — Master Plan (consolidated)

> Single source of truth. Supersedes the retired `PLAN_texture_regime_extension.md` and
> `PLAN_validation_and_writing.md` (folded in below; their full step/loader detail is in git history).
> Spec: `hypothesis/HYPOTHESES_v3.md`. Findings: `findings/FINDINGS_step{2,4,5_siglip,6_sa_lp}.md`.
> Heavy GPU tasks run on the cluster; analysis tasks run locally (`python3` has sklearn/scipy/openpyxl).

## Status — DONE (the core paper is experimentally complete)

- **Main grid** — DINOv3/CLIP/MAE × {LeJEPA, SimCLR, DIET, MAE}-CP × 15 datasets + FROM-SCRATCH baseline. ✓
- **C1 regime-gating** — sphere (DINOv3/CLIP/SigLIP) vs off-sphere MAE *sign-flip* (contests Sorkhei 2025;
  genuine sign flip, not magnitude). ✓ (FINDINGS_step2 / step4)
- **C2 decision rule** — pre-CP geometry → sign(Δ); validated leave-one-dataset-out AND cross-encoder on
  the held-out 4th encoder **SigLIP-2: sign(ΔkNN) 87 %, ΔLP 80 %** (always-help baseline 73 %). Per-encoder
  standardization is the transfer mechanism (LOO 0.667 → 0.778). The ΔFT-reversal does NOT transfer
  cross-encoder (disclosed). ✓ (FINDINGS_step5_siglip; `predictor.csv`, `c2_siglip_score.csv`,
  `predictor_standardize_ablation.csv`, `preregister_siglip.csv`)
- **C3 two-force mechanism** — spread (Δuniformity, helps frozen) + collision (Δoverlap, hurts);
  off-sphere/norm-CV alternative REFUTED. BH-FDR: spread robust on both sphere encoders, collision robust
  on CLIP but marginal on DINOv3 (disclose). ✓ (`forces_combined.csv`, `postcp_offsphere.csv`, `stats_pass.csv`)
- **F2 mechanism / Exp B** — MAE-CP's catastrophic frozen-feature degradation is mostly an **aggregation
  failure**: a learned Selective-Aggregation pool recovers **~61 %** of the readout collapse (residual
  info-loss only at the tail). ✓ (FINDINGS_step6_sa_lp; `run_exp_b.py` → `eval/outputs/exp_b/*`,
  `sa_lp_recovery.csv`)
- **F3 growth** — post-CP geometry vs CP data size. ✓ (`postcp_growth_analysis.csv`)

## Remaining

### R1 — Exp D: texture sub-regime (the ONLY remaining experiment)
Turn the DTD texture exception (predicted-HURT-but-HELPs, reproduced on the SigLIP C2 scoring) into a
*characterized boundary*, or confirm it is a lone oddity. Add 2 texture/material datasets whose TASK is
orthogonal to object identity but whose IMAGES embed near ImageNet.
- [ ] Datasets: **KTH-TIPS-2b** (11 materials, ~4752) + **FMD** (10 materials, 1000); folder-of-images-per-class.
- [ ] Loaders: `stable-datasets/stable_datasets/images/{kth_tips2b,fmd}.py` (copy the DTD template); export +
  register in `stable_cp/data/datasets.py`, `eval/geometry_metrics.py` DS_REGISTRY, `eval/load_results.py`.
- [ ] Pre-CP geometry (cheap, no training): DINOv3+CLIP on the 2 new datasets → extend `geometry_15.csv` →
  `geometry_17.csv`. **Premise guard**: confirm they are *embedded* (overlap ~ DTD/food101, not ≈0); else re-pick.
- [ ] Minimal CP grid (cluster): `{KTH-TIPS-2b, FMD} × {DINOv3, CLIP} × {LeJEPA, SimCLR} × MAX × 3 seeds`
  ≈ 24–72 short runs (sphere encoders + invariance methods only — the regime where F1 holds and DTD breaks it).
- [ ] Adjudicate: `eval/texture_regime_test.py` — fit Δknn ~ uniformity on the original 15, project the new
  datasets; PREDICTION: positive high-rank residual on BOTH DINOv3 and CLIP (DTD-like) → SUPPORTED (secondary
  finding); on-line/negative → REFUTED (DTD a lone characterized exception). Either way closes the caveat.

### R2 — Small analysis TODOs (no GPU)
- [ ] Extend `eval/load_results.load_long()` to include the `By Method (SigLIP)` CP sheet so `predictor.py`
  auto-reproduces the cross-encoder 87 % (currently scored manually in `c2_siglip_score.csv`).
- [ ] Make per-encoder-standardize the DEFAULT predictor variant (better AND consistent with the cross-encoder
  pre-registration); keep global-standardize as the reported ablation.

### R3 — Paper assembly
- [ ] Figures: (1) regime-gating sphere-vs-MAE sign-flip; (2) decision-rule out-of-sample ROC vs baselines +
  the SigLIP cross-encoder 87 %; (3) two-force mechanism (spread vs collision); (4) growth/transport;
  (5) SA-recovery (Exp B aggregation-failure).
- [ ] Stats: foreground OUT-OF-SAMPLE metrics (LOO bal-acc, cross-encoder 87 %) over in-sample ρ; report the
  BH-FDR q-values (`stats_pass.csv`); disclose the caveats — FT-non-transfer, DINOv3-collision-marginal,
  Exp-B-residual-loss, n=15.
- [ ] Sections: Intro (C1–C3) / Related (Sorkhei foil, Kumar 2022 LP-FT, **Beyond-[CLS]** Przewiezlikowski 2024
  for the SA aggregation mechanism) / Setup / one-bold-claim-header per finding / Guidelines decision-tree /
  Limitations.
- [ ] Wording discipline: every geometric "fact" traces to a theorem (Wang–Isola Prop 1, Gretton MMD) or is
  marked "our empirical claim"; "ρ>0.83" must NOT appear (the headline is the out-of-sample numbers now).

## Paper-planning reference

**Title (recommended):** *When Does Continued Pretraining Help? A Geometric Perspective.*

**Sections (Cole-style — title = a decisional question; each finding = one bold quotable claim used as a
paragraph header; each experiment isolates one variable; NO new method is introduced):**
1. Introduction · 2. Background & Related Work · 3. Benchmark Setup · 4. C1 regime-gating · 5. C2 decision
rule · 6. C3 two-force mechanism (+ F2 aggregation) · 7. Practical Guidelines · 8. Limitations.

**Benchmark:** 4 CP objectives (LeJEPA/SimCLR/MAE/DIET-CP) + FROM-SCRATCH; encoders DINOv3/CLIP/MAE
(+ **SigLIP-2 held-out**), all ViT-B/16, 768-d; eval kNN (k=20, cosine, macro-F1) / Linear Probe / Fine-tune;
sizes: ALL 15 datasets carry 3–6 size tiers (100/500/1k/…/MAX; the original 8 additionally have
10k/25k tiers; the 7 expansion datasets were back-filled with 100–1000 tiers after the initial
MAX-only pass — see `eval/outputs/cp_long.csv` for the authoritative coverage). 15 datasets =
9 OOD + 6 fine-grained.

**Guidelines decision-flow (Section 7):**
- Step 1 — sphere-native (L2-norm CV < 5 %, DINOv3/CLIP/SigLIP) vs off-sphere (MAE). This sets the SIGN of
  every geometric prediction.
- Step 2 (sphere) — pre-CP angular position: isolated/OOD (overlap≈0) → kNN/LP ↑, FT ↓; embedded/fine-grained
  (overlap > 0) → kNN/LP ↓, FT ↑; **texture = embedded-but-task-orthogonal exception** (Exp D / DTD).
- Step 2 (off-sphere MAE) — CP usually raises kNN/LP (gap effect); avoid MAE-CP on sphere encoders
  (catastrophic, e.g. Food101 MAX ΔkNN ≈ −0.76 — though mostly an *aggregation failure*, per Exp B).
- Step 3 — pick CP data scale by monitoring geometric growth (Finding 3), not only val accuracy.
