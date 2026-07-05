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
- **Reruns + second-axis phase (steps 7/8, 2026-07-05/06)** — the 60-cell reruns landed and
  REPLICATE the old behavior (test1-4 all green); F2/F3 numbers final on the fixed sweep
  (`postcp_sweep_fixed.csv`); **collision is invariance-specific** (DIET null −0.11); DIET
  folded into Δ@MAX (F1 point estimates up on DINOv3: +0.668/+0.757/−0.618; reversal p<0.05
  on all six pairs); MAE inversion survives the SA-readout ablation (`geometry_mae_sa.csv`);
  4-axis gate signature (`geometry_class_15.csv`: rankme/α-ReQ/TwoNN-ID/norm-CV); SigLIP
  position law +0.746; class-CDNV bookkeeping channel −0.461; exploratory packing axis
  (FG-7 +0.893/+0.714, dose confound refuted); Exp B on repaired references 0.594
  (test3). ✓ (`findings/FINDINGS_step7_second_axis.md`, `_step8_final_integration.md`;
  refreshed behavior in `cp_long_refreshed.csv`; adversarial verification 9/9)
- **Spectrum & Transport phase (2026-07-07, in flight)** — design + code complete
  (`eval/DESIGN_spectrum_transport.md`, `eval/PLAN_spectrum_transport.md`). Design 1
  (bilinear unified law) RAN locally: V1 PASS (single-term x·θ LOO +0.497 ties binary gate
  +0.485, beats position-only +0.437; block-bootstrap M4−M2 CI [−0.15, +0.09]); V2 PASS
  (θ_cdnv: MAE 0.168 ≪ SigLIP 0.386 < CLIP 0.596 < DINOv3 0.693); **V3 FAIL as
  pre-registered** (SigLIP ρ(unif, knn_pre)=+0.171 breaks the strict sphere-negative sign
  flip — the D3/CLIP(−) vs MAE(+0.59) contrast survives but is downgraded to suggestive;
  gate evidence count stays at 3). `bilinear_law.csv`. Exp I (layer-wise virtual encoders,
  incl DIET + SigLIP grids) and Exp J (transport-field decomposition, all 4 methods) are
  COded + slurm-ready, awaiting cluster: `run/slurm/eval/exp_i_layerwise_pre.sh`,
  `exp_i_layerwise_post.sh`, `exp_j_transport.sh`; scorers
  `eval/adjudicate/{layerwise_law,transport_law}.py` smoke-tested on synthetic data.

## Remaining

### R1 — Exp D: texture sub-regime (DEPRIORITIZED 2026-07-01, optional)
Status update: the P-B spectral-task-alignment explanation of DTD was tested on existing data
and **REFUTED** (DTD ranks 3/7 and 7/7 on task_energy_in_top50 — step 7); the exploratory
**packing axis** absorbs DTD instead (2nd-loosest FG class packing on both sphere encoders →
muted response). Exp D remains the only way to make the texture boundary CONFIRMATORY rather
than exploratory, but is deprioritized by decision. Original design kept below for when/if it
is revived. Turn the DTD texture exception (predicted-HURT-but-HELPs, reproduced on the SigLIP
C2 scoring) into a *characterized boundary*, or confirm it is a lone oddity. Add 2
texture/material datasets whose TASK is orthogonal to object identity but whose IMAGES embed
near ImageNet.
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
- [ ] Figures: (1) regime-gating sphere-vs-MAE sign-flip + the 4-axis gate-signature table
  (rankme/α-ReQ/TwoNN-ID/norm-CV, `geometry_class_15.csv`); (2) decision-rule out-of-sample ROC
  vs baselines + the SigLIP cross-encoder 87 %; (3) two-force mechanism — REVISED framing:
  spread universal / collision invariance-specific (DIET null); (4) growth/transport;
  (5) SA-recovery (Exp B aggregation-failure, repaired references: 0.594); (6) NEW: MAE
  SA-readout ablation panel (inversion survives a learned readout — C1 defense);
  (7) optional: packing-axis panel (CUB-vs-Flowers margins + DTD) marked EXPLORATORY.
- [ ] Stats: foreground OUT-OF-SAMPLE metrics (LOO bal-acc, cross-encoder 87 %) over in-sample ρ; report the
  BH-FDR q-values (`stats_pass.csv` + `second_axis_stats.csv`); disclose the caveats — FT-non-transfer,
  DINOv3-collision-marginal, Exp-B-residual-loss, n=15.
- [ ] Wording discipline additions (steps 7/8, from adversarial verification): DIET-integration
  gains are POINT-ESTIMATE statements (n=15, never "significantly stronger"); state per-method
  force definitions explicitly (spread = plain Spearman, collision = partial given spread);
  MAE force-law statistics per-method only (pooled = Simpson artifact) and as all-size-trajectory
  claims; predictor LODO is dataset-grouped (15 folds, both encoder rows held out together);
  packing axis reported with BOTH residualization variants (global-fit +0.893/+0.714 vs
  within-FG null) and labeled exploratory; d_cdnv presented as a labeled bookkeeping/consistency
  channel with the tautology disclosure, never as an independent discovery; SA-readout ablation
  tests POOLING (overlap re-measurement caveat: SA-vs-meanpool overlap rank-corr 0.97 —
  uniformity/MMD carry the independent weight).
- [ ] Sections: Intro (C1–C3) / Related (Sorkhei foil [τ_w=0.73, labeled kNN — see corrected
  CORE_PAPERS entry], Kumar 2022 LP-FT, **Beyond-[CLS]** Przewiezlikowski 2024
  for the SA aggregation mechanism; cite Galanti 2022 vs Harun 2025 as the live
  NC-vs-transfer controversy; cite & differentiate arXiv:2603.03530 directional-CDNV
  [few-shot LEVELS ≠ our CP-Δ]; pre-empt SimCLR App. B.7 + He2022 §4.3 [known LP–FT
  decoupling ACROSS methods vs our signed within-pipeline reversal]) / Setup / one-bold-claim-header
  per finding / Guidelines decision-tree / Limitations.
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
