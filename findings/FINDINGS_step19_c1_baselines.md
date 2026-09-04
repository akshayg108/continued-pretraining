# FINDINGS step 19 — C1 pressure test: the dataset-type baseline ties the geometry predictor

Date: 2026-09-03. Trigger: GPT5.6 audit of theory.docx; every number below was
independently recomputed by us from the frozen files (c2_siglip_score.csv,
preregister_siglip.csv, vitl_score.csv, int2_results.csv, results.xlsx).
Type labels: 9 OOD (5 MedMNIST, galaxy10, eurosat, plant_village, dtd) vs
6 fine-grained (cars196, cub200, fgvc_aircraft, flowers102, oxford_pet,
food101) — the PLAN_iclr benchmark definition.

## Binary sign endpoint (the pre-registered C1 endpoint)

| held-out panel | geometry predictor | uniformity only | OOD/FG type baseline | always-HELP |
|---|---|---|---|---|
| SigLIP-2 (15) | 13/15 | 13/15 | 13/15 | 11/15 |
| ViT-L (7) | 6/7 | — | 7/7 | 4/7 |

SigLIP misses: geometry {dtd, food101}; type {flowers102, food101}. McNemar
geometry vs type: 1 vs 1 discordant, p = 1.0. Geometry vs always-HELP: 4 vs 2,
p = 0.6875. CONSEQUENCE: the frozen two-variable predictor's binary accuracy
does NOT exceed a coarse dataset-type baseline on either held-out panel; the
overlap term (beta_o = 0.19) adds nothing (uniformity-only ties). The claim
"the geometry predictor beats simple alternatives / establishes a distinct
geometric law" is NOT licensed.

## Post-hoc salvage (NOT the pre-registered endpoint; label as such)

On SigLIP, the CONTINUOUS frozen score correlates with realized dkNN at
Spearman +0.764; after residualizing both on the OOD/FG dummy the partial
Spearman is +0.811 (our linear-residual convention; GPT5.6's rank-partial
convention gives +0.695 — same conclusion), stratified permutation (within
type) p = 0.0003 (20000 draws; GPT5.6: 0.0031). Licensed wording: "the
frozen geometry score transfers across encoders and scales and retains
type-independent CONTINUOUS ranking information (post-hoc pressure test); its
binary sign accuracy does not exceed a coarse dataset-type baseline."

## The spectral bridge on the 135 REALIZED cells (not the 484 proxy cells)

sign(seed-mean d log RankMe) vs sign(realized dkNN), |d log RankMe| >= 0.01:
all 97/134 (72.4%); DINOv3+CLIP 53/89 (59.6%); MAE 44/45 (97.8%). The pooled
association is carried by the MAE regime; on sphere encoders the natural
spectral direction carries essentially no sign information. The bridge
predictor -> spectrum -> outcome is NOT supported; the two evidence layers
stay parallel. (CORRECTED step20: the MAE 44/45 is a base-rate degeneracy — every MAE cell HELPs and nearly every MAE SSL cell expands; agreement equals the majority base rate on every stratum, so the natural spectral direction carries no sign information anywhere.)

## INT2 matched-dose effect is not carried by one encoder (SSL, seed 42, common eligible)

DINOv3 +0.00501 [+0.00052, +0.01052] (n=30); CLIP +0.00451 [+0.00103,
+0.00806] (n=30); MAE +0.00614 [+0.00065, +0.01142] (n=26). "Small, stable,
direction-specific causal response axis" stands; no mediation share is
licensed (matched-dose ABSOLUTE effect is small — that is all we know).

## Consequences for the ICLR mainline (adopted)

Neither layer can be the sole headline: C1's binary endpoint ties a trivial
baseline; C2 is a small axis with whitening-literature precedent (Jegou &
Chum 2012; RankMe; alpha-ReQ). Adopt the conditional-response framing:
benchmark -> pre-CP prediction WITH the full baseline table -> spectrum census
(encoder x objective) -> controlled intervention (INT1 dose figure + INT2 raw
2x2 table) -> boundaries (FT reversal is an evaluator/encoder boundary, not a
universal law: 6/6 on DINOv3/CLIP, SigLIP FT 4/15; natural mediation
unconfirmed; placement NO VERDICT). theory.docx stays the evidence
monograph; a separate compact ICLR_MAINLINE draft carries the paper.
The highest-value new experiment is a NEW dataset panel with three rules
frozen before CP (type baseline, geometry predictor, type+geometry) — the
only way to re-establish C1's binary headline.
