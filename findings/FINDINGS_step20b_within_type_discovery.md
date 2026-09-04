# FINDINGS step 20b — within-type check of "uniformity turns type into a continuous position" on the DISCOVERY encoders

Date: 2026-09-04. Trigger: review of theory.docx Option B (2026-09-04), whose Result 2
claims pre-CP uniformity carries continuous information beyond the OOD/FG type; the
supporting within-type evidence was SigLIP-only (post-hoc). Computed from
cp_long_refreshed.csv (3-objective LeJEPA/SimCLR/DIET mean at MAX) + geometry_15.csv.

| encoder | overall rho(U, dkNN) | within-OOD (n=9) | within-FG (n=6) | type-residual partial Spearman | stratified permutation p (within-type, 20k) |
|---|---|---|---|---|---|
| DINOv3 | +0.668 | +0.617 | **-0.771** | **+0.036** | 0.215 |
| CLIP | +0.750 | +0.617 | +0.314 | +0.504 | 0.074 |
| MAE | -0.354 | -0.033 | -0.143 | -0.200 | 0.847 |
| SigLIP (step19, post-hoc) | +0.764 | — | — | +0.811 / +0.695 | 0.0003 / 0.0031 |

READING: on the discovery encoders, uniformity's overall correlation is carried by the
OOD/FG split. Within-type information is ENCODER-DEPENDENT: absent on DINOv3 (partial
+0.04; within-FG strongly NEGATIVE), suggestive on CLIP (partial +0.50, p = 0.07),
significant only on the held-out SigLIP panel (post-hoc). Within-OOD ranking is
consistent on both sphere discovery encoders (+0.62, +0.62); the fine-grained side
(n=6) is where the sign flips on DINOv3.

CONSEQUENCE: the Result-2 wording "uniformity turns dataset type into a continuous
position / is more informative than the dataset name" is NOT licensed as a general
claim. Licensed: "pre-CP uniformity is a label-free, encoder-computable score that
recovers the OOD/FG frozen-risk split and transfers to held-out encoders/scales;
within-type continuous ranking is detectable on SigLIP (post-hoc) and weakly on CLIP,
not on DINOv3." The SigLIP x DIET extension is the only pre-registrable test of
within-type ranking left — its within-type analysis should be elevated from
sensitivity to a gatekept co-primary endpoint.

Size-tier robustness (Result 1): at NUM_DATA = 500 (the only other tier with full
15 x 3 coverage) the split and the overall correlation hold — DINOv3 OOD +0.069
(8/9 > 0) vs FG -0.142 (0/6), rho +0.568; CLIP OOD +0.120 (9/9) vs FG -0.037 (2/6),
rho +0.604.

Spot checks of theory.docx numbers (all reproduced): dkNN-dFT Spearman -0.511 /
-0.446 / -0.057 with opposite-sign counts 9/15, 5/15, 7/15; ViT-L DIET-only 6/7,
Spearman +0.857.
