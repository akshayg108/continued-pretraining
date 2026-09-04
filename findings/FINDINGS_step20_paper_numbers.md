# FINDINGS step 20 — cheap-analysis package for the ICLR mainline (no new training)

Date: 2026-09-03. Computed locally from frozen files (results.xlsx via load_long, geometry_15.csv, geometry_vitL.csv, c2_siglip_score.csv, vitl_score.csv, int1_results.csv, int2_results.csv). Conventions: behavioral dkNN/dFT = 2-method (LeJEPA/SimCLR) mean at MAX unless noted; Spearman with 2000-draw dataset bootstrap; INT2 A_full on the common eligible SSL set per seed.

KEY READINGS
- Regression to the mean is NOT the driver of C1: knn_pre -> dkNN is weak and CI-spanning-zero on DINOv3/CLIP/MAE, and on SigLIP the pre-kNN median split scores 5/15 (Spearman -0.12). The geometry score's continuous association (SigLIP +0.764 [+0.32,+0.94]) survives this check.
- Single-variable MEDIAN splits do not beat always-HELP on SigLIP (11/15 each); the frozen score's 13/15 comes from its frozen threshold; the OOD/FG type baseline also scores 13/15 (step19). MMD is not significant on SigLIP (+0.48 [-0.04,+0.84]).
- The SSL 'mixed' spectral direction is FULLY resolved by encoder regime: on sphere encoders (DINOv3/CLIP/SigLIP) LeJEPA/SimCLR contract in the large majority (DINOv3 35/6, 39/6; CLIP 32/9, 33/10; SigLIP 29/15, 33/12), on MAE they EXPAND (0/43, 3/42); DIET expands everywhere (45/45, 44/44, 45/45). Two-factor census: objective (DIET always expands) x regime (SSL contracts on sphere, expands off-sphere).
- The spectral bridge sits AT the majority-sign base rate everywhere: all 72.4% vs base 73.9%; sphere 59.6% vs 60.7%; MAE 97.8% vs 100% (every MAE cell HELPs and nearly every MAE SSL cell expands, so the 44/45 is a base-rate degeneracy, NOT a sign carrier — this corrects the step19 remark). Natural spectral direction carries no sign information beyond base rates; the two evidence layers stay parallel.
- INT2 matched-dose A_full is positive on every encoder in every seed (one seed-44 MAE CI touches 0): a small, stable, direction-specific axis, not a one-encoder effect.
- Discovery correlations carry WIDE n=15 CIs (several lower bounds near +0.05); the theory.docx headline values (+0.668/+0.757/-0.618) are the 3-METHOD delta@MAX convention (DIET folded in; FINDINGS_step8 'STRENGTHENED' table), whereas the 2-method (LeJEPA/SimCLR) convention used by INT1-4/INT2 gives +0.596/+0.693/-0.537 on DINOv3. Both are documented; the paper must state ONE convention per table and never mix them in one figure.

RAW OUTPUT

```
=== 1. Discovery correlations (2-method mean at MAX; Spearman, 2000-draw dataset bootstrap 95% CI) ===
  DINOv3 uniformity -> dknn: rho +0.596 [+0.06,+0.89] (n=15)
  DINOv3 uniformity -> dft: rho -0.532 [-0.87,-0.02] (n=15)
  DINOv3        MMD -> dknn: rho +0.693 [+0.24,+0.90] (n=15)
  DINOv3        MMD -> dft: rho -0.582 [-0.82,-0.16] (n=15)
  DINOv3    overlap -> dknn: rho -0.537 [-0.83,-0.04] (n=15)
  DINOv3    overlap -> dft: rho +0.661 [+0.18,+0.93] (n=15)
  DINOv3 regression-to-mean check knn_pre -> dknn: rho -0.339 [-0.76,+0.23]
    CLIP uniformity -> dknn: rho +0.754 [+0.29,+0.95] (n=15)
    CLIP uniformity -> dft: rho -0.486 [-0.88,+0.13] (n=15)
    CLIP        MMD -> dknn: rho +0.743 [+0.32,+0.95] (n=15)
    CLIP        MMD -> dft: rho -0.661 [-0.94,-0.11] (n=15)
    CLIP    overlap -> dknn: rho -0.673 [-0.90,-0.25] (n=15)
    CLIP    overlap -> dft: rho +0.549 [-0.04,+0.91] (n=15)
    CLIP regression-to-mean check knn_pre -> dknn: rho -0.154 [-0.65,+0.39]
     MAE uniformity -> dknn: rho -0.336 [-0.77,+0.20] (n=15)
     MAE uniformity -> dft: rho -0.500 [-0.91,+0.13] (n=15)
     MAE        MMD -> dknn: rho -0.229 [-0.70,+0.37] (n=15)
     MAE        MMD -> dft: rho -0.574 [-0.89,-0.01] (n=15)
     MAE    overlap -> dknn: rho +0.500 [-0.07,+0.81] (n=15)
     MAE    overlap -> dft: rho +0.325 [-0.33,+0.84] (n=15)
     MAE regression-to-mean check knn_pre -> dknn: rho -0.221 [-0.84,+0.37]

=== 2. SigLIP held-out: continuous variables vs realized dkNN (n=15) + median-split sign rules ===
  frozen geometry score                sign acc 13/15 (HELP if s>0); Spearman with dkNN +0.764 [+0.32,+0.94]
  uniformity (higher->HELP)            sign acc 11/15 (median split); Spearman with dkNN +0.746 [+0.32,+0.92]
  MMD (higher->HELP)                   sign acc 11/15 (median split); Spearman with dkNN +0.482 [-0.04,+0.84]
  overlap (lower->HELP)                sign acc 11/15 (median split); Spearman with dkNN +0.625 [+0.10,+0.91]
  pre-kNN level (lower->HELP, proxy)   sign acc 5/15 (median split); Spearman with dkNN -0.121 [-0.59,+0.42]
  OOD/FG type baseline                 sign acc 13/15; always-HELP 11/15

=== 3. ViT-L (n=7): geometry vs dkNN ===
  uniformity -> dkNN: rho +0.750 (n=7)
         MMD -> dkNN: rho +0.750 (n=7)
     overlap -> dkNN: rho -0.739 (n=7)

=== 4. Spectral direction census: encoder x objective (484 checkpoints) ===
                contract  expand  undef
encoder method                         
CLIP    DIET           0      44      0
        LeJEPA        32       9      0
        SimCLR        33      10      2
DINOv3  DIET           0      45      0
        LeJEPA        35       6      0
        SimCLR        39       6      0
MAE     DIET           0      45      0
        LeJEPA         0      43      0
        SimCLR         3      42      0
SigLIP  LeJEPA        29      15      1
        SimCLR        33      12      0

=== 5. Bridge on 135 realized cells with majority-sign base rates ===
  all                agreement 97/134 = 72.4% | majority-sign base rate 73.9%
  sphere (D3+CLIP)   agreement 53/89 = 59.6% | majority-sign base rate 60.7%
  MAE                agreement 44/45 = 97.8% | majority-sign base rate 100.0%

=== 6. INT2 matched-dose A_full per encoder (SSL, common eligible), seeds 42/43/44 ===
  seed 42: DINOv3 +0.0050 [+0.0005,+0.0105] n=30 | CLIP +0.0045 [+0.0010,+0.0081] n=30 | MAE +0.0061 [+0.0007,+0.0114] n=26
  seed 43: DINOv3 +0.0060 [+0.0011,+0.0118] n=28 | CLIP +0.0040 [+0.0002,+0.0084] n=27 | MAE +0.0081 [+0.0049,+0.0120] n=25
  seed 44: DINOv3 +0.0057 [+0.0015,+0.0110] n=28 | CLIP +0.0032 [+0.0002,+0.0074] n=27 | MAE +0.0038 [-0.0006,+0.0079] n=27
```
