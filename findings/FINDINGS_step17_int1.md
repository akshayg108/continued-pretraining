# FINDINGS step 17 — INT1 (Intervention-1): first causal-layer result

Date: 2026-07-19. Protocol: INT1_PREREG.md consolidated v1.6 (frozen pre-data
through 4 Codex review rounds). Features: 60/60 cells manifest-accepted (cluster
+ local verify). Run: 900 rows (60 cells x 15 transforms), zero alpha_cp clamps,
zero WARN. Full adjudicator output: eval/outputs/int1_verdict.txt.

## Gates

- G0 PASS (exact 4x15 grid, 15 transforms/cell, power_cp 60/60).
- G-NC PASS at the maximum: joint |dkNN| & |dLP| < 0.005 on 60/60 cells, both
  seeds (max drift 0.0020). The harness is clean.
- G-P: every power/power_cp/demote/shuffle transform passes (R <= 0.07 except one
  power_cp cell at R = 0.41, passing); combo[2.0|256] EXCLUDED (median capture
  drift 0.1479 > 0.05). Demote-family transform-level capture drift medians sit
  at 0.036-0.039 — under the 0.05 stop but with 25 individual rows over it (see
  INT1-2).
- G2-LP: rho = +0.909 >= 0.9 -> lp_connectable. The deterministic sklearn probe
  reproduces the paper's PyTorch-LP pre-CP ordering at the 45 MAX levels;
  BORDERLINE (0.009 above the gate) — treat as ordering-level connectability,
  nothing stronger. SCOPE: this validates the IDENTITY-BASELINE ordering only;
  it does not validate surgery deltas or effect magnitudes against the paper LP.

## Readouts

INT1-1 SPECTRUM EFFECT: kNN YES, LP YES (both primary doses significant at
Bonferroni 97.5%). Monotone dose-response AT THE AGGREGATE-MEAN level across all
five alphas, both readouts:
  alpha  0.25   0.5    0.75   1.5    2.0
  dkNN  +0.072 +0.055 +0.024 -0.038 -0.075
  dLP   +0.125 +0.122 +0.078 -0.149 -0.267
Flattening the spectrum (alpha < 1, RankMe up) improves frozen readouts;
sharpening (alpha > 1, RankMe down) degrades them.
SCOPE (audit round 5): the invariances (span, mode identity/order, capture, no
labels) hold EXACTLY in the raw feature space; the evaluators' per-row L2
normalization re-couples the geometry, and power's post-L2 |d cC_K| reaches
0.14-0.23 in individual cells. POST-HOC ROBUSTNESS (labelled as such, not
pre-registered): excluding cells with post-L2 |d cC_K| > 0.05, both primary
doses stay significant — alpha 0.25 (n=46): kNN +0.071 [+0.017,+0.110], LP
+0.132 [+0.086,+0.172]; alpha 2.0 (n=57): kNN -0.075 [-0.094,-0.060], LP
-0.264 [-0.332,-0.205] (97.5% block CIs; independently recomputed, matches the
external audit bit-for-bit).
PER-CELL HETEROGENEITY: strict 6-point per-cell monotonicity holds in 35/60
cells (kNN) and 32/60 (LP; 33 under tie-tolerant counting); per-dose sign
agreement with the aggregate direction is 51-59/60. The claim is an aggregate
causal effect with majority per-cell consistency, not a per-cell law.
Accurate claim level: a controlled, label-free spectral power intervention
causally moves the two SPECIFIED frozen readouts; it does not establish that
RankMe per se is the causal variable (the surgery reshapes the whole spectrum).

INT1-2 T3 (iso-spectral eigendirection-scale reassignment): NO VERDICT
(pre-declared undecidable branch, NOT a negative).
  - demote[64] failed first-stage (median |d cC_K| = 0.0199 < 0.05 — the mild
    grade does not actually move placement).
  - demote[256]/[512]/shuffle[0] each lost the SAME 25 rows to the per-row
    capture screen -> 35 cells < the 45-cell coverage floor. Composition
    (corrected, audit round 5): CLIP 7, MAE 7, SigLIP 7, DINOv3 4 — NOT a full
    4-encoder x 7-dataset block. The screened datasets are dermamnist, eurosat,
    galaxy10, octmnist, organamnist, pathmnist, plant_village (with DINOv3
    escaping on three of them).
  Mechanism status: POST-HOC HYPOTHESIS only. Capture drift anti-correlates
  with class count (rho = -0.48), consistent with "few classes -> class-mean
  energy concentrated in top directions -> demotion drags post-L2 capture",
  but exceptions in both directions (breastmnist K=2 never screened;
  plant_village K=38 screened) mean this is not established. What IS
  established: the round-1 surgery cannot separate placement from post-L2
  capture in those cells. Placement causality remains OPEN.

INT1-3 OPERATOR SPECIFICITY: power family operator-specific (pooled +0.026,
CI98.33 [+0.005, +0.044]); demote/shuffle NO VERDICT (coverage). The per-dose
panel is the informative part: the contrast is NEGATIVE at alpha < 1 (LP gains
more from flattening) and POSITIVE at alpha > 1 (LP loses more from
sharpening) — |dLP| / |dkNN| ranges 1.73-3.94 across doses. Accurate claim:
the two FIXED evaluation procedures have demonstrably different spectral
sensitivity under the same intervention. This supports the operator-matching
axis but does not by itself prove the operator-matching MECHANISM.

INT1-4 RANKME-MATCHED POWER-PATH CALIBRATION: pre-registered POOLED POSITIVE
RANK ASSOCIATION, no pass threshold: rho = +0.463 CI [+0.222, +0.639], n = 60
(zero calibration rejections; max RankMe relative error 4.2e-7). DECOMPOSITION
(audit round 5, independently reproduced): main-grid 45 cells rho = +0.552;
SigLIP-15 rho = +0.214; per-encoder CLIP +0.164, DINOv3 +0.232, MAE -0.071,
SigLIP +0.214; within-encoder pooled-rank rho = +0.135. The pooled association
is therefore carried mostly by BETWEEN-encoder structure; the path does NOT
reproduce within-encoder CP orderings, and no "explained share" language is
licensed. Interpretation restriction applies: this bounds THIS power path only.
POST-HOC SENSITIVITY (method convention): the realized target uses the frozen
nd1 2-method (LeJEPA/SimCLR) convention; recomputing the main-45 association
with a 3-method mean including DIET-CP gives rho +0.528 vs +0.552 — the
conclusion does not hinge on excluding DIET. (3-method rank targets would shift
the alpha_cp calibration by median 3.3%, max 20.3%; not rerun.)

INT1-5 INTERACTION: NO VERDICT (combo[2.0|256] G-P-excluded on capture;
combo[0.5|256] contrast blocked by demote[256] coverage). Open.

## Decision-tree mapping (v1.6 reading)

Closest branch: "T2 works" — with T3 undecidable rather than refuted. Plus the
spectrum-axis kNN-vs-LP differential (operator-sensitivity evidence). Combined
claim the data licenses (audit-round-5 calibrated):

  A controlled, label-free spectral power intervention on frozen features
  causally moves both specified frozen readouts, monotonically at the
  aggregate level (flatten -> improve, sharpen -> degrade; raw-space
  invariances exact, post-L2 recoupling disclosed and robustness-checked);
  the two evaluators differ in spectral sensitivity by 1.7-3.9x (LP more
  sensitive); and the RankMe-matched power path shows a pre-registered pooled
  positive rank association with the realized CP pattern (rho +0.46), carried
  mostly by between-encoder structure. NOT licensed: "RankMe is the cause",
  "real CP works through spectral contraction", any "explained share" figure,
  or per-cell lawfulness. Placement causality and the spectrum x placement
  interaction remain undecided (tool limitation, pre-declared).

Paper impact: upgrades C3's correlational "rank contraction tracks dknn drop"
to a causal statics statement at the readout layer (geometry -> readout;
NOT yet CP -> geometry -> readout). LP claims are paper-connectable via G2-LP
(borderline 0.909 — say "ordering-level", identity-baseline only).

## Independent audit trail

External audit (Codex, post-results): 60/60 manifest reproduced; identity kNN
vs ND12 bit-identical (max diff 0); verdict rerun SHA256-identical to
eval/outputs/int1_verdict.txt; 71/71 tests; power_cp max RankMe rel. error
4.22e-7, zero clamps. Every audit number above (post-hoc CIs, monotonicity
counts, sensitivity ratios, INT1-4 decomposition, screen composition,
rho(capture drift, K) = -0.48) was independently recomputed by us and matched.
Feature + results checksums: eval/outputs/int1_checksums.sha256.

## Open items for a possible INT2 round (not committed)

- Placement surgery that protects post-L2 capture per cell (e.g. class-energy-
  preserving reassignment, or restricting swaps to within-class-complement
  directions) to reopen INT1-2/5 on the 25 screened cells.
- Milder/graded demotes between 64 and 256 to pass first-stage without
  capture leak (the current gap between fs=0.02 and capture-leak grades is
  where a usable dose might live).
- combo re-design with capture stop in mind (alpha 2.0 composition leaked).
