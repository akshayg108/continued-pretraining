# FINDINGS step 18 — INT2: the spectral axis is functionally live in real post-CP representations

INTERPRETATION CORRECTED 2026-08-10 (external audit; CORRECTIONS.md #16): the
original text of this file misread the A statistic as "reverting toward pre
improves the readout in both directions". The raw sign table refutes that:
on contraction cells reversion FLATTENS and improves (+0.00595) while the
wrong arm sharpens and hurts (-0.00416); on expansion cells reversion SHARPENS
and HURTS (-0.00411) while the wrong arm flattens further and IMPROVES
(+0.00545) — screened common-eligible set, 50 + 36 cells.
The correct reading, verified independently twice: INT1's statics law —
flatter spectrum -> higher frozen readout, sharper -> lower — CONTINUES TO
HOLD inside real post-CP representations. A > 0 encodes sign-consistency with
that law along the axis real CP moved; it does not mean "pre is attractive".
G < 0 means readouts move numerically closer to PRE VALUES, which on expansion
cells (e.g. DIET: pre 0.5983, post 0.6897, reverted 0.6765) is a performance
LOSS. "Restoration" is restoration of the pre readout LEVEL, never a synonym
for improvement. All conclusions below are stated in this corrected frame.

Date: 2026-07-22. Protocol: INT2_PREREG.md consolidated operative v1.2 (frozen
pre-data through 12 external review rounds). Features: 484/484 post-CP cells,
cluster-frozen sidecar verified locally (read-only). Run: 3872 rows, zero
warnings. Dose accounting (audit round 2): 92 wrong and 15 over TARGETS were
capped to the attainable range (pre-declared TARGET CAPPING); of those, 25
wrong cells were judged dose-INFEASIBLE and rejected at calibration — capping
and rejection are distinct steps. Full output:
eval/outputs/int2_verdict.txt; checksums: eval/outputs/int2_checksums.sha256.

## Gates

- G0 PASS. G-NC joint 165/165 and 164/165 (sham clean).
- G-POST: rho +1.0000 on all three columns (kNN, vote_margin, vote_pos_frac),
  worst per-cell |diff| 0.0013 across ALL 484 checkpoints, n_test 484/484 —
  the dumped features numerically reproduce the audited ND12 post state within
  the frozen tolerances (worst per-cell diff 0.0013).
- Calibration: 25/484 wrong-arm cells REJECTED as dose-infeasible (distinct
  from the target capping accounted above; concentrated in DIET as predicted), 0 rejections elsewhere incl. transplant
  (profile err, amplification, rank-ratio all within gates).
- G-P: all arms R <= 0.02, median capture drift <= 0.0016 — the surgery is
  extremely clean on real post features.
- Direction census: S=+1 (contraction) 204, S=-1 (expansion) 277, undefined 3
  across 484 —
  real CP EXPANDS RankMe more often than it contracts once DIET and all seeds
  are counted; the bidirectional design was necessary, not decorative.

## Primary verdicts (seed 42, kNN)

INT2-1 FUNCTIONAL PARTICIPATION: **YES** (scope = LeJEPA+SimCLR; DIET's
common eligible coverage is 31/45 < 34 -> its formal INT2-1 is NO VERDICT;
"all three objective families confirmed" is NOT licensed):
  A_full +0.0052 CI[+0.0020, +0.0081] (n = 86 common eligible cells)
  A_half +0.0019 (dose-consistent)
  A_wrong -0.0047 CI[-0.0076, -0.0017]
  Corrected reading: A > 0 and A_wrong < 0 jointly say the frozen readout
  responds to spectral re-weighting on real post features with the
  INT1-consistent sign (flatten up, sharpen down) — per-direction table on
  the FORMAL screened common-eligible set (86 = 50 contraction + 36
  expansion; 4 MAE expansion cells leave via the placement screen):
  contraction full +0.00595 / wrong -0.00416; expansion full -0.00411 /
  wrong +0.00545. The artifact explanation ("any power transform improves")
  remains excluded: sharpening hurts ON AGGREGATE (per-cell strict
  monotonicity was only 35/60 kNN in INT1 — the law is an aggregate law).

INT2-2 NET RESTORATION: **RESTORATION licensed** (G-scope = all three
methods): G[full] -0.0063 CI[-0.0091, -0.0034] — reversion moves readout
VALUES significantly closer to the pre level. On expansion cells this is a
performance DECREASE toward pre (DIET example above); restoration-of-level,
not improvement.

Per-method strata (98.33% CIs): LeJEPA A +0.0045 *, SimCLR +0.0061 *,
DIET +0.0132 * (secondary full-arm statistic — DIET's formal INT2-1 is NO
VERDICT; licensed wording: "DIET's secondary full-arm response is consistent
with the same flatten-up / sharpen-down law"). DIET-vs-mean(SSL) contrast
positive (CI [+0.0019, +0.0145]). Direction symmetry: -0.0032
CI[-0.0103, +0.0038] — NO ASYMMETRY DETECTED (no equivalence test was
pre-registered; this is absence of evidence for asymmetry, not evidence of
symmetry).
SigLIP panel (separate): A +0.0092 CI[+0.0055, +0.0135].
MAE-backbone stratum: A +0.0138 (off-sphere regime shows the effect too).
Capture-screen 0.03 sensitivity: identical verdicts.

## Secondary readouts

INT2-3 OVERSHOOT: R_over +0.0099 CI[+0.0057, +0.0148]. Corrected reading:
at the single tested extra dose, the signed slope CONTINUES through pre — the
flatten-up/sharpen-down response does not saturate at the pre level. With one
dose beyond pre, "the optimum lies beyond pre" is NOT licensed; only "no
saturation detected at pre at the tested dose".

INT2-5 TRANSPLANT (profile-matched): A +0.0248 CI[+0.0159, +0.0347],
G -0.0219 CI[-0.0318, -0.0127]. On the SAME screened matched set (n=133) the
descriptive response is ROUGHLY THREEFOLD the power path (+0.0248 vs +0.0078;
the earlier 5x compared different analysis sets), and the doses are not
matched — licensed wording: "the full-profile surgery shows a roughly
threefold larger descriptive response than the RankMe-matched path"; "most
functional content lies beyond RankMe" is NOT licensed. ("Matches the normalized pre-spectrum profile within the frozen
tolerances".)

LP [proxy] mirrors (pre-registered scope INT2-1/2/4 only): replicate with
LARGER magnitudes (A_full +0.0207, wrong -0.0189, method full-arm strata
significant, G restoration) —
consistent with INT1's finding that LP is the more spectrum-sensitive operator.

## Cross-seed replication (pre-registered)

seed 43 kNN: INT2-1 YES, G restoration, all three METHOD-SPECIFIC FULL-ARM
strata significant (DIET secondary; formal INT2-1 NO VERDICT), DIET contrast
positive. seed 44 kNN: INT2-1 YES, G restoration, DIET +0.0138 *,
SimCLR *, LeJEPA attenuated (+0.0024, n.s.); DIET contrast positive. LP: YES +
restoration in both seeds. The primary conclusions replicate across seeds.

## What is now established (audit-calibrated claim)

INT1's spectral statics law — flatter spectrum raises, sharper lowers the
specified frozen readouts — CONTINUES TO HOLD inside real post-CP
representations, along the axis real CP actually moved, dose-consistently,
with the anti-directional control behaving with the opposite sign, replicated
across 3 seeds (pooled scope LeJEPA+SimCLR; DIET secondary full-arm
consistent). Reversion additionally moves readout VALUES toward the pre level
(restoration-of-level; a performance loss on expansion cells). What INT2
establishes is: constructed spectral re-weighting -> specified frozen readout,
live in real post representations. NOT established: CP training -> natural
spectral mediation -> total behavioral effect. Effect sizes are SMALL (pooled
A_full ~ +0.005 kNN); no "share" quantification is licensed.

NOT established / bounded: no saturation detected at pre at the single tested
overshoot dose (optimum location unresolved); the full-profile surgery shows a
roughly threefold larger descriptive response (matched-cell, n=133) than the RankMe-matched path
(dose-unmatched; no "beyond-RankMe content" claim); rotation/translation
channels untouched (a spectral-only account remains excluded by design); LP
claims are [proxy]; DIET's formal INT2-1 is NO VERDICT (coverage).

Paper impact: closes the INT1->INT2 arc — INT1 gave geometry -> readout
statics on pre features; INT2 shows those statics remain live inside real
post-CP representations along the axis real CP moved (small effects, pooled
scope LeJEPA+SimCLR, DIET secondary consistent). The licensed causal chain is
"constructed spectral re-weighting -> specified frozen readout"; the chain
"CP training -> natural spectral mediation -> total behavioral effect" remains
open. DIET remains the objective-family direction contrast (134/134 expand)
with a consistent secondary interventional response.

## Provenance

INT2_PREREG v1.2 (12 review rounds, all pre-data); sidecar-verified features;
G-POST numerically reproduced within frozen tolerances; 124-test suite green; results + verdict SHA256 in
eval/outputs/int2_checksums.sha256.
