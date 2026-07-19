# INT1 pre-registration — CONSOLIDATED OPERATIVE RULES v1.6

Frozen 2026-07-19, BEFORE any real INT1 data exists (the feature dump has not been
launched). Sections 1-6 below are the SINGLE authoritative protocol; the amendment
log (section 7) records how they evolved through pre-data review rounds (Codex
rounds 1-3) and is historical — where any older wording conflicts, sections 1-6
govern.

Intervention-1: counterfactual surgery on FROZEN pre-CP features (GPT5.6 design,
adopted with additions from this project's audit history). Question: does geometry
(spectrum shape, eigendirection scale assignment) CAUSALLY matter for frozen
readouts at fixed information content — "geometry -> kNN/LP" statics. Explicitly
NOT tested: why CP produces these geometry changes (the later CP-training
intervention), and capture necessity (invertible linear maps preserve col(X);
goes to a later matched-deletion round).

## 1. Data & protocol

Pre-CP features only; 4 ViT-B encoders x 15 datasets = 60 cells. Bank = the ND12
vote bank (standard <= 5000 seed-42 train protocol; galaxy10: evaluator
manual-split train, seed 42). Query = test split <= 2000 (ND12 protocol). One GPU
features-dump pass saves .npz per cell; ALL surgery and evaluation are local CPU.

Readouts (both, always):
  kNN  vote_operator_metrics (k=20, cosine, inverse-distance — the G2-validated
       proxy, rho = +0.984 against the paper's 45 pre-CP MAX levels)
  LP   standardized deterministic probe: L2 normalize, LogisticRegression
       (max_iter=1000, C=1.0, lbfgs), macro-F1 (the zero_shot_eval.
       linear_probe_evaluate protocol). DISCLOSURE: the paper's recorded
       lp_pre/dlp come from a DIFFERENT probe — linear_probe_pytorch_evaluate,
       an unregularized >= 10k-step Adam probe with no seed (nondeterministic).
       Connectability is decided by gate G2-LP below; without it, every LP
       conclusion carries the [proxy-internal] tag.

Transform construction rule: SVD fitted on the BANK features only; the SAME fixed
linear map applied to bank and query rows. No labels enter transform construction.
Every non-rotation map acts as the IDENTITY on the orthogonal complement of the
bank row space (M += I - V^T V): a no-op for n >= d cells; for n < d cells
(breastmnist 546 x 768) the out-of-span query component passes through unchanged,
so identity-vs-surgery deltas isolate the in-span effect. query_oos_frac is a
reported dose-dilution DIAGNOSTIC, not an exclusion. All preservation claims are
re-checked after the evaluator's L2 normalization and reported per cell.

## 2. Frozen transform grid (15 per cell; power_cp optional)

T0  IDENTITY — anchor (alpha = 1).
T1  ORTHOGONAL ROTATION, seeds {0, 1} — negative control (exact isometry).
T2  SPECTRAL POWER sigma -> sigma^alpha, alpha in {0.25, 0.5, 0.75, 1.5, 2.0} —
    spectrum shape / RankMe moves; span, mode identity and order preserved
    (alpha > 0), hence eigendirection scale assignment; capture preserved pre-L2.
T2c CP-CALIBRATED alpha per cell: bisect alpha so the surgered bank's raw RankMe
    matches the realized post-CP rank (nd7 LeJEPA/SimCLR median; nd1 convention).
    Skipped with a note where no nd7 target exists.
T3  ISO-SPECTRAL EIGENDIRECTION-SCALE REASSIGNMENT (graded, label-free): swap the
    sigma values of the top B = 16 directions with the block at depth D in
    {64, 256, 512}, plus T3f FULL SHUFFLE (seed 0) as the extreme. Preserves the
    sigma multiset, RankMe, numerical rank and capture; changes WHICH directions
    carry large scale. NAMING RULE: this family is called "placement surgery" in
    interpretation ONLY if INT1-2 passes its gates (first-stage included);
    otherwise reports must use the mechanical name above. T3f noise caveat: real
    s_max/s_min reaches ~1e5, so the full shuffle promotes near-noise tail
    directions — the clean claim leans on the graded demotes; breastmnist's
    demote-512 is near-tail (546 modes) and is read as shuffle-like.
T4  COMBO (interaction arm): demote(block 16, depth 256) then power(alpha),
    alpha in {0.5, 2.0}, built from one bank SVD. Bank spectrum multiset equals
    the pure-power arm's (same RankMe); scale assignment equals the demoted one —
    the factorial cell for INT1-5.

## 3. Gates (fail -> stop before interpretation)

G0    CENSUS: exact 4 x 15 cell set (membership, not count); unique (encoder,
      dataset, transform, param) keys; full mandatory grid per cell; NO rows
      beyond the frozen grid (stale/foreign transforms fail); power_cp REQUIRED
      on every cell (ND7 provides 60/60 rank targets) with finite rankme_target;
      ALL numeric metric columns finite (knn_f1, lp_f1, rankme_raw_bank,
      rankme_l2_bank, capture_l2, cC_K_l2, query_oos_frac). Fail -> stop.
G-NC  rotation control, JOINT per cell: a cell passes iff |delta kNN-F1| < 0.005
      AND |delta LP-F1| < 0.005; required on >= 58/60 cells for both seeds.
      Fail -> harness bug, stop. (Unscreened — the negative control is raw.)
G-P   contamination gate, per transform: excluded from causal readouts iff
        (a) cross-family drift ratio R > 0.5, where
            power family:   R = median|post-L2 cC_K drift| / median|cC_K movement
                            under shuffle|
            demote/shuffle: R = median|post-L2 log-RankMe drift| / median
                            |log-RankMe movement under extreme alphas|
            combo:          intends both axes — only (b) applies; OR
        (b) median |post-L2 capture drift| > 0.05.
      PER-ROW CAPTURE SCREEN: additionally, any (cell, dose) ROW with
      |capture_l2 drift| > 0.05 leaves the effect estimates (INT1-1/2/3/4/5);
      clean doses of the same cell STAY (the screening unit is the row, not the
      cell). Screened counts and (cell, dose) lists are always reported.
      COVERAGE FLOOR: a screened estimate carries decision weight only if >= 45
      unique cells spanning >= 12 datasets remain; below the floor the affected
      readout is NO VERDICT (undecidable), never "no".
      Absolute drifts always reported; pre-L2 preservation exact by construction.
G2-LP LP proxy reproduction: Spearman rho between identity-cell sklearn LP F1 and
      the paper pipeline's lp_pre at the 45 MAX levels (is_max only, main-grid
      3 encoders; exact count of 45 asserted). rho >= 0.9 -> lp_connectable: LP
      conclusions may reference the paper LP story at ordering level. Otherwise
      EVERY LP conclusion (INT1-1/2/3/5 LP flags) carries [proxy-internal] and
      paper-LP claims are not licensed. No stop either way; SigLIP never enters.
T3-FS FIRST-STAGE gate, per T3 grade: the grade enters the INT1-2 trend/decision
      only if its median |cC_K_l2 drift| >= 0.05, computed on the SAME capture-
      screened row subset that enters the F1 estimate (first stage and effect use
      identical eligibility); dropped grades are reported with their value.

## 4. Pre-registered readouts

All verdicts are THREE-STATE (v1.6): YES / no / NO VERDICT — a readout whose
decision doses are all excluded, uncomputable, or below the coverage floor is
UNDECIDABLE and must never be reported as a negative result.

INT1-1 SPECTRUM EFFECT: PRIMARY decision at the two frozen extreme doses alpha in
       {0.25, 2.0}, each read at a Bonferroni-adjusted 97.5% dataset-block
       bootstrap CI (2000 draws, seed 0; family-wise 5% per readout). "Functional
       spectrum effect" = CI excludes 0 at either primary dose (per readout,
       reported separately); NO VERDICT if no primary dose is evaluable (G-P +
       coverage). alphas {0.5, 0.75, 1.5} form the descriptive dose-response
       panel (95% CIs, no decision weight). Direction not pre-specified.
INT1-2 T3 EFFECT: decided at the DEEPEST grade passing G-P, T3-FS and the
       coverage floor (95% CI excludes 0) AND |mean effect| monotone
       non-decreasing across the surviving DEMOTE grades, evaluated SEPARATELY
       per readout (kNN and LP each use their own trend). NO VERDICT if no grade
       survives. Interpretation as "placement effect" only on pass (naming rule).
INT1-3 OPERATOR SPECIFICITY: paired per-cell contrast (delta kNN - delta LP).
       DECISION: family-pooled CI at Bonferroni-adjusted 98.33% level (3 families
       — power, demote, shuffle — family-wise 5%); "operator-specific response" =
       pooled CI excludes 0 for at least one family; per family NO VERDICT on
       full exclusion or coverage failure. Per-dose panel always reported at 95%
       (descriptive).
INT1-4 RANKME-MATCHED POWER-PATH CALIBRATION (T2c): Spearman rho(surgical
       dknn(alpha_cp), realized dknn) over power_cp cells surviving BOTH G-P and
       the calibration acceptance: achieved bank RankMe within 2% relative error
       of the nd7 target AND alpha_cp off the search bounds (<= 0.06 or >= 3.99
       rejected as clamped); rejected cells reported with reasons. (realized =
       2-method mean dknn; SigLIP cells from c2_siglip_score.) Reported with
       dataset-block bootstrap CI (2000 draws, seed 0); no pass threshold.
       INTERPRETATION RESTRICTION: rho ~ 0 rules out THIS RankMe-matched power
       path only — it does NOT establish that the missing factor is capture,
       local topology, or non-linear information.
INT1-5 INTERACTION: per readout, factorial contrast delta(combo) - delta(power
       alpha) - delta(demote 256) at the 2 combo doses, each at a Bonferroni-
       adjusted 97.5% block CI (family-wise 5% per readout over the 2 doses).
       "Interaction present" (per readout) = CI excludes 0 at either computable
       dose; NO VERDICT if neither dose is computable (exclusion/coverage).

## 5. Decision tree (declared verbatim, GPT5.6)

- Only T2 works: rank/spectrum has functional effect; placement possibly epiphenomenal.
- Only T3 works: where task information sits matters more than RankMe.
- Both work with interaction: strongest support for spectrum x placement = readout
  alignment.
- kNN and LP respond differently: causal evidence for operator matching.
- Neither works: the current theory remains a descriptive account.
- Surgery cannot reproduce realized CP changes (INT1-4 ~ 0): the missing factor is
  capture, local topology, or non-linear information.
(Read with the T3 naming rule, the G2-LP scope tag, and the INT1-4 interpretation
restriction: the last branch's "missing factor" list is the hypothesis space the
next experiment discriminates, NOT a conclusion INT1-4 itself licenses.)

## 6. Declared limitations

- Statics only: geometry -> readout at fixed information; not CP mechanism.
- Surgical geometry changes need not mimic CP's coupled changes (except T2c).
- LP is the standardized deterministic probe; its link to the paper's PyTorch LP
  is an empirical gate (G2-LP), not an assumption. No C sweep in round 1.
- n = 60 cells, dataset blocks as bootstrap units; multiplicity handled by frozen
  primary doses / Bonferroni families as specified per readout (no global FDR).
- Post-L2 preservation is approximate for non-orthogonal maps; G-P quantifies it,
  the per-cell capture screen bounds the third-axis leak at cell level.
- A positive INT1-1/2 supports "this linear surgery family causally moves the
  readout", with disclosed post-L2 covariate drift — single-axis attribution is
  bounded by G-P, not absolute.

## 7. Amendment log (historical; sections 1-6 govern)

- v1.0 (2026-07-19): initial freeze — grid T0-T3f, G-NC, absolute-drift G-P,
  INT1-1..4 at 95% "any dose" (later retired), LP described as "evaluator mirror".
- v1.1: G-P absolute thresholds -> unit-free cross-family contamination ratio
  (synthetic smoke showed dimension dependence at d=48).
- v1.2: T3f noise caveat; n<d out-of-span EXCLUSION rule (superseded in v1.4);
  INT1-4 CI; INT1-2 monotonicity enforced in code.
- v1.3: monotonicity clause restricted to the graded demote family.
- v1.4 (Codex round-2): identity-on-complement construction (replaces the v1.2
  exclusion; oos becomes a diagnostic); combo interaction arm + INT1-5; G0 gate;
  G-NC made joint per cell; G-P capture stop (median > 0.05); INT1-1 primary
  doses {0.25, 2.0} at Bonferroni 97.5% (retired "any alpha at 95%"); INT1-3
  pooled-decision clarification; INT1-4 exclusion consistency; LP protocol
  disclosure + G2-LP gate; dump structural acceptance + CUDA fail-fast.
- v1.5 (Codex round-3): G2-LP restricted to the 45 MAX levels (is_max, count
  asserted — mixed-size aggregation gave rho 0.925 vs the MAX ordering and could
  flip the 0.9 gate); G0 strict (exact cell-set membership + all metric columns
  finite; was fail-open to capture_l2 NaN); INT1-3 decision moved to Bonferroni
  98.33% pooled CIs; INT1-5 doses at Bonferroni 97.5% with per-readout verdict;
  lp_connectable is an explicit status propagated to every LP conclusion (was
  print-only); per-cell capture screen; T3 first-stage gate + naming rule
  ("iso-spectral eigendirection-scale reassignment" until INT1-2 passes);
  prereg consolidated into this single operative rule set.
- v1.6 (Codex round-4): capture screen unit corrected to the ROW (cell x dose) —
  the v1.5 cell-level drop removed clean doses of a contaminated cell from
  pooled estimates (verified: 4-row repro kept 2 instead of 3); G0 additionally
  rejects rows beyond the frozen grid and REQUIRES power_cp per cell (ND7 has
  60/60 targets) with finite rankme_target; run resume switched to an exact
  key-set check (the sentinel + row-count heuristic could false-complete a cell
  with a stale extra row); three-state verdicts (YES / no / NO VERDICT) for
  INT1-1/2/3/5; coverage floor (45 cells / 12 datasets) for decision-bearing
  screened estimates; T3 first stage computed on the same screened subset as
  the effect; INT1-4 renamed "RankMe-matched power-path calibration" with an
  explicit calibration acceptance (2% RankMe tolerance, clamped alphas
  rejected, run records rankme_target) and an interpretation restriction.
