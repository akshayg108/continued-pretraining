# INT2 pre-registration — RankMe-matched spectral REVERSION on real post-CP representations (CONSOLIDATED OPERATIVE v1.2)

FROZEN 2026-07-21, before any INT2 data exists (no post-CP feature dump has been
launched). Sections 1-6 are the SINGLE authoritative protocol at v1.2 (v1.1/v1.2
review-round amendments MERGED into the body; the amendment log at the bottom is
audit history only — where older wording conflicts, the body governs). Design
skeleton: GPT5.6; gate discipline: INT1_PREREG.md v1.6. OPERATIVE v1.2 CLAUSES
FOLDED INTO THE BODY: manifest identity locked (484 rows, method census 180/170/
134, seed census 165/160/159, unique ckpts, file SHA256 = 352247ca2aefefd6c74ef
3ffe80f3aaa55c50923ac02c54336cfbf1549e4069f); wrong/over doses capped to the
attainable calibration range with a numeric feasible flag (G0 enforces {0,1});
INT2-1 decided on the COMMON eligible intersection of full/half/wrong with
method-level claim scoping — a method failing its own 34/45 eligible floor
leaves the pooled primary and its stratum is NO VERDICT; the pooled claim names
its included methods (cross-family wording forbidden); the S!=0 filter applies
to A-statistics ONLY (INT2-2's G uses all cells); "no" requires significant
anti-alignment or a CI inside the +/-0.005 equivalence margin; INT2-strict npz
schema (float32, 768-dim, exact field set, finite labels, no foreign files);
ckpt full preflight before any model load; G-POST gated on ALL 484 cells with
PER-CELL worst-diff tolerances (kNN/vote_margin/vote_pos_frac 0.01 each) plus
n_test equality, alongside rho >= 0.98 and median < 0.005; pre-side provenance =
exactly 60 checksummed npz + int1_results.csv, verified before resume skipping;
transplant acceptance = RankMe 2% + normalized-spectrum profile error <= 0.02 +
max amplification <= 1e4 (without the profile gate the arm may only be called
"RankMe-matched transplant"); per-method G at Bonferroni-3 98.33%; duplicate-
preserving block bootstraps; seeds 43/44 replicate INT2-1/2/4; LP mirrors INT2-1/2/4 on
the main panels ([proxy]; narrowed round-9).

## 0. Question and honest hypothesis map

INT1 established the STATICS: a label-free spectral power surgery on frozen
PRE-CP features causally moves both frozen readouts (flatten -> improve,
sharpen -> degrade, aggregate level). NOT yet established: whether the spectral
change that REAL CP produces carries functional weight inside the REAL post-CP
representation.

INT2 asks exactly that, in both directions of real CP's spectral motion:
LeJEPA/SimCLR typically CONTRACT the spectrum; DIET-CP EXPANDS it (RankMe
higher than the 2-method median in 45/45 main-grid cells, median +203 — step17
follow-up). Each checkpoint is reverted toward ITS OWN matched pre geometry.

What the design can and cannot separate (declared up front):
- "The spectral axis is functionally live in real post representations, in the
  direction real CP moved it" — INT2-1 tests this, with the DIET stratum as the
  opposite-direction stress test of the same statistic.
- "Reversion moves readouts back TOWARD the pre level" — INT2-2 tests this
  (restoration language gated on it).
- "The pre level is special (vs. monotone flatter-is-better statics)" — only
  the OVERSHOOT arm (A6) discriminates this, as a SECONDARY readout (INT2-3).
  The wrong-direction arm (A4) excludes surgery-artifact explanations ("any
  power transform helps"); it does not by itself separate the hypotheses.
- A null INT2-1 with clean gates = the real spectral change is functionally
  inert at the real dose in the real representation — a strong, publishable
  negative that bounds INT1's statics to pre features.

## 1. Data

POST-CP features: ALL 484 MAX checkpoints, driven by the audited ND12 manifest
(eval/outputs/nd12_operator.csv: key columns method/encoder/dataset/seed + the
`ckpt` path column; NO re-discover()). Composition (verified 2026-07-21):
SimCLR 180, LeJEPA 170, DIET 134. Bank/query protocol IDENTICAL to INT1/ND12
(bank = vote bank <= 5000 seed-42 train, galaxy10 = evaluator manual split
seed 42; query = test <= 2000). One GPU dump pass (~8 GiB); all surgery and
evaluation local CPU.

PRIMARY DECISION SET, frozen: the complete seed-42 panel = 165 cells
(main panel: 3 methods x 3 encoders x 15 datasets = 135; SigLIP extension:
LeJEPA/SimCLR x 15 = 30; DIET has no SigLIP cells). Seeds 43/44 are a
pre-registered CROSS-SEED ROBUSTNESS replication of the same statistics —
never pooled as additional independent samples.

PRE-CP side: reused from INT1's 60 accepted cells (checksums in
eval/outputs/int1_checksums.sha256). Per (encoder, dataset): pre bank RankMe
and the full pre bank singular-value vector (from the INT1 bank SVD), and the
pre readouts Y_pre = INT1 identity rows. Y_pre is shared across methods and
seeds of the same (encoder, dataset) — declared.

## 2. Surgery arms (per post-CP cell)

All maps are fitted on the POST bank SVD only, label-free, applied identically
to bank and query, and act as the identity on the bank row-space complement
(INT1 v1.4 construction). Naming rule: A2-A4/A6 are "RankMe-matched power-path"
arms — the word "spectrum shape" is reserved for the transplant arm A5.

A0 identity — post baseline.
A1 rotation sham, seeds {0, 1} — negative control.
A2 HALF reversion — power alpha calibrated so log RankMe(surgered post bank)
   hits the midpoint of log RankMe_post and log RankMe_pre.
A3 FULL reversion — target = RankMe_pre (the cell's INT1 pre bank RankMe).
A4 WRONG-DIRECTION — target: log RankMe moved AWAY from pre by the magnitude
   of the full reversion: log target = log post + (log post - log pre); CAPPED
   to the attainable calibration range (2% margin) and flagged infeasible when
   the capped magnitude is below half the required dose (numeric feasible flag,
   G0-enforced to {0,1}). Same capping applies to A6.
A5 SPECTRUM TRANSPLANT (secondary diagnostic) — replace the post bank singular
   values by the PRE bank singular values in rank order (directions unchanged):
   s_post[i] <- s_pre[i] for i < min(n_pre, n_post) ranks; s_post[i] kept for
   larger i. The only arm that matches the normalized pre-spectrum profile
   within the frozen tolerances (never claim exact 'full shape' restoration).
A6 OVERSHOOT (secondary; this project's addition to the GPT5.6 grid) — target:
   log RankMe moved BEYOND pre by the same magnitude: log target = log pre +
   (log pre - log post). Discriminates "pre level is special" from "monotone
   statics" (INT2-3).

7 arm rows + identity = 8 rows per cell; 484 x 8 = 3872 rows, local CPU.
Calibration: calibrate_alpha bisection; ACCEPTANCE (INT1 v1.6): achieved bank
RankMe within 2% relative error of target AND alpha off the search bounds
(<= 0.06 or >= 3.99 rejected as clamped). Rejected arms are WRITTEN with their
values and excluded at verdict with reasons. For DIET cells the reversion
direction is sharpening (alpha > 1) — same machinery, direction emerges from
the data, never hard-coded per method.

## 3. Readouts

kNN PRIMARY: vote_operator_metrics (k=20, cosine, inverse-distance) — the
G2-validated proxy, additionally bound to the recorded post levels by G-POST.
LP SECONDARY, [proxy] UNCONDITIONALLY this round: sklearn C=1.0 protocol.
G2-LP (INT1) validated the PRE identity ordering only (rho +0.909); there is no
post-delta bridge, so every LP statement in INT2 carries the [proxy] tag and no
gate can upgrade it (stricter than INT1).
Per row: method, encoder, dataset, seed, arm, param, knn_f1, lp_f1,
rankme_raw_bank, rankme_l2_bank, capture_l2, cC_K_l2, alpha, rankme_target,
query_oos_frac.

## 4. Gates (fail -> stop before interpretation)

G0    census: EXACT key set from the frozen manifest (dump: 484 cells; primary
      verdict set: the 165 seed-42 cells), unique keys, full 8-row grid per
      cell, no rows beyond the grid, all metric columns finite, rankme_target
      finite on calibrated arms.
G-NC  rotation sham on post features, JOINT per cell: |d kNN| AND |d LP|
      < 0.005; required on >= ceil(98%) of each panel's cells (162/165 primary),
      both seeds. Unscreened.
      PER-ROW SCREENS (apply to every estimate): capture |drift| > 0.05 OR
      placement |d cC_K_l2| > 0.5 x 0.3393 removes the row.
      TRANSPLANT ACCEPTANCE (four gates): RankMe 2%; normalized-spectrum
      profile error <= 0.02 (finite, >= 0); map-scale amplification finite, positive and <= 1e4
      (a purely shrinking map legitimately has max scale < 1); SYMMETRIC numerical-rank match |achieved - target|/target <=
      0.02 at the surgery's own mask scale (smax * max(n, d) * eps), recorded
      as 'achieved/target' in spec_rank_ratio. Result columns must equal the
      frozen FIELDS contract exactly (incl. vote_margin, vote_pos_frac,
      n_query, feasible, spec_profile_err, spec_max_amp, spec_rank_ratio).
G-POST reproduction: gated on ALL 484 cells AND the seed-42 panel, on THREE
      columns (knn_f1 vs knn_f1_hat_post, vote_margin vs vote_margin_post,
      vote_pos_frac vs vote_pos_frac_post): per-cell WORST |diff| < 0.01 each,
      plus Spearman rho >= 0.98 and median < 0.005; n_test equality per cell;
      join count must equal the panel exactly (no NaN slack; required columns
      asserted). Fail -> the dumped post features are not the audited ND12
      features; stop.
G-P   contamination, per arm: all non-rotation arms intend the SPECTRUM axis;
      unintended placement drift ratio R = median|post-L2 cC_K drift| /
      0.3393, where 0.3393 is the FROZEN placement-movement scale measured by
      INT1's shuffle family on pre features (INT2 fields no placement-intended
      family of its own; cross-experiment constant declared here, pre-data).
      Excluded iff R > 0.5 OR median |capture_l2 drift| > 0.05.
      PER-ROW capture screen: any (cell, arm) row with |capture_l2 drift| >
      0.05 leaves the estimates; screened counts and lists reported.
COVERAGE floor (rescaled from INT1): a decision-bearing screened estimate
      needs >= 75% of its stratum's cells AND >= 12 datasets (strata: pooled
      main 135 -> 102; per-method main 45 -> 34; SigLIP panel 30 -> 23).
      Below floor -> NO VERDICT. All verdicts three-state (YES/no/NO VERDICT).

## 5. Pre-registered readouts

Sign convention (frozen): per cell, S = sign(log RankMe_pre - log RankMe_post).
S > 0: real CP contracted the spectrum (reversion flattens). S < 0: real CP
expanded it (reversion sharpens; the DIET regime). Cells with |log RankMe_pre -
log RankMe_post| < 0.01 are direction-undefined and excluded from A-statistics
(count reported; no other magnitude screening).

INT2-1 FUNCTIONAL PARTICIPATION (PRIMARY, kNN): per cell A = S x
      (Y_A3 - Y_A0). Computed on the COMMON eligible intersection of the
      full/half/wrong cells (each past calibration acceptance, feasibility and
      the per-row screens), with method-level claim scoping: any method failing
      its own 34/45 eligible floor leaves the pooled panel (its stratum = NO
      VERDICT) and the pooled claim NAMES the included methods only. Verdict
      YES iff (a) 95% block CI of mean A excludes 0 positively; (b) dose
      consistency (same sign, |A_half| <= |A_full|) — an ALIGNED but dose-
      inconsistent result is NO VERDICT (mixed evidence), never "no"; (c) the
      wrong-direction CI does not lie entirely above 0 (else NO VERDICT).
      "no" requires significant anti-alignment OR a CI inside the +/-0.005
      equivalence margin; a wide null is NO VERDICT.
INT2-2 NET RESTORATION (PRIMARY, kNN; its own G-scope = methods whose
       screened FULL-arm rows pass their 75% floor, direction filter NOT
       applied; per-method G is reported independently of the A scope): G =
       mean over cells of
      (|Y_A3 - Y_pre| - |Y_A0 - Y_pre|). "Restoration/rescue" language is
      licensed IFF the dataset-block 95% CI of G lies entirely below 0, on the
      pooled seed-42 main panel. Otherwise the allowed claim stops at INT2-1's.
      The per-cell rescue fraction is a RESTRICTED DESCRIPTIVE quantity only
      (reported with its instability disclosed; no decision weight).
INT2-3 OVERSHOOT DISCRIMINATOR (secondary, kNN): R_over = mean(S x (Y_A6 -
      Y_A3)), block 95% CI. CI < 0 -> "pre level is special" (curvature);
      CI > 0 -> monotone statics continue past pre; else undecided. Reported,
      no pass threshold.
INT2-4 STRATA AND CONTRASTS (secondary, pre-registered): per-method effects
      (LeJEPA, SimCLR, DIET) on A and G at Bonferroni-3 98.33% CIs; the
      family contrast A_DIET vs mean(A_LeJEPA, A_SimCLR); the direction-
      symmetry check (mean A over S>0 cells vs over S<0 cells, difference CI);
      SigLIP panel separately (2 methods, never pooled with the DIET panel);
      MAE-BACKBONE stratum reported separately from the MAE-CP METHOD stratum
      (nd12 has no MAE-CP method rows; the distinction is terminological
      discipline: "off-sphere" refers to the backbone regime only).
INT2-5 TRANSPLANT DIAGNOSTIC (secondary): A- and G-statistics with A5 in
      place of A3. Only this arm licenses "matching the normalized pre-spectrum profile within the frozen tolerances"
      wording; reported, no threshold.
ROBUSTNESS (pre-registered, no decision weight): seeds 43/44 replication of
      INT2-1/2/4 (same scoping, real per-panel denominators); capture-screen-
      tightened rerun (thr 0.03) with its OWN scope at the same thr.
LP mirrors are NARROWED (round-9) to INT2-1/2/4 on the main panels, [proxy].
Secondary readouts (overshoot, SigLIP, transplant) carry the same coverage
floor as their panel — no single-cell CIs.

## 6. Decision map (frozen)

- INT2-1 YES and INT2-2 CI < 0: real CP's spectral motion carries functional
  weight AND its reversion moves readouts back toward pre levels (restoration
  language allowed, kNN scope).
- INT2-1 YES, INT2-2 fails: the spectral axis is live in the real
  representation, but reversion does not net-restore — the remaining CP
  changes dominate the distance to pre.
- INT2-1 no (CI excludes 0 with NEGATIVE mean, or CI inside the equivalence
  margin; dose-inconsistent-but-aligned is NO VERDICT, not no) with clean gates: the real spectral change is functionally inert (or
  anti-aligned) at the real dose — INT1's statics do not transfer to real
  post representations; the spectral path within real CP is bounded out.
- DIET stratum anti-aligned while contraction strata align (INT2-4): no
  unified participation claim; report direction-asymmetry as the finding.
- Any NO VERDICT branch: report as undecidable, never as negative.

## 7. Engineering plan (implementation AFTER freeze; TDD throughout)

- int2_features_dump.py: manifest-driven (read nd12_operator.csv keys + ckpt
  paths; assert 484), load_cp_backbone per checkpoint, INT1 dump machinery
  reused (validate_npz acceptance on resume, per-task manifest, CUDA hard
  assert, --verify-only torch-free). File name: {enc}__{ds}__{method}__{seed}
  .npz. Slurm: array over datasets, %5 concurrency, node-local staging as in
  the nd10/nd12 job.
- int2_run.py (local): one calibration-side SVD of the post bank per cell
  (map construction re-decomposes inside fit_surgery — deterministic identical
  result; an implementation detail, not a protocol quantity); arms via fit_surgery
  extended with kind="transplant" (NEW -> failing tests first) and the three
  calibrated targets; complete-cell resume via the exact-key-set check.
- int2_verdict.py: gates + readouts above; reuses INT1 helpers (gnc_joint,
  coverage_ok, tri_verdict, calibration_ok, screened_deltas, block_ci).
- External review loop (Codex) before any real data enters the verdict.

## 8. Declared limitations

- Statics on frozen post representations; CP training dynamics untouched.
- Power + transplant arms only: reversion cannot restore rotated/translated
  components (the A(kappa) rotation channel is untouched); a null bounds the
  SPECTRAL path only.
- LP is proxy-only for post deltas (no bridge; declared above).
- Y_pre comes from the public pre encoders, shared across methods/seeds.
- The overshoot discriminator is a single extra dose, not a dose ladder; a
  curvature CLAIM would need a dedicated follow-up.
- The G-P placement-movement scale is imported from INT1's pre-feature shuffle
  family (declared cross-experiment constant).

## Amendment log

- v1.1 (2026-07-21, Codex round-6, still before any INT2 data): manifest census
  LOCKED in code (484 = SimCLR 180 / LeJEPA 170 / DIET 134, seeds {42,43,44});
  wrong/over targets CAPPED to the attainable calibration range and flagged
  infeasible below half the required dose magnitude (real-data fact: 31 seed-42
  wrong targets were unattainable, DIET 26/45); INT2-1 decided on the COMMON
  eligible intersection of full/half/wrong cells with its own coverage check —
  an unavailable control arm yields NO VERDICT, never YES; INT2-2 routed
  through the per-row screens; per-row PLACEMENT screen added (|d cC_K_l2| >
  0.5 x 0.3393); "no" now requires either significant anti-alignment or a CI
  inside the +/-0.005 equivalence margin (CI straddling 0 beyond it = NO
  VERDICT; "functionally inert" wording only within the margin); exact 8-row
  grid enforced (duplicate same-arm params rejected); S defined over ALL seeds
  (cross-seed replication was silently empty); provenance gates: INT1 pre npz
  verified against int1_checksums.sha256 at run start, per-cell pre/post label-
  sequence and dim equality enforced, G-POST gated on ALL 484 cells; transplant
  leaves numerically-null directions untouched and gets an achieved-spectrum
  acceptance (2%); INT2-4 completed (per-method G, DIET-vs-SSL contrast CI,
  direction-symmetry CI, MAE-backbone stratum), INT2-5 transplant G, 0.03
  capture sensitivity, full LP mirrors.

- v1.2 body consolidation completed after Codex round-8 (this file's sections
  1-6 are the single operative protocol; round-8 additions: shard-scoped dump
  census vs full-set --verify-only, cross-seed G via the all-seed identity
  frame, full INT2-1/2/4 replication for seeds 43/44 and LP (with scoping),
  strict G-POST column/count asserts, unique-file pre-checksum verification,
  integer-label/min-sample/n_test npz checks, readable-file ckpt preflight,
  transplant numerical-rank-ratio gate 0.98 + map-scale amplification, exact
  pseudo-grid rejection in resume).
