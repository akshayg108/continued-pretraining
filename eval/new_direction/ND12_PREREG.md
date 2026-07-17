# ND12 pre-registration (FROZEN 2026-07-16, before any ND12 data exists)

Redefined after the Codex critique of the first draft (2026-07-16): the draft's
"evaluator-matched" graph was not evaluator-matched (1-d weights vs the evaluator's
inverse-distance; symmetric within-sample graph vs directed test-to-train voting;
one-hot energy balancing vs macro-F1), and closing the purity gap is NOT a legitimate
primary target (quantities closer to the evaluator are trivially closer to its output).

## Instruments (one 12h MAX-checkpoint pass, extended nd10 driver)

I1  VOTE-OPERATOR PROXY (round-3 amendment: renamed from "true vote operator" — the
    bank is the STANDARDIZED <= 5000 seed-42 subsample shared across methods/seeds,
    NOT each run's (size, seed) training subset; this is a classifier-protocol-matched
    standardized proxy, and G2 measures empirically how faithful it is):
    train-split features = bank, test-split features = queries (test subsampled to
    <= 2000 stratified, seed 42; bank capped by the standard <= 5000 protocol);
    sklearn KNeighborsClassifier(n_neighbors=min(20, n_bank), metric="cosine",
    weights="distance") — the exact protocol of
    stable_cp/evaluation/zero_shot_eval.py::knn_evaluate. Recorded:
      knn_f1_hat   recomputed macro-F1 (torchmetrics-equivalent macro over all classes)
      vote_margin  mean over queries of [weighted vote share of the true class minus
                   the largest other-class share]; sklearn zero-distance convention.
    ROLE RESTRICTION (declared): vote-margin quantities are KIN to the outcome. They
    serve ONLY (a) the evaluator-reproduction gate and (b) descriptive bridge lines.
    They are NOT features in any evidence-bearing model and NOT part of any criterion.
I2  GRAPH PLACEMENT (secondary operator-placement metric, unchanged from the draft):
    graph_cC_K on the symmetric cosine-kNN graph; FROZEN evidence variant = m20b
    (k=20, unweighted adjacency, class-balanced label energy). m20wb and legacy k10
    are sensitivity columns only — no picking the best-performing variant.

## Gates

G1  nd10 consistency: legacy graph_cC_K_pre joined on the full cell key reproduces
    nd10_operator.csv (Spearman > 0.99 AND max |diff| < 0.02). Fail -> stop.
G2  EVALUATOR REPRODUCTION (round-4 clarification: the proxy's pre side is
    method/seed-invariant, so the declared unit is the 45 (encoder, dataset)
    pre-CP LEVELS — pooled n=45, per-encoder n=15): pooled Spearman
    rho(knn_f1_hat_pre, knn_pre from results.xlsx) > 0.95, per-encoder rho > 0.90 (level
    reproduction up to bank/query subsampling; mean |diff| reported, no threshold —
    absolute F1 shifts under subsampling). Fail -> the vote operator is NOT
    evaluator-faithful; stop before interpreting anything built on it.

## Primary judgment (the missing held-out increment)

ND12-1  LEAVE-ONE-DATASET-OUT INCREMENTAL VALUE. Unit = dataset block (15 folds; all
    9 main-grid (method, encoder) cells of the held-out dataset are predicted per
    fold; cells are never split across train/test). Model = LogisticRegression
    (max_iter=1000, default C) on z-scored features; target = sign(dknn > 0) at the
    (method, encoder, dataset) cell level (per-method dknn from results.xlsx is_max).
    Feature sets (FROZEN):
      GLOBAL       d_rankme, d_cC_K, dA_spec, dA_rot   (round-3 amendment: includes
                   the strongest tested global quantities — dA_spec alone reaches
                   ~80.7% sign agreement; dA_total = spec + rot is not duplicated.
                   PASS therefore means "beyond ALL tested globals", not "beyond
                   rank + feature placement")
      GLOBAL+GRAPH GLOBAL + d graph_cC_K_m20b
      GRAPH-ONLY   d graph_cC_K_m20b
    Criterion (round-3 amendment, conjunctive): pooled LODO sign-accuracy
    (GLOBAL+GRAPH) - accuracy(GLOBAL) >= +0.02 AND dataset-block bootstrap
    (2000 draws, seed 0) P(diff > 0) >= 0.90. Population gate: exactly 135 cells =
    15 datasets x 9; fold-level wins reported.
    PASS  -> claimable: "the graph operator carries held-out sign information beyond
             the pre-registered four-feature linear global baseline" (round-4
             wording: NOT "beyond all global quantities" — the baseline is a
             specific linear model over four features).
    Key census gate (round-4): before any seed-averaging, nd12 raw keys must be
    one-to-one with the ND10 pass's 484 unique keys (no duplicates, exact set
    equality) — a missing seed must fail loudly, not vanish into a cell mean.
    FAIL  -> only claimable: "graph placement has stronger marginal correlation with
             dknn than feature placement" — nothing about irreducibility or locality.

## Secondary / sensitivity (no criteria, no cherry-picking)

ND12-2  descriptive bridge: pooled and per-encoder sign agreement + correlation of
    d vote_margin with dknn (expected near-ceiling BY KINSHIP — reported to quantify
    the bridge, never cited as evidence).
ND12-3  sensitivity panel: the ND12-1 comparison re-run with m20wb and legacy k10 in
    place of m20b, and k=10/50 vote operators if collected — all reported, none
    criterion-bearing.

## Declared limitations

- The graph subsample (n_max=2000, seed 42) is NON-STRATIFIED — retained as a
  protocol-continuity choice with ND10 (round-4 note: G1 only checks the legacy k10
  columns, so stratifying the m20b variant WAS technically possible; we keep one
  shared subsample for simplicity and declare it). Known, accepted.
- Test split availability is validated FAIL-FAST at shard start (all loaders built
  before the checkpoint loop). galaxy10 ships only a train asset (round-4 confirmed):
  its vote bank AND query come from stable_cp.data.datasets._split_single_dataset
  (80/10/10) — the real evaluator's own split FUNCTION, but with the STANDARDIZED
  seed 42 (round-5 correction: training runs passed their per-run seed 1/2/3, so
  results.xlsx's galaxy10 numbers average three different partitions; our fixed-42
  split matches none of them exactly — consistent with the standardized-proxy
  convention, and G2 is rank-level). Bank/query disjoint by construction; the G1
  full-cloud loader is untouched.

## What ND12 cannot conclude regardless of outcome (declared)

No causal statement (no intervention); no "sign lives only locally"; no upgrade of
purity-family quantities to evidence. After ND12, the next step is intervention
(move the cloud), not further correlational metrics.
