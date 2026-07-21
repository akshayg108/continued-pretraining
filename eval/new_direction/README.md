# new_direction/ — ND1-ND4 experiments (theory sweep 2026-07-09)

Tests derived from `papers/new_direction/NEW_DIRECTION.md` §H. Each experiment is a GPU
pass (cluster, slurm under `run/slurm/new_direction/`) + a CPU adjudicator (local, reads
the returned CSVs). Pre-registered readouts are declared in each adjudicator's docstring
(written 2026-07-10, before any data existed).

| # | Question | GPU pass -> CSV | Adjudicator |
|---|---|---|---|
| ND1 | A4: is the SigLIP-2 level-channel sign flip a metric artifact (Tsitsulin template) or a model property? | `nd1_precp_spectral.py` -> `nd1_precp_spectral.csv` | `nd1_verdict.py` |
| ND2 | F2: do the two forces have the Li et al. spectral signature (expansion / consolidation) along the size axis? | `nd2_spectral_sweep.py` -> `nd2_spectral_sweep.csv` | `nd2_verdict.py` |
| ND3 | A3: is DTD augmentation-dominated along its class-discriminative directions (Jing Thm 1)? | `nd3_augvar.py` -> `nd3_augvar.csv` | `nd3_verdict.py` |
| ND4 | A2: does the method head buffer spectral reshaping away from the backbone (projector-as-buffer), tracking the gate? | `nd4_projector_spectra.py` -> `nd4_projector_spectra.csv` | `nd4_verdict.py` |
| ND6 | Round-2 theory (NEW_DIRECTION_R2.md): does task-model alignment C(rho) give a universal level law and explain the SigLIP coupling anomaly; does the omniscient LP risk beat rankme/alpha; is hubness a universal kNN-level correlate? | `nd6_alignment.py` -> `nd6_alignment.csv` | `nd6_verdict.py` |
| ND7 | The story's spine: does CP move class information UP the ranking (Delta cC_K on MAX ckpts), does placement mediate the rank->kNN link, and is there a placement attractor? | `nd7_placement.py` -> `nd7_placement.csv` | `nd7_verdict.py` (needs nd6 + nd1 CSVs local) |
| ND8 | Round-3 (NEW_DIRECTION_R3.md): does hubness-corrected/centered overlap (or Sun kNN-distance) cure the position feature's shape-entanglement, and does swapping it into the frozen tool beat v1 on the SigLIP + ViT-L holdouts? | `nd8_overlap_upgrade.py` -> `nd8_overlap.csv` (needs ImageNet-val bank; 5 encoders incl. DINOv3L) | `nd8_verdict.py` (algorithms TDD-tested in `test_nd8_position_metrics.py`; ND8-2 scoring repaired 2026-07-15, CORRECTIONS #13) |
| ND9 | Operator-transport theory #1 (theory_unification_2026-07-15): is the SigLIP-2 alignment deficit a CAPTURE deficit (total task power in the span — the factor cC(rho) normalizes away) or conditional-placement-only; does ridge-weighted accessibility A(kappa) serve the LP level? | `nd9_capture.py` -> `nd9_capture.csv` (pre-CP, 4 encoders x 15 ds, nd6 protocol) | `nd9_verdict.py` (metrics TDD-tested in `test_nd9_task_operator.py`) |
| ND10 | Operator-transport theory #2: decompose the pre->post accessibility change into eigenvalue flow (dA_spec) vs basis rotation (dA_rot) on matched samples — the operator version of "rank thermostat + placement tide"; are they independent trackers of dknn? | `nd10_operator_transport.py` -> `nd10_operator.csv` (MAX ckpts, nd7 population) | `nd10_verdict.py` (ND10-1..3) |
| ND11 | Operator-transport theory #3 (the T3 slot): does a LOCAL cosine-kNN graph quantity (neighbour label purity / graph placement) carry the kNN benefit sign that global spectra provably cannot? | same pass as ND10 (graph columns in `nd10_operator.csv`) | `nd10_verdict.py` (ND11-1..3) |
| ND12 | REDEFINED 2026-07-16 (Codex critique adopted; frozen in `ND12_PREREG.md`): (a) TRUE vote operator — test-to-train, inverse-distance k=20, exact zero_shot_eval protocol — as evaluator-reproduction gate + bridge (role-restricted, never evidence); (b) PRIMARY = leave-one-dataset-out incremental value of graph placement (m20b frozen) over global scalars {d_rankme, d_cC_K}; gap-closing demoted (circularity). | extended `nd10_operator_transport.py` (test-split extraction + vote columns) -> `nd12_operator.csv` | `nd12_verdict.py` (G1/G2 gates + ND12-1..3) |

| INT1 | Intervention-1 (post-ND causal phase, `INT1_PREREG.md` consolidated v1.6): counterfactual surgery on frozen pre-CP features — rotation negative control / spectral power (incl. CP-calibrated alpha) / graded iso-spectral eigendirection-scale reassignment / combo interaction arm — does geometry causally matter for kNN/LP at fixed information? | `int1_features_dump.py` -> `int1_features/*.npz` (GPU dump; surgery + eval run LOCALLY: `int1_run.py` -> `int1_results.csv`) | `int1_verdict.py` (G0/G-NC/G-P/G2-LP gates + INT1-1..5, three-state verdicts; TDD in `test_int1_surgery.py`, `test_int1_dump.py`, `test_int1_run_helpers.py`, `test_int1_verdict_helpers.py`) |
| INT2 | Intervention-2 (`INT2_PREREG.md` v1.0, frozen 2026-07-21): RankMe-matched spectral REVERSION on real post-CP features — does real CP's spectral change carry functional weight? 484-ckpt manifest dump (nd12_operator.csv; SimCLR 180/LeJEPA 170/DIET 134), seed-42 primary panel 165; arms identity/rotation/half/full/wrong-direction/overshoot/transplant; INT2-1 directional alignment + INT2-2 net restoration (rescue fraction demoted to descriptive); DIET = full member (expansion direction). | `int2_features_dump.py` -> `int2_features/*.npz` (GPU; ckpts via `load_cp_backbone`) then LOCAL `int2_run.py` -> `int2_results.csv` | `int2_verdict.py` (G0/G-NC/G-POST/G-P gates + INT2-1..5, three-state; TDD in `test_int2_helpers.py`) |

Local (CPU, no cluster): `f5_label_aware_challenge.py` — replication of the v3 re-audit's
label-aware feature challenge on the SigLIP holdout (run 2026-07-16: v1 13/15, cC_K-only
10/15, knn_pre-only 8/15, v1+cC_K 13/15 — reproduces the re-audit exactly; the T2 slot is
closed: even label-aware placement does not beat the label-free tool).

Shared metric definitions: `spectral_metrics.py` (rankme / alpha-ReQ / coherence / VCI,
implemented verbatim from the papers in `papers/new_direction/`; run it directly for the
synthetic self-test). ND6's omniscient-risk implementation (Wei 2022 Eq. 1+4) is validated
against Monte-Carlo ridge regression: `python eval/new_direction/nd6_alignment.py --selftest`.
ND6 runs like ND1 (array 0-14, no ckpts, ~10-20 min/task); its verdict additionally needs
nd1_precp_spectral.csv (baselines) and results.xlsx (knn_pre / lp_pre levels).

## Run order

1. `sbatch run/slurm/new_direction/nd1_precp_spectral.sh`   (array 0-14, ~1-2 h/task; no ckpts)
2. `sbatch run/slurm/new_direction/nd3_augvar.sh`           (array 0-14, ~2-4 h/task; no ckpts)
3. `sbatch run/slurm/new_direction/nd4_projector.sh`        (array 0-14, ~4-8 h/task; MAX ckpts)
4. `sbatch run/slurm/new_direction/nd2_spectral_sweep.sh`   (array 0-15, the big one: all
   pretrained cp ckpts; ~12-20 h/shard. Queue with `--array=0-15%6` if needed.)

ND1 must finish before adjudicating ND2/ND4 (they join its pre-CP baseline), but the GPU
passes are independent and can run concurrently. Each pass is resumable (rerun the same
array task; done rows are skipped).

## After each array finishes (login node)

Concat shards (the per-experiment command is in the slurm header comment), copy the four
CSVs into local `eval/outputs/`, then run the adjudicators locally:

```
python eval/new_direction/nd1_verdict.py
python eval/new_direction/nd2_verdict.py
python eval/new_direction/nd3_verdict.py
python eval/new_direction/nd4_verdict.py
```
