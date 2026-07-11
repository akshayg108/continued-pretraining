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
