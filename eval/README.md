# `eval/` — CP geometry analysis, organized by finding

All analysis + experiment code for the manifold/geometry study of the CP benchmark. This is the
**single source** (the old top-level `CP/eval/` copy was removed). Run everything from the repo
root `continued-pretraining/` (paths below are relative to it). Narrative docs live at the project
root: `../../hypothesis/HYPOTHESES_v3.md` (current), `../../findings/FINDINGS_step{2,4}.md`.

```
eval/
├── load_results.py          ┐ shared library (imported by the analysis scripts)
├── geometry_metrics.py      │  load_results = results.xlsx → tidy long DF (+ size_canon)
├── postcp_features.py       ┘  geometry_metrics = pre/post-CP geometry; postcp_features = ckpt loader
│
├── run_postcp_sweep.sh      ┐ experiment runners (produce the data)
├── postcp_sweep.py          │  sweep = post-CP geometry over ALL cp/ ckpts (GPU, SLURM array)
├── prebuild_datasets.py     │  prebuild = build all dataset caches once (avoids runtime extraction)
├── download_imagenet_val.py │  one-off: ImageNet-val for overlap metrics
├── run_postcp_analysis.sh   ┘  CPU: merge sweep shards → F2 + F3 analysis (no GPU)
│
├── f1_position/             ── Finding 1: starting-position predictor (the initial condition)
│   ├── correlate.py            geometry→Δ per-encoder: position→ΔkNN law + kNN/FT reversal + MMD diag
│   ├── bivariate.py            position vs pre-CP baseline (two-direction partials, adj-R²)
│   └── delta_structure.py      pure Δ-structure from results.xlsx (reversal quadrants, Δ@MAX ranking)
│
├── f2_mechanism/            ── Finding 2: the two forces (spread vs collision), reconstructed
│   ├── postcp_offsphere.py     Δcv/Δuniformity/Δoverlap → ΔkNN; off-sphere norm-CV REFUTED, uniformity governs
│   └── forces_combined.py      combined two-force rank model (spread + collision; incremental R²)
│  (Exp B / SA-LP = eval/run_exp_b.py: re-eval post-CP encoders with a learned aggregation pool → outputs/exp_b/)
│
├── f3_growth/               ── Finding 3: transport dynamics over CP data size
│   └── postcp_growth_analysis.py  spread (uniformity↓) + collision (overlap↑) vs size; peak-before-collision
│
├── PLAN_iclr.md                       master plan — status (done) + remaining (Exp D, paper assembly)
└── outputs/                 all CSVs + console logs
```

## Findings ↔ files (current model: `HYPOTHESES_v3.md`, manifold transport + two forces)

| Finding (v3) | Folder | Scripts | Status |
|---|---|---|---|
| **F1** starting position predicts transfer | `f1_position/` | correlate, bivariate, delta_structure | verified (FINDINGS_step2) |
| **F2** two forces: spread (uniformity↓, helps) vs collision (overlap↑, hurts); norm-CV refuted | `f2_mechanism/` | postcp_offsphere, forces_combined (+ Exp B: `run_exp_b.py`) | verified (FINDINGS_step4 / step6); off-sphere/norm-CV **refuted** |
| **F3** transport dynamics (spread then collide with data) | `f3_growth/` | postcp_growth_analysis | verified (FINDINGS_step4) |

## Run order

```bash
cd /scratch/gs4133/zhd/CP/continued-pretraining      # (or your repo root)

# 1. Pre-CP geometry (GPU, needs stable_datasets) → outputs/geometry_15.csv
python eval/geometry_metrics.py --imagenet-dir <imagenet_val> \
    --download-dir <downloads> --processed-dir <processed> --output eval/outputs/geometry_15.csv

# 2. Prereqs for the post-CP sweep: ImageNet-val + clean dataset caches
python eval/download_imagenet_val.py --save-dir <.../data/imagenet_val>   # one-off (HF login)
srun ... python eval/prebuild_datasets.py                                 # build all caches once

# 3. Post-CP geometry sweep (GPU, SLURM array; N = max array index + 1)
sbatch --array=0-11 eval/run_postcp_sweep.sh        # → outputs/postcp_sweep_*.csv

# 4. Analysis (CPU, no GPU): merge shards → F2 (off-sphere) + F3 (growth)
bash eval/run_postcp_analysis.sh                    # → postcp_sweep.csv, postcp_offsphere.csv, postcp_growth_analysis.csv

# 5. F1 (anywhere, pandas): position law, bivariate, Δ-structure
python eval/f1_position/correlate.py --geometry eval/outputs/geometry_15.csv
python eval/f1_position/bivariate.py
python eval/f1_position/delta_structure.py

# 6. Exp B (SA-LP) — re-evaluate post-CP encoders with a learned aggregation pool (one job per dataset)
python eval/run_exp_b.py --ckpt-root <.../ckpts/cp> --cache-dir <.../data> --datasets <ds> --out eval/outputs/exp_b/<ds>.csv
# then: recovery fraction in eval/outputs/sa_lp_recovery.csv (aggregation-failure vs information-loss verdict)
```

## Conventions
- The analysis scripts in `f{1,2,3}_*/` add `eval/` to `sys.path` (for `load_results`) and set
  `ROOT` = repo root, so they run from any cwd. The runners use `python` (cluster conda env);
  locally substitute `python3`.
- Geometry on a ≤5000-sample stratified subset; Δ at MAX (mean over LeJEPA-CP+SimCLR-CP by default).
- `uniformity_t2`: more negative = more spread (Wang–Isola). `size_canon` reconciles MAX-label drift.

## Dependencies
- `f{1,2,3}_*/`, `load_results.py`, `run_postcp_analysis.sh`: pandas, numpy, scipy, scikit-learn, openpyxl.
- `geometry_metrics.py`, `postcp_features.py`, `postcp_sweep.py`, `prebuild_datasets.py`: also torch,
  timm, torchvision, datasets, `stable_datasets` (vendored under `../stable-datasets`, `pip install -e`).
