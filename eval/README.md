# eval/ — analysis layer, organized by the paper's findings

Data lives in `outputs/` (all CSVs). Shared modules stay at this root; analysis and
producer scripts live in one folder per finding (numbering matches story.docx /
results_full.xlsx: F1=position law, F2=transport forces, F3=size dynamics,
F4=encoder gate, F5=decision score, F6=aggregation failure, F7=exploratory packing).

## utils/ (shared modules + cross-finding tools)
- `utils/geometry_metrics.py`   loaders + unlabeled geometry (uniformity/overlap/MMD/norm-CV)
- `utils/geometry_class.py`     labeled class + spectral geometry (CDNV/margin/... ; Exp E)
- `utils/postcp_features.py`    checkpoint loading (`load_cp_backbone`)
- `utils/layerwise_geometry.py` per-block feature taps + internal kNN (Exp I pre side)
- `utils/postcp_class_sweep.py` MAX-checkpoint discovery (shared) + Exp H class sweep
- `utils/load_results.py`       results.xlsx -> long behavior table
- `utils/final_integration.py`  master refresh: folds test4 reruns, recomputes headline stats
- `utils/build_results_full.py` builds ../../results_full.xlsx from outputs/

## Per-finding folders
- `F1_position_law/`      correlate/bivariate/delta_structure, stats_pass (FDR), dft_snr
- `F2_forces/`            postcp_sweep (producer), forces_combined, postcp_offsphere,
                          suppressor_robustness, transport_field + transport_law (Exp J)
- `F3_dynamics/`          postcp_growth_analysis, merge_rest_geometry
- `F4_gate/`              bilinear_law, layerwise_postcp + layerwise_law (Exp I),
                          mae_sa_geometry (Exp F), class_forces (d_cdnv), second_axis_fdr
- `F5_decision_score/`    predictor (frozen rule), preregister_siglip (2026-06-19 freeze),
                          margin_predictor, geometry_vitL + vitl_score (held-out scale)
- `F6_aggregation/`       run_exp_b (SA-recovery)
- `F7_packing_exploratory/` heterogeneity_probe (dose control), correlate_second_axis,
                          margin analyses — exploratory grade only

Path convention: scripts inside F-folders add `eval/utils/` to sys.path, so shared imports
(`from geometry_metrics import ...`) resolve; run them from the repo root.
Archived (superseded/one-off verification): see ../_archive/eval/ at the project root.
