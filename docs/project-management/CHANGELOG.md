# SBi-FAGAT: Changelog & Audit Modification Record

## [v2.0-rc1] - PR #1 Reconciliation, EGM Hardening & Paper Tooling

### Reconciled & Merged (`5820e71` + v2 + EGM `F1–F6`)
- **`src/data/arctic_shift_vn.py`**: Preserved `PR #1` salted SHA-256 pseudonymization (`SBIF_HASH_SALT` / `.author_hash_salt.secret` + `UserWarning`) alongside `v2` `cluster_vn_cascade_topics` and `--before-epoch`.
- **`scripts/predict_live_thread.py`**: Preserved `PR #1` untrained live guard, `--allow-untrained`, and corrupt checkpoint error handling alongside `v2` 5-class discourse inference, 8D `cascade_stats`, and `node_backbone` selection; supported both `--reddit-url`/`--url` and `--gnn-ckpt`/`--checkpoint` CLI aliases.
- **`run_all.sh` & `scripts/run_phase2.sh`**: Unified `PIPELINE_VERSION=v1|v2` gating with consistent `MIN_TOTAL_REPLIES=25`, `VN_BEFORE_EPOCH=1763164800`, and `human_audit_vi_300.csv`.
- **6 EGM Hardening Fixes (`F1–F6`)**:
  - Restored `DualHeadCrossEncoder` / `extract_features.py` `node_backbone` consistency.
  - Restored LOO teacher isolation (`edge_source="teacher"`).
  - Enforced `DEFAULT_FORECAST_MIN_REPLIES=25` uniformly across ingestion, splits, and trainers.
  - Hardened finite-tensor assertions and checkpoint state validation.
- **`docs/`**: Removed broken absolute links, documented empirical rejection of RWC/Modularity on $K=20$ cascades, and reconciled all 7 files in `docs/project-management/` to match actual codebase symbols and dependency bounds.

### Added
- **`.github/workflows/ci.yml` & `.github/pull_request_template.md`**: Automated CPU `pytest` + Forbidden Binary/Data File Gate + offline smoke checks on push/PR to `main`.
- **`scripts/export_dataset_stats.py`**: Leakage-free Table 1 dataset EDA exporter and Early-$K$ Persistence Baseline evaluator (`paper/tables_dataset_eda.tex`, `paper/results_dataset_eda.json`).
- **`scripts/benchmark_efficiency.py`**: Added `format_efficiency_latex` and `--output-table` to emit `paper/tables_efficiency.tex`.
- **`paper/main.tex`**: Zero-fabrication `IEEEtran` conference manuscript skeleton wired to `paper/references.bib` and dynamic `\IfFileExists` LaTeX tables.
- **`tests/test_sota_v2_upgrades.py`**: Added regression tests covering Persistence baseline, dataset EDA LaTeX formatting, efficiency LaTeX formatting, and `--url`/`--checkpoint` CLI aliases.
