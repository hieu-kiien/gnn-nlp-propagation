# SBi-FAGAT: Project Status & Module Verification Matrix

**Branch:** `main` (Unified 2D + EGM 5-Goldfish Audit Hardening complete)  
**Test Suite:** 15 test modules in `tests/` (see [TEST_REPORT.md](TEST_REPORT.md) for latest observed run)

---

## 1. Module Implementation & Verification Status

| Module / Path | Purpose & Function | Status | Verification | Audit Remarks |
| :--- | :--- | :--- | :--- | :--- |
| `src/data/schema.py` | Arrow/Parquet contract, Pydantic models (`NodeRecord`, `EdgeRecord`), zero-leakage & DAG checks | Implemented | Unit tested (`test_schema.py`) | Enforces UTC timestamps, `FORBIDDEN_LEAKAGE_COLUMNS`, acyclic trees, and temporal order. |
| `src/data/pheme_loader.py` | PHEME Figshare downloader/parser (`PHEME_veracity.tar.bz2`), MD5/SHA-256 verification | Implemented | Unit tested (`test_pheme_tree.py`) | Reconstructs trees (`source-tweets`, `reactions`, `structure.json`); inductive event grouping. |
| `src/data/arctic_shift_vn.py` | Arctic Shift Reddit VN parser, salted SHA-256 pseudonymization, `--before-epoch`, `cluster_vn_cascade_topics` | Implemented | Unit tested (`test_vn_tree.py`) | Supports `SBIF_HASH_SALT` and `.author_hash_salt.secret` with `UserWarning` fallback. |
| `src/data/pyg_dataset.py` | `EarlyCascadeDataset`, `CascadeData`, `DEFAULT_FORECAST_MIN_REPLIES=25`, canonical `topic_splits.json`, 10D `struct_x`, 8D `cascade_stats` | Implemented | Unit tested (`test_no_leakage.py`, `test_egm_fixes.py`, `test_step3_training_and_audit_fixes.py`) | Enforces $K \le 20$ truncation, `future_only` $y_{\text{polar}}$, LOO `edge_source="teacher"`, and strict `topic_splits.json` adherence even on in-memory/$K=50$ subsets. |
| `src/nlp/discourse_labels.py` | Canonical 5-class discourse label schema & stance collapse mappings | Implemented | Unit tested (`test_kaggle_t4x2_v2_pipeline.py`) | Shared across live inference and Kappa audit scripts. |
| `src/nlp/dual_edge_scorer.py` | `NLIStanceTeacherScorer` (Symmetric 2-Hypothesis Verbalizer, 384-tok reply budget), `HostilityTeacherScorer` (`UIT-ViHSD` class-weighted / `twitter-roberta-base-offensive`), segment-isolated `DualHeadCrossEncoder`, blind audit CSV exporter | Implemented | Unit tested (`test_nlp_and_features.py`, `test_sota_v2_upgrades.py`, `test_step3_training_and_audit_fixes.py`) | Unified social normalization (`vi`/`en` URL + whitespace cleanup), `BCEWithLogitsLoss` + $T=2.0$ KL, `--sanity-check-shuffle`. |
| `src/nlp/extract_features.py` | Offline frozen **Ancestor-Conditioned SpanPool** (`uitnlp/visobert` / `cardiffnlp/twitter-roberta-base`) & 2D edge attribute exporter (`nlp_features.pt`) | Implemented | Unit tested (`test_nlp_and_features.py`, `test_step3_training_and_audit_fixes.py`) | Strict finite-tensor checks; skips redundant deep Student forward pass when full teacher `root_s_uv` coverage is cached. |
| `src/models/sbi_fagat.py` | Dual-channel Dir-GNN `SBiFAGAT`, `StanceGuidedFAConv`, `GlobalAttentionReadout`, 64-hop BFS root-aligned camp dipole readout, `extract_edge_gates` | Implemented | Unit tested (`test_overfit_batch.py`, `test_overfit_20.py`, `test_step3_training_and_audit_fixes.py`) | Default `hidden_dim=64`, 2D `[s_uv, h_uv]` edges, Polarity + Excitation channels, TD–BU Dir-GNN mixing. |
| `src/models/baselines.py` | `TwoStreamNodeStem`, `MultiTaskHead` (with `y_polar_logit`), & 7 fair neural baselines | Implemented | Unit tested (`test_overfit_batch.py`, `test_overfit_20.py`, `test_step3_training_and_audit_fixes.py`) | Uniform `hidden_dim=64`, matched `2*hidden_dim=128` heads, fair topology masking in `TextMLP`. |
| `src/training/metrics.py` | Huber + KL-calibrated `BCEWithLogitsLoss` + Smooth-L1 + Cross-Entropy, evaluation metrics | Implemented | Unit tested (`test_overfit_batch.py`, `test_step3_training_and_audit_fixes.py`) | Exact zero minimum at $\sigma(z)=y$ and train-only target scale balance. |
| `src/training/trainer.py` | Train-only target scaling & bias init, linear warmup + `ReduceLROnPlateau` + EMA (`0.985`), Early Stopping (`task_loss`), best-val seed selection across 5 seeds | Implemented | Unit tested (`test_overfit_batch.py`, `test_sota_v2_upgrades.py`, `test_step3_training_and_audit_fixes.py`) | Enforces `OFFICIAL_SEEDS = [42, 100, 2024, 777, 999]` and excludes non-learned baselines from comparator selection. |
| `src/training/reporting.py` | LaTeX / Markdown table exporters & serialized JSON result summaries | Implemented | Unit tested (`test_pipeline_and_viz.py`) | Formats multi-seed `mean ± std` with Holm-Bonferroni markers. |
| `src/training/auto_loop.py` | Opt-in validation residual diagnostic & candidate audit loop (`hidden_dim=64`, `edge_dim=2`) | Implemented | Unit tested (`test_kaggle_t4x2_v2_pipeline.py`, `test_step3_training_and_audit_fixes.py`) | Never overrides fair fixed-hyperparam benchmark when used in default runner mode. |
| `src/utils/seed.py`, `text.py`, `visualize.py` | Deterministic seeding, NFC normalization, UMAP / $K$-curve / heatmap vector figures | Implemented | Unit tested (`test_pipeline_and_viz.py`) | Consistent preprocessing and `.pdf`/`.svg` figure exports. |

---

## 2. Scripts, Notebooks & Publication Tooling

| Path | Purpose | Status | Remarks |
| :--- | :--- | :--- | :--- |
| `scripts/kaggle_parallel_runner.py` | Dual T4 unified 2D orchestrator (`member1_vn`, `member2_en`, `model_nlp_only`, `model_gnn_only`) | Verified | `--smoke-test` exits 0 offline; fair fixed `--hidden-dim 64` (`auto_loop` opt-in via `--run-auto-loop`); strips `*_machine_key.csv`, `*.secret`, `.env` from zip bundles. |
| `scripts/predict_live_thread.py` | Live thread inference CLI | Verified | Blocks untrained live inference unless `--allow-untrained` or `--demo-offline`. |
| `scripts/benchmark_efficiency.py` | Complexity profiler (`#Params`, `ms/cascade`, peak RAM/VRAM) + LaTeX export | Verified | Default `hidden_dim=64`, `edge_dim=2`. |
| `scripts/export_dataset_stats.py` | Dataset EDA (`Table 1` LaTeX) + Early-$K$ Persistence Baseline | Verified | Reports full size + polarization persistence metrics cleanly. |
| `scripts/demo_research_walkthrough.py` | Offline CPU walkthrough (`<5s`) | Verified | Demonstrates `StandardGCN` vs. `SBiFAGAT`. |
| `scripts/run_phase2.sh`, `run_all.sh` | Unified 2D Phase 2 & Phase 3 pipelines | Verified | Default `hidden_dim=64`, `--edge-dim 2`, `--min-total-replies 25`. |
| `scripts/compute_kappa.py`, `audit_vn_50_trees.py` | Pair-level (300 pairs) & thread-level (50 VN Test trees) Cohen's Kappa, Machine-vs-Gold & LaTeX table exporter | Verified | Supports `--strict-adjudication` and `--output-table paper/tables_human_audit.tex`. |
| `scripts/assemble_paper_artifacts.py` | Multi-track EDA/efficiency LaTeX table merger & unescaped-`_` syntax validator | Verified | Produces unified `tables_dataset_eda.tex`, `tables_efficiency.tex`, and `assembly_manifest.json`. |
| `scripts/run_experiments.py`, `run_ablation.py`, `run_upfd_extension.py` | 5-seed benchmarks (+ Train-mean `Early-K Persistence`), fixed-horizon $K$-ablation + matched `BiEdgeGATv2`, UPFD structural transfer | Ready | Await full GPU execution in Step 4. |
| `notebooks/*.ipynb` | Kaggle dual-T4 Member 1/2 notebooks | Ready | Mirror `scripts/kaggle_parallel_runner.py` unified 2D workflow. |
| `paper/main.tex`, `paper/references.bib` | `IEEEtran` manuscript synced to unified 2D architecture | Ready | Tables populated via `\IfFileExists{tables_*.tex}` (zero fabricated results). |
