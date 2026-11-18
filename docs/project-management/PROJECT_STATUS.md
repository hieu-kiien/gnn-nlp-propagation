# SBi-FAGAT: Project Status & Module Verification Matrix

**Baseline Branch:** `main` (reconciled post-PR #1 `5820e71` + local v2 + 6 EGM fixes)  
**Test Suite:** 12 test modules in `tests/` (see [TEST_REPORT.md](TEST_REPORT.md) for latest observed run)

---

## 1. Module Implementation & Verification Status

| Module / Path | Purpose & Function | Status | Verification | Audit Remarks |
| :--- | :--- | :--- | :--- | :--- |
| `src/data/schema.py` | Arrow/Parquet contract, Pydantic models (`NodeRecord`, `EdgeRecord`), zero-leakage & DAG checks | Implemented | Unit tested (`test_schema.py`) | Enforces UTC timestamps, `FORBIDDEN_LEAKAGE_COLUMNS`, acyclic trees, and temporal order. |
| `src/data/pheme_loader.py` | PHEME Figshare downloader/parser (`PHEME_veracity.tar.bz2`), MD5/SHA-256 verification | Implemented | Unit tested (`test_pheme_tree.py`) | Reconstructs trees (`source-tweets`, `reactions`, `structure.json`); inductive event grouping. |
| `src/data/arctic_shift_vn.py` | Arctic Shift Reddit VN parser, salted SHA-256 pseudonymization, `--before-epoch`, `cluster_vn_cascade_topics` | Implemented | Unit tested (`test_vn_tree.py`) | Supports `SBIF_HASH_SALT` and `.author_hash_salt.secret` with `UserWarning` fallback. |
| `src/data/pyg_dataset.py` | `EarlyCascadeDataset`, `CascadeData`, `DEFAULT_FORECAST_MIN_REPLIES=25`, canonical `topic_splits.json`, 8D `cascade_stats` | Implemented | Unit tested (`test_no_leakage.py`, `test_egm_fixes.py`, `test_round2_research_fixes.py`) | Enforces $K \le 20$ truncation, `future_only` $y_{\text{polar}}$, and `edge_source="teacher"` LOO isolation. |
| `src/nlp/dual_edge_scorer.py` | Hostility + Stance teachers (`legacy_nli`), `DualHeadCrossEncoder`, blind audit CSV exporter, LOO CV | Implemented | Unit tested (`test_nlp_and_features.py`, `test_sota_v2_upgrades.py`) | Supports 2D/4D edge distillation and `--sanity-check-shuffle` anti-cheating check. |
| `src/nlp/qwen_teacher.py` | `Qwen/Qwen3.8-27B` QLoRA 1-pass G-Eval 5-class discourse teacher (`[s_uv, h_uv, r_uv, c_uv]`) | Implemented | Unit tested (`test_kaggle_t4x2_v2_pipeline.py`) | Includes FP16/4-bit NF4 loader, multimodal vision-tower freeze, and behavioral prototypes. |
| `src/nlp/extract_features.py` | Offline frozen node (`[N, 770]`) and edge attribute exporter (`nlp_features.pt`) | Implemented | Unit tested (`test_nlp_and_features.py`, `test_pipeline_and_viz.py`) | Strict finite-tensor checks; separates NLP extraction from GNN training. |
| `src/models/sbi_fagat.py` | Proposed `SBiFAGAT`, `StanceGuidedFAConv`, `GlobalAttentionReadout`, `extract_edge_gates` | Implemented | Unit tested (`test_overfit_batch.py`, `test_sota_v2_upgrades.py`, `test_egm_fixes.py`) | Supports 2D/4D edges, `tanh`/`sigmoid` gate, TD/BU, `cascade_stats`, and `beta-NLL` uncertainty. |
| `src/models/baselines.py` | 7 fair baselines (`TextMLP`, `TemporalStanceMLP`, `StandardGCN`, `StandardGATv2`, `EdgeGATv2`, `FAGCNBaseline`, `BiGCN`) & `MultiTaskHead` | Implemented | Unit tested (`test_overfit_batch.py`, `test_egm_fixes.py`) | Uniform `hidden_dim=128` and shared `MultiTaskHead`. |
| `src/training/metrics.py` | Huber + CE + $\beta$-NLL loss (`compute_multitask_loss`), RMSE, MAE, $R^2$, Pearson $r$, Spearman $\rho$, Macro-F1, AUROC | Implemented | Unit tested (`test_sota_v2_upgrades.py`, `test_round2_research_fixes.py`) | Detached residual weighting for $\beta$-NLL uncertainty calibration. |
| `src/training/trainer.py` | Early Stopping (`task_loss`), 5-seed runner (`run_multi_seed_benchmark`), paired t-test & Holm-Bonferroni | Implemented | Unit tested (`test_overfit_batch.py`, `test_egm_fixes.py`) | Enforces `OFFICIAL_SEEDS = [42, 100, 2024, 777, 999]`. |
| `src/training/reporting.py` | LaTeX / Markdown table exporters & serialized JSON result summaries | Implemented | Unit tested (`test_pipeline_and_viz.py`) | Formats multi-seed `mean ± std` with significance markers. |
| `src/training/auto_loop.py` | Budgeted hyperparameter search loop (`T4 x 2` / automated tuning) | Implemented | Unit tested (`test_kaggle_t4x2_v2_pipeline.py`) | Guarded parameter search over validation splits. |
| `src/utils/seed.py` | Deterministic seeding (`OFFICIAL_SEEDS`) | Implemented | Unit tested (`test_overfit_batch.py`) | Configures Python, NumPy, and PyTorch RNGs. |
| `src/utils/text.py` | Unicode NFC & shared text normalization helpers | Implemented | Audited | Used identically across ingestion and NLP extractors. |
| `src/utils/visualize.py` | UMAP embeddings, early-$K$ curves, cascade gate heatmaps (`paper/figures/`) | Implemented | Unit tested (`test_pipeline_and_viz.py`) | Exports `.pdf` and `.svg` vector graphics. |

---

## 2. Scripts, Notebooks & Publication Tooling

| Path | Purpose | Status | Remarks |
| :--- | :--- | :--- | :--- |
| `scripts/kaggle_parallel_runner.py` | Dual T4 orchestrator (`member1_vn`, `member2_gnn`, `t4x2_all`) | Verified | `--smoke-test` exits 0 offline. |
| `scripts/predict_live_thread.py` | Live thread inference CLI (`--reddit-url`/`--url`, `--gnn-ckpt`/`--checkpoint`) | Verified | Blocks untrained live inference unless `--allow-untrained` or `--demo-offline`. |
| `scripts/benchmark_efficiency.py` | Complexity profiler (`#Params`, `ms/cascade`, peak RAM/VRAM) + LaTeX export | Verified | Uses controlled tensor shapes strictly for speed/memory profiling (`--output-table`). |
| `scripts/export_dataset_stats.py` | Dataset EDA (`Table 1` LaTeX) + Early-$K$ Persistence Baseline | Verified | Zero-leakage check on `y_polar(G_K)` vs. future `y_polar`. |
| `scripts/demo_research_walkthrough.py` | Offline CPU walkthrough (`<5s`) | Verified | Educational sanity check comparing `StandardGCN` vs. `SBiFAGAT`. |
| `scripts/run_phase2.sh`, `run_all.sh` | Profile-gated shell pipelines (`PIPELINE_VERSION=v1|v2`) | Reconciled | Default `v1` (`--edge-dim 2`, `--min-total-replies 25`); `v2` opt-in (`--edge-dim 4`). |
| `scripts/compute_kappa.py` | Cohen's Kappa & gold human audit evaluator (`human_audit_300.csv` / `human_audit_vi_300.csv`) | Ready | Evaluates inter-annotator & teacher-vs-gold agreement. |
| `scripts/run_experiments.py`, `run_ablation.py`, `run_upfd_extension.py` | 5-seed benchmarks, $K$-horizon ablation, and UPFD evaluation | Ready | Await real extracted `.pt` feature banks on GPU/Kaggle. |
| `notebooks/*.ipynb` | Kaggle dual-T4 & Member 1/2 runnable notebooks | Ready | Mirrors `scripts/kaggle_parallel_runner.py` workflows. |
| `paper/main.tex`, `paper/references.bib` | `IEEEtran` manuscript skeleton & 25 verified BibTeX references | Ready | Tables populated via `\IfFileExists{tables_*.tex}` (zero fabricated results). |

---

## 3. Repository Data State on Git

| Directory | Tracked on Git | Purpose |
| :--- | :--- | :--- |
| `data/raw/` | `.gitkeep` only | Destination for downloaded PHEME archive & raw VN Reddit dumps |
| `data/interim/` | `.gitkeep` only | Validated `nodes.parquet` / `edges.parquet` and audit CSVs (local `vn_sample5/` gitignored) |
| `data/processed/` | `.gitkeep` only | Pre-extracted `nlp_features.pt` and canonical `topic_splits.json` |
| `checkpoints/` | `.gitkeep` only | Trained GNN / Student / QLoRA checkpoints (gitignored) |
| `results/` | `.gitkeep` only | 5-seed experimental JSONs, LaTeX tables, and paired test reports (gitignored) |
