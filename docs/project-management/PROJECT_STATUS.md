# SBi-FAGAT: Project Status & Module Verification Matrix

**Last Audit Date:** Current Session  
**Baseline Git Commit:** `f4be964b0307aee5032223760b3e9d1b3d39df8f`  
**Test Suite Status:** 79/79 Passed across 11 test modules (100% Pass Rate)

---

## 1. Module Implementation & Verification Status

| Module / Path | Purpose & Function | Implementation Status | Verification Level | Notes & Audit Remarks |
| :--- | :--- | :--- | :--- | :--- |
| `src/data/schema.py` | Data contract, Pydantic models (`PostNode`, `EdgeInteraction`, `CascadeTree`) | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_schema.py`). Validates node and edge schema, enforces timestamp sorting and max $K=20$ constraints. |
| `src/data/pheme_loader.py` | PHEME raw JSON parser, inductive event split generator | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_pheme_loader.py`). True inductive split across 9 breaking news events. |
| `src/data/arctic_shift_vn.py` | Reddit VN crawler/parser, pseudonymization, heuristic split | Fully Implemented | **VERIFIED & TESTED** | Updated during audit: converted author hash from "anonymization" to "pseudonymization" with salt support (`SBIF_HASH_SALT`). Grouped inductive split by week and title keyword. |
| `src/data/pyg_dataset.py` | PyTorch Geometric `InMemoryDataset` conversion, burst stats | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_pyg_dataset.py`). Zero-leakage verified: target labels are excluded from $X$ and edge attributes. |
| `src/nlp/dual_edge_scorer.py` | NLI/stance scoring on edges ($u \to v$) | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests. Supports `legacy_nli` (2D) and `qwen_qlora` (4D). |
| `src/nlp/qwen_teacher.py` | Qwen LLM teacher for 5-class stance extraction | Fully Implemented | **READY / CODE AUDITED** | Implements 4-bit NF4 quantization loader. Requires GPU and downloaded Qwen checkpoint to run live distillation. |
| `src/nlp/extract_features.py` | Batch feature extraction pipeline (text to `.pt`) | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_pipeline_and_viz.py`). Extracts node and edge features offline. |
| `src/models/sbi_fagat.py` | Proposed SBi-FAGAT model (bidirectional, frequency-adaptive) | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_models.py`, `test_overfit.py`). Validated forward pass, low/high-pass attention, multi-task heads, and MC Dropout. |
| `src/models/baselines.py` | GCN, GAT, BiGCN, BiGAT, FAGCN, StanceGCN, StanceGAT | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_models.py`). All 7 baselines conform to uniform input/output interface. |
| `src/training/metrics.py` | Metric calculations: MSE, MAE, Pearson $r$, Macro-F1, Acc | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_metrics.py`). Robust handling of constant predictions. |
| `src/training/trainer.py` | Unified multi-task trainer, early stopping, LR scheduler | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_overfit.py`). Validated training convergence on synthetic tensors. |
| `src/training/reporting.py` | Latex tables, markdown tables, bootstrap p-values | Fully Implemented | **VERIFIED & TESTED** | Bootstrap paired t-test and Wilcoxon signed-rank test implemented. |
| `src/training/auto_loop.py` | Autonomous hyperparameter loop and trial orchestrator | Fully Implemented | **READY / CODE AUDITED** | Code audited. Runs training over parameter grids. |
| `src/utils/seed.py` | Deterministic seeding (`torch`, `numpy`, `random`) | Fully Implemented | **VERIFIED & TESTED** | Enforces reproducible seeds `[42, 100, 2024, 777, 999]`. |
| `src/utils/visualize.py` | UMAP embeddings, cascade tree plotting, loss curves | Fully Implemented | **VERIFIED & TESTED** | Passed unit tests (`test_pipeline_and_viz.py`). |

---

## 2. Scripts & Execution Entrypoints

| Script | Purpose | Status | Audit Findings & Verification |
| :--- | :--- | :--- | :--- |
| `scripts/kaggle_parallel_runner.py` | Dual T4 orchestration (`member1_vn`, `member2_gnn`, etc.) | **VERIFIED** | Smoke test verified: `python -m scripts.kaggle_parallel_runner --role member1_vn --smoke-test` exited with code 0. |
| `scripts/predict_live_thread.py` | Inference CLI for live Reddit/social threads | **VERIFIED** | Audited and hardened: Added `--allow-untrained` safeguard to prevent accidental inference on real posts with random weights. Demo offline verified (`--demo-offline`). |
| `scripts/benchmark_efficiency.py` | Model parameter counts, inference latency, RAM/VRAM | **VERIFIED** | Audited. Benchmarks all 8 models across synthetic graph batches. |
| `scripts/demo_research_walkthrough.py` | Quick CPU walkthrough (<5s) comparing GCN vs. SBi-FAGAT | **VERIFIED** | Audited. Educational and reproducible sanity check. |
| `scripts/run_phase2.sh` | Shell script for Frozen-Teacher NLP feature extraction | **UPDATED** | Enhanced with `PIPELINE_VERSION=v1|v2` variable. Default `v1` uses `legacy_nli` (2D). |
| `run_all.sh` | End-to-end execution script | **UPDATED** | Enhanced with `PIPELINE_VERSION=v1|v2` variable. Default `v1` passes `--edge-dim 2`. |
| `scripts/compute_kappa.py` | Inter-annotator agreement (SDQC, RumourEval) | **READY** | Implements Cohen's Kappa, Machine-vs-Gold, and confusion matrices. |
| `scripts/run_experiments.py` | 5-seed benchmark runner for 8 models | **READY** | Requires processed dataset `.pt` files to execute. |
| `scripts/run_ablation.py` | Component ablation study ($K \in \{5, 10, 20, 50\}$) | **READY** | Requires processed dataset `.pt` files to execute. |
| `scripts/run_upfd_extension.py` | UPFD benchmark extension | **READY** | Requires UPFD dataset download to execute. |

---

## 3. Workspace Data & Artifact Reality

| Directory | Expected Content | Actual State on Disk | Impact / Assessment |
| :--- | :--- | :--- | :--- |
| `data/raw/` | PHEME tarballs, raw Reddit JSON dumps | Contains `.gitkeep` only | No raw data stored locally. Must be ingested via `pheme_loader.py` or `arctic_shift_vn.py`. |
| `data/interim/` | Parsed Parquet tables, cleaned trees | Contains `.gitkeep` only | No interim tables stored locally. |
| `data/processed/` | PyG `.pt` tensor files for Train/Val/Test | Contains `.gitkeep` only | Requires NLP feature extraction pass. |
| `checkpoints/` | Trained model weights (`.pt`) | Directory exists (empty or gitignored) | No pre-trained checkpoints exist in the repository checkout. |
| `outputs/` / `results/` | Final benchmark tables, figures, CSVs | Contains `.gitkeep` only | No empirical result files exist. All published tables must be generated from real training runs. |
