# SBi-FAGAT: Changelog & Audit Modification Record

## [Steps 1–3] - Unified 2D Architecture, Optimization & Leakage/Fairness Audits

### Step 1 — Core GNN Upgrade (`SBi-FAGAT` & 8-Model Parity)
- Upgraded `StanceGuidedFAConv` to dual-channel `[Signed Polarity || Unsigned Excitation]` message passing with `1/sqrt(2d)` attention scaling and Dir-GNN TD–BU cross-mixing.
- Added `10`-D causal `struct_x` (`pyg_dataset.py`) + `TwoStreamNodeStem` (`baselines.py`, topology masking for `TextMLP`) + `64`-hop BFS root-aligned camp dipole readout and standardized `hidden_dim = 64` across all 8 neural models.

### Step 2 — Unified 2D NLP Distillation & Teacher Coverage
- Replaced LLM QLoRA teacher path with 100%-coverage bilingual teachers: `mDeBERTa-v3-base-xnli` Symmetric 2-Hypothesis Stance Verbalizer + class-weighted `ViSoBERT` (`UIT-ViHSD`) / `twitter-roberta-base-offensive`.
- Added segment-isolated pooling (`BCEWithLogitsLoss` + $T=2.0$ KL + differential AdamW + deep `(root, reply)` pairs) in `DualHeadCrossEncoder`, switched EN node/student backbone to `cardiffnlp/twitter-roberta-base`, implemented causal **Ancestor-Conditioned SpanPool** in `extract_features.py`, extracted `src/nlp/discourse_labels.py`, and updated `GEMINI.md` & Kaggle notebooks.

### Step 3 — Multi-Task Optimization, 4 Audit Fixes & Docs Sync
- **Multi-Task Training:** Exposed `y_polar_logit` in `MultiTaskHead`; replaced sigmoid Huber with zero-minimum KL-calibrated `BCEWithLogitsLoss + SmoothL1(beta=0.08)`; added train-only target scaling & bias init, linear warmup + `ReduceLROnPlateau` + EMA (`0.985`), and validation-best seed checkpointing in `trainer.py`.
- **Fairness & Security (`auto_loop.py`, `kaggle_parallel_runner.py`):** Centered `auto_loop` on `hidden_dim=64`, kept 8-model benchmark hyperparameters uniform (`--run-auto-loop` diagnostic opt-in only), and unconditionally stripped `*_machine_key.csv`, `*.secret`, `.env` from zip bundles.
- **Canonical Split Lock (`pyg_dataset.py`, `run_ablation.py`):** Removed `used_disk_tables` bypass and fallback re-splitting so `topic_splits.json` is enforced uniformly across in-memory tables and `K=50` subsets.
- **NLP Text & Pipeline Cleanups (`dual_edge_scorer.py`, `extract_features.py`, `sbi_fagat.py`):** Extended URL/whitespace normalization to `lang="vi"`, raised NLI scorer budget to `max_length=384` with reply-preserving token allocation, skipped redundant deep Student forward pass when full teacher `root_s_uv` is cached, and raised BFS camp propagation to `max_hops=64`.
- **Evaluation & Tooling:** Attached `Early-K Persistence` to main `run_experiments.py` outputs (excluded from neural comparator ranking), added disagreement/coverage warnings & `--strict-adjudication` in `compute_kappa.py`, loosened `numpy<3.0.0`, unified `run_all.sh`, and synced `paper/main.tex`, `README.md`, and `docs/`.

---

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
