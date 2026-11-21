# SBi-FAGAT: Changelog & Audit Modification Record

## [EGM Hardening] - Full 5-Goldfish Audit Remediation (`B1–B10`, `M1–M8`)

### Critical Blockers Resolved (`B1–B10`)
- **`B1` (`extract_features.py`, `run_phase2.sh`, `kaggle_parallel_runner.py`):** Added `--min-total-replies` scoping to Transformer encoding and 100% Teacher coverage checks (`full_teacher_coverage`, `has_full_teacher_root`) so PHEME LOO `teacher_edge_stance` / `teacher_node_root_stance` populate cleanly over eligible cascades ($\ge 25$ replies) while preserving raw Parquet tables for EDA accounting.
- **`B2` (`pyg_dataset.py`, `run_ablation.py`, `run_experiments.py`):** Removed the split-brain `edge_source="auto" and custom_topic_splits is not None -> teacher` heuristic; inductive benchmarks and ablations now consistently use Student edges (`edge_source="auto"`), while LOO explicitly enforces `edge_source="teacher"`.
- **`B3` (`dual_edge_scorer.py`):** Replaced $O(N^2)$ full-JSON rewrites with $O(N)$ append-only score-only `.checkpoint.jsonl` delta persistence, wrapped both Teachers with `_AutoCastLogitsWrapper` + `nn.DataParallel` for dual-T4 FP16 acceleration, and moved `_cuda_autocast` inside `DualHeadCrossEncoder.forward()` for multi-GPU Student distillation.
- **`B4` (`arctic_shift_vn.py`, `pheme_loader.py`):** Aligned VN crawl defaults (`min_comments=25`, `min_vi_ratio=0.65`, `target_count=1500`, `max_pages_per_sub=300`), added resilient per-thread error isolation and `.state.json` cursor/rejection persistence, enabled `TfidfVectorizer` stopwords, and upgraded `pheme_loader.py` to salted SHA-256 + retry download.
- **`B5` (`kaggle_parallel_runner.py`, `requirements.txt`, `notebooks/*.ipynb`):** Pinned `datasets>=2.18.0,<4.0.0`, preserved Kaggle CUDA PyTorch during `pip install`, added stage-level skip-if-exists (`--force` to override), isolated optional UPFD execution AFTER primary bundle creation, and added persistent salt initialization to Notebook 2.
- **`B6` (`reporting.py`, `paper/main.tex`, `paper/references.bib`, `scripts/assemble_paper_artifacts.py`):** Escaped `_` in LaTeX p-value headers (`ref_esc`/`cmp_esc`), added `table*` + `\resizebox` for IEEEtran fit, wired all 6 vector PDF figures and `tables_human_audit.tex` into `paper/main.tex`, added verified `rossi2023edge` (Dir-GNN) and fixed `li2017deepcas` DOI while removing unused v2 BibTeX entries, and added `scripts/assemble_paper_artifacts.py`.
- **`B7` (`dual_edge_scorer.py`, `compute_kappa.py`, `scripts/audit_vn_50_trees.py`):** Fixed NLI double-contradiction neutral attribution (`double_contra = min(c_sup, c_opp)`), upgraded 2D audit export to joint `stance x hostility` stratification, added 50-tree VN blind audit export/evaluation (`scripts/audit_vn_50_trees.py`), and added `--output-table` LaTeX export to `compute_kappa.py`.
- **`B8` (`trainer.py`, `reporting.py`, `run_ablation.py`, `run_upfd_extension.py`):** Made significance stars strictly direction-aware, added exact $2^n$ sign-flip permutation test for $n < 6$ (LOO folds), applied `(count+1)/(B+1)` bootstrap smoothing and topic-cluster bootstrap (`>= 6` topics), scoped `primary_hypothesis_keys` across ablation & UPFD, and synchronized per-seed DataLoader shuffle order.
- **`B9` (`pyg_dataset.py`, `run_ablation.py`):** Upgraded `_compute_full_tree_polarization` to continuous finite-sample-unbiased intensity-weighted variance + balance (`ddof=1` + $\frac{n}{n-1} \cdot 4 m_+ m_-$), added `polar_future_start_k` so $K$-curve sensitivity evaluates all horizons on an identical fixed future target window.
- **`B10` (`sbi_fagat.py`, `baselines.py`, `run_ablation.py`, `metrics.py`):** Fixed `gate_mode="sigmoid"` camp dipole centering (`2*gamma - 1`), added `use_camp_dipole`, `use_dir_mix`, `use_dual_channel` ablations plus matched `BiEdgeGATv2`, masked topology columns `[3, 4]` in `TextMLP`'s `cascade_stats`, wired `y_assortativity` & `early_node_root_stance` onto `CascadeData`, and sanitized JSON exports (`allow_nan=False`).

---

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
