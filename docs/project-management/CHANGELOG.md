# SBi-FAGAT: Changelog & Audit Modification Record

## [v3.1 — Kaggle Version #5 VN Temporal Holdout, v3 ReZero Ablation & Notebook Guard] - `results/kaggle-v5-vn-v3-temporal-ablation`

- **Kaggle Version #5 Track 1 VN Completion (`scriptVersionId=373122344`, `1h 32m 24s`, `0%` local CPU):**
  - Executed `scripts.run_experiments --split-mode temporal --edge-source teacher` (`225` Test cascades): `Early-K Persistence` collapses under chronological shift ($R^2_{\text{polar}} = -5.4386$, RMSE $= 0.0874$), whereas all learned architectures generalize positively ($R^2_{\text{polar}} \in [0.0481, 0.0815]$); `SBiFAGAT` leads Spearman polarization rank correlation ($\rho = 0.3058 \pm 0.0187$, RMSE $= 0.0334 \pm 0.0009$, $r = 0.3338 \pm 0.0319$), validation-selected `TemporalStanceMLP` reaches RMSE $= 0.0330 \pm 0.0004$ ($R^2_{\text{polar}} = 0.0815 \pm 0.0198$, $r = 0.3377 \pm 0.0112$, Holm--Bonferroni $p_{\text{adj}} = 1.00$), and `FAGCN` leads size regression (MSLE $= 0.1881 \pm 0.0129$, $R^2_{\text{size}} = 0.2586 \pm 0.0509$).
  - Executed refreshed 5-seed v3 ReZero `scripts.run_ablation` (`tables_ablation_vn.tex`, `results_ablation_vn.json`, `figures/vn/fig2_early_k_curve.{pdf,svg}`): `SBiFAGAT (Full)` achieves MSLE $= 0.1580 \pm 0.0070$, $R^2_{\text{size}} = 0.2925 \pm 0.0315$, RMSE $= 0.0369 \pm 0.0003$, $R^2_{\text{polar}} = 0.0777 \pm 0.0141$, $r = 0.2939 \pm 0.0223$, $\rho = 0.2335 \pm 0.0180$, comparable to matched `BiEdgeGATv2 (Matched)` ($R^2_{\text{polar}} = 0.0789 \pm 0.0172$, $p_{\text{adj}} = 1.00$); removing Camp Dipole \& Moments drops $R^2_{\text{polar}}$ to $0.0503 \pm 0.0208$ and removing the signed Conflict Gate drops it to $0.0527 \pm 0.0258$.
  - Integrated all 6 Kaggle artifacts into `paper/`, re-ran `scripts.assemble_paper_artifacts --paper-dir paper` (`missing_tables=[]`, `missing_figures=[]`, `latex_underscore_issues={}`), and updated `paper/main.tex` (~93% overall completion; 100% Track 1 VN automated deliverables).
- **Notebook Tier 2A Guard (`notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`):** Hardened Cell 5 (`colab-v3-pheme-tier2a`) to check `_meta.min_total_replies_extracted` in `data/processed/pheme/nlp_features.pt` before launching Tier 2A (`min_total_replies=15`), cleanly skipping if the attached bundle reflects legacy Cohort B (`min_total_replies_extracted=25`) so Cell 6 (`sbifagat_v3_gpu_results.zip`) still completes.

## [v3.0 — EGM Full Mathematical, Audit & Cloud-GPU Overhaul] - `fix/egm-audit-hardening`

- **PII, Doxxing & Spam Hygiene (`arctic_shift_vn.py`, `dual_edge_scorer.py`):** Scrubbed `112` crypto/spam rows across local `data/{interim,processed}/vn/nodes.parquet` while preserving cascade IDs/edges, and blocked `[deleted]`/spam/PII/redacted strings from blind audit exports.
- **v3 Mathematical & Modeling Fixes (`pyg_dataset.py`, `baselines.py`, `sbi_fagat.py`, `metrics.py`, `trainer.py`, `reporting.py`):**
  - Linearized `_compute_full_tree_polarization` and attached causal `early_y_polar` + `root_ts` to `CascadeData`, alongside chronological `compute_temporal_topic_splits` & `compute_embedding_cluster_splits` (both enforcing `edge_source="teacher"` in `run_experiments.py`).
  - Replaced raw Persistence-Residual scaling with ReZero zero-initialized `tanh(persist_weight)` over train-centered `[0.02, 0.98]` clamped `early_y_polar` logits (`persist_ep_logit_ref`) symmetrically across stance-aware models (`TemporalStanceMLP`, `EdgeGATv2`, `BiEdgeGATv2`, `SBiFAGAT`); fixed `predict_uncertainty` parity across all 8 models; zeroed single-camp Camp Dipole (`both_camps`); tightened Smooth-L1 `beta=0.05`; moved `comparator_model="auto"` strictly to Validation `val_summary_stats`; added `r2_polar` to primary tables/hypothesis keys.
- **Streamlined Gate 1 `20/120/30` Binary Audit & UI (`compute_kappa.py`, `audit_vn_50_trees.py`, `human_audit_ui.py`):**
  - Standardized hostility annotation to binary `clean` vs. `uncivil` (with `ROC-AUC`/`PR-AUC`/`weighted_macro_f1`, always-on `--strict-adjudication`, Kendall $\tau_b$, AUROC $@\ge 2$, dynamic `y_polar_quintile_grade`).
  - Added `--lock-role` HTTP `403` enforcement, per-role isolated notes/`uncertain` fields, `grandparent_text`, queued race-free UI `saveQueue`, JSON reply rendering, `[Q]/[W]` hotkeys, VN rubric, `gold` confirmation & `disagreements_count`.
  - Exported clean Test-only disjoint VN blind packages (`20` pilot [`6` uncivil / `14` clean], `120` pairs [`47` / `73`], `30` trees [`6 x 5` grades]) with machine keys sealed in `data/interim/sbifagat_member1_vn_sealed_machine_keys.zip`.
- **Verified v3 VN Canonical + Cloud GPU Offload (`notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`):** Executed v3 VN EDA, CPU efficiency profiler, canonical 5-seed VN benchmark (`SBiFAGAT RMSE=0.0370±0.0003, R^2_polar=0.0742±0.0157`), observational sensitivity analysis (`analyze_mechanisms.py` with consistent `early_y_polar` recomputation), and `assemble_paper_artifacts.py` (multi-corpus non-overwriting audit merge, `latex_underscore_issues={}`); added `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb` (plus updated Notebook 2 Tier 2A `--k-horizon 10 --min-total-replies 15`) to run remaining heavy multi-seed benchmarks on Google Colab / Kaggle GPU without taxing local CPU.

## [Track 1 & Track 2 Results + Blind Audit UI] - `2575295`..`5332750`

- **Model, Loss & Ingestion Hardening (`2575295`, `3215ede`, `11637ed`):** Added CleanRes Camp Dipole readout, decoupled `polar_head`, balanced 2:1 multi-task loss, shallow depth-6 `/kaggle/input` PHEME auto-discovery, and parallelized GCS-FUSE thread parsing (`os.scandir` + `ThreadPoolExecutor`).
- **Track 1 VN & Track 2 PHEME+UPFD Empirical Artifacts (`338651b`, `5332750`):** Executed both Kaggle `T4 x 2` tracks across `[42, 100, 2024, 777, 999]`, exported unified EDA, efficiency, 5-seed main/LOO/ablation/mechanisms & UPFD (GossipCop + PolitiFact) LaTeX/JSON artifacts plus 8 vector figures (`pheme` + `vn`), verified via `scripts/assemble_paper_artifacts.py` (`0` missing tables/figures, `0` unescaped `_`).
- **Strict-Blind Gate 1 Web Studio (`5332750`):** Added localhost-only Vietnamese Web UI (`scripts/human_audit_ui.py`) with 5-CSV allowlist (`403` guard on sealed machine keys), dual-annotator blinding, atomic `utf-8-sig` saves + `.bak`, native `<dialog closedby="any">`, IME-safe Enter handling, and `tests/test_human_audit_ui.py` (`17` test modules / `124` passing tests).

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
