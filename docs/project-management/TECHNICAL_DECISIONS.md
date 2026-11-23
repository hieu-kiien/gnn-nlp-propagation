# SBi-FAGAT: Architecture Decision Records (ADRs)

## ADR-01: Pseudonymization vs. Anonymization and Salt Management
- **Context:** Reddit author usernames cannot be deterministically hashed without secret salt and labeled "anonymous", as static SHA-256 hashes remain susceptible to dictionary precomputation.
- **Decision:** Use the accurate term **pseudonymization** in `src/data/arctic_shift_vn.py`, resolve salt from `SBIF_HASH_SALT` or `data/interim/.author_hash_salt.secret`, and emit a `UserWarning` when unsalted hashing is triggered.
- **Status:** Accepted & Implemented.

---

## ADR-02: Silver-Standard Nature and Linear-Amplitude Mathematical Form of $y_{\text{polar}}$
- **Context:** Unlike observed final reply counts ($y_{\text{size}}$) or human-annotated PHEME veracity labels ($y_{\text{rumor}}$), future-tree polarization ($y_{\text{polar}}$) is computed from NLP stance scores on replies after $t_K$ (`polar_target_mode="future_only"`). The prior quadratic formulation collapsed VN/PHEME targets into $[0.001, 0.07]$ with median $\sim 0.01$.
- **Decision:**
  1. Disclose $y_{\text{polar}}$ explicitly as a **silver-standard model-derived label**.
  2. Define linear-amplitude $y_{\text{polar}}$ in `src/data/pyg_dataset.py` (`_compute_full_tree_polarization`, $\lambda_p = 0.5$) from root-aligned future stances $\tilde{s}_v \in [-1, 1]$:
     $$y_{\text{polar}} = \operatorname{clip}\!\left(\lambda_p \min(1, s_{\tilde{s}, \text{ddof}=1}) + (1 - \lambda_p)\min\!\left(1, 2\sqrt{\frac{n}{n-1} m_+ m_-}\right),\, 0,\, 1\right)$$
  3. Validate reliability via blind 20-pilot + 120-pair VN Test audit (`scripts/compute_kappa.py`, reporting ROC-AUC/PR-AUC & inverse-propensity `sampling_weight`), RumourEval SDQC for English, 30-thread VN manual audit (`scripts/audit_vn_50_trees.py`, Kendall $\tau_b$ & AUROC $@ \ge 2$), Sarle's Bimodality Coefficient, Signed Assortativity, and comparison against the Early-$K$ Persistence Baseline (`scripts/export_dataset_stats.py`).
- **Status:** Accepted & Implemented (v3 Overhaul).

---

## ADR-03: Unified 2D Dual-Teacher Architecture & Model Parity
- **Context:** Scaling to an oversized quantized 27B generative LLM teacher on Kaggle `T4 x 2` caused excessive VRAM pressure and time budgets that forced `--max-teacher-pairs` sampling, which would introduce train/test teacher–student distribution gaps and fallback risk. Meanwhile, fair GNN benchmarking across 8 models requires uniform capacity and non-leaky node/edge encoders.
- **Decision:** Standardize the repository on a single **Unified 2D Architecture**:
  1. Freeze `uitnlp/visobert` (VN) and `cardiffnlp/twitter-roberta-base` (EN) for **Ancestor-Conditioned SpanPool** (`[root | parent | reply]` pooling over reply tokens) paired with `10`-D causal `struct_x` and `8`-D `cascade_stats`.
  2. Use dual full-coverage teachers (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` with Symmetric 2-Hypothesis Stance Verbalizer inside a 384-token reply-preserving budget + class-weighted `uitnlp/visobert` on `UIT-ViHSD` / `cardiffnlp/twitter-roberta-base-offensive`) to distill a segment-isolated `DualHeadCrossEncoder` (`BCEWithLogitsLoss` + $T=2.0$ KL).
  3. Benchmark all 8 neural models fairly at fixed `hidden_dim = 64` (matching `2 * hidden_dim = 128` heads, symmetric `predict_uncertainty`, symmetric Causal Persistence-Residual anchor `early_y_polar` on stance-aware models `TemporalStanceMLP`, non-latent `EdgeGATv2`/`BiEdgeGATv2`, and non-ablated `SBiFAGAT`, Validation-only comparator selection via `val_summary_stats`, and `auto_loop` disabled unless explicitly passed `--run-auto-loop`), evaluated against non-learned `Early-K Persistence`.
- **Status:** Accepted & Implemented (v3 Overhaul).

---

## ADR-04: Rejection of RWC and Modularity for Sparse Early Trees ($K=20$)
- **Context:** Random Walk Controversy (RWC) and Modularity ($Q$) require denser community graphs than a $K=20$ shallow propagation tree, where random walks fail to mix and modularity degenerates.
- **Decision:** Reject RWC/$Q$ in favor of node/edge stance dispersion, Sarle's Bimodality Coefficient, and Signed Assortativity (`docs/math_spec.md`).
- **Status:** Accepted & Documented.

---

## ADR-05: Live Predictor Untrained Guard & CLI Compatibility
- **Context:** `scripts/predict_live_thread.py` must prevent scientific misuse of random weights on real URLs.
- **Decision:** Raise `ValueError`/`RuntimeError` when `--gnn-ckpt` (alias `--checkpoint`) is omitted for a live URL unless `--allow-untrained` or `--demo-offline` is passed; validate checkpoint readability and `model_state_dict`; keep backward-compatible CLI aliases (`--reddit-url`/`--url` and `--gnn-ckpt`/`--checkpoint`).
- **Status:** Accepted & Implemented.

---

## ADR-06: Heuristic Temporal-Keyword / TF-IDF Split & Holdouts for Vietnamese Reddit Data
- **Context:** Vietnamese Reddit posts lack canonical news-event IDs like PHEME.
- **Decision:** Partition VN cascades via `cluster_vn_cascade_topics` (30-day temporal window + TF-IDF cosine `>= 0.30` clustering + title keywords) and accurately report this as a **Heuristic Temporal-Keyword Grouped Inductive Split**, verified against `compute_temporal_topic_splits` and `compute_embedding_cluster_splits` (`edge_source="teacher"`).
- **Status:** Accepted & Implemented.

---

## ADR-07: Canonical Forecasting Minimum Reply Floor (`min_total_replies = 25`)
- **Context:** Computing meaningful `future_only` polarization targets beyond the $K=20$ observation prefix requires at least 5 future replies beyond $K=20$.
- **Decision:** Standardize `DEFAULT_FORECAST_MIN_REPLIES = 25` across `pyg_dataset.py`, `dual_edge_scorer.py`, `run_experiments.py`, `run_ablation.py`, `run_all.sh`, and `scripts/run_phase2.sh` (with Tier 2A notebook option `K=10, min_total_replies=15` for sparse external corpora such as PHEME).
- **Status:** Accepted & Implemented.

---

## ADR-08: Binary Reply Incivility & Strict Gate 1 Blinding / PII Hygiene
- **Context:** Separating `offensive` vs. `hate` on Vietnamese social media replies caused ambiguity and asymmetric numeric bug (`0.5` vs `1.0`), while minor/doxxing PII and `LMGX` spam leaked into unscreened crawls and Pilot splits collapsed to 100%-hostile samples.
- **Decision:** Standardize pair-level `h_uv` construct to **Binary Reply Incivility** (`clean` vs. `uncivil`, keeping legacy alias compatibility in `compute_kappa.py`), scrub phone/bank/CCCD/FB-profile/doxxing/`LMGX` before persistence and audit export, export balanced 50/50 uncivil/clean `20`-pair pilot + `120`-pair Test + `30`-tree Test sheets with `grandparent_text` & `sampling_weight`, and isolate role notes/blinding in `human_audit_ui.py --lock-role`.
- **Status:** Accepted & Implemented (v3 Overhaul).
