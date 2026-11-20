# SBi-FAGAT: Architecture Decision Records (ADRs)

## ADR-01: Pseudonymization vs. Anonymization and Salt Management
- **Context:** Reddit author usernames cannot be deterministically hashed without secret salt and labeled "anonymous", as static SHA-256 hashes remain susceptible to dictionary precomputation.
- **Decision:** Use the accurate term **pseudonymization** in `src/data/arctic_shift_vn.py`, resolve salt from `SBIF_HASH_SALT` or `data/interim/.author_hash_salt.secret`, and emit a `UserWarning` when unsalted hashing is triggered.
- **Status:** Accepted & Implemented.

---

## ADR-02: Silver-Standard Nature and Mathematical Form of $y_{\text{polar}}$
- **Context:** Unlike observed final reply counts ($y_{\text{size}}$) or human-annotated PHEME veracity labels ($y_{\text{rumor}}$), future-tree polarization ($y_{\text{polar}}$) is computed from NLP stance scores on replies after $t_K$ (`polar_target_mode="future_only"`).
- **Decision:**
  1. Disclose $y_{\text{polar}}$ explicitly as a **silver-standard model-derived label**.
  2. Define $y_{\text{polar}}$ in `src/data/pyg_dataset.py` (`_compute_full_tree_polarization`, $\lambda_p = 0.5$) from root-aligned future stances $\tilde{s}_v \in [-1, 1]$ with neutral threshold $\tau = 0.1$:
     $$y_{\text{polar}} = \operatorname{clip}\!\left(\lambda_p \, \sigma_{\tilde{s}}^2 + (1 - \lambda_p)(4 p_+ p_-),\, 0,\, 1\right)$$
  3. Validate reliability via blind 300-pair audit (`scripts/compute_kappa.py`), 50-thread VN manual audit, Sarle's Bimodality Coefficient (`compute_bimodality_coefficient`), Signed Assortativity (`compute_signed_assortativity`), and comparison against the Early-$K$ Persistence Baseline (`scripts/export_dataset_stats.py`).
- **Status:** Accepted & Implemented.

---

## ADR-03: Unified 2D Dual-Teacher Architecture & Model Parity
- **Context:** Scaling to an oversized quantized 27B generative LLM teacher on Kaggle `T4 x 2` caused excessive VRAM pressure and time budgets that forced `--max-teacher-pairs` sampling, which would introduce train/test teacher–student distribution gaps and fallback risk. Meanwhile, fair GNN benchmarking across 8 models requires uniform capacity and non-leaky node/edge encoders.
- **Decision:** Standardize the repository on a single **Unified 2D Architecture**:
  1. Freeze `uitnlp/visobert` (VN) and `cardiffnlp/twitter-roberta-base` (EN) for **Ancestor-Conditioned SpanPool** (`[root | parent | reply]` pooling over reply tokens) paired with `10`-D causal `struct_x` and `8`-D `cascade_stats`.
  2. Use dual full-coverage teachers (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` with Symmetric 2-Hypothesis Stance Verbalizer inside a 384-token reply-preserving budget + class-weighted `uitnlp/visobert` on `UIT-ViHSD` / `cardiffnlp/twitter-roberta-base-offensive`) to distill a segment-isolated `DualHeadCrossEncoder` (`BCEWithLogitsLoss` + $T=2.0$ KL).
  3. Benchmark all 8 neural models fairly at fixed `hidden_dim = 64` (matching `2 * hidden_dim = 128` heads, `auto_loop` disabled for benchmark tuning unless explicitly passed `--run-auto-loop` for diagnostics only), evaluated against non-learned `Early-K Persistence`.
- **Status:** Accepted & Implemented (Steps 1–3).

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

## ADR-06: Heuristic Temporal-Keyword / TF-IDF Split for Vietnamese Reddit Data
- **Context:** Vietnamese Reddit posts lack canonical news-event IDs like PHEME.
- **Decision:** Partition VN cascades via `cluster_vn_cascade_topics` (14-day temporal window + TF-IDF cosine clustering + title keywords) and accurately report this as a **Heuristic Temporal-Keyword Grouped Inductive Split**.
- **Status:** Accepted & Implemented.

---

## ADR-07: Canonical Forecasting Minimum Reply Floor (`min_total_replies = 25`)
- **Context:** Computing meaningful `future_only` polarization targets beyond the $K=20$ observation prefix requires at least 5 future replies beyond $K=20$.
- **Decision:** Standardize `DEFAULT_FORECAST_MIN_REPLIES = 25` across `pyg_dataset.py`, `dual_edge_scorer.py`, `run_experiments.py`, `run_ablation.py`, `run_all.sh`, and `scripts/run_phase2.sh`.
- **Status:** Accepted & Implemented.
