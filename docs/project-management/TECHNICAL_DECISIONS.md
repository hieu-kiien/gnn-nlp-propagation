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

## ADR-03: Stable Profile `v1` Default with Opt-In Profile `v2`
- **Context:** Running `Qwen/Qwen3.8-27B` QLoRA requires `T4 x 2` (32 GB VRAM) and `transformers >= 5.8.0`, whereas `v1` runs cleanly on single 8–16 GB GPUs or CPU CI.
- **Decision:** Keep `PIPELINE_VERSION=v1` (`--teacher-backend legacy_nli --edge-dim 2`, deterministic point heads) as default across `run_all.sh` and `scripts/run_phase2.sh`, and gate `v2` (`--teacher-backend qwen_qlora --edge-dim 4 --v2` with 8D `cascade_stats` and $\beta$-NLL uncertainty) behind explicit opt-in (`SBIF-27`, `SBIF-28` resolved).
- **Status:** Accepted & Implemented.

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
