# SBi-FAGAT: Architecture Decision Records (ADRs)

This document formalizes key architectural, methodological, and security decisions made in the SBi-FAGAT project.

---

## ADR-01: Pseudonymization vs. Anonymization and Salt Management

### Context
Social media research involving Reddit usernames requires compliance with ethical guidelines and privacy regulations (GDPR, IRB). The codebase originally referred to truncated SHA-256 hashes of author usernames as "anonymization".

### Decision
1. **Terminology Correction:** Replaced all claims of "anonymization" with **"pseudonymization"**. Under international data protection frameworks, static deterministic hashing of public identifiers without a secret salt is pseudonymized, not anonymized, because it is vulnerable to rainbow-table precomputation and dictionary attacks.
2. **Salt Implementation:** Updated `src/data/arctic_shift_vn.py` to support a cryptographic salt via the `SBIF_HASH_SALT` environment variable or `data/interim/.author_hash_salt.secret`.
3. **Safe Fallback:** If no salt is configured, the system logs a `UserWarning` explicitly stating that unsalted SHA-256 hashes are vulnerable to re-identification.

### Status
**ACCEPTED & IMPLEMENTED**

---

## ADR-02: Nature of $y_{\text{polar}}$ as a Silver-Standard Model-Derived Label

### Context
The target metric $y_{\text{polar}} \in [0, 1]$ represents the community opinion polarization across the full future cascade. Unlike $y_{\text{size}}$ (which is directly observed from counts) or $y_{\text{rumor}}$ (which is human-annotated in PHEME), $y_{\text{polar}}$ is computed from stance distributions produced by a teacher NLP model across future comments.

### Decision
1. **Scientific Clarification:** Formally designate $y_{\text{polar}}$ as a **silver-standard label** (pseudo-ground truth) rather than human gold ground truth.
2. **Mathematical Definition:** Ensure $y_{\text{polar}}$ is defined as the normalized Popoviciu variance combined with Sarle's Bimodality Coefficient:
   $$y_{\text{polar}} = \text{clamp}\left(4 \cdot \text{Var}(s) \cdot \text{BC}(s), 0, 1\right)$$
3. **Reporting Integrity:** Paper drafts and reports must clearly disclose that polarization evaluation reflects agreement with the teacher's synthesized distribution over future interactions.

### Status
**ACCEPTED & IMPLEMENTED**

---

## ADR-03: Backward Compatible Pipeline Default (v1) with Opt-in v2

### Context
The repository includes both a lightweight, reproducible v1 pipeline (frozen RoBERTa/mDeBERTa NLI teacher, 2D edge scores) and an advanced v2 pipeline (fine-tuned Qwen 27B QLoRA, 4D edge scores, 8D burst features, Monte Carlo Dropout). Running v2 requires multi-GPU infrastructure (Kaggle dual T4 or 32GB+ VRAM), while v1 can run on standard 16GB GPUs or CPU smoke tests.

### Decision
1. **Default Stability:** Maintain **Profile v1** as the default configuration across `run_all.sh`, `scripts/run_phase2.sh`, and unit tests.
2. **Explicit Opt-In:** Require explicit opt-in for **Profile v2** via `export PIPELINE_VERSION=v2` or `--v2` CLI arguments.
3. **No Automatic Downgrade/Upgrade:** Ensure that datasets extracted with 2D edges fail gracefully with clear error messages if fed to 4D models, rather than silently slicing or padding tensors.

### Status
**ACCEPTED & IMPLEMENTED**

---

## ADR-04: Rejection of RWC and Modularity for Sparse Cascades ($K=20$)

### Context
Early literature review suggested graph-based polarization metrics such as Random Walk Controversy (RWC) or Newman's Modularity ($Q$).

### Decision
1. **Empirical Analysis & Rejection:** On early-stage cascades with only $K = 20$ nodes, the graph topology is predominantly a shallow tree with $E \approx 20$. Random walks on such sparse trees fail to achieve stationary mixing, and community detection algorithms degenerate or yield singletons.
2. **Adoption of Node-Centric Stance Statistics:** Adopted Sarle's Bimodality Coefficient (BC) and Signed Newman Assortativity on stance attributes instead of RWC/Modularity.
3. **Documentation:** Reconciled `docs/math_spec.md` to state that RWC and Modularity were analyzed and rejected for $K=20$ early prediction.

### Status
**ACCEPTED & IMPLEMENTED**

---

## ADR-05: Untrained Model Guard on Live Predictor CLI

### Context
`scripts/predict_live_thread.py` is intended to allow researchers to test the model on live Reddit threads. Previously, if `--checkpoint` was omitted, the script initialized a random SBi-FAGAT model and outputted predictions without warning, creating a high risk of misinterpretation.

### Decision
1. **Mandatory Checkpoint for Live Data:** Block live thread inference when no checkpoint is supplied, raising a `RuntimeError`.
2. **Explicit Bypass Flag:** Add `--allow-untrained` for researchers who explicitly wish to perform structural/dry-run checks with randomly initialized weights.
3. **Prominent CLI Output Tagging:** When `--allow-untrained` or `--demo-offline` is active, prominently print warning banners in the terminal output stating that predictions are generated by an untuned model.
4. **Checkpoint Integrity Check:** Add validation to verify that checkpoint files are readable and contain a valid `model_state_dict`.

### Status
**ACCEPTED & IMPLEMENTED**

---

## ADR-06: Heuristic Temporal-Keyword Grouped Inductive Split for Vietnamese Data

### Context
In the PHEME benchmark, threads are grouped naturally into 9 distinct breaking events (Ferguson, Sydney Siege, etc.), enabling true inductive event splitting. In the Reddit Vietnam dataset, posts do not carry explicit event IDs.

### Decision
1. **Grouping Mechanism:** Group threads by ISO calendar week (`YYYY-WW`) and top title keywords (e.g., `kinh_te`, `chinh_tri`, `xa_hoi`).
2. **Naming Precision:** Designate this split as a **Heuristic Temporal-Keyword Grouped Inductive Split**.
3. **Disclosure:** Disclose in research documentation that while this grouping strongly prevents cross-split thread leakage, it does not guarantee complete semantic independence to the same degree as real-world crisis event splits.

### Status
**ACCEPTED & IMPLEMENTED**
