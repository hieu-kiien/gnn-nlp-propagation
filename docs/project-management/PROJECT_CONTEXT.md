# SBi-FAGAT: Project Context & Architecture Specification

## 1. Problem Statement & Research Objectives
This project addresses early-stage modeling of community opinion polarization and information propagation scale on social media.
The research title is:
**"Graph Neural Networks and Natural Language Processing in Modeling Opinion Polarization and Information Propagation on Social Media"** (*Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội*).

### Core Challenge
Early-stage prediction of viral cascades and polarization is intrinsically challenging because:
1. **Sparsity & Cold-Start:** Only the source post and the first $K = 20$ interactions ($t \le t_{20}$) are available at inference time.
2. **Structural & Semantic Coupling:** Cascade propagation is driven both by structural topology (tree depth, branching factor, speed) and semantic interaction (agreement, disagreement, toxic escalation, support).
3. **Multi-task Objectives:** The model must predict:
   - $y_{\text{size}} = \log(1 + N_{\text{total}})$: Log-scale final cascade size (Regression).
   - $y_{\text{polar}} \in [0, 1]$: Community polarization intensity across the complete future cascade (Regression, silver-standard label).
   - $y_{\text{rumor}} \in \{0, 1\}$: Rumor verification classification (Binary, on PHEME dataset).

---

## 2. Proposed Architecture: SBi-FAGAT
**SBi-FAGAT** stands for **Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network**.

### Key Architectural Components
1. **Node Feature Representation ($X \in \mathbb{R}^{|V| \times d_{\text{in}}}$):**
   - 768-dimensional contextual text embeddings extracted from frozen language models (e.g., `vinai/phobert-base-v2` for Vietnamese, `sentence-transformers/all-mpnet-base-v2` or `roberta-base` for English).
   - Augmented by root-relative temporal and topological node attributes.
2. **Stance-Guided Edge Modulation ($e_{uv} \in \mathbb{R}^{d_{\text{edge}}}$):**
   - Encodes stance interaction between parent node $u$ and child reply $v$.
   - **Profile v1 ($d_{\text{edge}} = 2$):** Signed agreement/disagreement score $s_{uv} \in [-1, +1]$ and confidence $c_{uv} \in [0, 1]$ computed via legacy NLI teacher (`FacebookAI/roberta-large-mnli` / `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`).
   - **Profile v2 ($d_{\text{edge}} = 4$):** 5-class stance probabilities projected via Qwen QLoRA (`support, refute, query, comment, toxic`) yielding signed polarity, confidence, structural decay, and stance entropy.
3. **Bidirectional Message Passing:**
   - Downward edge propagation ($\mathcal{E}_{\text{down}}$: post $\to$ response) captures cascade dispersion.
   - Upward edge propagation ($\mathcal{E}_{\text{up}}$: response $\to$ parent) captures aggregation and collective consensus/conflict.
4. **Frequency-Adaptive Attention (FAGAT Filtering):**
   - Learns low-pass coefficients ($\beta_l > 0$) for homophilic opinion clusters (reinforcing stance agreement).
   - Learns high-pass coefficients ($\beta_l < 0$) for heterophilic cross-ideological friction (disagreement, controversy).
5. **Readout & Multi-Task Predictor:**
   - Mean/Max/Sum graph-level pooling.
   - Optional burst statistics (8-dimensional temporal velocity features) in v2.
   - Multi-layer perceptron heads with optional Monte Carlo Dropout for epistemic uncertainty estimation.

---

## 3. Data Contract & Ingestion Pipelines

### Datasets
1. **PHEME Dataset (English Benchmark):**
   - Contains rumor and non-rumor cascades across 9 breaking events (e.g., Ferguson, Sydney Siege, Charlie Hebdo).
   - Ingested via `src/data/pheme_loader.py`.
   - Truly inductive splits: Train/Val/Test split grouped strictly by distinct real-world events.
2. **Vietnamese Social Media Dataset (Arctic Shift / Reddit VN):**
   - Crawled from Reddit Vietnam subreddits (e.g., `r/VietNam`, `r/TroChuyenLinhTinh`) via `src/data/arctic_shift_vn.py`.
   - Heuristic Temporal-Keyword Grouped Inductive Split: Threads partitioned by ISO week and title keywords to prevent overlap.
3. **UPFD Benchmark (GossipCop / PolitiFact):**
   - Optional transfer learning benchmark ingested via PyG UPFD datasets.

### Strict Data Ingestion Guardrails (5 Iron Laws)
- **Zero Data Leakage:** Node inputs strictly limited to root post and earliest $K=20$ interactions ($t_i \le t_K$).
- **Label Independence:** Target metrics $y_{\text{size}}$ and $y_{\text{polar}}$ are derived from full-cascade metadata but never included in node or edge feature matrices.
- **Pseudonymization:** User identifiers are pseudonymized with SHA-256 and optional secret salt (`SBIF_HASH_SALT`); plain unsalted hashes are tagged with clear methodological warnings.

---

## 4. Pipeline Execution Profiles (v1 vs. v2)

| Dimension | Profile v1 (Default & Stable) | Profile v2 (Advanced & Opt-In) |
| :--- | :--- | :--- |
| **Philosophy** | Reproducible baseline, low compute overhead, strict paper compatibility | Experimental scaling, LLM-based stance distillation, uncertainty quantification |
| **NLP Edge Teacher** | Frozen NLI teacher (`legacy_nli`) | Qwen 2.5 / 3.8 / 27B QLoRA teacher (`qwen_qlora`) |
| **Edge Dimension ($d_{\text{edge}}$)** | 2 ($s_{uv}, c_{uv}$) | 4 (polarity, confidence, decay, entropy) |
| **Burst Statistics** | Disabled (standard graph readout) | Enabled (8-dimensional cascade velocity features) |
| **Uncertainty Estimation** | Standard point predictions | Monte Carlo Dropout (variance across $M=20$ passes) |
| **Hardware Requirement** | CPU or 1x 8-16GB GPU | Dual GPU (Kaggle T4 x 2, 32GB VRAM total) |
| **Activation Flag** | Default (`PIPELINE_VERSION=v1`) | `PIPELINE_VERSION=v2` or `--v2` |
