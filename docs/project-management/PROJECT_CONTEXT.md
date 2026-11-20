# SBi-FAGAT: Project Context & Architecture Specification

## 1. Problem Statement & Research Objectives
This project addresses early-stage modeling of community opinion polarization and information propagation scale on social media.
The research title is:
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"** (*Graph Neural Networks and Natural Language Processing for Modeling Opinion Polarization and Information Propagation on Social Media*).

### Core Challenge
Early-stage prediction of viral cascades and polarization is intrinsically challenging because:
1. **Sparsity & Cold-Start:** Only the source post and the earliest $K = 20$ chronological interactions ($t_i \le t_K$) are visible at inference time (`min_total_replies = 25` enforced on forecasting splits so non-trivial future growth exists past $t_K$).
2. **Structural & Semantic Coupling:** Cascade propagation is driven both by tree topology (depth, branching, arrival order, temporal lag) and pairwise discourse dynamics (stance agreement/opposition and hostility intensity).
3. **Multi-Task Objectives (`MultiTaskHead`):**
   - $y_{\text{size}} = \ln(1 + N_{\text{total}})$: Log-scale final cascade size regression (`F.softplus` output, Huber loss).
   - $y_{\text{polar}} \in [0, 1]$: Silver-standard community polarization intensity over future interactions (`torch.sigmoid` output with `"y_polar_logit"`, KL-calibrated `BCEWithLogitsLoss` + Smooth-L1 loss).
   - $y_{\text{rumor}}$: Rumor veracity classification on PHEME (`veracity4` 4-class or `binary`, class-weighted Cross-Entropy loss; masked out via `-1` when predicting Vietnamese Reddit threads without rumor labels).

---

## 2. Proposed Architecture: Unified 2D SBi-FAGAT
**SBi-FAGAT** stands for **Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network** (`src/models/sbi_fagat.py`), compared fairly against 7 neural baselines sharing `hidden_dim = 64` and `2 * hidden_dim = 128` head capacity (`src/models/baselines.py`):
`TextMLP`, `TemporalStanceMLP`, `StandardGCN`, `StandardGATv2`, `EdgeGATv2`, `FAGCNBaseline` (`FAGCN`), and `BiGCN`, alongside the non-learned `Early-K Persistence` baseline.

### Key Architectural Components
1. **Causal Node Representation ($X \in \mathbb{R}^{|V_K| \times 770}$ + `struct_x` $\in \mathbb{R}^{|V_K| \times 10}$):**
   - `768`-dimensional contextual text embeddings extracted offline via frozen **Ancestor-Conditioned SpanPool** (`[CLS] root [SEP] parent [SEP] reply [SEP]`, pooling strictly over reply tokens from `uitnlp/visobert` for Vietnamese or `cardiffnlp/twitter-roberta-base` for English PHEME).
   - Concatenated with `2` causal temporal/depth features (`t_norm` and `depth_norm`, `TOTAL_NODE_FEATURE_DIM = 770`) and paired with `10`-dimensional causal structural features `struct_x` projected uniformly through `TwoStreamNodeStem` (topology columns masked for `TextMLP`).
2. **Stance-Guided Edge Modulation ($e_{uv} = [s_{uv}, h_{uv}]^\top \in \mathbb{R}^{2}$):**
   - **Teacher Backends (`dual_nli`):** Stance polarity $s_{uv} \in [-1, 1]$ via `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` using a Symmetric 2-Hypothesis Stance Verbalizer (within a 384-token reply-preserving budget), paired with hostility $h_{uv} \in [0, 1]$ from class-weighted `uitnlp/visobert` on `UIT-ViHSD` (VI) or `cardiffnlp/twitter-roberta-base-offensive` (EN).
   - **Student Encoder (`DualHeadCrossEncoder`):** Distilled strictly on $\mathcal{T}_{\text{train}}$ with target-segment-isolated pooling for `h_uv` (`BCEWithLogitsLoss` + $T=2.0$ KL + deep `(root, reply)` augmentation + differential AdamW).
   - Also provides `LatentConflictMLP(x_u, x_v, x_u - x_v)` for edge-text-free datasets such as PyG `UPFD` (`scripts/run_upfd_extension.py`).
3. **Dual-Channel Dir-GNN Attention (`StanceGuidedFAConv`):**
   - Computes scaled node--edge signed gates $\gamma_{uv} \in (-1, +1)$ (with `sigmoid` ablation option) and unsigned excitation gates $\alpha_{uv}^{\text{exc}} \in (0, 1)$, splitting messages evenly across Polarity and Excitation channels under symmetric degree normalization $((d_{\text{out}}(u)+1)(d_{\text{in}}(v)+1))^{-1/2}$.
   - Runs Top-Down (`edge_index_td`) and Bottom-Up (`edge_index_bu`) message passing stacks with initial residual (`eps * h_0`), root anchoring, odd-symmetric stabilization (`raw + 0.2 * tanh(raw)`), and learnable Dir-GNN TD–BU cross-mixing.
4. **Readout, Cascade Summary, & Multi-Task Optimization (`MultiTaskHead`):**
   - Aggregates root states, attention pooling, node dispersion, root-aligned BFS (`max_hops = 64`) camp dipoles, and 6 scalar gate moments (`5 * hidden_dim + 6 -> 2 * hidden_dim = 128`).
   - Supports optional 8-dimensional early structural/temporal `cascade_stats` (`--use-cascade-stats`) passed symmetrically to all 8 neural models.
   - Optimizes Huber (`y_size`), KL-calibrated `BCEWithLogitsLoss + SmoothL1` (`y_polar`), and class-weighted CE (`y_rumor`) using train-only target scaling and bias initialization, linear warmup + `ReduceLROnPlateau`, EMA (`decay = 0.985`), Early Stopping on validation `task_loss`, and validation-selected best checkpoints across 5 seeds.

---

## 3. Data Contract & Ingestion Pipelines

### Datasets
1. **PHEME Dataset (English Benchmark):**
   - Rumor and non-rumor cascades ingested from Figshare article `6392078` via `src/data/pheme_loader.py` into validated `nodes.parquet` / `edges.parquet`.
   - Truly inductive topic split grouped strictly by breaking news event (`topic_id`) + LOO (`edge_source="teacher"`).
2. **Vietnamese Social Media Dataset (Arctic Shift / Reddit VN):**
   - Ingested from Reddit Vietnam subreddits (e.g., `r/VietNam`, `r/TroChuyenLinhTinh`) via `src/data/arctic_shift_vn.py` with `--before-epoch` cutoff.
   - Uses a **Heuristic Temporal-Keyword / 14-day TF-IDF Grouped Inductive Split** (`cluster_vn_cascade_topics`).
3. **UPFD Benchmark (GossipCop / PolitiFact):**
   - External benchmark ingested via `scripts/run_upfd_extension.py` using `LatentConflictMLP`.

### Five Iron Laws (`GEMINI.md`)
- **Zero Synthetic Results:** Never report synthetic/mock numbers in experimental paper tables; synthetic tensors are strictly restricted to unit tests (`tests/`) and profiling (`scripts/benchmark_efficiency.py`).
- **Zero Data Leakage:** Strictly limit inputs to root + earliest $K = 20$ replies, isolate post-hoc labels (`FORBIDDEN_LEAKAGE_COLUMNS`), share canonical `topic_splits.json` across distillation, benchmarks, and $K$-ablation subsets, and isolate LOO teacher edges (`edge_source="teacher"`).
- **Verified References Only:** All citations in `paper/references.bib` must carry verified authors, venues, and DOIs/arXiv IDs.
- **GPU Optimization & Reproducibility:** Offline NLP `.pt` extraction, mixed precision (`torch.amp`), unified 2D pipeline (`hidden_dim = 64`), and mandatory 5-seed evaluation (`[42, 100, 2024, 777, 999]`).
- **Test-Gated Delivery:** Every change must pass `pytest` before reporting completion.
