# SBi-FAGAT: Project Context & Architecture Specification

## 1. Problem Statement & Research Objectives
This project addresses early-stage modeling of community opinion polarization and information propagation scale on social media.
The research title is:
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"** (*Graph Neural Networks and Natural Language Processing for Modeling Opinion Polarization and Information Propagation on Social Media*).

### Core Challenge
Early-stage prediction of viral cascades and polarization is intrinsically challenging because:
1. **Sparsity & Cold-Start:** Only the source post and the earliest $K = 20$ chronological interactions ($t_i \le t_K$) are visible at inference time on VN (`min_total_replies = 25`), with reduced-horizon evaluation $K = 10$, `min_total_replies = 15` on PHEME so non-trivial future growth exists past $t_K$.
2. **Structural & Semantic Coupling:** Cascade propagation is driven both by tree topology (depth, branching, arrival order, temporal lag) and pairwise discourse dynamics (stance agreement/opposition and hostility intensity).
3. **Multi-Task Objectives (`MultiTaskHead`):**
   - $y_{\text{size}} = \ln(1 + N_{\text{total}})$: Log-scale final cascade size regression (`F.softplus` output, Huber loss).
   - $y_{\text{polar}} \in [0, 1]$: Linear-amplitude silver-standard community polarization intensity over future interactions (`torch.sigmoid` output with `"y_polar_logit"`, KL-calibrated `BCEWithLogitsLoss` + Smooth-L1 $\beta=0.05$, and causal `Persistence-Residual` anchor `early_y_polar` on stance-aware architectures).
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
   - Aggregates root states, attention pooling, node dispersion, root-aligned BFS (`max_hops = 64`) camp dipoles with `both_camps` zero-masking and linear `camp_balance`, and 6 scalar gate moments (`3 * hidden_dim + 6 -> 2 * hidden_dim = 128`).
   - Supports optional 8-dimensional early structural/temporal `cascade_stats` (`--use-cascade-stats`) and symmetric uncertainty prediction (`predict_uncertainty`) passed across all 8 neural models.
   - Optimizes Huber (`y_size`), KL-calibrated `BCEWithLogitsLoss + SmoothL1(beta=0.05)` (`y_polar`), and class-weighted CE (`y_rumor`) using train-only target scaling and bias initialization, linear warmup + `ReduceLROnPlateau`, EMA (`decay = 0.985`), Early Stopping on validation `task_loss`, and Validation-only comparator selection (`val_summary_stats`) across 5 seeds.

### Tier 2: Macro Signed User-Interaction Social Network (`src/data/user_network.py`)
1. **Macro Polarization Geometry (`scripts/run_macro_signed_network.py --analysis-only`):** Projects valid non-deleted, non-self-loop reply pairs $(u, v)$ across cascades onto $\mathcal{G}_{\text{macro}}$, partitions the largest signed WCC via the normalized Signed Laplacian (`kunegis2010spectral`), and evaluates Structural Balance ($z$-score vs. 100 uniform edge-sign permutations), Signed Modularity ($Q_{\text{signed}}$), Garimella RWC, and cross- vs. within-camp opposition/incivility rates (executed on both VN [`13,718` users, $Q_{\text{signed}} = 0.4280$] and PHEME [`50,062` users, $Q_{\text{signed}} = 0.8232$, RWC `0.6218`]).
2. **Chronological Signed Link Prediction (`build_temporal_signed_link_dataset`):** Splits signed interactions ($|s_{uv}| \ge 0.1$) chronologically (`70/15/15`) at non-overlapping UTC cutpoints ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$) using $772$-D user representations ($\le t_{\text{train}}^{\max}$ SpanPool + `4`-D unsigned activity features, zeroed for cold-start users without pre-cutoff posts) and mini-batch pair masking across 6 equal-width ($d=64$) architectures (`UserTextMLP`, `StandardGCN`, `StandardGATv2`, `SGCN`, `FAGCN`, `SBiFAGAT-Link`).

---

## 3. Data Contract, Ingestion & 7-Day Delivery Scope

### Datasets
1. **PHEME Dataset (English Benchmark):**
   - Rumor and non-rumor cascades (`6,425` raw threads, `101,342` nodes, `9` news events) ingested from Figshare article `6392078` via `src/data/pheme_loader.py` into validated `nodes.parquet` / `edges.parquet`, supporting both canonical $K=20, N \ge 25$ and executed reduced-horizon $K=10, N \ge 15$ (`2,664` eligible cascades, `1786/544/334` split) + 5-fold LOO (`edge_source="teacher"`).
2. **Vietnamese Social Media Dataset (Arctic Shift / Reddit VN):**
   - Ingested from Vietnamese Reddit subreddits (`r/vozforums` `87,788` nodes, `r/VietNam` `32` nodes; `1,500` eligible cascades at $K=20, N \ge 25$) via `src/data/arctic_shift_vn.py` with `--before-epoch` cutoff, `--min-comments 25`, `--min-vi-ratio 0.65`, PII/doxxing/crypto-spam filtering & redaction, and resumable `.state.json` cursor persistence.
   - Uses a **Heuristic Temporal-Keyword / 30-day TF-IDF Grouped Inductive Split** (`cluster_vn_cascade_topics`, cosine `>= 0.30`, yielding `1,496` clusters), cross-checked via Temporal Holdout and Embedding-Cluster Holdout (`edge_source="teacher"`).
3. **UPFD & RumourEval-2019 Benchmarks:**
   - External post-hoc text-free structural transfer (`GossipCop` + `PolitiFact`) via `scripts/run_upfd_extension.py`, and SemEval-2019 RumourEval SDQC stance validation (`1,675` Test pairs) via `scripts/eval_rumoureval_sdqc.py`.

### 7-Day Submission Scope Freeze & Mandatory Deliverables (`ADR-10`)
- **Scope Freeze:** No additional data crawling and no core pipeline rewrites prior to submission; honestly report `w/o Hostility` ablation parity (`0.0757` vs. `0.0777`) showing 3-class stance is the primary signal.
- **Mandatory Deliverables:** (1) Finalized English IEEE paper (`paper/main.pdf`), (2) full 4-chapter Vietnamese Research Report + defense slides/Q&A handbook synced 100% with verified metrics, and (3) runnable local Live Demo Web UI MVP featuring tree-preserving junk filtering and 3-class stance + Tier 2 camp visualization.

### Five Iron Laws (`GEMINI.md`)
- **Zero Synthetic Results:** Never report synthetic/mock numbers in experimental paper tables; synthetic tensors are strictly restricted to unit tests (`tests/`) and profiling (`scripts/benchmark_efficiency.py`).
- **Zero Data Leakage:** Strictly limit inputs to root + earliest $K = 20$ replies, isolate post-hoc labels (`FORBIDDEN_LEAKAGE_COLUMNS`), share canonical `topic_splits.json` across distillation, benchmarks, and $K$-ablation subsets, and isolate LOO teacher edges (`edge_source="teacher"`).
- **Verified References Only:** All citations in `paper/references.bib` must carry verified authors, venues, and DOIs/arXiv IDs.
- **GPU Optimization & Reproducibility:** Offline NLP `.pt` extraction, mixed precision (`torch.amp`), unified 2D pipeline (`hidden_dim = 64`), and mandatory 5-seed evaluation (`[42, 100, 2024, 777, 999]`).
- **Test-Gated Delivery:** Every change must pass `pytest` before reporting completion.
