# SBi-FAGAT: Project Context & Architecture Specification

## 1. Problem Statement & Research Objectives
This project addresses early-stage modeling of community opinion polarization and information propagation scale on social media.
The research title is:
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"** (*Graph Neural Networks and Natural Language Processing for Modeling Opinion Polarization and Information Propagation on Social Media*).

### Core Challenge
Early-stage prediction of viral cascades and polarization is intrinsically challenging because:
1. **Sparsity & Cold-Start:** Only the source post and the earliest $K = 20$ chronological interactions ($t_i \le t_K$) are visible at inference time (`min_total_replies = 25` enforced on forecasting splits so non-trivial future growth exists past $t_K$).
2. **Structural & Semantic Coupling:** Cascade propagation is driven both by tree topology (depth, branching, temporal delay) and pairwise discourse dynamics (stance agreement, constructive opposition, fallacy, hostility).
3. **Multi-Task Objectives (`MultiTaskHead`):**
   - $y_{\text{size}} = \log(1 + N_{\text{total}})$: Log-scale final cascade size regression (`F.softplus` output, Huber loss).
   - $y_{\text{polar}} \in [0, 1]$: Silver-standard community polarization intensity over future interactions (`torch.sigmoid` output, Huber loss).
   - $y_{\text{rumor}}$: Rumor veracity classification on PHEME (`veracity4` 4-class or `binary`, Cross-Entropy loss; masked out via `-1` when predicting Vietnamese Reddit threads without rumor labels).

---

## 2. Proposed Architecture: SBi-FAGAT
**SBi-FAGAT** stands for **Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network** (`src/models/sbi_fagat.py`), compared fairly against 7 baselines sharing `hidden_dim = 128` (`src/models/baselines.py`):
`TextMLP`, `TemporalStanceMLP`, `StandardGCN`, `StandardGATv2`, `EdgeGATv2`, `FAGCNBaseline` (`FAGCN`), and `BiGCN`.

### Key Architectural Components
1. **Node Feature Representation ($X \in \mathbb{R}^{|V_K| \times 770}$):**
   - `768`-dimensional contextual text embeddings extracted offline via masked mean pooling from frozen backbones (`uitnlp/visobert` for Vietnamese, `roberta-base` for English PHEME).
   - Concatenated with `2` causal intra-$K$ structural/temporal features (`t_norm` using fixed 7-day logarithmic reference `TEMPORAL_REF_SECONDS`, and `depth_norm`), forming `TOTAL_NODE_FEATURE_DIM = 770`.
2. **Stance-Guided Edge Modulation ($e_{uv} \in \mathbb{R}^{d_{\text{edge}}}$):**
   - Encodes directed parent-reply interaction $(u, v)$.
   - **Profile `v1` (Default MVP, $d_{\text{edge}} = 2$):** `e_uv = [s_uv, h_uv]` where `s_uv in [-1, 1]` is stance polarity from `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` and `h_uv in [0, 1]` is hostility intensity from `uitnlp/visobert` (`ura-hcmut/UIT-ViHSD`) for VI or `cardiffnlp/twitter-roberta-base-offensive` for EN, distilled into 2-head `DualHeadCrossEncoder`.
   - **Profile `v2` (Opt-In Scaling, $d_{\text{edge}} = 4$):** `e_uv = [s_uv, h_uv, r_uv, c_uv]` (stance polarity, hostility intensity, irrationality/fallacy intensity, constructive reasoning quality) projected from 5-class discourse probabilities (`support`, `neutral`, `constructive_oppose`, `irrational_fallacy`, `hostile_conflict`) via `Qwen/Qwen3.8-27B` QLoRA 1-pass G-Eval logit scoring (`src/nlp/qwen_teacher.py`) and distilled into 4D `DualHeadCrossEncoder` (`uitnlp/visobert` / `microsoft/deberta-v3-base`).
   - Also provides `LatentConflictMLP(x_u, x_v, x_u - x_v)` for edge-text-free datasets such as PyG `UPFD` (`scripts/run_upfd_extension.py`).
3. **Bidirectional Frequency-Adaptive Attention (`StanceGuidedFAConv`):**
   - Computes bounded signed edge gates $\gamma_{uv} = \tanh(\mathbf{a}^\top [z_u^{\text{norm}} \Vert z_v^{\text{norm}} \Vert e_{uv}]) \in (-1, +1)$ (with `sigmoid` ablation option) and symmetric degree normalization $((d_{\text{out}}(u)+1)(d_{\text{in}}(v)+1))^{-1/2}$.
   - Runs independent Top-Down (`edge_index_td`) and Bottom-Up (`edge_index_bu`) message passing stacks with initial residual (`eps * h_0`), root anchoring (`lin_root(h_root_broadcast)`), and nonlinear stabilization (`raw + 0.2 * tanh(raw)`).
4. **Readout, Cascade Summary, & Uncertainty (`MultiTaskHead`):**
   - Concatenates 4 graph vectors `[z_root_td, z_att_td, z_root_bu, z_att_bu]` (`4 * hidden_dim = 512`) using `GlobalAttentionReadout` (or mean pool ablation).
   - Optional 8-dimensional early structural/temporal `cascade_stats` projection (`use_cascade_stats=True` in `v2`).
   - Optional heteroscedastic $\beta$-NLL uncertainty head (`predict_uncertainty=True` in `v2`) outputting bounded `size_logvar` and `polar_logvar` (`[-6, 3]`) trained with detached residuals (`beta_nll = 0.5`, Seitzer et al., ICLR 2022).

---

## 3. Data Contract & Ingestion Pipelines

### Datasets
1. **PHEME Dataset (English Benchmark):**
   - Rumor and non-rumor cascades ingested from Figshare article `6392078` via `src/data/pheme_loader.py` into validated `nodes.parquet` / `edges.parquet`.
   - Truly inductive topic split grouped strictly by real-world breaking news event (`topic_id`).
2. **Vietnamese Social Media Dataset (Arctic Shift / Reddit VN):**
   - Ingested from Reddit Vietnam subreddits (e.g., `r/VietNam`, `r/TroChuyenLinhTinh`) via `src/data/arctic_shift_vn.py` with temporal cutoff `--before-epoch` support.
   - Uses a **Heuristic Temporal-Keyword / 14-day TF-IDF Grouped Inductive Split** (`cluster_vn_cascade_topics`) to avoid thread overlap across Train/Val/Test while disclosing that it is heuristic rather than event-level ground truth.
3. **UPFD Benchmark (GossipCop / PolitiFact):**
   - External benchmark ingested via `scripts/run_upfd_extension.py` using `LatentConflictMLP`.

### Five Iron Laws (`GEMINI.md`)
- **Zero Synthetic Results:** Never report synthetic/mock numbers in experimental paper tables; synthetic tensors are strictly restricted to unit tests (`tests/`) and computational complexity profiling (`scripts/benchmark_efficiency.py`).
- **Zero Data Leakage:** Strictly limit graph inputs to root + earliest $K = 20$ replies, isolate post-hoc labels (`FORBIDDEN_LEAKAGE_COLUMNS` in `src/data/schema.py`), share canonical `topic_splits.json`, and isolate LOO teacher edges (`edge_source="teacher"`).
- **Verified References Only:** All citations in `paper/references.bib` must carry verified authors, venues, and DOIs/arXiv IDs.
- **GPU Optimization & Reproducibility:** Offline NLP extraction (`.pt` feature banks), mixed precision (`torch.amp`), and mandatory 5-seed evaluation (`[42, 100, 2024, 777, 999]`) with Early Stopping on `task_loss`.
- **Test-Gated Delivery:** Every change must pass `pytest` before reporting completion.

---

## 4. Pipeline Execution Profiles (`v1` vs. `v2`)

| Dimension | Profile `v1` (Default MVP) | Profile `v2` (Opt-In Scaling) |
| :--- | :--- | :--- |
| **Activation** | Default (`PIPELINE_VERSION=v1`) | `export PIPELINE_VERSION=v2` or `--v2` |
| **NLP Edge Teacher** | Frozen NLI (`legacy_nli`: `mDeBERTa-v3` + `ViHSD`/`twitter-roberta`) | `Qwen/Qwen3.8-27B` QLoRA 1-pass G-Eval (`qwen_qlora`) |
| **Edge Dimension ($d_{\text{edge}}$)** | `2`: `[s_uv, h_uv]` | `4`: `[s_uv, h_uv, r_uv, c_uv]` |
| **Student Edge Encoder** | 2D `DualHeadCrossEncoder` | 4D `DualHeadCrossEncoder` (`visobert` / `deberta-v3-base`) |
| **Cascade Burst Stats** | Off (`use_cascade_stats=False`) | On (8-dimensional `cascade_stats`) |
| **Uncertainty Estimation** | Deterministic point prediction | Heteroscedastic $\beta$-NLL (`beta_nll=0.5`, detached residuals) |
| **Target Compute** | CPU or 1x 8–16 GB GPU | Dual GPU (`Kaggle T4 x 2`, 32 GB total VRAM) |
