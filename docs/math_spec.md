# Đặc tả Toán học Chính thức & Thiết kế Hệ thống: Kiến trúc SBi-FAGAT

Tài liệu này trình bày đánh giá thiết kế tổng quan (theo chuẩn *Heilmeier Catechism*), bảng ký hiệu thống nhất, đặc tả trích xuất đặc trưng NLP ngoại tuyến, phương trình truyền thông điệp hai kênh – hai hướng thích ứng tần số và giao thức tối ưu đa nhiệm của kiến trúc **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network*) theo **kiến trúc 2D hợp nhất**.

---

## 0. Tóm tắt Thiết kế & Đánh giá Kỹ thuật (Heilmeier Assessment)

### 0.1. Mục tiêu Phi kỹ thuật & Đối tượng Thụ hưởng
- **Bài toán thực tế & Phạm vi phân tích**: Khi một thông tin nóng hoặc tin thất thiệt bắt đầu lan truyền trên mạng xã hội, nền tảng kiểm duyệt (*Trust & Safety*) và nhóm ứng phó khủng hoảng truyền thông chỉ có một cửa sổ thời gian rất ngắn (tối đa $K = 20$ bình luận đầu tiên) trước khi làn sóng chia rẽ hoặc tin giả bùng phát vượt kiểm soát. Đề tài xác định rõ phạm vi ở **cấp chuỗi thảo luận vi mô (cascade-level, $|\mathcal{V}_K| \le 21$)**, đo lường đồng thời *phân cực lập trường* ($s_{uv}$) và *phân cực cảm xúc/gay gắt* ($h_{uv}$), thay vì mô hình hóa cấu trúc buồng vang người dùng vĩ mô xuyên nhiều chủ đề qua thời gian dài.
- **Mục tiêu**: Xây dựng mô hình dự đoán sớm từ bài đăng gốc và tối đa $20$ tương tác đầu để ước lượng đồng thời: (1) quy mô lan rộng cuối kỳ ($y_{\text{size}}$), (2) mức độ chia rẽ hai phe ủng hộ–phản bác về cuối kỳ ($y_{\text{polar}}$), (3) xác thực tin đồn/tin giả ($y_{\text{rumor}}$ trên PHEME), và (4) phân tích cơ chế tương tác & kiểm thử can thiệp giả định (*counterfactual moderation*) hậu huấn luyện (`scripts/analyze_mechanisms.py`).

### 0.2. Hiện trạng & Điểm mù Kỹ thuật
- Các mô hình chuỗi thời gian/đồ thị lan truyền truyền thống (*DeepCas*, *GCN*, *GATv2*, *Bi-GCN*) bỏ qua thái độ đồng tình hay phản bác giữa từng cặp bình luận và dùng bộ lọc trung bình (low-pass) làm nhòa sự đối lập giữa hai phe.
- Các thước đo phân cực cấu trúc (như *RWC* hay *Modularity*) đòi hỏi đồ thị quy mô lớn đã trưởng thành, bị suy biến khi chỉ quan sát một cây hội thoại nhỏ có $\le 21$ nút (mã nguồn vì vậy loại bỏ RWC/Modularity, thay bằng chỉ số phân cực lưỡng cực quy chiếu gốc đối chiếu chéo với *Sarle's Bimodality Coefficient* và *Signed Newman Assortativity*). Chi tiết đối sánh các công trình nền tảng được tổng hợp tại [docs/literature_review.md](literature_review.md).

### 0.3. Kiến trúc Hợp nhất 2D & Phương án Thay thế bị Bác bỏ
- **Ý tưởng cốt lõi**: Kết hợp nhúng nút *Ancestor-Conditioned SpanPool* đóng băng (`uitnlp/visobert` cho Tiếng Việt, `cardiffnlp/twitter-roberta-base` cho Tiếng Anh), $10$ đặc trưng cấu trúc–thời gian nhân quả `struct_x`, mô hình thầy kép NLI + Hostility chưng cất về `DualHeadCrossEncoder` cô lập phân đoạn tạo vector cạnh 2D $e_{uv} = [s_{uv}, h_{uv}]^\top$, rồi đưa vào mạng hai hướng **Dir-GNN** với `StanceGuidedFAConv` hai kênh (phân cực có dấu vs. kích thích không dấu) và `hidden_dim = 64` đồng đều cho cả 8 mô hình.
- **Phương án thay thế đã cân nhắc & bác bỏ**:
  1. *Fine-tune end-to-end Transformer + GNN cùng lúc*: Với mỗi cây có $K=20$ văn bản ($L_{\text{tok}} \le 384$), chi phí bộ nhớ $\mathcal{O}(K \cdot L_{\text{tok}}^2)$ vượt quá giới hạn 16GB VRAM khi chạy multi-seed và ablation. Tách pha NLP ra tệp `.pt` giúp giai đoạn GNN tiêu thụ $< 500\text{ MB}$ VRAM.
  2. *Sinh nhãn Teacher bằng LLM 27B QLoRA (`Qwen`)*: Bị bác bỏ vì trên Kaggle `T4 x 2`, chạy phủ 100% tập dữ liệu gây tràn VRAM logits và vượt trần thời gian 12 giờ, còn cắt mẫu `--max-teacher-pairs` làm tập Test bị gán nhãn tự chấm. Teacher kép NLI (`mDeBERTa-v3-xnli` với *Symmetric Stance Verbalizer*) + Hostility (`ViSoBERT` tinh chỉnh class-weighted trên `UIT-ViHSD` / `twitter-roberta-base-offensive`) đạt độ phủ 100% sạch và ổn định.
  3. *Chỉ truyền cặp thô `(parent, reply)` vào NLI hoặc `[CLS]` chung đoạn cho `h_uv`*: Làm $\sim 85\%$ bình luận bị dồn về `Neutral` hoặc bị câu chửi ở `parent` lây sang `h_uv` của `reply`. Việc dùng *Symmetric 2-Hypothesis Verbalizer* trong ngân sách 384 token bảo toàn đoạn Reply và *reply-span pooling* giảm thiểu đáng kể điểm nghẽn này.

### 0.4. Rủi ro, Giới hạn & Biện pháp Kiểm soát
1. **Rò rỉ dữ liệu & Tái phân hoạch tại $K=50$**:
   - Mặc định `DEFAULT_FORECAST_MIN_REPLIES = 25` ($\ge 5$ bình luận tương lai sau $K=20$). Toàn bộ pipeline (`dual_edge_scorer.py`, `extract_features.py`, `EarlyCascadeDataset`, `run_ablation.py` kể cả khi đọc từ bộ nhớ đệm hoặc quét $K=50$) **bắt buộc tôn trọng** phân hoạch chuẩn hóa `topic_splits.json`, cấm chia lại ngầm giữa Train và Test.
   - Trên dữ liệu Việt Nam, `cluster_vn_cascade_topics` gộp chuỗi liên thông theo TF-IDF cosin ($> 0.55$) trong cửa sổ 14 ngày (*Heuristic Temporal-Keyword / TF-IDF Grouped Inductive Split*). Trên PHEME LOO, `edge_source="teacher"` cô lập Student khỏi fold giữ lại.
2. **Suy giảm đạo hàm `y_polar` & Công bằng 8 mô hình**:
   - Đạo hàm của `SmoothL1/Huber` sau `sigmoid` ở hai đuôi bị co rút mạnh; chuyển sang hàm mất mát **KL-calibrated `BCEWithLogitsLoss` + Smooth-L1 ($\beta = 0.08$)** kết hợp cân bằng phương sai mục tiêu chỉ trên tập Train và khởi tạo bias đầu ra.
   - Không chạy vòng lặp dò siêu tham số thiên vị riêng `SBiFAGAT` trên benchmark chính: tất cả 8 mô hình thi đấu công bằng ở `hidden_dim = 64`, cùng `TwoStreamNodeStem` (`struct_x` 10 chiều), đầu dự đoán `2 * hidden_dim` (`128` chiều), và `cascade_stats` 8 chiều.
3. **Giả danh hóa PII (*Pseudonymization*) & An toàn tệp**: Định danh tác giả được băm SHA-256 có salt (`SBIF_HASH_SALT` hoặc `.author_hash_salt.secret`). Hàm `_bundle_outputs` chặn tuyệt đối đóng gói `*_machine_key.csv`, `*.secret`, `.env` kể cả khi bật `--bundle-weights`.
4. **Tối ưu suy luận NLP bảo toàn bất biến thứ tự & ranh giới cây đơn lẻ**: Các bộ trích xuất `HostilityTeacherScorer`, `NLIStanceTeacherScorer`, `encode_node_embeddings`, và `encode_edges_with_cross_encoder` phân lô theo độ dài chuỗi (*length-bucketing*) và hỗ trợ đa GPU `DataParallel` (`use_multi_gpu=True`), sau đó khôi phục chính xác thứ tự đầu vào ban đầu (bất biến toán học tuyệt đối). Về phạm vi đồ thị, mạng chỉ nhìn từng chuỗi thảo luận rời rạc $\le 21$ nút nên cấu trúc buồng vang người dùng vĩ mô xuyên chủ đề được đặt làm hướng mở rộng cho nghiên cứu tiếp theo.

### 0.5. Cổng Nghiệm thu Bắt buộc
- **Gate 0 (Contract & Split Parity)**: `pytest` xác nhận nhân quả UTC, nhất quán `topic_splits.json` ở mọi $K \in \{10, 20, 30, 50\}$ và LOO `edge_source="teacher"`.
- **Gate 1 (ViHSD & Human Audit)**: kiểm định mù 300 cặp $(u, v)$ (phân tầng đồng thời `stance x hostility`) kèm thống kê `unadjudicated_disagreements` và `adjudication_coverage` (`compute_kappa.py`), 50 cây Test VN chấm tay (`scripts/audit_vn_50_trees.py`), và đối chiếu **Early-K Persistence Baseline**.
- **Gate 2 (Distillation & Anti-Cheating)**: độ phủ 100% nhãn Teacher kép sạch trên toàn bộ chuỗi đủ điều kiện ($\ge 25$ phản hồi), Student khớp Teacher 2D trên Val split, và `--sanity-check-shuffle` đẩy độ khớp về ngẫu nhiên.
- **Gate 3 (Statistical Benchmark)**: 5 hạt giống $\mathcal{S} = \{42, 100, 2024, 777, 999\}$, Early Stopping theo `task_loss`, chọn checkpoint tốt nhất theo Validation trong từng mô hình, kiểm định `ttest_rel` + Sample/Topic-Cluster Paired Bootstrap (hoặc kiểm định hoán vị dấu chính xác khi $F < 6$ trên LOO) + Holm-Bonferroni có hướng (chỉ gắn sao khi vượt trội theo chiều tốt hơn) đối chứng với mô hình học máy mạnh nhất (loại trừ baseline phi tham số `Early-K Persistence`).

---

## 1. Bảng Ký hiệu Toán học Thống nhất

| Ký hiệu | Miền giá trị / Kích thước | Ý nghĩa |
| :--- | :--- | :--- |
| $\mathcal{G}_K = (\mathcal{V}_K, \mathcal{E}_K)$ | Đồ thị cây có hướng | Cây lan truyền sớm cắt tại ngưỡng $K \in \{10, 20, 30, 50\}$ (mặc định $K=20$) |
| $r, N_{\text{total}}$ | $r \in \mathcal{V}_K, \mathbb{Z}_{\ge 0}$ | Nút gốc tại $t_r$ và tổng số phản hồi cuối kỳ $N_{\text{total}} = |\mathcal{V}_{\text{full}}| - 1$ |
| $\mathbf{e}_v^{\text{Span}}$ | $\mathbb{R}^{768}$ | Vector ngữ nghĩa nút từ **Ancestor-Conditioned SpanPool** đóng băng (`visobert` / `twitter-roberta-base`) |
| $\mathbf{x}_v, \mathbf{s}_v$ | $\mathbb{R}^{770}, [0, 1]^{10}$ | Đặc trưng nút $[\mathbf{e}_v^{\text{Span}} \parallel \tilde{\Delta t}_v \parallel \tilde{\ell}_v]$ và $10$-D đặc trưng cấu trúc nhân quả `struct_x` |
| $\mathbf{c}_{\mathcal{G}_K}$ | $\mathbb{R}^{8}$ | Vector thống kê nhịp độ lan truyền sớm 8 chiều `cascade_stats` |
| $\mathbf{h}_v^{(0)}$ | $\mathbb{R}^{d}$ | Biểu diễn nút sau `TwoStreamNodeStem` ($d = 64$ mặc định) |
| $\mathbf{a}_{uv} = [s_{uv}, h_{uv}]^\top$ | $[-1, 1] \times [0, 1]$ | Vector cạnh 2D (lập trường $s_{uv}$ + độ gay gắt $h_{uv}$) |
| $\gamma_{uv}^{(l, \delta)}, \alpha_{uv}^{\text{exc}}$ | $(-1, 1), (0, 1)$ | Cổng tần số có dấu (kênh phân cực) và cổng kích thích không dấu (kênh cường độ) |
| $\mathbf{z}_{\mathcal{G}_K}$ | $\mathbb{R}^{2d}$ | Biểu diễn toàn đồ thị chiếu từ $5d + 6$ chiều sau pooling TD/BU và camp dipoles |
| $y_{\text{size}}, y_{\text{polar}}, y_{\text{rumor}}$ | $\mathbb{R}_{\ge 0}, [0, 1], \{0..C-1\}$ | Quy mô $\ln(1+N_{\text{total}})$, chỉ số phân cực tương lai (`future_only`), nhãn tin đồn |

---

## 2. Định nghĩa Cây Lan truyền Sớm $\mathcal{G}_K$ & Bất biến Chống Rò rỉ

Xét cây hội thoại hoàn chỉnh $\mathcal{G}_{\text{full}} = (\mathcal{V}_{\text{full}}, \mathcal{E}_{\text{full}})$ thỏa nhân quả thời gian $t_u \le t_v$. Sắp xếp phản hồi theo bộ khóa từ điển $(t_v, \ell_v, \texttt{node\_id}(v))$ và lọc $N_{\text{total}} \ge \texttt{min\_total\_replies} = 25$:
$$\mathcal{V}_K = \{r\} \cup \left\{v_1, \dots, v_{\min(K, N_{\text{total}})}\right\}, \qquad \mathcal{E}_K = \left\{(u, v) \in \mathcal{E}_{\text{full}} \mid u \in \mathcal{V}_K \land v \in \mathcal{V}_K\right\}.$$

Mọi thành phần đặc trưng trong `EarlyCascadeDataset` đều tuân thủ bất biến nhân quả tại $t \le t_K$:
1. **Nhúng nút Ancestor-Conditioned SpanPool**: Với mỗi nút phản hồi $v$ có cha trực tiếp $u$ và gốc $r$, chuỗi đầu vào `[CLS] r [SEP] u [SEP] v [SEP]` (cắt gốc theo tỷ lệ 60% đầu + 40% cuối khi vượt ngân sách) đi qua `uitnlp/visobert` (VN) hoặc `cardiffnlp/twitter-roberta-base` (EN), sau đó lấy trung bình ẩn tầng cuối **chỉ trên mặt nạ token thuộc $v$**. Việc điều kiện hóa theo tổ tiên nhân quả ($t_r \le t_u \le t_v \le t_K$) giải nghĩa các câu trả lời ngắn kiểu *"Hoàn toàn sai!"* mà hoàn toàn không nhìn thấy nút tương lai.
2. **Đặc trưng cấu trúc 10 chiều `struct_x`**: (0) `t_norm` — thời gian trễ chuẩn hóa log so với gốc theo $T_{\text{ref}} = 7\text{ ngày}$; (1) `dt_parent_norm` — độ trễ chuẩn hóa log so với nút cha trực tiếp; (2) `rel_phase_k` — pha đến tương đối trong $[t_0, t_K]$; (3) `rank_norm_k` — thứ tự thời gian chuẩn hóa trong $V_K$; (4) `d_norm` — độ sâu chuẩn hóa $\ell_v / K$; (5) `is_root` — cờ nút gốc; (6) `early_out_deg_norm` — bậc ra sớm chuẩn hóa trong $V_K$; (7) `is_early_leaf` — cờ nút lá sớm trong $V_K$; (8) `is_op_reentry` — cờ tác giả bài gốc quay lại bình luận; (9) `author_repeat_norm` — số lần xuất hiện trước đó của cùng tác giả trong $V_K$ (trong `TextMLP`, các cột tô-pô `{1, 4, 6, 7}` của `struct_x` và `{3, 4, 7}` của `cascade_stats` bị khóa bằng 0 để giữ nguyên nghĩa phi cấu trúc).

---

## 3. Trích xuất Teacher–Student 2D Ngoại tuyến

### 3.1. Teacher Kép NLI & Hostility Độ phủ 100%
- **Chuẩn hóa văn bản xã hội (`normalize_social_text`)**: Rút gọn URL $\to$ `http` và gộp khoảng trắng cho **cả Tiếng Việt và Tiếng Anh**; ẩn `@handle` $\to$ `@user` cho Tiếng Anh (giữ nguyên `@handle` trên ngữ cảnh Việt).
- **Symmetric NLI Stance Verbalizer (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`)**: Với ngân sách `max_length = 384`, cấp phát token ưu tiên bảo toàn phần `Reply` trước khi cắt `Parent` sao cho `truncation="only_first"` không làm mất câu trả lời. Đánh giá đối xứng hai giả thuyết (`H_sup`: ủng hộ bình luận trước vs. `H_opp`: phản đối bình luận trước) và gán phần mâu thuẫn kép $m = \min(c_{\text{sup}}, c_{\text{opp}})$ về lớp trung lập trên đơn hình 3 lớp:
  $$p_{\text{sup}} = \tfrac{1}{2}\left(e_{\text{sup}} + (c_{\text{opp}} - m)_+\right), \quad p_{\text{opp}} = \tfrac{1}{2}\left(e_{\text{opp}} + (c_{\text{sup}} - m)_+\right), \quad p_{\text{neu}} = \tfrac{1}{2}(n_{\text{sup}} + n_{\text{opp}}) + m,$$
  sau đó tính $s_{uv} = p_{\text{sup}} - p_{\text{opp}} \in [-1, 1]$ kèm nhãn `root_s_uv` đối với nút sâu (`depth >= 2`). Checkpoint chấm điểm ghi delta JSONL gia tăng $\mathcal{O}(N)$ và tận dụng song song đa GPU `T4 x 2`.
- **Hostility Teacher ($h_{uv} \in [0, 1]$)**: Tinh chỉnh `uitnlp/visobert` với trọng số lớp cân bằng căn bậc hai nghịch đảo tần suất, warmup + cosine trên `UIT-ViHSD` (Tiếng Việt) hoặc dùng `cardiffnlp/twitter-roberta-base-offensive` (Tiếng Anh).

### 3.2. Chưng cất Student *Reply-Span Pooled* (`DualHeadCrossEncoder`)
- Đầu vào `(context_u, v)` (`context_u = root_head_tail [SEP] parent` theo ngân sách token khi bật `--include-root-context`, bảo toàn 60% đầu + 40% cuối bài gốc và ít nhất nửa ngân sách cho `reply`).
- Đầu lập trường gộp 5 khối tương tác quan hệ trên mặt nạ ngữ cảnh $\mathbf{h}_{\text{ctx}}$ (`context_mask == 1`) và mặt nạ phản hồi $\mathbf{h}_{\text{reply}}$ (`reply_mask == 1`) để dự đoán 3 logits `stance`:
  $$\mathbf{r}_{\text{rel}} = \left[\mathbf{h}_{\text{CLS}} \;\parallel\; \mathbf{h}_{\text{ctx}} \;\parallel\; \mathbf{h}_{\text{reply}} \;\parallel\; (\mathbf{h}_{\text{ctx}} \odot \mathbf{h}_{\text{reply}}) \;\parallel\; |\mathbf{h}_{\text{ctx}} - \mathbf{h}_{\text{reply}}|\right] \in \mathbb{R}^{5 d_{\text{PLM}}}$$
  đi qua phép chiếu `Linear -> LayerNorm -> GELU -> Dropout`; trong khi đầu gay gắt $h_{uv}$ **chỉ đọc vector phân đoạn phản hồi (`reply-span pooling`)** `Dropout(LayerNorm(h_reply))` để giảm thiểu nhiễu ngôn từ gay gắt từ bài cha sang bình luận con hiền hòa.
- Tối ưu $T^2 \cdot \operatorname{KL}(T=2.0) + 0.5 \cdot \operatorname{SmoothL1}(s_{uv}) + \operatorname{BCEWithLogitsLoss}(h_{uv})$ trên $\mathcal{T}_{\text{train}}$, tăng cường thêm cặp sâu `(root, deep_reply)` và dùng AdamW phân tầng tốc độ học (encoder vs. heads) hỗ trợ FP16 trên `nn.DataParallel`. Khi `scored_cache_json` đã phủ 100% các chuỗi đủ điều kiện (`--min-total-replies 25`), `extract_features.py` chỉ mã hóa các chuỗi đủ điều kiện và lưu giữ `teacher_edge_*` / `teacher_node_root_stance` sạch cho giao thức PHEME LOO.

---

## 4. Kiến trúc GNN SBi-FAGAT (Kênh Kép & Dir-GNN)

Tại tầng $l \in \{1, \dots, L\}$ ($L=2$, $d=64$) trên hai nhánh có hướng độc lập $\delta \in \{\text{TD}, \text{BU}\}$:
$$\mathbf{z}_u^{(l, \delta)} = \mathbf{W}^{(l, \delta)} \mathbf{h}_u^{(l-1, \delta)}, \qquad \tilde{\mathbf{z}}_u^{(l, \delta)} = \operatorname{LayerNorm}\left(\mathbf{z}_u^{(l, \delta)}\right)$$
$$\gamma_{uv}^{(l, \delta)} = \tanh\left(\frac{\left(\mathbf{a}_{\text{node}}^{(l,\delta)}\right)^\top [\tilde{\mathbf{z}}_u^{(l,\delta)} \parallel \tilde{\mathbf{z}}_v^{(l,\delta)}]}{\sqrt{2d}} + \omega \cos(\tilde{\mathbf{z}}_u^{(l,\delta)}, \tilde{\mathbf{z}}_v^{(l,\delta)}) + \left(\mathbf{a}_{\text{edge}}^{(l,\delta)}\right)^\top \mathbf{a}_{uv} + \delta_\gamma(\mathbf{a}_{uv})\right)$$
với khởi tạo $\mathbf{a}_{\text{node}}^{(l,\delta)}$ mức gain $0.2$ và $\omega^{(0)} = 0.10$.
$$\alpha_{uv}^{\text{exc}} = \sigma\left(0.8|s_{uv}| + 0.8 h_{uv} + \delta_\alpha(\mathbf{a}_{uv})\right)$$

Thông điệp chia 2 nửa kênh: nửa đầu ($d/2$ chiều) truyền phân cực có dấu $\gamma_{uv} \mathbf{z}_u[:d/2]$; nửa sau ($d/2$ chiều) truyền kích thích không dấu $\alpha_{uv}^{\text{exc}} \mathbf{z}_u[d/2:]$, chuẩn hóa bậc đối xứng $c_{uv}^\delta = \sqrt{(d_{\text{out}}^\delta(u)+1)(d_{\text{in}}^\delta(v)+1)}$:
$$\mathbf{r}_v^{(l, \delta)} = 0.5\,\mathbf{h}_v^{(l-1, \delta)} + \varepsilon \mathbf{h}_v^{(0)} + \mathbf{m}_v^{(l, \delta)} + \mathbf{W}_{\text{root}}^{(l, \delta)} \mathbf{h}_r^{(0)} + \mathbf{b}_{\text{root}}^{(l, \delta)}, \qquad \mathbf{h}_v^{(l, \delta)} = \operatorname{Dropout}\left(\operatorname{LayerNorm}\left(\mathbf{r}_v^{(l,\delta)} + 0.2\tanh(\mathbf{r}_v^{(l,\delta)})\right)\right).$$

Sau mỗi tầng, **Dir-GNN Cross-Mixing** ($\eta_l^{(0)} = 0.10$) trộn hai hướng với hệ số học được $\tanh(\eta_l)$:
$$\mathbf{h}^{\text{TD}} \leftarrow \mathbf{h}^{\text{TD}} + \tanh(\eta_l) \mathbf{W}_{\text{mix}}^{\text{TD}} \mathbf{h}^{\text{BU}}, \qquad \mathbf{h}^{\text{BU}} \leftarrow \mathbf{h}^{\text{BU}} + \tanh(\eta_l) \mathbf{W}_{\text{mix}}^{\text{BU}} \mathbf{h}^{\text{TD}}.$$

Ở tầng Readout (**CleanRes Continuous Camp Dipole**), ngoài nhúng nền $\mathbf{z}_{\text{base}} = [\mathbf{z}_{\text{root}} \parallel \mathbf{z}_{\text{att}}] \in \mathbb{R}^{2d}$, độ phân tán nút $\mathbf{z}_{\text{disp}}$ và hàm `_propagate_root_aligned_edge_signal` (tối đa **64 bước BFS**) quy chiếu dấu cổng tầng cuối về gốc $r$ bằng hệ số định hướng cha tách gradient $p_{\text{sign}}(u) = \operatorname{clip}(\operatorname{detach}(u_{\text{ori}})/0.1, -1, 1) \in [-1, 1]$ (tránh bùng nổ Jacobian qua chuỗi sâu) để tính cặp cực phe tuyến tính mảnh liên tục:
$$w_{uv}^+ = \operatorname{ReLU}(\hat{\gamma}_{uv}), \quad w_{uv}^- = \operatorname{ReLU}(-\hat{\gamma}_{uv}), \quad \boldsymbol{\mu}_\pm = \frac{\sum w_{uv}^\pm \mathbf{h}_v}{\sum w_{uv}^\pm + 1}, \quad b = 4 f_+ f_-,$$
$$\mathbf{z}_{\text{dipole}}^{\text{polar}} = b\sqrt{(\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-)^2 + 10^{-8}}, \qquad \mathbf{z}_{\text{dipole}}^{\text{signed}} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-,$$
cùng 6 mô-men cổng vô hướng (`g_mean, g_std, g_abs_mean, frac_pos, frac_neg, camp_balance`). Khối thắt cổ chai $\operatorname{dipole\_proj}: \mathbb{R}^{3d+6} \to \mathbb{R}^{\max(16, d/2)} \to \mathbb{R}^{2d}$ điều biến phần dư qua $\tanh(\beta_{\text{dip}})$ ($\beta_{\text{dip}}^{(0)} = 0.25$) để tạo biểu diễn đồ thị $\mathbf{z}_{\text{graph}} = \operatorname{LayerNorm}(\mathbf{z}_{\text{base}} + \tanh(\beta_{\text{dip}})\,\operatorname{dipole\_proj}([\mathbf{z}_{\text{disp}} \parallel \mathbf{z}_{\text{dipole}}^{\text{polar}} \parallel \mathbf{z}_{\text{dipole}}^{\text{signed}} \parallel \mathbf{g}_{\text{mom}}]))$ (khi tắt *w/o Camp Dipole & Moments*, nhánh thắt cổ chai bị lược bỏ và chỉ chuẩn hóa $\operatorname{LayerNorm}(\mathbf{z}_{\text{base}})$). Bên trong `MultiTaskHead`, đầu `polar_head` được **tách hoàn toàn khỏi `cascade_stats`** (chỉ nhận `graph_emb` cho mọi mô hình) nhằm đảm bảo dự đoán phân cực dựa thuần túy vào biểu diễn cấu trúc–lập trường.

---

## 5. Nhãn Mục tiêu & Tối ưu Đa nhiệm

1. **Nhãn mục tiêu liên tục khử chệch cỡ mẫu trước phép kẹp `[0, 1]`**:
   $$y_{\text{size}} = \ln(1 + N_{\text{total}}), \qquad y_{\text{polar}} = \operatorname{clip}\!\left(0.5\,\min(1, s_{\tilde{s}, \text{ddof}=1}^2) + 0.5\,\min\!\left(1, \frac{4n}{n-1} m_+ m_-\right),\, 0,\, 1\right),$$
   trong đó $m_+ = \frac{1}{n}\sum_i \max(\tilde{s}_i, 0)$, $m_- = \frac{1}{n}\sum_i \max(-\tilde{s}_i, 0)$ tính trên $n \ge 2$ tương tác tương lai $t > t_K$ (`polar_target_mode="future_only"`, với cửa sổ đích cố định $t > K_{\max}$ trong đường cong độ nhạy $K$; hệ số $\frac{n}{n-1}$ khử đúng độ chệch cỡ mẫu của tích $m_+ m_-$ *trước* phép kẹp $\min(1, \cdot)$ dưới giả định rút mẫu i.i.d.). Hai biến kiểm chứng cấu trúc độc lập `y_bimodality` (Sarle BC) và `y_assortativity` (Signed Newman Assortativity) được gắn kèm để báo cáo tương quan `construct_corr_bc` và `construct_corr_assort`.
2. **Hàm mất mát Đa nhiệm Chuẩn hóa Hiệu chỉnh KL**:
   - Đầu $y_{\text{polar}}$ xuất cả logit $z_{\text{polar}}$ và $\hat{y}_{\text{polar}} = \sigma(z_{\text{polar}})$.
   - Sử dụng **KL-calibrated `BCEWithLogitsLoss`** (trừ đi entropy mục tiêu $H(y_{\text{polar}})$ để giá trị cực tiểu đúng tại $\sigma(z)=y$ bằng chính xác `0.0`) cộng `SmoothL1(..., beta=0.08)`:
     $$\mathcal{L}_{\text{polar}} = \left(\operatorname{BCEWithLogits}(z_{\text{polar}}, y_{\text{polar}}) - H(y_{\text{polar}})\right)_+ + \mathcal{L}_{0.08}^{\text{SmoothL1}}(\hat{y}_{\text{polar}}, y_{\text{polar}}).$$
   - Trọng số cân bằng chuẩn hóa theo loss cơ sở trung bình hằng trên `train_loader` theo tỷ lệ 2:1 ($s_{\text{size}} = \operatorname{clip}(1.0/\max(L_{\text{base}}^{\text{size}}, 10^{-4}), 0.5, 16.0)$, $s_{\text{polar}} = \operatorname{clip}(0.5/\max(L_{\text{base}}^{\text{polar}}, 10^{-4}), 0.5, 250.0)$) và khởi tạo bias cho `size_head`/`polar_head` (kẹp $p \in [10^{-3}, 0.99]$) được ước lượng **thuần túy trên `train_loader`**.
   - Lịch học **Linear Warmup + `ReduceLROnPlateau`** cùng **EMA (`decay=0.985`, bật khi `epochs >= 10`)**, dừng sớm theo `val task_loss` và chọn seed có `val_task_loss` nhỏ nhất làm checkpoint chính thức của mỗi mô hình qua 5 hạt giống `[42, 100, 2024, 777, 999]`.
