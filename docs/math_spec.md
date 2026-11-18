# Đặc tả Toán học Chính thức & Thiết kế Hệ thống: Kiến trúc SBi-FAGAT

Tài liệu này trình bày đánh giá thiết kế tổng quan (theo chuẩn *Heilmeier Catechism*), bảng ký hiệu thống nhất, đặc tả trích xuất đặc trưng NLP ngoại tuyến, phương trình truyền thông điệp hai hướng thích ứng tần số và giao thức thực nghiệm của kiến trúc **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network*).

---

## 0. Tóm tắt Thiết kế & Đánh giá Kỹ thuật (Heilmeier Assessment)

### 0.1. Mục tiêu Phi kỹ thuật & Đối tượng Thụ hưởng
- **Bài toán thực tế**: Khi một thông tin nóng hoặc tin thất thiệt bắt đầu lan truyền trên mạng xã hội, nền tảng kiểm duyệt và nhóm ứng phó khủng hoảng truyền thông chỉ có một cửa sổ thời gian rất ngắn (khoảng $K = 20$ bình luận đầu tiên) trước khi làn sóng chia rẽ hoặc tin giả bùng phát vượt kiểm soát. Nếu chờ hàng nghìn lượt tương tác để thống kê hay phân tích cộng đồng đầy đủ, mọi biện pháp cảnh báo sớm đều trở nên muộn màng.
- **Mục tiêu**: Xây dựng mô hình dự đoán sớm từ bài đăng gốc và tối đa $20$ tương tác đầu để ước lượng đồng thời: (1) mức độ lan rộng cuối cùng của chuỗi hội thoại, (2) mức độ chia rẽ hai phe ủng hộ–phản bác về cuối kỳ, và (3) khả năng đây là tin đồn/tin giả.

### 0.2. Hiện trạng & Điểm mù Kỹ thuật
- Các mô hình chuỗi thời gian/đồ thị lan truyền truyền thống (*DeepCas*, *GCN*, *GATv2*, *Bi-GCN*) bỏ qua thái độ đồng tình hay phản bác giữa từng cặp bình luận và dùng bộ lọc trung bình (low-pass) làm nhòa sự đối lập giữa hai phe.
- Các thước đo phân cực cấu trúc (như *RWC* hay *Modularity*) đòi hỏi đồ thị mạng lưới bạn bè/người theo dõi quy mô lớn đã trưởng thành, bị suy biến khi chỉ quan sát một cây hội thoại nhỏ có $\le 21$ nút (mã nguồn vì vậy không tính RWC/Modularity mà triển khai hệ số lưỡng cực *Sarle's Bimodality Coefficient* và *Signed Newman Assortativity* trên cây). Chi tiết đối sánh 12 công trình nền tảng được tổng hợp tại [docs/literature_review.md](literature_review.md).

### 0.3. Phương pháp Đề xuất & Phương án Thay thế bị Bác bỏ
- **Ý tưởng cốt lõi**: Kết hợp bộ mã hóa ngữ nghĩa đơn câu đóng băng trên từng nút, mô hình suy luận cặp câu (Cross-Encoder) để gán tín hiệu lập trường $s_{uv} \in [-1, 1]$ và độ gay gắt $h_{uv} \in [0, 1]$ cho từng cạnh trả lời, rồi đưa trực tiếp cặp tín hiệu này vào cổng thích ứng tần số có dấu $\gamma_{uv} \in (-1, 1)$ trên hai nhánh cây có hướng độc lập Top-Down ($\text{TD}$) và Bottom-Up ($\text{BU}$).
- **Phương án thay thế đã cân nhắc & bác bỏ**:
  1. *Fine-tune end-to-end Transformer + GNN cùng lúc*: Với mỗi cây có $K=20$ văn bản ($L_{\text{tok}} \le 256$), chi phí bộ nhớ $\mathcal{O}(K \cdot L_{\text{tok}}^2)$ vượt quá giới hạn 16GB VRAM khi huấn luyện multi-seed và ablation hàng chục lượt. Tách pha NLP ra tệp `.pt` giúp giai đoạn GNN tiêu thụ $< 500\text{ MB}$ VRAM.
  2. *Suy luận lập trường rời rạc nhị phân $\{-1, +1\}$ (SGCN)*: Đánh mất phần lớn bình luận trung lập/hỏi thăm trên mạng xã hội và giả định cân bằng cấu trúc vô hướng không hợp với cây trả lời có hướng.
  3. *Chỉ dùng teacher NLI không tách kiểm soát chưng cất*: Việc giữ `DualHeadCrossEncoder` chưng cất cô lập chỉ trên $\mathcal{T}_{\text{train}}$ (tuyệt đối không chạm $\mathcal{T}_{\text{test}}$) cho phép kiểm định tổng quát hóa lẫn chạy kiểm chứng phản chứng `--sanity-check-shuffle`, trong khi vectơ nút $\mathbf{e}_v$ luôn lấy từ backbone đơn câu đóng băng gốc.

### 0.4. Rủi ro, Giới hạn & Biện pháp Kiểm soát
1. **Tuần hoàn nhãn $y_{\text{polar}}$, suy biến phương sai tại $N=21$ và lệch phân hoạch giữa Distillation–GNN**:
   - Nếu lấy `min_total_replies = 21` với $K=20$ và `polar_target_mode="future_only"`, mỗi cây biên $N_{\text{total}}=21$ chỉ còn đúng **1 bình luận tương lai**, làm $\sigma_{\tilde{s}}^2 = 0$ và ép `4 p_+ p_- = 0` giả tạo. Hệ thống nay mặc định `DEFAULT_FORECAST_MIN_REPLIES = 25` ($\ge 5$ bình luận tương lai) kèm phân hoạch chủ đề **chuẩn hóa duy nhất** `topic_splits.json` dùng chung từ `dual_edge_scorer.py` tới `EarlyCascadeDataset` (chặn tuyệt đối rò rỉ khi có nhiều cascade cùng `topic_id` qua ngưỡng lọc khác nhau).
   - Trên dữ liệu Việt Nam, `cluster_vn_cascade_topics` gộp các chủ đề liên thông theo tương đồng cosin TF-IDF ($> 0.55$) trong cửa sổ trượt 14 ngày. Trong giao thức LOO trên PHEME, `edge_source="teacher"` cô lập Student Cross-Encoder khỏi mọi fold giữ lại.
2. **Chất lượng nhãn Teacher 4D & Tránh sập phương sai dị biến (*heteroscedastic NLL*)**:
   - Teacher `Qwen/Qwen3.8-27B` QLoRA (hoặc `Qwen/Qwen3-8B` trên GPU 16GB đơn) dùng **G-Eval Single-Pass Logit Scoring** (`scoring_mode="logit"` đọc logit 1 lượt trên chữ số `1..5` + kỳ vọng prototype `[s_uv, h_uv, r_uv, c_uv]`) giúp **0% lỗi JSON** và tăng tốc 10–20$\times$.
   - Đầu bất định dùng **$\beta$-NLL** ($\beta=0.5$, Seitzer et al., ICLR 2022) với phần dư tách gradient `detach()`, biên log-phương sai mượt `-6 + 9*sigmoid(...)`, và điểm tự tin $\exp(-\hat{\sigma}_{\text{polar}} / 0.25)$ trong $(0, 1]$ kèm dừng sớm thuần `task_loss`.
3. **Ẩn danh PII & Công bằng 8 mô hình**: Mọi mã tác giả `author` được băm SHA-256 trước khi xuất; đặc trưng tốc độ `cascade_stats` (8 chiều) được mở công bằng cho **cả 8 mô hình** khi bật `--v2`.

### 0.5. Chi phí Hạ tầng (16GB VRAM / Kaggle `T4 x 2`), Lộ trình & Cổng Nghiệm thu
- **Chi phí tính toán**: `Qwen/Qwen3.8-27B` NF4 chạy 4-bit phân mảnh tự động trên Kaggle `T4 x 2` với QLoRA có mặt nạ prompt (`labels[:prompt_len] = -100`) và G-Eval 1-pass logit (~15–25 phút thay vì 12–20 giờ sinh chuỗi 96 token); Student 4D `ViSoBERT`/`DeBERTa-v3-base` + `node_embeddings` lưu sẵn xuống `.pt` nên toàn bộ pha GNN Phase 3 chỉ dùng $< 0.5$ GB VRAM.
- **Các Cổng Nghiệm thu Bắt buộc**:
  - **Gate 0 (Contract & Split Parity)**: `pytest` xác nhận mọi cây có 1 gốc duy nhất, nhân quả UTC, `topic_splits.json` đồng nhất giữa Teacher distillation và GNN.
  - **Gate 1 (ViHSD & Human Audit)**: kiểm định mù 300 cặp $(u, v)$ cả stance 3 lớp lẫn `behavior` 5 lớp (`compute_kappa.py`).
  - **Gate 2 (Distillation & Anti-Cheating)**: `fallback_rate <= max_fallback_rate`, Student khớp Teacher 4D trên Val split và `--sanity-check-shuffle` đẩy sai số về ngẫu nhiên.
  - **Gate 3 (Statistical Benchmark)**: 5 hạt giống $\mathcal{S} = \{42, 100, 2024, 777, 999\}$ với `alpha_polar = 4.0`, `EarlyStopping` theo `task_loss`, LOO cô lập `edge_source="teacher"`, kiểm định `ttest_rel` + Paired Bootstrap **từng seed rồi lấy trung bình trên $S$ seeds** kèm điều chỉnh Holm-Bonferroni.

---

## 1. Bảng Ký hiệu Toán học Thống nhất

| Ký hiệu | Miền giá trị / Kích thước | Ý nghĩa |
| :--- | :--- | :--- |
| $\mathcal{G}_{\text{full}} = (\mathcal{V}_{\text{full}}, \mathcal{E}_{\text{full}})$ | Đồ thị cây có hướng | Cây lan truyền đầy đủ ở trạng thái cuối kỳ |
| $\mathcal{G}_K = (\mathcal{V}_K, \mathcal{E}_K)$ | Đồ thị cây có hướng | Cây lan truyền sớm cắt tại ngưỡng $K \in \{10, 20, 30, 50\}$ tương tác đầu |
| $r$ | $r \in \mathcal{V}_K$ | Nút gốc (bài đăng khởi nguồn) tại thời điểm $t_r$ |
| $N_{\text{total}}$ | $\mathbb{Z}_{\ge 0}$ | Tổng số tương tác phản hồi cuối kỳ: $N_{\text{total}} = |\mathcal{V}_{\text{full}}| - 1$ |
| $t_v, \Delta t_v$ | $\mathbb{R}_{\ge 0}$ (giây UTC) | Dấu thời gian tạo nút $v$ và độ trễ tuyệt đối $\Delta t_v = \max(0, t_v - t_r)$ |
| $T_{\text{ref}}$ | $604\,800\text{ giây}$ (7 ngày) | Ngưỡng tham chiếu thời gian nhân quả cố định chuẩn hóa $\tilde{\Delta t}_v$ |
| $\ell_v$ | $\{0, 1, \dots, N_{\text{total}}\}$ | Độ sâu tô-pô của nút $v$ tính từ nút gốc $r$ ($\ell_r = 0$) |
| $L_{\text{tok}}(v), m_{v,i}, \mathbf{u}_{v,i}$ | $\mathbb{Z}_{>0}, \{0,1\}, \mathbb{R}^{768}$ | Độ dài token (tối đa $256$), mặt nạ chú ý và vector ẩn tầng cuối PLM tại token $i$ |
| $\mathbf{e}_v^{\text{nlp}}$ | $\mathbb{R}^{768}$ | Vector ngữ nghĩa nút từ **encoder đơn câu đóng băng gốc** (`roberta-base` / `visobert`) |
| $\mathbf{x}_v$ | $\mathbb{R}^{770}$ | Vector đặc trưng nút lắp ghép động tại $K$: $[\mathbf{e}_v^{\text{nlp}} \parallel \tilde{\Delta t}_v \parallel \tilde{\ell}_v]$ |
| $\mathbf{h}_v^{(0)}$ | $\mathbb{R}^{d}$ | Biểu diễn nút khởi tạo sau lớp chiếu đầu vào ($d = 128$ mặc định) |
| $s_{uv}, \tilde{s}_v$ | $[-1, 1]$ | Điểm lập trường cạnh cục bộ $(u \to v)$ và lập trường nút $v$ **quy chiếu về gốc $r$** |
| $h_{uv}$ | $[0, 1]$ | Điểm độ gay gắt ngôn từ (*hostility intensity*) của phản hồi $v$ trên cạnh $(u, v)$ |
| $\mathbf{a}_{uv}$ | $\mathbb{R}^2$ | Vector thuộc tính cạnh 2 chiều: $\mathbf{a}_{uv} = [s_{uv}, h_{uv}]^\top$ |
| $\delta \in \{\text{TD}, \text{BU}\}$ | Chỉ số hướng | Hướng truyền thông điệp Top-Down ($\text{TD}$) và Bottom-Up ($\text{BU}$) |
| $\gamma_{uv}^{(l, \delta)}$ | $(-1, 1)$ | Cổng thích ứng tần số không-trạng-thái ở tầng $l$, hướng $\delta$ |
| $q_v^{\delta}, \beta_v^{\delta}$ | $\mathbb{R}, [0, 1]$ | Điểm chú ý và trọng số chuẩn hóa softmax tại nút $v$ trong Global Attention Readout |
| $\mathbf{z}_{\mathcal{G}_K}$ | $\mathbb{R}^{4d}$ | Vector biểu diễn toàn đồ thị ghép nối 2 hướng $\times$ 2 kiểu pooling |
| $y_{\text{size}}, \hat{y}_{\text{size}}$ | $[0, \infty)$ | Quy mô lan truyền cuối kỳ $y_{\text{size}} = \ln(1 + N_{\text{total}})$ và dự đoán |
| $y_{\text{polar}}, \hat{y}_{\text{polar}}$ | $[0, 1]$ | Chỉ số phân cực bạc (*silver-standard bipolar*) và giá trị dự đoán |
| $y_{\text{rumor}}, \hat{\mathbf{p}}_{\text{rumor}}$ | $\{0, \dots, C-1\}$, $\Delta^{C-1}$ | Nhãn tin đồn/xác thực trên PHEME (`veracity4` $C=4$ hoặc `binary` $C=2$) |

---

## 2. Định nghĩa Cây Lan truyền Sớm $\mathcal{G}_K$ ($K=20$) & Bất biến Chống Rò rỉ

### 2.1. Xây dựng Cây Lan truyền Sớm $\mathcal{G}_K = (\mathcal{V}_K, \mathcal{E}_K)$
Xét một cây hội thoại hoàn chỉnh ở thời điểm kết thúc thu thập dữ liệu $\mathcal{G}_{\text{full}} = (\mathcal{V}_{\text{full}}, \mathcal{E}_{\text{full}})$ có gốc duy nhất $r \in \mathcal{V}_{\text{full}}$, trong đó mỗi cạnh có hướng $(u, v) \in \mathcal{E}_{\text{full}}$ biểu thị nút $v$ trả lời hoặc trích dẫn trực tiếp nút cha $u$, thỏa mãn tính nhân quả thời gian:

$$t_u \le t_v, \quad \forall (u, v) \in \mathcal{E}_{\text{full}}$$

Gọi $N_{\text{total}} = |\mathcal{V}_{\text{full}}| - 1$ là tổng số tương tác phản hồi hạ nguồn. Sắp xếp tất cả các nút phản hồi $\mathcal{V}_{\text{full}} \setminus \{r\} = \{v_1, v_2, \dots, v_{N_{\text{total}}}\}$ theo thứ tự từ điển tăng dần của bộ khóa $(t_v, \ell_v, \texttt{node\_id}(v))$ sao cho:

$$t_{v_1} \le t_{v_2} \le \dots \le t_{v_{N_{\text{total}}}}$$

Với ngưỡng quan sát sớm $K \in \{10, 20, 30, 50\}$ (mặc định $K = 20$) và bộ lọc dự báo sớm chuẩn $N_{\text{total}} \ge \texttt{min\_total\_replies} = 25$ (bảo đảm $\ge 5$ bình luận tương lai ngoài $K=20$ để tránh suy biến phương sai tại $N=21$), cây lan truyền sớm $\mathcal{G}_K = (\mathcal{V}_K, \mathcal{E}_K)$ được trích xuất theo đúng thứ tự thời gian:

$$\mathcal{V}_K = \{r\} \cup \left\{v_1, v_2, \dots, v_{\min(K, N_{\text{total}})}\right\}$$

$$\mathcal{E}_K = \left\{(u, v) \in \mathcal{E}_{\text{full}} \;\middle|\; u \in \mathcal{V}_K \land v \in \mathcal{V}_K\right\}$$

**Bổ đề 1 (Bảo toàn cấu trúc cây có gốc liên thông).** *Do mọi nút $v \in \mathcal{V}_K \setminus \{r\}$ có nút cha $u$ thỏa mãn $t_u \le t_v$ (bộ phân giải áp dụng kẹp nhân quả `effective_dt` nếu đồng hồ lệch nhẹ) và thứ tự phá vỡ đồng hạng ưu tiên $\ell_u < \ell_v$, mọi tổ tiên của $v$ đều nằm trong $\mathcal{V}_K$. Vì vậy, $\mathcal{G}_K$ là một cây có hướng liên thông tại gốc $r$, với $1 \le |\mathcal{V}_K| \le K + 1$ và $|\mathcal{E}_K| = |\mathcal{V}_K| - 1$.*

### 2.2. Tiên đề Chống Rò rỉ Dữ liệu (*Zero Data Leakage*)
Để bảo đảm tính hợp lệ khoa học tuyệt đối trong dự đoán sớm:
1. **Chia tập chuẩn hóa theo chủ đề/sự kiện (`topic_splits.json`)**:
   - Phân hoạch được tính một lần từ đồ thị nguồn sau lọc `min_total_replies` và chia sẻ đồng nhất cho `dual_edge_scorer.py`, `extract_features.py` và `EarlyCascadeDataset` (chặn hoàn toàn lệch chủ đề giữa distillation và GNN).
   - Trên mạng xã hội Việt Nam (`r/vozforums`, `r/VietNam`): `cluster_vn_cascade_topics` gom nhóm các cây có tương đồng TF-IDF $> 0.55$ trong cửa sổ trượt $14$ ngày (kết hợp từ khóa tiêu đề & tuần ISO, gọi là *Heuristic Temporal-Keyword / TF-IDF Grouped Inductive Split* nhằm chặn rò rỉ chuỗi thảo luận cùng sự kiện trong cửa sổ thời gian dù không bảo đảm độc lập ngữ nghĩa tuyệt đối như các tin tức lớn), chia 70/15/15 với $\mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{val}} = \mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{test}} = \mathcal{T}_{\text{val}} \cap \mathcal{T}_{\text{test}} = \emptyset$.
   - Trên PHEME: hỗ trợ cả *Inductive Topic Split* cân bằng lớp lẫn **Leave-One-Event-Out (LOO)** với `edge_source="teacher"` cô lập Student khỏi mọi fold giữ lại.
2. **Cô lập tuyệt đối quá trình chưng cất Cross-Encoder & QLoRA**: Teacher QLoRA chỉ hiệu chỉnh trên nguồn ngoài độc lập (`UIT-ViHSD`, `RumourEval`, 25 rubric anchors cân bằng 5 lớp); Student `DualHeadCrossEncoder` chỉ chưng cất trên $\mathcal{T}_{\text{train}}$, không chạm $\mathcal{T}_{\text{test}}$.
3. **Cô lập đầu vào nhân quả ở chân trời $K$**: Toàn bộ vector đặc trưng nút, ma trận kề, thuộc tính cạnh $\mathbf{a}_{uv}$ và vector tốc độ sớm 8 chiều `cascade_stats` chỉ tính trên $\mathcal{G}_K$; biến hậu nghiệm ($N_{\text{total}}$, $y_{\text{size}}$, $y_{\text{polar}}$) tuyệt đối chỉ dùng tính Loss.

---

## 3. Trích xuất Đặc trưng NLP Ngoại tuyến (Teacher 4D & Student Cross-Encoder)

Nhằm vận hành gọn nhẹ trong giới hạn VRAM 16GB / Kaggle `T4 x 2`, giai đoạn mã hóa ngôn ngữ tự nhiên chạy ngoại tuyến một lần (bật Mixed Precision `torch.amp`) và lưu sẵn xuống `.pt`.

### 3.1. Nhánh Teacher G-Eval 4D & Chưng cất Cross-Encoder Thống nhất
Hệ thống hỗ trợ Teacher 4 chiều mặc định (`qwen_qlora`, `edge_dim=4`) và nhánh 2 chiều dự phòng (`legacy_nli`, `edge_dim=2`):
1. **G-Eval Single-Pass Logit Teacher (`Qwen/Qwen3.8-27B` QLoRA, Kaggle `T4 x 2`, hoặc `Qwen/Qwen3-8B` GPU 16GB đơn)**:
   Với bộ ba $(r, u, v)$, chỉ thực hiện **1 lượt forward** trích xác suất softmax $q \in \Delta^4$ trên 5 token chữ số `1..5` (`support`, `neutral`, `constructive_oppose`, `irrational_fallacy`, `hostile_conflict`) và chiếu kỳ vọng ra vector hành vi 4D $\mathbf{a}_{uv} = [s_{uv}, h_{uv}, r_{uv}, c_{uv}]^\top \in [-1, 1] \times [0, 1]^3$.
2. **Chưng cất Cross-Encoder Thống nhất (`DualHeadCrossEncoder`)**:
   Sử dụng `uitnlp/visobert` (`vi`) hoặc `microsoft/deberta-v3-base` (`en`, chuyển tiếp an toàn `token_type_ids`). Tùy chọn `--include-root-context` ghép ngữ cảnh gốc `root[:220] | parent` vào vế trái. Ngoài 2 đầu $(s_{uv}, h_{uv})$, nhánh 4D bổ sung đầu hành vi 5 lớp ($\mathbf{z}_{\text{beh}}$), độ vô lý $r_{uv}$ và tính xây dựng $c_{uv}$ chưng cất bằng `KL + SmoothL1` trên $\mathcal{T}_{\text{train}}$.

### 3.2. Biểu diễn Đặc trưng Nút Khởi tạo $\mathbf{h}_v^{(0)}$
Mỗi nút $v \in \mathcal{V}_K$ được gán ba thành phần đặc trưng không rò rỉ dữ liệu:

1. **Vector ngữ nghĩa văn bản $\mathbf{e}_v^{\text{nlp}} \in \mathbb{R}^{768}$**: Bắt buộc lấy từ phép gộp trung bình có mặt nạ (*masked mean pooling*) của tầng ẩn cuối cùng từ **mô hình PLM đơn câu đóng băng gốc** (`roberta-base` / `uitnlp/visobert`, tuyệt đối không dùng encoder của student đã chưng cất để chặn rò rỉ):

$$\mathbf{e}_v^{\text{nlp}} = \frac{\sum_{i=1}^{L_{\text{tok}}(v)} m_{v,i} \, \mathbf{u}_{v,i}}{\max\left(10^{-8}, \sum_{i=1}^{L_{\text{tok}}(v)} m_{v,i}\right)} \in \mathbb{R}^{768}$$

2. **Độ trễ thời gian nhân quả bảo toàn vận tốc ($\tilde{\Delta t}_v \in [0, 1]$)**: Sử dụng hằng số tham chiếu nhân quả cố định $T_{\text{ref}} = 7 \text{ ngày} = 604\,800\text{ giây}$ để bảo toàn tốc độ lan truyền tuyệt đối (phân biệt rõ chuỗi bùng phát trong vài phút với chuỗi kéo dài nhiều ngày mà không phụ thuộc độ trễ nút tương lai):

$$\tilde{\Delta t}_v = \min\left(1.0, \;\frac{\ln(1 + \Delta t_v)}{\ln(1 + T_{\text{ref}})}\right) \in [0, 1]$$

*(Trường hợp nút gốc $r$: $\Delta t_r = 0 \implies \tilde{\Delta t}_r = 0$.)*

3. **Độ sâu cây chuẩn hóa**: Với khoảng cách đường đi ngắn nhất từ gốc $r$ đến nút $v$ ký hiệu là $\ell_v = \operatorname{dist}_{\mathcal{G}_{\text{full}}}(r, v)$ (bằng bất biến tô-pô nhân quả trên cây sớm $\mathcal{V}_K$):

$$\tilde{\ell}_v = \min\left(1.0, \;\frac{\ell_v}{K}\right) \in [0, 1]$$

Ghép nối động ba thành phần trong `EarlyCascadeDataset` thu được vector đặc trưng thô $\mathbf{x}_v \in \mathbb{R}^{770}$ và chiếu vào không gian ẩn $d$ chiều (`Linear -> GELU -> LayerNorm`):

$$\mathbf{x}_v = \left[\mathbf{e}_v^{\text{nlp}} \;\parallel\; \tilde{\Delta t}_v \;\parallel\; \tilde{\ell}_v\right] \in \mathbb{R}^{770}, \qquad \mathbf{h}_v^{(0)} = \operatorname{LayerNorm}\left(\operatorname{GELU}\left(\mathbf{W}_{\text{in}} \mathbf{x}_v + \mathbf{b}_{\text{in}}\right)\right) \in \mathbb{R}^{d}$$

---

## 4. Truyền Thông điệp Hai hướng Thích ứng Tần số (SBi-FAGAT)

### 4.1. Hai Kênh Đồ thị Có hướng Độc lập (Top-Down & Bottom-Up)
Từ cây $\mathcal{G}_K = (\mathcal{V}_K, \mathcal{E}_K)$, tách thành hai đồ thị có hướng chạy song song với hai bộ trọng số độc lập:
- **Nhánh Top-Down ($\delta = \text{TD}$)**: Giữ nguyên chiều nhân quả từ bài cha xuống bình luận con, mô hình hóa *đà lan truyền thông tin*:

$$\mathcal{E}_K^{\text{TD}} = \left\{(u \to v) \;\middle|\; (u, v) \in \mathcal{E}_K\right\}, \quad s_{uv}^{\text{TD}} = s_{uv}$$

- **Nhánh Bottom-Up ($\delta = \text{BU}$)**: Đảo chiều cạnh từ bình luận con dội ngược về bài cha, mô hình hóa *phản hồi, kiểm chứng và áp lực phản bác từ cộng đồng* với cùng thuộc tính cặp câu $\mathbf{a}_{vu}^{\text{BU}} = \mathbf{a}_{uv} = [s_{uv}, h_{uv}]^\top$:

$$\mathcal{E}_K^{\text{BU}} = \left\{(v \to u) \;\middle|\; (u, v) \in \mathcal{E}_K\right\}, \quad \mathbf{a}_{vu}^{\text{BU}} = \mathbf{a}_{uv}$$

Trên cả hai nhánh, khởi tạo trạng thái tầng $0$ giống nhau: $\mathbf{h}_v^{(0, \text{TD})} = \mathbf{h}_v^{(0, \text{BU})} = \mathbf{h}_v^{(0)}$ cho mọi $v \in \mathcal{V}_K$.

### 4.2. Cổng Thích ứng Tần số Tiền-Chuẩn hóa Dẫn hướng bởi $\mathbf{a}_{uv}$
Tại tầng $l \in \{1, \dots, L\}$ ($L=2$ mặc định) của nhánh $\delta \in \{\text{TD}, \text{BU}\}$, xét cạnh có hướng $(u \to v) \in \mathcal{E}_K^\delta$ mang vector thuộc tính $\mathbf{a}_{uv}^\delta = [s_{uv}, h_{uv}, r_{uv}, c_{uv}]^\top \in \mathbb{R}^4$ (hoặc 2D $[s_{uv}, h_{uv}]^\top$, biến `edge_attr`). Cổng thích ứng tần số thực hiện **Pre-Norm** trước tích vô hướng (Platonov et al., ICLR 2023) và khởi tạo trọng số cạnh 4D theo tiền nghiệm miền `[+0.5, -0.25, -0.25, +0.25]` (`[1/sqrt(2), 1/sqrt(2)]` cho 2D):

$$\mathbf{z}_u^{(l, \delta)} = \mathbf{W}^{(l, \delta)} \mathbf{h}_u^{(l-1, \delta)}, \quad \tilde{\mathbf{z}}_u^{(l, \delta)} = \operatorname{LayerNorm}\left(\mathbf{z}_u^{(l, \delta)}\right), \quad \mathbf{W}^{(l, \delta)} \in \mathbb{R}^{d \times d}$$

$$\gamma_{uv}^{(l, \delta)} = \tanh\left(\left(\mathbf{a}_{\text{att}}^{(l, \delta)}\right)^\top \left[\tilde{\mathbf{z}}_u^{(l, \delta)} \;\middle\|\; \tilde{\mathbf{z}}_v^{(l, \delta)} \;\middle\|\; \mathbf{a}_{uv}^\delta\right]\right) \in (-1, 1)$$

trong đó $\mathbf{a}_{\text{att}}^{(l, \delta)} \in \mathbb{R}^{2d + d_{\text{edge}}}$ là vector trọng số chú ý có dấu. Các biến thể cắt bỏ trong nghiên cứu Ablation gồm:
- `w/o Stance Score`: gán $s_{uv} = 0$.
- `w/o Hostility Score`: gán $h_{uv} = 0$.
- `w/o Irrationality` / `w/o Constructive`: gán $r_{uv} = 0$ hoặc $c_{uv} = 0$ (khi `edge_dim=4`).
- `w/o Edge Signals`: gán đồng thời toàn bộ thành phần $\mathbf{a}_{uv} = \mathbf{0}$.
- `w/o Conflict Gate`: thay $\tanh(\cdot)$ bằng $\sigma(\cdot) \in (0, 1)$, loại bỏ khả năng đảo dấu.
- `w/o Bottom-Up`: tắt hoàn toàn nhánh $\text{BU}$, nhân đôi vectơ nhánh $\text{TD}$ ở readout để bảo toàn dung lượng đầu dự đoán.
- `w/o Attention Readout`: thay Global Attention Readout bằng `global_mean_pool`.

### 4.3. Phương trình Cập nhật Nút Đối xứng Dấu & Neo Đặc trưng Gốc
Thông điệp gom về nút $v$ theo hướng $\delta$ sử dụng chuẩn hóa bậc có hiệu chỉnh $+1$ (tương ứng thêm trọng số self-loop ảo tránh chia $0$ tại nút gốc/nút lá trên đồ thị có hướng):

$$\mathbf{m}_v^{(l, \delta)} = \sum_{u \in \mathcal{N}_{\text{in}}^\delta(v)} \frac{\gamma_{uv}^{(l, \delta)}}{c_{uv}^\delta} \, \mathbf{z}_u^{(l, \delta)}, \qquad c_{uv}^\delta = \sqrt{\left(d_{\text{out}}^\delta(u) + 1\right)\left(d_{\text{in}}^\delta(v) + 1\right)}$$

*(Lưu ý hình học có hướng: trên nhánh $\text{TD}$, nút gốc có $d_{\text{in}}^{\text{TD}}(r)=0 \implies \mathbf{m}_r^{(l,\text{TD})} = \mathbf{0}$, nên trạng thái lan truyền phản hồi về gốc $\mathbf{z}_{\text{root}}^{\text{BU}}$ chủ yếu do nhánh $\text{BU}$ đảm nhiệm, còn nhánh $\text{TD}$ tổng hợp lan truyền qua $\mathbf{z}_{\text{att}}^{\text{TD}}$.)*

Kết hợp kết nối thặng dư tầng trước ($0.5\,\mathbf{h}_v^{(l-1,\delta)}$), neo tầng $0$ ($\varepsilon = 0.5$), neo nút gốc và **hàm kích hoạt lẻ đối xứng dấu** $\phi_{\text{sym}}(x) = x + 0.2 \tanh(x)$ (bảo toàn đối xứng $+\mathbf{h}$ vs $-\mathbf{h}$ giữa hai phe đối lập thay cho ELU bất đối xứng):

$$\mathbf{r}_v^{(l, \delta)} = 0.5\,\mathbf{h}_v^{(l-1, \delta)} + \varepsilon \mathbf{h}_v^{(0)} + \mathbf{m}_v^{(l, \delta)} + \mathbf{W}_{\text{root}}^{(l, \delta)} \mathbf{h}_r^{(0)} + \mathbf{b}_{\text{root}}^{(l, \delta)}$$

$$\mathbf{h}_v^{(l, \delta)} = \operatorname{Dropout}\left(\operatorname{LayerNorm}\left(\phi_{\text{sym}}\left(\mathbf{r}_v^{(l, \delta)}\right)\right)\right)$$

### 4.4. Cơ chế Đóng–Mở Tín hiệu Có dấu trên Cây Có hướng
Trên cây có hướng $\mathcal{G}_K^\delta$ (ma trận kề $\mathbf{A}_\delta$ dạng lũy linh có mọi trị riêng bằng $0$), ta không áp đặt phân tích phổ đối xứng vô hướng mà xét trực tiếp tác động hình học của dấu $\gamma_{uv}$ trước phép chuẩn hóa lớp trên một cạnh $(u \to v)$ truyền thông điệp $\mathbf{m}_{u \to v} = \frac{\gamma_{uv}}{c_{uv}} \mathbf{z}_u$:
- **Khi $\gamma_{uv} > 0$ (cổng đồng thuận / làm mịn thông thấp cục bộ)**: Thông điệp $+\frac{|\gamma_{uv}|}{c_{uv}} \mathbf{z}_u$ dịch biểu diễn trước chuẩn hóa của $v$ về gần cùng bán không gian với $\mathbf{z}_u$ ($\langle \mathbf{m}_{u \to v}, \mathbf{z}_u \rangle > 0$), cho phép tích hợp tương đồng giữa các nút chung quan điểm.
- **Khi $\gamma_{uv} < 0$ (cổng đối kháng / tương phản thông cao cục bộ)**: Thông điệp $-\frac{|\gamma_{uv}|}{c_{uv}} \mathbf{z}_u$ mang thành phần **ngược chiều** với nguồn gửi ($\langle \mathbf{m}_{u \to v}, \mathbf{z}_u \rangle < 0$). Khi $\mathbf{z}_v$ gần trạng thái của cha đối lập ($\langle \mathbf{z}_v, \mathbf{z}_u \rangle > 0$), hạng tử trừ kéo vector ra khỏi hướng $\mathbf{z}_u$, giúp biên độ tách biệt cụm quan điểm ở đầu ra `LayerNorm` duy trì sự phân rã giữa hai phe ủng hộ và phản bác thay vì bị kéo về trung bình lồi như `GCN`/`GATv2` ($\alpha_{ij} > 0$). Vector cạnh $\mathbf{a}_{uv} = [s_{uv}, h_{uv}]^\top$ cung cấp tín hiệu định hướng mạnh cho dấu và biên độ của $\gamma_{uv}$, được học kết hợp cùng biểu diễn đầu mút $(\mathbf{z}_u, \mathbf{z}_v)$.

---

## 5. Hợp nhất Toàn Đồ thị (Readout) & Học Đa nhiệm

### 5.1. Root Node Pooling & Attention Readout trên Hai Hướng
Sau $L$ tầng truyền thông điệp, mỗi nhánh $\delta \in \{\text{TD}, \text{BU}\}$ tạo ra tập biểu diễn đỉnh $\{\mathbf{h}_v^{(L, \delta)}\}_{v \in \mathcal{V}_K}$. Ta trích xuất hai đặc trưng bổ trợ cho nhau:

1. **Root Node Pooling** (nắm bắt trạng thái bài gốc sau khi tiếp nhận phản hồi từ nhánh $\text{BU}$ hoặc truyền tải xuống nhánh $\text{TD}$):

$$\mathbf{z}_{\text{root}}^{\delta} = \mathbf{h}_r^{(L, \delta)} \in \mathbb{R}^d$$

2. **Global Attention Readout** (tổng hợp có trọng số toàn bộ tối đa $21$ nút của $\mathcal{G}_K$, làm nổi bật các cụm bình luận kích hoạt xung đột; tương ứng tầng `GlobalAttentionReadout`):

$$q_v^{\delta} = \mathbf{q}_\delta^\top \tanh\left(\mathbf{U}_\delta \mathbf{h}_v^{(L, \delta)} + \mathbf{b}_\delta\right), \quad \beta_v^{\delta} = \frac{\exp(q_v^{\delta})}{\sum_{w \in \mathcal{V}_K} \exp(q_w^{\delta})}$$

$$\mathbf{z}_{\text{att}}^{\delta} = \sum_{v \in \mathcal{V}_K} \beta_v^{\delta} \mathbf{h}_v^{(L, \delta)} \in \mathbb{R}^d, \qquad \mathbf{U}_\delta \in \mathbb{R}^{d \times d}, \; \mathbf{b}_\delta, \mathbf{q}_\delta \in \mathbb{R}^d$$

Ghép nối cả hai thành phần trên cả hai hướng tạo thành biểu diễn toàn đồ thị $\mathbf{z}_{\mathcal{G}_K} \in \mathbb{R}^{4d}$ (khi tắt nhánh $\text{BU}$ trong ablation, $[\mathbf{z}_{\text{root}}^{\text{TD}} \parallel \mathbf{z}_{\text{att}}^{\text{TD}}]$ được phản chiếu sang hai vị trí sau):

$$\mathbf{z}_{\mathcal{G}_K} = \left[\,\mathbf{z}_{\text{root}}^{\text{TD}} \;\parallel\; \mathbf{z}_{\text{att}}^{\text{TD}} \;\parallel\; \mathbf{z}_{\text{root}}^{\text{BU}} \;\parallel\; \mathbf{z}_{\text{att}}^{\text{BU}}\,\right] \in \mathbb{R}^{4d}$$

### 5.2. Định nghĩa Nhãn Mục tiêu Cuối kỳ (trên $\mathcal{G}_{\text{full}}$ hoặc Tương lai $\mathcal{E}_{\text{full}} \setminus \mathcal{E}_K$)
Các nhãn đích chỉ phục vụ tính hàm mất mát và đánh giá:

1. **Quy mô lan truyền cuối kỳ ($y_{\text{size}}$)**:

$$y_{\text{size}} = \ln\left(1 + N_{\text{total}}\right) \in [0, \infty)$$

2. **Chỉ số phân cực lưỡng cực quy chiếu về gốc ($y_{\text{polar}} \in [0, 1]$, *silver-standard Cross-Encoder polarization*)**:
   - **Quy chiếu lập trường về bài gốc ($\tilde{s}_v$)**: Duyệt BFS từ gốc $r$ ($\tilde{s}_r = +1$), truyền dấu phản biện qua độ sâu cây để bình luận con phản bác ($s_{uv} < 0$) một bình luận đang phản bác gốc ($\tilde{s}_u < 0$) trở thành ủng hộ gốc ($\tilde{s}_v = \operatorname{sgn}(\tilde{s}_u) \cdot s_{uv}$ với $\operatorname{sgn}(x) = +1$ nếu $|x| \le 0.1$ hoặc $x > 0$, ngược lại $-1$).
   - Cho phép tính trên toàn tập cạnh $\mathcal{E}_{\text{full}}$ (`polar_target_mode="full"`) hoặc **chỉ trên các cạnh tương lai ngoài $K$**: $\mathcal{E}_{\text{target}} = \mathcal{E}_{\text{full}} \setminus \mathcal{E}_K$ (`polar_target_mode="future_only"` để cắt hoàn toàn tự quy chiếu trên $\mathcal{G}_K$).
   - Với $p_+ = \frac{1}{|\mathcal{E}_{\text{target}}|}\sum \mathbb{I}(\tilde{s}_v > 0.1)$ và $p_- = \frac{1}{|\mathcal{E}_{\text{target}}|}\sum \mathbb{I}(\tilde{s}_v < -0.1)$ cùng phương sai quần thể $\sigma_{\tilde{s}}^2 \in [0, 1]$ (bất đẳng thức Popoviciu trên $[-1, 1]$), định nghĩa chỉ số phân cực lưỡng cực đối xứng ($\lambda_p = 0.5$):

$$y_{\text{polar}} = \lambda_p \, \sigma_{\tilde{s}}^2 + (1 - \lambda_p) \, \left(4 \, p_+ p_-\right) \in [0, 1]$$

   *(Tính chất chuẩn hóa: cả đồng thuận 100% ủng hộ lẫn đồng thuận 100% bác bỏ đều cho $y_{\text{polar}} = 0$, trong khi chia rẽ đối kháng 50/50 giữa hai phe đạt cực đại $1.0$.)*

3. **Nhãn phân loại tin đồn cho PHEME ($y_{\text{rumor}}$)**: Hỗ trợ tách bạch cấu hình 4 lớp xác thực (`veracity4`: `non-rumor=0`, `true=1`, `false=2`, `unverified/rumor=3`) và 2 lớp nhị phân (`binary`: `non-rumor=0`, `rumor=1`). Dữ liệu tiếng Việt gán $y_{\text{rumor}} = -1$ (bỏ qua nhánh phân loại).

### 5.3. Ba Nhánh Đầu ra, Đầu Tự tin $\beta$-NLL & Hàm Mất mát Đa nhiệm
Từ biểu diễn đồ thị $\mathbf{z}_{\mathcal{G}_K} \in \mathbb{R}^{4d}$ (ghép nối thêm biểu diễn đặc trưng tốc độ sớm 8 chiều `cascade_stats` khi bật `--v2`, áp dụng công bằng cho cả 8 mô hình), các đầu MLP 2 tầng dự đoán:

$$\hat{y}_{\text{size}} = \operatorname{Softplus}\left(\operatorname{MLP}_{\text{size}}(\mathbf{z}_{\text{reg}})\right) \in (0, \infty), \qquad \hat{y}_{\text{polar}} = \sigma\left(\operatorname{MLP}_{\text{polar}}(\mathbf{z}_{\text{reg}})\right) \in (0, 1)$$

$$\hat{\mathbf{p}}_{\text{rumor}} = \operatorname{Softmax}\left(\operatorname{MLP}_{\text{rumor}}(\mathbf{z}_{\mathcal{G}_K})\right) \in \Delta^{C-1}$$

Khi bật `predict_uncertainty=True` (`--v2`), đầu bất định xuất log-phương sai giới hạn mượt $\log \hat{\sigma}_t^2 = -6 + 9\,\sigma(u_t) \in (-6, 3)$ và điểm tự tin hiệu chỉnh liên tục $c_{\text{conf}} = \exp(-\hat{\sigma}_{\text{polar}} / 0.25) \in (0, 1]$.

Sử dụng Huber Loss tách ngưỡng ($\delta_{\text{size}} = 1.0$, $\delta_{\text{polar}} = 0.15$), Cross-Entropy có trọng số lớp $\mathbf{w}$, và thành phần **$\beta$-NLL** ($\beta = 0.5$, Seitzer et al., ICLR 2022) với sai số tách gradient $\operatorname{sg}(\hat{y}_t) - y_t$ ($\alpha_{\text{size}}=1.0, \alpha_{\text{polar}}=4.0, \alpha_{\text{rumor}}=1.0$ mặc định trong huấn luyện):

$$\mathcal{L}_{\text{task}} = \alpha_{\text{size}} \mathcal{L}_{\delta_{\text{size}}}^{\text{Huber}}(\hat{y}_{\text{size}}, y_{\text{size}}) + \alpha_{\text{polar}} \mathcal{L}_{\delta_{\text{polar}}}^{\text{Huber}}(\hat{y}_{\text{polar}}, y_{\text{polar}}) + \mathbb{I}_{\text{PHEME}} \, \alpha_{\text{rumor}} \mathcal{L}_{\text{CE}}(\hat{\mathbf{p}}_{\text{rumor}}, y_{\text{rumor}}; \mathbf{w})$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{unc}} \sum_{t \in \{\text{size}, \text{polar}\}} \frac{1}{2} \operatorname{sg}\left(\hat{\sigma}_t^{2\beta}\right) \left(\exp(-\log\hat{\sigma}_t^2)(\operatorname{sg}(\hat{y}_t) - y_t)^2 + \log\hat{\sigma}_t^2\right)$$

### 5.4. Nhánh Mở rộng UPFD (`GossipCop` / `PolitiFact` — Classification-Only)
Trên bộ dữ liệu UPFD (`classification_only=True`, hỗ trợ `--feature {bert,spacy,profile,content}`), hàm mục tiêu rút gọn về $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{\mathbf{p}}, y; \mathbf{w})$. Suy luận cặp cạnh ẩn từ cặp đặc trưng nút qua `LatentConflictMLP` (dùng chung kiến trúc 2 tầng cho cả `SBiFAGAT (Latent)` và baseline cạnh `EdgeGATv2 (Latent)` để bảo đảm công bằng tham số).

---

## 6. Giao thức Lưu trữ `.pt`, Tối ưu 16GB VRAM & Bộ Độ đo Đánh giá

- **Cấu trúc tệp ngoại tuyến `nlp_features.pt`**: Lưu trọn vẹn theo đồ thị đầy đủ kèm `topic_splits.json` để `EarlyCascadeDataset` cắt động cho mọi $K \in \{10, 20, 30, 50\}$:
  - `node_embeddings [N_all, 768]` (từ encoder đơn câu đóng băng gốc) + `node_id_to_idx`
  - `edge_attr [E_all, 4]` (ghép $[s_{uv}, h_{uv}, r_{uv}, c_{uv}]$ hoặc `[E_all, 2]`) + `edge_id_to_idx`, `edge_stance`, `edge_hostility`, `edge_irrationality`, `edge_constructive`, `edge_oppose_flag`, `edge_pred_label`
  - `teacher_edge_*` (nếu có cache Teacher): lưu cô lập để phục vụ giao thức LOO (`edge_source="teacher"`) mà không rò rỉ Student Cross-Encoder qua các fold.
- **Tối ưu 16GB VRAM & Chạy lặp lại 5 Seeds**:
  - Phase 2 chạy FP16 (`torch.amp`) và giải phóng GPU trước Phase 3; Phase 3 dùng `batch_size = 32` (`64` trên UPFD), `epochs = 100`, `EarlyStopping(patience=10)` và xếp hạng `auto_loop` thuần theo $\mathcal{L}_{\text{task}}$ (không phụ thuộc quy mô $\lambda_{\text{unc}}$).
  - Mọi thực nghiệm chạy độc lập trên $\mathcal{S} = \{42, 100, 2024, 777, 999\}$, báo cáo $\text{Mean} \pm \text{Std}$ cùng `construct_corr_bc` tùy chọn.
  - Kiểm định ý nghĩa thống kê với đối thủ mạnh nhất (`--comparator-model auto`): `scipy.stats.ttest_rel` theo seeds/LOO folds kèm **Paired Bootstrap từng seed rồi lấy trung bình trên $S$ seeds** và hiệu chỉnh **Holm-Bonferroni**.
