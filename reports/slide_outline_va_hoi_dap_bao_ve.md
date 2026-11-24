# DÀN Ý 14 SLIDE & SỔ TAY 15 CÂU HỎI–ĐÁP BẢO VỆ ĐỀ TÀI

**Đề tài:** *Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội*  
**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU) — Đại học Thái Nguyên  
**Nhóm thực hiện:** Vũ Hiếu Kiên, Vũ Ngọc Ánh Thuyên  
**Đồng bộ 100% với:** [`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md) (`bao_cao_de_tai_vn.pdf`) và [`paper/main.pdf`](../paper/main.pdf)  
**Thời lượng thuyết trình dự kiến:** 15 phút (~1 phút/slide, dành thời gian cho Demo trực tiếp ở Slide 13)

---

## PHẦN A — DÀN Ý CHI TIẾT 14 SLIDE BẢO VỆ (MẠCH HAI TẦNG)

### Slide 1 — Trang bìa & Thông tin Đề tài
- **Tiêu đề slide:** Mạng Nơ-ron Đồ thị và Xử lý Ngôn ngữ Tự nhiên trong Mô hình hóa Phân cực Quan điểm và Lan truyền Thông tin trên Mạng Xã hội
- **Thông điệp chốt:** Giới thiệu khung đồ thị có dấu **Hai tầng** (`SBi-FAGAT` Tầng 1 + Mạng người dùng vĩ mô Tầng 2) vận hành thực nghiệm trung thực trên Reddit Tiếng Việt và PHEME Tiếng Anh.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Đơn vị:** Khoa Công nghệ Thông tin — Trường ĐH CNTT & Truyền thông (ICTU), Đại học Thái Nguyên.
  - **Nhóm tác giả:** Vũ Hiếu Kiên, Vũ Ngọc Ánh Thuyên.
  - **Kiến trúc đề xuất:** `SBi-FAGAT` (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network*) + Phân tích Mạng xã hội Người dùng Có dấu Vĩ mô ($\mathcal{G}_{\text{macro}}$).
  - **Phạm vi công bố:** Song ngữ Việt–Anh, tối ưu GPU $16\text{ GB}$ VRAM / dual T4, kèm Demo Web cục bộ kiểm chứng trình duyệt.
- **Hình ảnh / Bảng chèn vào:** Logo ICTU & Sơ đồ tóm tắt 2 hộp ghép nối Tầng 1 (Cây thảo luận sớm) $\to$ Tầng 2 (Bản đồ cộng đồng 2 phe) trích từ `Hình 2.1` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Kính thưa Quý thầy cô trong Hội đồng, hôm nay nhóm chúng em xin trình bày đề tài nghiên cứu kết hợp Xử lý ngôn ngữ tự nhiên (NLP) và Mạng nơ-ron đồ thị (GNN) theo kiến trúc Hai tầng để dự báo sớm quy mô lan truyền, mức độ phân cực và lập bản đồ chia rẽ cộng đồng trên mạng xã hội. Điểm xuyên suốt báo cáo của nhóm là tính minh bạch thực nghiệm: công bố đầy đủ cả kết quả đạt được lẫn các giới hạn chẩn đoán trên dữ liệu thật.

---

### Slide 2 — Đặt vấn đề & 3 Khoảng trống Kỹ thuật
- **Tiêu đề slide:** Thách thức Dự báo Sớm và Phân cực: Vì sao GNN Truyền thống gặp Giới hạn?
- **Thông điệp chốt:** Mô hình hóa bình luận sớm ($11$–$21$ nút) và phân cực vĩ mô bị cản trở đồng thời bởi giả định đồng nhất (*homophily*), lan truyền một chiều và rò rỉ dữ liệu.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Khoảng trống 1 — *Heterophily* & Làm trơn (*Over-smoothing*):** `GCN` / `GATv2` dùng trọng số $\alpha_{ij} > 0$, làm nhòa ranh giới giữa hai bình luận phản bác nhau.
  - **Khoảng trống 2 — Một chiều & Mù diễn ngôn:** Bỏ qua chiều phản hồi Bottom-Up hoặc chỉ nhúng từng bình luận rời rạc, thiếu quan hệ cặp `[stance, hostility]` và nhầm lẫn văng tục bên thứ ba.
  - **Khoảng trống 3 — Rò rỉ dữ liệu & Suy biến thước đo:** Lẫn thông tin tương lai/sự kiện khi tách tập; RWC và Modularity suy biến trên cây đơn nhỏ ($\le 21$ nút, không chu trình).
- **Hình ảnh / Bảng chèn vào:** Bảng đối chiếu gọn trích từ `Bảng 1.1` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)): `GCN` vs `GATv2` vs `Bi-GCN` vs `SGCN` vs `FAGCN` vs `SBi-FAGAT`.
- **Ghi chú lời dẫn cho người thuyết trình:** Trên mạng xã hội, các bình luận nối tiếp nhau thường xuyên tranh cãi trái chiều chứ không đồng nhất như đồ thị trích dẫn bài báo, khiến bộ lọc thông thấp truyền thống tự làm mờ tín hiệu đối lập. Bên cạnh đó, những thước đo kinh điển như Random Walk Controversy hay Modularity chỉ có nghĩa trên đồ thị người dùng lớn có vòng lặp, chứ không thể áp dụng trực tiếp lên một cây hội thoại mới chỉ có $10$ đến $20$ bình luận đầu tiên.

---

### Slide 3 — Mục tiêu & Tổng quan Kiến trúc Hợp nhất Hai tầng
- **Tiêu đề slide:** Giải pháp Tổng thể: Kiến trúc Đồ thị Có dấu Hai tầng Chống Rò rỉ
- **Thông điệp chốt:** Tách biệt rõ hai cấp độ phân tích — dự báo sớm từng chuỗi ở Tầng 1 và đo lường/dự báo đối đầu cộng đồng ở Tầng 2 — trên cùng một đường ống NLP ngoại tuyến.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Pha NLP Ngoại tuyến (tách khỏi vòng lặp 5-seed GNN):** Trích xuất vector nút đóng băng *SpanPool* ($768$-D) và chưng cất cạnh 2D $e_{uv} = [s_{uv}, h_{uv}]^\top$.
  - **Tầng 1 — Early Cascade (`SBi-FAGAT`):** Chỉ quan sát bài gốc + tối đa $K$ bình luận đầu ($K=20$ VN / $K=10$ PHEME) để dự báo $y_{\text{size}}$, $y_{\text{polar}}$ và $y_{\text{rumor}}$.
  - **Tầng 2 — Macro User Network ($\mathcal{G}_{\text{macro}}$):** Chiếu tương tác hợp lệ xuyên chủ đề lên mạng người dùng để tính $Q_{\text{signed}}$, Cân bằng cấu trúc, RWC và dự báo cạnh đối đầu (`SBiFAGAT-Link`).
  - **Cam kết *Zero Data Leakage*:** Chặn nhân quả $t \le t_K$, khóa `topic_splits.json` 70/15/15, cô lập cạnh Teacher ở LOO/Temporal Holdout.
- **Hình ảnh / Bảng chèn vào:** Sơ đồ luồng Hai tầng (`Hình 2.1` trong [`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Để giải quyết triệt để hai bài toán và phù hợp với GPU $16\text{ GB}$ VRAM, chúng em tách quy trình thành một pha trích xuất NLP ngoại tuyến dùng chung cho cả hai tầng. Ở Tầng 1, `SBi-FAGAT` chỉ được phép nhìn tối đa $K$ bình luận sớm nhất; còn ở Tầng 2, toàn bộ cạnh trả lời hợp lệ được chiếu xuyên suốt các chủ đề lên đồ thị người dùng toàn cục để quan sát cấu trúc hai phe.

---

### Slide 4 — Đóng khung Dữ liệu, Giả danh hóa SHA-256 & Lọc rác *Path Contraction*
- **Tiêu đề slide:** Bộ dữ liệu Song ngữ, Bảo mật Danh tính & Lọc rác Bảo toàn Cấu trúc Cây
- **Thông điệp chốt:** Chuẩn hóa bộ dữ liệu Reddit VN ($1{,}500$ chuỗi) và PHEME rút gọn ($2{,}664$ chuỗi), bảo vệ riêng tư bằng SHA-256 có muối và giữ nguyên nhánh con qua *Path Contraction*.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Reddit VN (Arctic Shift):** $1{,}500$ chuỗi ($87{,}820$ nút; $99.96\%$ `r/vozforums`, $32$ nút `r/VietNam`), $K=20, N_{\text{reply}} \ge 25$; $1{,}496$ cụm heuristic (30 ngày, cosine TF-IDF $\ge 0.30$), $1{,}050/225/225$ + Temporal Holdout.
  - **PHEME EN (Reduced Horizon):** $2{,}664 / 6{,}425$ chuỗi ($K=10, N_{\text{reply}} \ge 15$; $68.2\%$ `non-rumor`, Test-334 $46.4\%$ `non-rumor`), $9$ sự kiện ($1{,}786/544/334$) + `5-fold LOO`.
  - **Bảo mật dữ liệu:** `hash_author_sha256` có muối (`SBIF_HASH_SALT`) + `redact_pii_text` + chặn `*_machine_key.csv` và bài *doxxing*.
  - ***Path Contraction* (2 lượt, ứng dụng Demo):** Co rút nút giữa rác (`[deleted]`, `"hóng"`, `"chấm"`) nối lên tổ tiên hợp lệ (chống chu trình), giữ nguyên câu 1–2 từ (`"Đồng ý"`, `"Phản đối"`), tách bạch `junk_mode="prune"` của benchmark.
- **Hình ảnh / Bảng chèn vào:** `Bảng 3.1` + Sơ đồ *Prune* vs *Contract* (`Hình 2.2` trong [`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Dữ liệu thực tế chứa rất nhiều bình luận giữ chỗ như "chấm", "hóng" hoặc đã bị xóa ở giữa chuỗi thảo luận. Thay vì cắt bỏ làm đứt cả nhánh con bên dưới như cách thông thường, mô-đun ứng dụng dùng thuật toán *Path Contraction* hai lượt để co rút nút rác trung gian mà vẫn bảo vệ nguyên vẹn các lập trường ngắn như "Đồng ý" hay "Phản đối", trong khi toàn bộ bảng đối chứng khoa học vẫn giữ nguyên chuẩn `prune` đã khóa.

---

### Slide 5 — NLP Tầng sâu: *Ancestor-Conditioned SpanPool* & Chưng cất 2D `[s_uv, h_uv]`
- **Tiêu đề slide:** Trích xuất Ngữ nghĩa Nhân quả & Chưng cất Cross-Encoder 2D Cô lập Phân đoạn
- **Thông điệp chốt:** Chưng cất NLI đối xứng và bộ chấm công kích cân bằng lớp về Student 2D gọn nhẹ giúp tăng độ tin cậy lập trường và loại bỏ nhiễu công kích từ bình luận cha.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Nhúng nút nhân quả (*SpanPool*):** `[CLS] root [SEP] parent [SEP] reply [SEP]` ($256$ token, cắt gốc 60% đầu + 40% cuối), chỉ mean-pool trên token thuộc `reply` (`uitnlp/visobert` VN / `twitter-roberta-base` EN) + $10$-D `struct_x` + $8$-D `cascade_stats`.
  - **Teacher kép mở:** NLI đa ngữ đối xứng `mDeBERTa-v3-xnli` ($384$ token, khử mâu thuẫn kép $m_{\text{dc}} = \min(c_{\text{sup}}, c_{\text{opp}})$) + Teacher công kích (`ViSoBERT` tinh chỉnh `UIT-ViHSD` Val Macro-F1 $= 0.7390$ / `twitter-roberta-base-offensive`); loại bỏ `Qwen` QLoRA để giữ 100% nhãn sạch trong $16\text{ GB}$ VRAM.
  - **Student `DualHeadCrossEncoder` cô lập phân đoạn:** Nhánh stance ghép $5$ khối $(\mathbf{h}_{\text{CLS}}, \mathbf{h}_{\text{ctx}}, \mathbf{h}_{\text{reply}}, \odot, |\cdot|)$; nhánh hostility $h_{uv}$ **chỉ đọc $\mathbf{h}_{\text{reply}}$** (chống lây chửi từ cha).
  - **Kết quả chưng cất (Val Pearson $s_{uv} / h_{uv}$):** VN $0.6601 / 0.8196$; PHEME $0.8104 / 0.9834$ (`Loss` $T^2 \cdot \operatorname{KL}_T + 0.5\,\operatorname{SmoothL1} + \operatorname{BCE}$ với $T=2.0$, trọng số $T^2=4.0$).
- **Hình ảnh / Bảng chèn vào:** Công thức `Double-Contradiction` & `Bảng 2.2` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Một vấn đề tinh tế khi chấm cặp bình luận là nếu bình luận cha dùng lời lẽ nặng nề còn câu trả lời lại ôn hòa, bộ mã hóa cặp thông thường dễ gán nhầm mức công kích cho câu trả lời. Bằng cách tách nhánh hostility chỉ đọc đúng phân đoạn token của câu trả lời, Student `DualHeadCrossEncoder` khắc phục hiện tượng lây lan này và đạt độ tương quan trên tập Validation tới $0.81$–$0.98$.

---

### Slide 6 — Kiến trúc `SBi-FAGAT` Tầng 1
- **Tiêu đề slide:** Cơ chế Hai kênh Phân cực–Kích thích, Dir-GNN và Camp Dipole trong `SBi-FAGAT`
- **Thông điệp chốt:** `SBi-FAGAT` kết hợp hai kênh tách biệt (có dấu $\gamma_{uv}$ và không dấu $\alpha_{uv}^{\text{exc}}$), trộn hai chiều `Dir-GNN`, readout lưỡng cực `Camp Dipole` và neo `Persistence-Residual`.
- **Nội dung gạch đầu dòng (trên slide):**
  - **`TwoStreamNodeStem` ($d=64, L=2$):** Chiếu riêng $768\to 48$ và $10\to 16$ qua `Linear + GELU + LayerNorm`.
  - **Tách 2 nửa kênh trong `StanceGuidedFAConv`:** Kênh phân cực $\gamma_{uv}^{\xi} \in (-1, 1)$ (chú ý $\sqrt{2d}$ + cosine + $e_{uv}$ + GELU basis $\delta_\gamma$) khuếch đại bất đồng; kênh kích thích $\alpha_{uv}^{\text{exc}} \in (0, 1)$ đo độ sôi nổi lan truyền.
  - **Trộn `Dir-GNN` & Readout `Camp Dipole` ($\mathbb{R}^{3d+6}$):** BFS 64 bước quy chiếu dấu về gốc tính $(\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-)$, $b = 2\sqrt{f_+ f_-}$ và 6 mô-men cổng.
  - **Neo & Tối ưu đa nhiệm:** `Persistence-Residual Anchor` quanh $\bar{\ell}_{\text{train}}$, `KL-calibrated BCE + SmoothL1`, cân bằng 2:1, AdamW + EMA (`0.985`), đúng 5 seeds `[42, 100, 2024, 777, 999]`.
- **Hình ảnh / Bảng chèn vào:** Công thức $\gamma_{uv}^{\xi,(l)}$, $\alpha_{uv}^{\text{exc},(l)}$ và mô tả khối `Camp Dipole` (Mục 2.4, [`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Kiến trúc `SBi-FAGAT` có $137{,}086$ tham số, phân tách không gian ẩn thành hai nửa kênh: nửa phân cực dùng cổng $\tanh$ có dấu để tách các nhánh phản biện ra xa thay vì làm mờ như GCN, còn nửa kích thích giữ trọng số dương để bảo toàn sức nóng cuộc tranh luận phục vụ dự báo quy mô. Khối Camp Dipole đóng vai trò tổng hợp khoảng cách hai cực trước khi qua đầu ra đa nhiệm.

---

### Slide 7 — Kết quả Tầng 1 trên Reddit Việt Nam ($K=20$)
- **Tiêu đề slide:** Thực nghiệm Tầng 1 VN: Vượt xa Persistence, Tương đương các Mô hình Ngang sức
- **Thông điệp chốt:** Tất cả mô hình học sâu nâng $R^2_{\text{polar}}$ từ $-3.1986$ lên $\sim 0.074$, trong đó `SBi-FAGAT` hòa thống kê với `TextMLP` trên giao thức Inductive và dẫn Spearman $\rho$ ở Temporal Holdout.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Inductive ($N=225$ Test, 5 seeds):** `SBiFAGAT` đạt $R^2_{\text{polar}} = 0.0742$ (`RMSE = 0.0370`), vượt xa `Early-K Persistence` ($-3.1986$); nhưng ngang bằng thống kê với `TextMLP` ($0.0741$, $r=0.2989, \rho=0.2501$) và `TemporalStanceMLP` ($0.0608$, $p_{\text{adj}} \ge 0.574$).
  - **Dự báo quy mô VN Inductive:** `StandardGCN` dẫn đầu (`MSLE = 0.1542`, $R^2_{\text{size}} = 0.3097$) so với `SBiFAGAT` (`0.1580` / `0.2926`).
  - **Temporal Holdout (cô lập cạnh Teacher):** `Persistence` tụt xuống $-5.4386$; `SBiFAGAT` dẫn đầu Spearman ($\rho = 0.3058$, $p_{\text{adj}} = 1.00$), nhưng xếp sau `TemporalStanceMLP` ở $R^2_{\text{polar}}$ ($0.0609$ vs $0.0815$) và `FAGCN` ở $R^2_{\text{size}}$ ($0.1914$ vs $0.2586$).
- **Hình ảnh / Bảng chèn vào:** [`fig1_umap_polarization.svg`](../paper/figures/vn/fig1_umap_polarization.svg) (UMAP VN Test) + số liệu tóm tắt từ `Bảng 3.2` & `Bảng 3.3` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Trên tập Reddit Việt Nam, điểm nổi bật đầu tiên là `Early-K Persistence` thất bại hoàn toàn với $R^2 = -3.1986$, chứng minh không thể ngoại suy trực tiếp $K$ bình luận đầu ra cuối kỳ. Khi học sâu vào cuộc, `SBi-FAGAT` đạt mức $0.0742$, nhưng chúng em báo cáo thẳng thắn rằng trên cây nông tối đa $21$ nút này, `SBi-FAGAT` không tạo ra cách biệt thống kê trước `TextMLP` hay `TemporalStanceMLP` vì nhúng ngữ cảnh SpanPool đã mang hầu hết thông tin dự báo.

---

### Slide 8 — Kết quả Tầng 1 trên PHEME ($K=10$): Inductive & 5-Fold LOO
- **Tiêu đề slide:** Thực nghiệm Tầng 1 PHEME ($K=10$): Lợi thế chọn lọc ở Tin đồn & Chuyển dịch Sự kiện
- **Thông điệp chốt:** Ở cửa sổ rút gọn $K=10$, `SBi-FAGAT` có lợi thế phân loại tin đồn Inductive và giữ ổn định tương đối qua `5-fold LOO`, nhưng thua `TextMLP` và `EdgeGATv2` ở hồi quy phân cực Inductive.
- **Nội dung gạch đầu dòng (trên slide):**
  - **PHEME Inductive (Test-334):** `SBiFAGAT` dẫn đầu `Macro-F1 = 0.3541` (boot $p_{\text{adj}} = 8.96\text{e-}02$, paired-$t$ $p_{\text{adj}} = 1.00$) và `Acc = 0.6359**` (boot $p_{\text{adj}} = 7.20\text{e-}03$ vs `EdgeGATv2 0.5988`; $+17.2\text{ pp}$ trên đa số `non-rumor` $46.4\%$, sau `TemporalStanceMLP 0.6443`).
  - **Hồi quy PHEME Inductive:** `TextMLP` dẫn $R^2_{\text{polar}} = 0.0932$ (vs `SBiFAGAT 0.0793`); `EdgeGATv2` dẫn $r/\rho$ ($0.4008 / 0.4451$); `FAGCN` dẫn `MSLE` ($0.0986$, $R^2_{\text{size}} = 0.0697$).
  - **Kiểm định `5-fold LOO` (cô lập Teacher):** `SBiFAGAT` dẫn đầu `RMSE 0.0445`, $R^2_{\text{polar}} = 0.0512$, `Macro-F1 = 0.2978`, `Acc = 0.6203` (Exact Perm. $p = 0.0625, p_{\text{adj}} = 0.5625$); nhưng sau `Persistence` ($0.3171/0.3822$) & `EdgeGATv2` ($0.2783/0.2783$) về $r/\rho$ (`0.2739/0.2737`), và `MSLE 0.1600` (vs `EdgeGATv2 0.1594`).
- **Hình ảnh / Bảng chèn vào:** [`fig1_umap_polarization.svg`](../paper/figures/pheme/fig1_umap_polarization.svg) + trích dẫn `Bảng 3.4` & `Bảng 3.5` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Chuyển sang PHEME tiếng Anh ở cửa sổ rút gọn $K=10$, `SBi-FAGAT` phát huy điểm mạnh ở nhiệm vụ phân loại 4 lớp tin đồn khi dẫn đầu `Macro-F1` cả ở chế độ Inductive lẫn `5-fold LOO` chuyển dịch sự kiện. Tuy nhiên, cần lưu ý ký hiệu hai sao ở Inductive PHEME xuất phát từ bootstrap cấp cây trong phạm vi $2$ sự kiện kiểm thử chứ chưa đạt ý nghĩa xuyên sự kiện trên $5$ hạt giống (`paired-t p_adj = 1.00`).

---

### Slide 9 — Cắt bỏ Thành phần, Đường cong $K$, UPFD & 3 Phát hiện Cốt lõi
- **Tiêu đề slide:** Chẩn đoán Trung thực: Kết quả Âm tính trên Cây nông & Chuyển giao UPFD
- **Thông điệp chốt:** Nghiên cứu cắt bỏ, $K$-curve và UPFD chỉ ra giới hạn trần thông tin trên cây nhỏ: **3 lớp lập trường** là cốt lõi, còn chồng phức tạp hóa mô hình dễ phản tác dụng trên cây $11$ nút.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Ablation VN (`Full 0.0777`):** Bỏ `Camp Dipole` tụt mạnh ($0.0503$), bỏ `Conflict Gate` ($0.0527$), bỏ `Burst Stats` tụt $R^2_{\text{size}}$ ($0.2925\to 0.2215$), **bỏ `Hostility` giữ nguyên ($0.0757$)**; ngang `BiEdgeGATv2 (Matched)` ($0.0789, p_{\text{adj}} = 1.00$).
  - **Ablation PHEME (`Full 0.0791`):** `Full` **thắng $0/9$ chỉ số**; `w/o Burst Stats` ($0.1015$), `w/o Dir-GNN` ($0.1010$), `w/o Bottom-Up` ($0.1009$) và `BiEdgeGATv2` ($0.0940$, boot $p_{\text{adj}} = 0.0072\text{--}0.0288$) đều vượt hồi quy phân cực.
  - **Đường cong $K$ ($N_{\text{reply}}\ge 55$) & `UPFD`:** VN MSLE giảm dần ($0.1443\to 0.1053$); PHEME ($n=10$) sụp đổ `Acc = 0.7000` (đa số). Trên `UPFD` không văn bản, `StandardGATv2` (`GossipCop 0.9684`) & `EdgeGATv2` (`PolitiFact 0.8253`) vượt `SBiFAGAT (Latent)` (`0.9486` / `0.8090`).
- **Hình ảnh / Bảng chèn vào:** [`vn/fig2_early_k_curve.svg`](../paper/figures/vn/fig2_early_k_curve.svg) & [`pheme/fig2_early_k_curve.svg`](../paper/figures/pheme/fig2_early_k_curve.svg) + số liệu `Bảng 3.6`, `3.7`, `3.8` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Slide này tóm tắt phát hiện trung thực quan trọng nhất của Tầng 1. Khi cây chỉ có $11$ nút trên PHEME, việc loại bỏ các khối trộn hai hướng hoặc `Camp Dipole` thậm chí làm tăng $R^2_{\text{polar}}$ lên $0.101$, và trên dữ liệu `UPFD` không có văn bản bình luận thì cổng có dấu mất tác dụng. Ngược lại, trên Việt Nam, việc cắt bỏ điểm công kích không làm mô hình yếu đi, chứng tỏ **3 lớp lập trường** mới là tín hiệu hạt nhân.

---

### Slide 10 — Kiểm định Nhãn bạc & Lỗi *Target-Misattributed Profanity*
- **Tiêu đề slide:** Kiểm định Nhãn bạc & Phát hiện Lỗi "Chửi Bên Thứ 3 để Ủng hộ"
- **Thông điệp chốt:** Độ khớp nhãn bạc còn thấp và NLI đảo ngược $57.1\%$ bình luận ủng hộ có văng tục bên thứ 3 — giải thích trọn vẹn vì sao bỏ công kích thô lại có lợi.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Kiểm định VN (`Gemini Flash-Lite` + 1 tác giả duyệt, $\text{IAA}=\text{--}$):** $20$ pilot stance `Acc 70.0%` (`Macro-F1 0.6640`), nhưng trên **$120$ cặp Test** tụt xuống `Acc 40.0%` / **weighted `Acc 24.79%`** (`Macro-F1 0.3760 / 0.2324`); còn `ViSoBERT` công kích đạt `ROC-AUC = 0.9841` (`PR-AUC = 0.2619`, pos-F1 `0.3509`).
  - **Chấm tay $30$ cây VN:** Tương quan $y_{\text{polar}}$ vs chia rẽ cảm nhận 0–4 là **$\rho = 0.0018$** ($\tau_b = 0.0026$) $\to$ Tầng 1 thực chất dự báo **độ phân tán lập trường sinh bởi NLI**.
  - **Đối chứng `RumourEval-2019` (`1,675` cặp):** $93.0\%$ dồn về `neutral`, recall lớp thiểu số chỉ $3.8\%$ (`support`) và $2.0\%$–$3.0\%$ (`oppose`).
  - ***Target-Misattributed Profanity* ($86{,}320$ cạnh VN):** $r(s, h) = -0.1832$; **$4/7 = 57.1\%$ câu `support+uncivil`** chửi bên thứ 3 để ủng hộ cha bị NLI kéo về âm (`mean s_uv = -0.2495`).
- **Hình ảnh / Bảng chèn vào:** [`vn/fig4_counterfactual_interventions.svg`](../paper/figures/vn/fig4_counterfactual_interventions.svg) + [`vn/fig3_cascade_heatmap.svg`](../paper/figures/vn/fig3_cascade_heatmap.svg) + `Bảng 3.12`.
- **Ghi chú lời dẫn cho người thuyết trình:** Để lý giải bản chất tín hiệu, chúng em tự tiến hành kiểm định lại nhãn bạc của Teacher trên $120$ cặp, $30$ cây VN và bộ chuẩn `RumourEval-2019`. Kết quả chỉ ra một lỗi ngữ dụng điển hình: khi người dùng Việt Nam chửi bới một nhân vật thứ ba trong tin tức để tỏ thái độ đồng tình với người viết bài, NLI zero-shot nhầm sự tức giận đó là phản đối tới $57.1\%$ số trường hợp. Điều này giải thích ngay vì sao mô hình hoạt động tốt hơn hoặc ngang bằng khi bỏ điểm công kích thô.

---

### Slide 11 — Bước ngoặt Tầng 2: Mạng Người dùng Có dấu Vĩ mô
- **Tiêu đề slide:** Lên quy mô Vĩ mô Tầng 2: Tách phổ Hai phe & Cấu trúc Phản cân bằng $(---)$
- **Thông điệp chốt:** Chiếu tương tác lên đồ thị người dùng xuyên chủ đề $\mathcal{G}_{\text{macro}}$ xóa bỏ trần cây nhỏ, bộc lộ tách phổ sắc nét và trạng thái xã hội phản cân bằng.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Quy mô mạng:** **VN** $13{,}718$ tổng người dùng ($10{,}974$ người dùng thuộc cạnh mạnh $|\bar{s}_{uv}| \ge 0.1$ trên $85{,}398$ tương tác; WCC $10{,}475$ nút = $76.4\%$, $5{,}143$ tam giác); **PHEME** $50{,}062$ tổng người dùng ($36{,}606$ người dùng tham gia cặp trả lời hợp lệ trên $72{,}858$ tương tác; $18{,}985$ người dùng thuộc cạnh có dấu mạnh $|\bar{s}_{uv}| \ge 0.1$; WCC thưa $10{,}364$ nút = $20.7\%$ tổng / $28.3\%$ hợp lệ, bậc mạnh $\le 3.09$, $212$ tam giác).
  - **Tách phổ cộng đồng:** PHEME đạt **$Q_{\text{signed}} = 0.8232$**, **$\text{RWC} = 0.6218$**, đối đầu liên phe tăng **$+54.5\text{ pp}$** ($60.6\%$ vs $6.0\%$), công kích gần gấp $3$ ($28.8\%$ vs $10.5\%$); VN đạt $Q_{\text{signed}} = 0.4280$, $\text{RWC} = 0.0957$, đối đầu liên phe tăng **$+19.1\text{ pp}$** ($57.0\%$ vs $37.9\%$).
  - **Phát hiện *Phản cân bằng (Anti-balanced)*:** Tỷ lệ cân bằng thấp hơn kỳ vọng ngẫu nhiên (VN $25.3\%$, $z = -15.17$; PHEME $35.4\%$, $z = -4.30$) do đa số tam giác thuộc loại $(---)$ ($68.7\%$ VN / $52.4\%$ PHEME).
- **Hình ảnh / Bảng chèn vào:** [`vn/fig5_macro_signed_network.svg`](../paper/figures/vn/fig5_macro_signed_network.svg) & [`pheme/fig5_macro_signed_network.svg`](../paper/figures/pheme/fig5_macro_signed_network.svg) + `Bảng 3.9`.
- **Ghi chú lời dẫn cho người thuyết trình:** Khi giải phóng bài toán khỏi giới hạn một cây đơn nhỏ bằng cách chiếu tất cả tương tác lên mạng người dùng vĩ mô Tầng 2, cấu trúc phân cực lập tức hiện ra rõ rệt. Đặc biệt, khác với hình dung kinh điển về hai buồng vang nội bộ yêu thương nhau và ghét bên ngoài, mạng thảo luận mở ở cả Việt Nam và PHEME đều có chỉ số cân bằng âm sâu vì ngập tràn các bộ ba tranh cãi tay ba đều có dấu âm.

---

### Slide 12 — Dự báo Liên kết Đối đầu `SBiFAGAT-Link`
- **Tiêu đề slide:** Dự báo Liên kết Có dấu Theo Thời gian: Xu hướng Cải thiện & Giải thích Overlap
- **Thông điệp chốt:** `SBiFAGAT-Link` dẫn đầu xu hướng trên các cạnh *transductive* của PHEME (`ROC-AUC = 0.9181 ± 0.0102`), đồng thời nhóm công bố minh bạch 3 chẩn đoán lặp cặp và ngưỡng `p_adj`.
- **Nội dung gạch đầu dòng (trên slide):**
  - **PHEME (`52.9%` oppose, $4{,}278$ All-Test / $212$ *transductive*):** `SBiFAGAT-Link` dẫn cả $3$ chỉ số *transductive*: **`ROC-AUC = 0.9181 ± 0.0102`**, `PR-AUC = 0.7616 ± 0.0600`, **`Macro-F1 = 0.8006 ± 0.0068`** (paired-$t$ chưa hiệu chỉnh $p = 0.0270$, Exact Perm. $p = 0.0625$, $p_{\text{adj}} = 0.1622 / 0.3750$); trên **All-Test ($4{,}278$ cạnh)** đạt `ROC-AUC = 0.7328` (ngang `UserTextMLP 0.7331`) và `PR-AUC = 0.6019`, sau `FAGCN` ở `Macro-F1 = 0.6209`.
  - **Reddit VN (`79.5%` oppose, $2{,}255$ cạnh *transductive*):** `SBiFAGAT-Link` dẫn All-Test `ROC-AUC = 0.5853`, `PR-AUC = 0.8455`, `Trans. Macro-F1 = 0.5526`; nhưng **`SGCN` dẫn `Trans. ROC-AUC = 0.6149`** (`PR-AUC 0.8848`), $p_{\text{adj}} \ge 0.2876$.
  - **3 chẩn đoán trung thực:** (1) Lặp cặp `{u, v}` khi mở đồ thị Train là **$21.7\%$ ($46/212$) trên PHEME** vs $3.2\%$ ($73/2{,}255$) trên VN; (2) $9.1\%$ VN / $6.6\%$–$10.1\%$ người dùng *unseen* giữ đặc trưng tiền mốc cắt từ chuỗi trung lập; (3) Chỉ `SBiFAGAT-Link` nhận $e_{uv}$ 2D liên tục.
- **Hình ảnh / Bảng chèn vào:** Bảng đối chiếu tóm tắt từ `Bảng 3.10` & `Bảng 3.11` ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Ở tác vụ dự báo cặp người dùng đối đầu theo thời gian chống rò rỉ, `SBiFAGAT-Link` đạt `Transductive ROC-AUC = 0.9181` trên PHEME. Tuy nhiên, để giữ chuẩn mực trung thực khoa học, chúng em đo luôn tỷ lệ cặp lặp lại giữa Train và Test *transductive*: tỷ lệ này lên tới $21.7\%$ trên PHEME, giải thích vì sao ngay cả `UserTextMLP` cũng đạt `0.9081`, đồng thời chứng minh lý do `SGCN` vẫn cạnh tranh rất tốt trên mạng VN dày hơn.

---

### Slide 13 — Trình diễn Demo Web Cục bộ 2 Tab (Gate 2 Browser QA)
- **Tiêu đề slide:** Ứng dụng Demo Web Cục bộ MVP: Kết nối Checkpoint Thật & Trung thực Trạng thái
- **Thông điệp chốt:** Demo Web `127.0.0.1:8765` minh họa trực quan dự báo Tầng 1, lọc rác *Path Contraction*, cầu nối đồ thị người dùng cấp thread và bản đồ vĩ mô Tầng 2.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Tab 1 — Chế độ A (`Bài Test VN Thật`, `preextracted_full_2d_ce`):** Đối chứng `vn_1n0hqn0` (bài gốc mang nhãn thay thế ẩn danh `[REDACTED_DOXXING_PII]` từ tập dữ liệu VN đã lọc/ẩn danh trên đĩa; $\hat{y}_{\text{polar}} = 0.1122$ vs `0.0290`) và `vn_1lg7ayf` ($\hat{y}_{\text{polar}} = 0.1159$ vs `0.2644`, `~41.8` vs `30`); nút chuyển **Đồ thị Người dùng của Thread** tách rõ `Phe 0 (5 người)` vs `Phe 1 (10 người)`.
  - **Tab 1 — Chế độ B (`Sandbox`, `Preset 1 · Phạt nguội AI`):** Thu gọn $15 \to 9$ (cắt `3` lá, gộp `3` giữa, nối `3` cạnh đứt nét cam `[Nối nhánh]`), giữ nguyên `"Đồng ý"`, `"Phản đối"`; dự báo `~50.6` bình luận, $\hat{y}_{\text{polar}} = 0.1313$ (con số `0.7031` trên thẻ là **`early_y_polar` nội tại trên $9$ nút giữ lại theo Heuristic**, không phải $t > K$); hiển thị **banner cảnh báo hổ phách không thể tắt (`cross_encoder_loaded = false`)**.
  - **Tab 2 — Mạng Vĩ mô (`VN / PHEME`):** Trình bày $Q_{\text{signed}} = 0.8232$, `RWC = 0.6218`, `z = -4.30`, `+54.5 pp` và hộp cảnh báo giới hạn ($n=212$, $21.7\%$ lặp cặp).
- **Hình ảnh / Bảng chèn vào:** 5 ảnh thật đã kiểm chứng tại Gate 2 Browser QA: [`demo_tab1_vn_tree.png`](figures/demo_tab1_vn_tree.png), [`demo_tab1_user_graph.png`](figures/demo_tab1_user_graph.png), [`demo_tab1_sandbox_banner.png`](figures/demo_tab1_sandbox_banner.png), [`demo_tab1_sandbox_contraction.png`](figures/demo_tab1_sandbox_contraction.png), [`demo_tab2_macro_network.png`](figures/demo_tab2_macro_network.png) + `Bảng 4.2`.
- **Ghi chú lời dẫn cho người thuyết trình:** Trên màn hình là 5 ảnh chụp thực tế từ hệ thống Demo Web cục bộ chạy tại `http://127.0.0.1:8765/`. Ở Chế độ A trên chuỗi `vn_1lg7ayf`, chỉ với một nút chuyển đổi, Hội đồng có thể quan sát trực tiếp cách 15 tác giả từ cây thảo luận Tầng 1 được chiếu thành hai phe $5$ vs $10$ người của Tầng 2. Ở Chế độ B Sandbox, hệ thống minh họa thuật toán *Path Contraction* nối lại $3$ nhánh con phía sau các nút `[deleted]`/`hóng ké` và chủ động bật cảnh báo màu hổ phách `cross_encoder_loaded = false` để không gây hiểu nhầm khi thiếu tệp trọng số Cross-Encoder cục bộ.

---

### Slide 14 — Kết luận, Giới hạn & 3 Hướng Phát triển Tiếp theo
- **Tiêu đề slide:** Kết luận Khoa học & Lộ trình 3 Nâng cấp Trọng tâm Tiếp theo
- **Thông điệp chốt:** Đề tài khép lại với bài học rõ ràng về ranh giới giữa cây nông và mạng vĩ mô, đồng thời định vị chính xác 3 nâng cấp mở khóa hiệu năng tiếp theo.
- **Nội dung gạch đầu dòng (trên slide):**
  - **Tổng kết đóng góp:** Hoàn thiện kiến trúc Hai tầng chống rò rỉ trên GPU $16\text{ GB}$ + chứng minh thực nghiệm vai trò cốt lõi của **3 lớp lập trường ($s_{uv}$)**, hiệu quả tách phổ Tầng 2 và Demo Web an toàn.
  - **Giới hạn đã công bố:** Trần thông tin cây $11$–$21$ nút, độ tin cậy nhãn bạc đơn tác giả ($\text{IAA}=\text{--}$) và lỗi *Target-Misattributed Profanity* ($57.1\%$), cụm VN 2 subreddit, cỡ mẫu nhỏ $K$-curve PHEME ($n=10$), giới hạn API `limit=500`.
  - **3 Hướng phát triển ưu tiên:**
    1. *Target-Aware Stance & Toxicity Modeling* (phân biệt chửi cha vs chửi bên thứ 3).
    2. *Multi-Turn Thread Context* ($\ge 5$ lượt tổ tiên liền kề chống trôi dấu phủ định/mỉa mai).
    3. Đồ thị dị thể `Reply–@mention` Tầng 2 + phân trang thu thập vượt trần `limit=500`.
- **Hình ảnh / Bảng chèn vào:** Bảng 3 dòng lộ trình nâng cấp trích từ Mục 2 phần *Kết luận & Hướng phát triển* ([`bao_cao_de_tai_vn.md`](bao_cao_de_tai_vn.md)).
- **Ghi chú lời dẫn cho người thuyết trình:** Tóm lại, đề tài đã hoàn thành trọn vẹn 3 sản phẩm cam kết: báo cáo khoa học tiếng Anh và tiếng Việt đồng bộ 100% số liệu, cùng hệ thống Demo chạy cục bộ. Quan trọng hơn cả, quá trình đối chứng khắt khe đã giúp nhóm chỉ ra đúng nút thắt nằm ở bài toán lập trường gắn mục tiêu và ngữ cảnh đa lượt $\ge 5$ bước cho giai đoạn nghiên cứu tiếp theo. Em xin trân trọng cảm ơn sự lắng nghe của Quý thầy cô và kính mong nhận được các câu hỏi góp ý từ Hội đồng.

---

## PHẦN B — SỔ TAY 15 CÂU HỎI–ĐÁP PHẢN BIỆN TIẾNG VIỆT

> **Nguyên tắc trả lời trước Hội đồng:** Đi thẳng vào vấn đề, thừa nhận trung thực mọi giới hạn định lượng, dẫn đúng con số từ `paper/main.tex` và nêu rõ ý nghĩa khoa học hoặc giải pháp kỹ thuật tương ứng. Tuyệt đối không vòng vo hay bao biện.

### Câu 1. Vì sao Tầng 1 `SBi-FAGAT` chỉ ngang `TextMLP`/`BiEdgeGATv2` trên VN (`0.0742` vs `0.0741`) và thắng `0/9` trên PHEME Ablation? Có thất bại không?
- **Trả lời thẳng:** Không phải thất bại về lập trình hay rò rỉ, mà là một **phát hiện âm tính có giá trị chẩn đoán (*diagnostic negative finding*)**: trên một cây hội thoại đơn nông chỉ có $11$–$21$ nút, nhúng văn bản điều kiện hóa tổ tiên (*SpanPool*) và thống kê lập trường sớm đã hấp thụ gần hết tín hiệu dự báo sẵn có.
- **Bằng chứng số liệu thật:** Trên VN Inductive, `SBiFAGAT` ($R^2_{\text{polar}} = 0.0742$) ngang bằng thống kê với `TextMLP` ($0.0741$, $p_{\text{adj}} \ge 0.574$) và `BiEdgeGATv2 (Matched)` ($0.0789$ vs Full $0.0777$, $p_{\text{adj}} = 1.00$); trên PHEME ($K=10$, $\le 11$ nút), `Full` thắng $0/9$ chỉ số cắt bỏ khi `w/o Burst Stats` ($0.1015$) và `BiEdgeGATv2` ($0.0940$) đều vượt hồi quy phân cực của `Full` ($0.0791$). Trong khi đó, `SBiFAGAT` vẫn giữ ưu thế chọn lọc ở phân loại tin đồn PHEME (`Macro-F1 = 0.3541` Inductive; `0.2978` 5-fold LOO).
- **Ý nghĩa nghiên cứu:** Phát hiện này bác bỏ giả định ngây thơ rằng "cứ thêm cơ chế GNN phức tạp lên cây nhỏ là sẽ tăng độ chính xác", từ đó lý giải vì sao nghiên cứu cần mở rộng lên **Tầng 2 Mạng người dùng vĩ mô** nơi cấu trúc đồ thị mới thực sự phát huy tác dụng.

### Câu 2. Tại sao chỉ số `y_polar` tương quan gần như `0` ($\rho = 0.0018$) với 30 cây chấm tay VN? Vậy mô hình Tầng 1 đang dự đoán cái gì?
- **Trả lời thẳng:** Chúng em khẳng định minh bạch rằng bộ hồi quy $y_{\text{polar}}$ của Tầng 1 đang học **dự báo phân tán lập trường cặp cha–con do mô hình NLI tạo ra** (`model-derived stance dispersion`), chứ **không** dự báo trực tiếp cảm nhận chia rẽ chủ quan của con người trên toàn bộ chủ đề.
- **Bằng chứng số liệu thật:** Trên $30$ cây Test VN chấm tay thang $0$–$4$, tương quan giữa $y_{\text{polar}} \in [0.0447, 0.2644]$ và nhãn người gần như bằng $0$ ($r = 0.0284, \rho = 0.0018, \tau_b = 0.0026, \kappa_w = 0.0192$, `ROC-AUC` $@\ge 2 = 0.5402$). Trong khi đó, cùng một định nghĩa mục tiêu, `SBiFAGAT` lại vượt cực mạnh `Early-K Persistence` ($-3.1986 \to 0.0742$ trên VN và $-0.3022 \to 0.0793$ trên PHEME).
- **Nguyên nhân cơ chế:** Công thức $y_{\text{polar}}$ đo tích cân bằng hai chiều $m_+ m_-$ của cặp câu trả lời cục bộ; vì thế các bài hỏi lời khuyên kiểu nhị phân (*"nên chọn A hay B"*) cho nhiều cặp `tán thành/phản đối` ôn hòa nên có $y_{\text{polar}}$ cao, trong khi các bài phẫn nộ ai cũng đồng lòng chửi chung một đối tượng ngoài cuộc lại dồn hết về một phía nên $m_+ m_-$ thấp.

### Câu 3. Dữ liệu kiểm định nhãn bạc chỉ do 1 tác giả duyệt sau khi AI (`Gemini Flash-Lite`) tiền chú thích (`IAA = --`), còn `mDeBERTa` trên RumourEval recall chỉ `2.0%–3.8%` — làm sao bảo vệ tính tin cậy?
- **Trả lời thẳng:** Nhóm **không** tuyên bố nhãn bạc này tương đương nhãn vàng đa giám định viên (`multi-annotator gold standard`); thay vào đó chúng em công bố đầy đủ độ lệch làm giới hạn bắt buộc và sử dụng kiểm định để **chứng minh nguồn gốc lỗi** thay vì che giấu.
- **Bằng chứng số liệu thật:** Nhãn VN chỉ gắn cờ `single_verified_gold` ($\text{IAA} = \text{--}$): trên $120$ cặp Test VN, độ chính xác lập trường 3 lớp có trọng số chỉ đạt $24.79\%$ (`Macro-F1` có trọng số $0.2324$, không trọng số $40.0\% / 0.3760$); còn trên `RumourEval-2019` ($1{,}675$ cặp độc lập), `mDeBERTa` đẩy $93.0\%$ về `neutral` khiến recall lớp thiểu số chỉ đạt $3.8\%$ (`support`) và $2.0\%$–$3.0\%$ (`oppose`). Riêng bộ `ViSoBERT` công kích giữ được thứ hạng tốt (`ROC-AUC = 0.9841` trên $120$ cặp Test, `PR-AUC = 0.2619`, pos-F1 `0.3509`).
- **Hướng bảo vệ kỹ thuật:** Do không dùng làm tiêu chí "vượt chuẩn người", kết quả kiểm định giúp khoanh vùng chính xác ba điểm cần xử lý ở thế hệ kế tiếp: (i) thiên lệch bảo thủ về `neutral` của NLI zero-shot trên cặp ngắn, (ii) hiện tượng văng tục bên thứ 3 và (iii) sự cần thiết của tập gán nhãn đa người chấm độc lập ($\kappa \ge 0.7$) trước khi triển khai thực tế.

### Câu 4. Tại sao `SBiFAGAT-Link` trên PHEME Transductive đạt `ROC-AUC = 0.9181` ($p_{\text{adj}} = 0.1622$), nhưng All-Test chỉ `0.7328` và lặp cặp chiếm tới `21.7%` ($46/212$)? Còn trên VN sao `SGCN` dẫn `0.6149`?
- **Trả lời thẳng:** Có ba yếu tố giải thích trung thực sự chênh lệch này: mật độ lịch sử người dùng quen (*recurring pairs*), sự khác biệt cấu trúc giữa Test tổng thể (gồm người dùng lạnh) và *Transductive*, cùng lợi thế bất đối xứng của vector cạnh 2D liên tục.
- **Bằng chứng số liệu thật:** Trên toàn tập kiểm tra PHEME (`all-test`, $4{,}278$ cạnh gồm cả người dùng mới chưa gặp ở Train), `SBiFAGAT-Link` đạt `ROC-AUC = 0.7328` (ngang `UserTextMLP` `0.7331`) và dẫn đầu `PR-AUC = 0.6019`, trong khi `FAGCN` dẫn đầu `Macro-F1 = 0.6209` (`SBiFAGAT-Link` đạt `0.5860`). Khi đánh giá tập *transductive* không che cạnh Train (phản ánh thực tế dự báo trên mạng xã hội hiện hữu), $46/212$ ($21.7\%$) cạnh là cặp $\{u, v\}$ đã từng tương tác ở Train khiến ngay cả `UserTextMLP` cũng tăng từ `0.7331` lên `0.9081`, còn `SBiFAGAT-Link` đạt xu hướng dẫn đầu `Trans. ROC-AUC = 0.9181 ± 0.0102`, `Trans. PR-AUC = 0.7616 ± 0.0600`, `Trans. Macro-F1 = 0.8006 ± 0.0068` (chưa hiệu chỉnh $p = 0.0270$, Exact Perm. $p = 0.0625$, Holm–Bonferroni $p_{\text{adj}} = 0.1622 / 0.3750$). Trên VN, tỷ lệ lặp cặp *transductive* chỉ là $3.2\%$ ($73/2{,}255$) và mạng dày đặc hơn ($5{,}143$ tam giác), giúp mô hình chuyên biệt cân bằng cấu trúc **`SGCN`** tận dụng hiệu quả các cạnh dấu rời rạc $\{-1, +1\}$ để dẫn `Trans. ROC-AUC = 0.6149` (so với `0.6001` của `SBiFAGAT-Link`).
- **Lưu ý rò rỉ tiền mốc cắt:** Cũng chỉ có $6.6\%$–$10.1\%$ người dùng unseen ở PHEME và $9.1\%$ ở VN mang đặc trưng từ chuỗi trung lập tiền mốc cắt, loại trừ hoàn toàn rò rỉ nhãn tương lai.

### Câu 5. Vì sao lại giữ phần phân loại Công kích (`h_uv`) trong khi `w/o Hostility` trên VN giữ nguyên ($0.0757$ vs $0.0777$) còn PHEME lại tăng ($0.0791 \to 0.0878$)? Và giải thích lỗi `57.1%` *chửi bên thứ 3 để ủng hộ*?
- **Trả lời thẳng:** Nhóm giữ đường ống 2D `[s_uv, h_uv]` cho phần lõi đã đóng băng để bảo toàn tính tái lập thực nghiệm trên toàn bộ các bảng đã chạy, đồng thời dùng chính sự đối chiếu có/không có $h_{uv}$ để rút ra phát hiện quan trọng về **Target-Misattributed Profanity**.
- **Bằng chứng số liệu thật:** Trên $86{,}320$ cạnh VN, $h_{uv}$ tương quan âm với $s_{uv}$ ($r = -0.1832$). Trên $N=140$ cặp kiểm định VN, **$4/7$ ($57.1\%$) bình luận `support+uncivil`** — tức đồng tình với tác giả bài cha bằng cách chửi một bên thứ ba trong câu chuyện — bị NLI kéo điểm lập trường về âm (`mean s_uv = -0.2495`, âm hơn cả `oppose+clean` $-0.1594$). Ngoài ra, `h_uv` có tương quan thuận yếu với quy mô cuối kỳ ($\rho = 0.2128$ VN; $0.2024$ PHEME) nên giữ vai trò phụ trợ ở nhánh kích thích $\alpha^{\text{exc}}$, nhưng làm nhiễu hồi quy phân cực khi đi cùng `Persistence`.
- **Ứng dụng trên Demo:** Chính từ phát hiện này, giao diện Demo Web tắt mặc định công tắc hiển thị chỉ báo công kích thô tục (`#toggle-profanity-badge`) để tránh người dùng hiểu nhầm.

### Câu 6. Tại sao mạng người dùng cả 2 tập lại phản cân bằng (`balance z = -15.17` và `-4.30`, đầy tam giác `(---)`) thay vì cân bằng Heider như lý thuyết? Vậy chỉ số $Q_{\text{signed}} = 0.8232$ trên PHEME WCC thưa (`212` tam giác) có nghĩa gì?
- **Trả lời thẳng:** Cân bằng Heider kinh điển thường xuất hiện trong mạng xã hội kết bạn lâu dài; ngược lại, diễn đàn thảo luận công khai (`r/vozforums`) và chuỗi tin tức nóng (`PHEME`) vận hành theo cơ chế **tranh luận đa chiều nơi nhiều người cùng phản bác chéo lẫn nhau**, sinh ra sự thống trị của bộ ba toàn âm $(---)$.
- **Bằng chứng số liệu thật:** Tam giác $(---)$ chiếm $68.7\%$ ($3{,}535/5{,}143$) trên VN ($z = -15.17$) và $52.4\%$ ($111/212$) trên PHEME ($z = -4.30$). Trên PHEME ($50{,}062$ tổng người dùng, trong đó $36{,}606$ người dùng tham gia cặp trả lời hợp lệ trên $72{,}858$ tương tác và $18{,}985$ người dùng thuộc cạnh có dấu mạnh $|\bar{s}_{uv}| \ge 0.1$; so với VN là $13{,}718$ tổng / $10{,}974$ cạnh mạnh), thành phần liên thông lớn nhất (WCC) của PHEME gồm $10{,}364$ nút ($20.7\%$ tổng $50{,}062$ / $28.3\%$ trong $36{,}606$ người dùng hợp lệ) mang cấu trúc **gần-cây cực thưa (*near-tree sparse topology*)** với bậc mạnh trung bình $\le 3.09$ và chỉ $212$ tam giác.
- **Diễn giải đúng $Q_{\text{signed}} = 0.8232$:** Giá trị $Q_{\text{signed}} = 0.8232$ ($\text{RWC} = 0.6218$) cho thấy phân đôi quang phổ đã tách các cạnh đối đầu $(s_{uv} \le -0.1)$ ra nằm giữa hai tập đỉnh đối lập cực kỳ hiệu quả ($60.6\%$ liên phe vs $6.0\%$ nội phe), chứ **không** đồng nghĩa sự tồn tại của hai buồng vang khép kín mật độ cao kiểu mạng retweet bầu cử — và chúng em ghi chú rõ đây là phép đo nội mẫu chưa đối chuẩn null-graph modularity.

### Câu 7. Tại sao không thu thập thêm dữ liệu hoặc dùng LLM lớn hơn (`Qwen` QLoRA) ngay cho bản nộp này?
- **Trả lời thẳng:** Có hai lý do kỹ thuật và kỷ luật khoa học: giới hạn phần cứng thực nghiệm $16\text{ GB}$ VRAM / dual T4 và nguyên tắc đóng băng phạm vi (*Scope Freeze*) nhằm bảo đảm đồng bộ tuyệt đối và 100% độ phủ nhãn sạch.
- **Chi tiết thực nghiệm:** Khi thử nghiệm hướng LLM tạo sinh trước đó (`Qwen3.8-27B` QLoRA), chi phí suy luận vượt quá giới hạn VRAM và thời gian phiên chạy Kaggle T4x2 khi xử lý hàng trăm nghìn cặp văn bản ($86{,}320$ cặp VN và $74{,}324$ cặp PHEME cùng các cặp sâu), dẫn tới nguy cơ rớt nhãn hoặc mất cân đối. Chuyển hoàn toàn sang mô hình thầy mở `mDeBERTa-v3-xnli` kết hợp `ViSoBERT`/`Twitter-RoBERTa` chưng cất về `DualHeadCrossEncoder` giúp đạt độ phủ $100\%$ nhãn không bị rớt mẫu và toàn bộ giai đoạn GNN 5 seeds chỉ mất `1.17 ms/chuỗi` trên CPU (`0.03 MB` RAM trên VN) hoặc `0.38 ms/chuỗi` với `27.09 MB` VRAM trên GPU (PHEME). Đồng thời, đóng băng dữ liệu trước hạn nộp giúp mọi bảng biểu giữa bài báo IEEE, báo cáo tiếng Việt và Demo khớp nhau tuyệt đối.

### Câu 8. Tại sao không giữ nguyên chuỗi cãi $K=20$ cho PHEME mà phải dùng reduced-horizon $K=10, N_{\text{reply}} \ge 15$?
- **Trả lời thẳng:** Vì đặc trưng phân bố độ dài tự nhiên của Twitter PHEME rất ngắn: nếu ép ngưỡng $K=20, N_{\text{reply}} \ge 25$ như Reddit VN, phần lớn dữ liệu tin tức PHEME sẽ bị loại bỏ hết, đặc biệt các sự kiện nhỏ sẽ không còn đủ mẫu để kiểm chứng chia theo sự kiện hoặc 5-fold LOO.
- **Bằng chứng số liệu thật:** Tập PHEME thô gồm $6{,}425$ chuỗi có trung vị chỉ **$11.0$ bình luận** (thời lượng trung vị $2.53\text{ giờ}$), so với trung vị **$44.0$ bình luận** ($95.22\text{ giờ}$) của Reddit VN. Lọc tại cửa sổ rút gọn $K=10, N_{\text{reply}} \ge 15$ chuyển từ $6{,}425$ chuỗi thô xuống còn **$2{,}664$ chuỗi đủ điều kiện** (trung vị $20.0$, trung bình $27.9$ bình luận; vẫn bảo đảm ít nhất $5$ bình luận tương lai $t > t_{10}$ để tính nhãn mục tiêu khử chệch Bessel), đủ kích thước $1{,}786 / 544 / 334$ cho $9$ sự kiện và phép giữ đủ **$5$ sự kiện lớn ($\ge 15$ chuỗi)** để chạy đánh giá chéo `5-fold LOO`.

### Câu 9. Cụm VN tách `1,496/1,500` topic singleton có bị rò rỉ ngữ nghĩa giữa Train và Test không? Tại sao phải kiểm chứng thêm Temporal Holdout?
- **Trả lời thẳng:** Có khả năng còn dư âm chủ đề vĩ mô chung của diễn đàn `r/vozforums`, vì phân hoạch `1,496` cụm thực chất là một phép gom cụm theo thời gian–từ khóa heuristic (*Heuristic Temporal-Keyword / TF-IDF Grouped Split*), chứ **không** tương đương sự kiện tin tức tách biệt hoàn toàn như PHEME.
- **Giải pháp đối chứng trực tiếp:** Thay vì ngộ nhận `1,496` cụm là *event-disjoint* tuyệt đối, nhóm bổ sung ngay phép thử **Temporal Holdout** (`Bảng 3.3`): sắp xếp toàn bộ $1{,}500$ chuỗi theo dấu thời gian, khóa $225$ chuỗi cuối kỳ làm tập Test và cô lập hoàn toàn cạnh từ Teacher (`edge_source="teacher"`) để Student không học bất kỳ ngữ cảnh tương lai nào. Kết quả cho thấy `SBiFAGAT` vẫn giữ vị trí dẫn đầu tương quan thứ hạng ($\rho = 0.3058$), trong khi `TemporalStanceMLP` dẫn $R^2_{\text{polar}}$ ($0.0815$ vs $0.0609$).

### Câu 10. Tại sao `Demo` lại tách riêng chế độ *Bài Test VN Thật* vs *Sandbox Heuristic*, và tại sao không giấu đi `cross_encoder_loaded = false` khi bảo vệ?
- **Trả lời thẳng:** Việc tách hai chế độ nhằm bảo đảm cả tính khoa học (chạy đúng đặc trưng $2\text{D}$ trích xuất trên $16\text{ GB}$ GPU với tập Test VN thật) lẫn tính tương tác thực tế (cho phép Hội đồng gõ bất kỳ bình luận tiếng Việt nào trên máy laptop thường không có file nhị phân Cross-Encoder nặng).
- **Lý do minh bạch `cross_encoder_loaded = false`:** Theo kỷ luật chống ảo giác kết quả của đề tài, nếu thiếu tệp `dual_head_weights.pt` trên máy trình diễn, các điểm cạnh trong Sandbox bắt buộc suy về luật Heuristic (`visobert_spanpool_heuristic_edges` hoặc `heuristic_hash_sandbox`). Việc khóa cứng **banner cảnh báo màu hổ phách không thể tắt** giúp Hội đồng phân biệt rạch ròi đâu là suy luận chuẩn của mô hình bài báo (Chế độ A: `preextracted_full_2d_ce`) và đâu là minh họa cơ chế *Path Contraction* với trọng số Heuristic (Chế độ B), tuyệt đối không đánh lừa người xem.

### Câu 11. Cơ chế *Path Contraction* xử lý bình luận rác khác gì với cắt bỏ thông thường (*Prune*), và làm sao tránh xóa nhầm câu ngắn kiểu `"Đồng ý"`, `"Phản đối"`?
- **Trả lời thẳng:** *Prune* xóa thẳng nút rác làm mọi bình luận con bên dưới nút đó đứt khỏi cha và biến mất khỏi cây BFS; ngược lại, *Path Contraction* leo ngược hai lượt qua bảng `raw_parent_map` để nối trực tiếp các nút con hợp lệ lên tổ tiên hợp lệ gần nhất, đồng thời bỏ bộ lọc độ dài từ.
- **Chi tiết cài đặt (`src/data/arctic_shift_vn.py`):**
  1. Lượt 1 ánh xạ toàn bộ cha thô trước khi lọc, sau đó `is_filler_junk_text` **không dùng ngưỡng đếm từ tối thiểu** mà đối chiếu chính xác vào danh sách $24$ cụm giữ chỗ tường minh (`"hóng"`, `"chấm"`, `"lót dép"`, `"ok"`, ...), chuỗi chỉ có emoji/dấu câu và spam (`[deleted]`, link rác), đồng thời giữ lại `"+1"`, `"-1"`, `"Đồng ý"`, `"Phản đối"`, `"Chuẩn bác"`.
  2. Lượt 2 (`resolve_contracted_parent_id`) có biến bảo vệ chống chu trình (`seen`, `probe_seen`) để đảm bảo cây luôn hội tụ về gốc an toàn — như đã kiểm chứng thực tế trên `Preset 1` ($15 \to 9$ bình luận, cắt `3` lá, gộp `3` nút giữa, nối lại `3` cạnh `[Nối nhánh]`).

### Câu 12. Vì sao Tầng 1 kết hợp `Persistence-Residual Anchor` ($\bar{\ell}_{\text{train}}$) và hàm mất mát KL-Calibrated BCE + 2:1 scaling thay vì MSE/BCE thuần?
- **Trả lời thẳng:** Kết hợp này giải quyết đồng thời hai khó khăn số học của hồi quy phân cực trên cây sớm: phân phối $y_{\text{polar}}$ tập trung ở dải nhỏ ($[0.04, 0.26]$) dễ gây suy giảm đạo hàm ở đuôi sigmoid, và chênh lệch thang đo giữa hai đầu ra $y_{\text{size}}$ vs $y_{\text{polar}}$.
- **Cơ chế cụ thể:**
  - `Persistence-Residual Anchor` đưa tín hiệu phân cực sớm đã hiệu chỉnh theo trung bình logit Train $\bar{\ell}_{\text{train}}$ qua cổng học $\tanh(w_p)$, giúp mô hình chỉ cần học phần dư chuyển dịch tương lai $\Delta z_{\text{polar}}$ thay vì phải tái dựng lại biên độ từ con số $0$.
  - `KL-Calibrated BCE` trừ đi entropy mục tiêu $H(y_{\text{polar}})$ (kẹp dương) cộng `SmoothL1` ($\beta=0.05$) giữ độ dốc ổn định quanh các giá trị liên tục trong $(0, 1)$, trong khi cân bằng đa nhiệm $2:1$ chuẩn hóa theo loss hằng số trên tập Train giúp cả đầu quy mô và đầu phân cực hội tụ đồng đều mà không bị một nhiệm vụ lấn át.

### Câu 13. Dấu `**` trong các bảng PHEME (`p < 0.01`) khác gì với $p_{\text{adj}} = 1.00$ của `paired-t` 5 seed?
- **Trả lời thẳng:** Dấu `**` trên PHEME Inductive phản ánh kết quả **bootstrap IID cấp cây** ($N = 334$ cây trên chỉ $2$ sự kiện Test là `ottawashooting` và `prince-toronto`), **không** phải ý nghĩa thống kê khái quát xuyên sự kiện.
- **Phân tích đối chiếu:** Vì tập Test PHEME chỉ có $2$ sự kiện ($< 6$ cụm tối thiểu để dùng cluster bootstrap an toàn), nhóm công bố đồng thời cả hai loại kiểm định: ở cấp cây nội bộ hai sự kiện này, `SBiFAGAT` vượt `EdgeGATv2` về `Acc` ($0.6359$ vs $0.5988$, boot $p_{\text{adj}} = 7.20\text{e-}03$); nhưng khi xét biến thiên qua $5$ hạt giống ngẫu nhiên ở cấp độ bài toán (`paired-t` trên 5 seeds), sự chênh lệch này không có ý nghĩa thống kê ($p_{\text{adj}} = 1.00$). Nhóm ghi chú rõ điều này ngay trong chú thích bảng để người đọc không hiểu quá mức.

### Câu 14. So sánh đóng góp thực chất của `SBi-FAGAT` trên **dữ liệu có văn bản lập trường** vs **dữ liệu chỉ có vector nhúng tĩnh UPFD** (`0.9486` vs `0.9684`)?
- **Trả lời thẳng:** `SBi-FAGAT` phụ thuộc bản chất vào **tín hiệu diễn ngôn cặp câu $e_{uv} = [s_{uv}, h_{uv}]^\top$ trích xuất từ văn bản bình luận**; khi mất văn bản gốc như ở `UPFD`, kiến trúc này bị bất lợi so với các GNN tiêu chuẩn ít tham số hơn.
- **Bằng chứng số liệu thật (`Bảng 3.8`):** Trên `UPFD` (chỉ có vector BERT $768$ chiều ở nút, không có văn bản để chạy `DualHeadCrossEncoder`), `SBiFAGAT` phải tự đoán cạnh ẩn qua `LatentConflictMLP`. Kết quả là trên `GossipCop`, `StandardGATv2` ($0.9684$, `ROC-AUC = 0.9929`) và `EdgeGATv2 (Latent)` ($0.9682$, `0.9937`) vượt có ý nghĩa trước `SBiFAGAT (Latent)` ($0.9486$, `0.9859`, $p_{\text{adj}} = 8.00\text{e-}04 / 1.30\text{e-}02$); trên `PolitiFact`, `EdgeGATv2 (Latent)` cũng dẫn (`0.8253` vs `0.8090`, boot $p_{\text{adj}} = 0.032$). Điểm tích cực duy nhất của nhánh cạnh ẩn là giảm mạnh phương sai giữa các seed trên `PolitiFact` ($\pm 0.0240$ so với $\pm 0.1450$ của `StandardGATv2`).

### Câu 15. Nếu phát triển tiếp sau đề tài này, 3 nâng cấp nào mang lại bước nhảy lớn nhất (*Target-Aware Stance*, *Multi-turn $\ge 5$*, *`@mention` + vượt giới hạn `500` API*)?
- **Trả lời thẳng:** Ba nâng cấp này đánh trực diện vào ba điểm nghẽn định lượng đã tìm ra trong Chương 3 và Chương 4, thay vì chạy theo tăng kích thước tham số GNN một cách mù quáng:
  1. ***Target-Aware Stance & Toxicity Modeling*:** Dạy bộ mã hóa phân biệt đối tượng bị công kích là *tác giả bài cha* hay *bên thứ 3 được nhắc tới*, giải quyết trực tiếp tỷ lệ đảo nhãn $57.1\%$ trên nhóm `support+uncivil` và mở khóa giá trị thực của kênh công kích $h_{uv}$.
  2. ***Multi-Turn Thread Context* ($\ge 5$ lượt tổ tiên):** Mở rộng ngữ cảnh từ cặp 2 lượt `[root, parent, reply]` lên tối thiểu $5$ lượt bình luận liền kề trong nhánh, giảm tỷ lệ dồn về `neutral` ($93.0\%$ trên RumourEval) và xử lý các câu mỉa mai, phủ định gián tiếp ở độ sâu $\ge 2$.
  3. **Đồ thị dị thể `Reply–@mention` Tầng 2 + phân trang vượt trần `limit=500`:** Thêm cạnh nhắc tên `@mention` giúp liên kết các thành phần rời rạc trên mạng người dùng thưa (như WCC $20.7\%$ của PHEME), đồng thời bổ sung cơ chế phân trang theo `before` timestamp trong `scripts/predict_live_thread.py` để không bị cắt cụt khi truy vấn các chủ đề trực tiếp có trên $500$ bình luận.
