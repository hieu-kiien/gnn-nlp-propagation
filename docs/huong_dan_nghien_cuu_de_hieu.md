# Hướng dẫn Nghiên cứu SBi-FAGAT từ Con số 0 (Dễ hiểu, Không cần biết GNN trước)

Nếu bạn đọc đề tài *"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"* mà thấy rối trước hàng loạt công thức và thư mục code, tài liệu này giải thích toàn bộ bức tranh bằng tiếng Việt đời thường trong khoảng **10 phút đọc**.

---

## 1. Bài toán thực tế nhóm mình đang giải là gì?

Hãy tưởng tượng lúc 8 giờ sáng, một người đăng một bài gây tranh cãi trên Reddit hoặc X/Twitter (ví dụ: *"Đề xuất thu phí xe máy vào nội đô"*). Mới chỉ có **20 bình luận đầu tiên** ($K = 20$) xuất hiện trong vài chục phút đầu.

**Câu hỏi nghiên cứu**: Chỉ nhìn vào bài gốc và 20 bình luận đầu đó, AI có thể đoán trước được:
1. **Quy mô lan truyền cuối kỳ ($y_{\text{size}}$)**: Sau này tổng cộng cả topic đó sẽ bùng nổ thành 50 bình luận hay 5.000 bình luận?
2. **Mức độ phân cực cộng đồng ($y_{\text{polar}}$)**: Cộng đồng rồi sẽ đồng thuận nhẹ nhàng ($\approx 0$) hay chia làm hai phe chửi nhau nảy lửa 50/50 ($\approx 1$)?
3. **Tin đồn / Thật-Giả ($y_{\text{rumor}}$)**: Trên bộ dữ liệu chuẩn quốc tế PHEME, chuỗi phản hồi ban đầu có đủ dấu hiệu nhận biết tin giả / tin đồn chưa kiểm chứng không?

---

## 2. Giải nghĩa từng ký hiệu & thành phần trong mô hình

Đừng lo ký hiệu toán học! Mỗi ký hiệu chỉ tương ứng với một ý nghĩa rất cụ thể:

| Ký hiệu | Tên gọi trong bài báo | Hiểu nôm na bằng đời thường | Nằm ở file nào? |
| :--- | :--- | :--- | :--- |
| $K = 20$ | Early Horizon Window | "Cửa sổ sớm": AI **chỉ được nhìn** bài gốc + tối đa 20 bình luận đến sớm nhất theo giờ UTC. Bình luận thứ 21 trở đi bị khóa kín để tính đáp án, **cấm rò rỉ**. | `src/data/pyg_dataset.py` |
| $h_i^{(0)}$ | Node Features (770 chiều) | Bộ não NLP (`ViSoBERT` tiếng Việt / `RoBERTa` tiếng Anh) đọc nội dung bình luận $i$ thành 768 con số, ghép thêm **thời gian đến trễ** ($t_{\text{norm}}$) và **độ sâu tầng cây** ($d_{\text{norm}}$). | `src/nlp/extract_features.py` |
| $\mathbf{a}_{uv} = [s, h, r, c]^\top$ | Behavioral Edge Vector (4D / 2D) | Đọc `(bài gốc, cha u, con v)`: $s_{uv} \in [-1, 1]$ (lập trường), $h_{uv} \in [0, 1]$ (gay gắt), $r_{uv} \in [0, 1]$ (vô lý/ngụy biện), $c_{uv} \in [0, 1]$ (lập luận xây dựng). | `qwen_teacher.py` $\to$ `dual_edge_scorer.py` |
| **TD / BU** | Top-Down & Bottom-Up | Hai luồng truyền tin: **TD** (từ bài gốc truyền xuống nhánh con) và **BU** (từ các cãi vã ở lá dội ngược lên bài gốc). | `src/models/sbi_fagat.py` |
| $\gamma_{uv} \in (-1, +1)$ | Signed Frequency Gate | **Trái tim của SBi-FAGAT v2**: Chiếc van thông minh tiền chuẩn hóa (`LayerNorm`) trên từng cạnh! Đồng thuận có lý lẽ $\to \gamma_{uv} > 0$; phản bác / công kích $\to \gamma_{uv} < 0$ đẩy hai phe tách xa nhờ hàm kích hoạt **lẻ đối xứng dấu** $x + 0.2\tanh(x)$. | `StanceGuidedFAConv` trong `src/models/sbi_fagat.py` |
| $y_{\text{polar}} \in [0, 1]$ | Polarization Target | Mức độ chia rẽ hai phe ở toàn cây cuối kỳ (kết hợp phương sai lập trường $\sigma_s^2$ và độ cân bằng 2 phe $4 p_+ p_-$). | `_compute_full_tree_polarization` |
| $BC \in [0, 1]$ | Sarle's Bimodality Coefficient | Chỉ số thống kê độc lập ($BC > 0.555$) kiểm tra xem cộng đồng thực sự tách thành **2 đỉnh (Ủng hộ vs Phản đối)** hay chỉ là nhiễu 1 đỉnh. | `compute_bimodality_coefficient` |

---

## 3. Tại sao GCN truyền thống thất bại, còn SBi-FAGAT lại hiệu quả?

* **Bệnh "San phẳng quan điểm" (Over-smoothing) của GCN thường (`StandardGCN`, `BiGCN`)**:  
  Các Mạng Nơ-ron Đồ thị (GNN) thông thường mặc định **cộng trung bình** vector của bạn với những người trả lời bạn. Kết quả là nếu Phe Ủng hộ ($+1$) và Phe Phản đối ($-1$) cãi nhau kịch liệt ở cùng một nhánh, GCN cộng trung bình lại thành... $0$ (**trung lập giả tạo**)! Nhìn bề ngoài cây cãi nhau nảy lửa, nhưng qua 2 lớp GCN thì vector cả đám trông giống nhau như một cuộc thảo luận hòa bình.
* **Cách SBi-FAGAT v2 giải quyết**:  
  Nhờ đọc thêm vector hành vi 4D $\mathbf{a}_{uv} = [s_{uv}, h_{uv}, r_{uv}, c_{uv}]^\top$, lớp `StanceGuidedFAConv` học cổng $\gamma_{uv} = \tanh(\dots)$ có thể nhận **giá trị ÂM** trên các cạnh đối lập/gay gắt. Khối cập nhật dùng kết hợp thặng dư $+ 0.5\,\mathbf{h}_v$ và hàm kích hoạt **lẻ đối xứng dấu** thay cho ELU (vốn làm lệch dấu dương/âm), giúp giữ ranh giới hai phe sắc nét ngay từ 20 bình luận đầu tiên!

👉 **Tự chạy minh họa 5 giây ngay trên máy CPU (không cần GPU hay tải dữ liệu lớn):**

```bash
python -m scripts.demo_research_walkthrough
python -m scripts.predict_live_thread --demo-offline
python -m scripts.benchmark_efficiency --v2 --edge-dim 4
```

---

## 4. Những nâng cấp SOTA & Chặt chẽ Khoa học (SBi-FAGAT v2)

Toàn bộ pipeline đã được nâng cấp để vừa chạy nhanh gấp 10–20$\times$ trên Kaggle `T4 x 2`, vừa đạt chuẩn kiểm định quốc tế:
1. **G-Eval 1-Pass Logit Teacher + QLoRA sạch**: Teacher `Qwen3.8-27B QLoRA` đọc phân phối xác suất chữ số `1..5` chỉ trong **1 lượt forward** (`--scoring-mode logit`) thay vì sinh JSON 96 token chậm/dễ lỗi, rồi chưng cất về Student nhanh (`ViSoBERT` / `DeBERTa-v3-base`) cô lập hoàn toàn khỏi tập Test qua `topic_splits.json`.
2. **Chặt chẽ `min_total_replies = 25` & Cô lập LOO**: Mọi cây đánh giá có tối thiểu $\ge 5$ bình luận tương lai sau $K=20$ (loại bỏ suy biến phương sai tại $N=21$); trên PHEME Leave-One-Event-Out dùng cô lập `edge_source="teacher"`.
3. **Đo phân cực tương lai (`future_only`) & Công bằng 8 mô hình**: Đáp án `y_polar` chỉ tính từ các bình luận tương lai $t > K$; trong chế độ `--v2`, đặc trưng tốc độ 8 chiều (`cascade_stats`) được cấp **công bằng cho cả 8 mô hình** (`TextMLP`, `TemporalStanceMLP`, ..., `SBiFAGAT`) để chứng minh phần thắng thực sự đến từ cấu trúc đồ thị có dấu.
4. **Đầu đo tự tin $\beta$-NLL & Dừng sớm theo `task_loss`**: Dùng $\beta$-NLL ($\beta=0.5$, `detach()` phần dư) để đầu đo độ tự tin $\exp(-\hat{\sigma}_{\text{polar}}/0.25)$ không làm hỏng đầu hồi quy, đồng thời Early Stopping và `auto_loop` luôn chọn mô hình tốt nhất dựa trên `task_loss` ($\alpha_{\text{polar}}=4.0$).
5. **Kiểm định thống kê chặt (`Holm-Bonferroni` + Sample-level Bootstrap từng seed)**: Tính Bootstrap trên từng hạt giống rồi lấy trung bình ($5$ seeds) kèm hiệu chỉnh Holm-Bonferroni và chấm đồng thuận người (`compute_kappa.py`) cho cả stance 3 lớp lẫn behavior 5 lớp.

---

## 5. Lộ trình 4 bước đọc code & thực hành (Từ dễ đến khó)

1. **Bước 1 — Chạy kiểm thử & Demo (5 phút, trên laptop CPU)**:  
   Chạy `pytest -q` và `python -m scripts.demo_research_walkthrough` để thấy dòng dữ liệu chạy qua SBi-FAGAT.
2. **Bước 2 — Đọc luồng dữ liệu chống rò rỉ (15 phút)**:  
   Đọc `src/data/schema.py` $\to$ `src/data/pheme_loader.py` $\to$ `src/data/arctic_shift_vn.py` $\to$ `src/data/pyg_dataset.py` (chú ý đoạn cắt $K=20$ và `future_only`).
3. **Bước 3 — Đọc mô hình SBi-FAGAT (15 phút)**:  
   Đọc `src/models/baselines.py` (xem `TemporalStanceMLP`, `BiGCN`) rồi sang `src/models/sbi_fagat.py` (`StanceGuidedFAConv` chỉ dài hơn 60 dòng!).
4. **Bước 4 — Chạy thực nghiệm thật trên GPU 16GB & Kiểm định người (Khi sẵn sàng)**:  
   Làm đúng các ticket Jira Sprint 2 & Sprint 3: tải PHEME/crawl 500 cây VN $\to$ 2 thành viên chấm pilot 30 câu rồi chấm mù độc lập 300 cặp câu (`compute_kappa.py`) $\to$ xuất `.pt` $\to$ chạy 5 seeds (`run_experiments.py`, `run_ablation.py`, `run_upfd_extension.py`). Chỉ khi có đủ số liệu thật sạch mới chuyển sang viết bài báo!
