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
| $s_{uv} \in [-1, +1]$ | Stance Score (Lập trường) | AI đọc cặp `(bình luận cha u, bình luận con v)`: nếu con đồng tình với cha $\to +1$, trung lập $\to 0$, phản bác cha $\to -1$. | `src/nlp/dual_edge_scorer.py` |
| $h_{uv} \in [0, 1]$ | Hostility Score (Độ gay gắt) | AI đo độ công kích / thù ghét / mỉa mai nặng lời của bình luận $v$ ($0$ là lịch sự, $1$ là chửi bới/công kích mạnh). | `src/nlp/dual_edge_scorer.py` |
| **TD / BU** | Top-Down & Bottom-Up | Hai luồng truyền tin: **TD** (từ bài gốc truyền xuống nhánh con) và **BU** (từ các cãi vã ở lá dội ngược lên bài gốc). | `src/models/sbi_fagat.py` |
| $\gamma_{uv} \in (-1, +1)$ | Signed Frequency Gate | **Trái tim của SBi-FAGAT**: Chiếc van thông minh trên từng cạnh trả lời! Nếu hai người **đồng ý**, $\gamma_{uv} > 0$ (gộp gần nhau); nếu hai người **cãi nhau/công kích**, $\gamma_{uv} < 0$ (**đẩy vector hai phe tách xa ra**). | `StanceGuidedFAConv` trong `src/models/sbi_fagat.py` |
| $y_{\text{polar}} \in [0, 1]$ | Polarization Target | Mức độ chia rẽ hai phe ở toàn cây cuối kỳ (kết hợp phương sai lập trường $\sigma_s^2$ và độ cân bằng 2 phe $4 p_+ p_-$). | `_compute_full_tree_polarization` |
| $BC \in [0, 1]$ | Sarle's Bimodality Coefficient | Chỉ số thống kê độc lập ($BC > 0.555$) kiểm tra xem cộng đồng thực sự tách thành **2 đỉnh (Ủng hộ vs Phản đối)** hay chỉ là nhiễu 1 đỉnh. | `compute_bimodality_coefficient` |

---

## 3. Tại sao GCN truyền thống thất bại, còn SBi-FAGAT lại hiệu quả?

* **Bệnh "San phẳng quan điểm" (Over-smoothing) của GCN thường (`StandardGCN`, `BiGCN`)**:  
  Các Mạng Nơ-ron Đồ thị (GNN) thông thường mặc định **cộng trung bình** vector của bạn với những người trả lời bạn. Kết quả là nếu Phe Ủng hộ ($+1$) và Phe Phản đối ($-1$) cãi nhau kịch liệt ở cùng một nhánh, GCN cộng trung bình lại thành... $0$ (**trung lập giả tạo**)! Nhìn bề ngoài cây cãi nhau nảy lửa, nhưng qua 2 lớp GCN thì vector cả đám trông giống nhau như một cuộc thảo luận hòa bình.
* **Cách SBi-FAGAT giải quyết**:  
  Nhờ đọc thêm cặp tín hiệu ngôn ngữ $e_{uv} = [s_{uv}, h_{uv}]$, lớp `StanceGuidedFAConv` học cổng $\gamma_{uv} = \tanh(\dots)$ có thể nhận **giá trị ÂM** trên các cạnh đối lập/gay gắt. Khi $\gamma_{uv} < 0$, bộ lọc tần số cao sẽ **giữ sự khác biệt** giữa hai phe thay vì hòa tan chúng, đồng thời vẫn cho phép các nhánh đồng thuận kết nối mượt với nhau!

👉 **Tự chạy minh họa 5 giây ngay trên máy CPU (không cần GPU hay tải dữ liệu lớn):**

```bash
python -m scripts.demo_research_walkthrough
```

Và xem chi phí tham số / tốc độ của cả 8 mô hình:

```bash
python -m scripts.benchmark_efficiency
```

---

## 4. Những nâng cấp Chặt chẽ Khoa học (Vòng Phản biện Kỹ thuật)

Trước khi chạy GPU chính thức và viết bài báo, toàn bộ pipeline đã được rà soát để vá dứt điểm các lỗi dễ bị phản biện quốc tế bác bài:
1. **Không rò rỉ nhãn khi chấm cạnh (`Frozen Teacher Zero-Leakage`)**: Thay vì chưng cất một mô hình Cross-Encoder nhỏ có nguy cơ học lỏm phân bố tập dữ liệu, ta dùng trực tiếp Teacher mở lớn (`mDeBERTa-v3-xnli` + `ViSoBERT-ViHSD` / `twitter-roberta-offensive`) chấm một lần ra cache JSON (`--score-only` $\to$ `--scored-cache-json`) với cơ chế lưu shard chống sập Colab giữa chừng.
2. **Chấm lập trường trực tiếp với bài gốc ở tầng sâu (`node_root_stance`)**: Khi bình luận C phản bác B (người đang phản bác bài gốc A), phép nhân dấu $(-1)\times(-1)=+1$ dễ sai do mỉa mai; pipeline nay chấm NLI trực tiếp cặp `(A, C)` cho bình luận sâu tầng $\ge 2$.
3. **Đo phân cực tương lai (`--polar-target-mode future_only`) & Baseline `TemporalStanceMLP`**: Chỉ tính đáp án phân cực từ bình luận $t > K$ và ép SBi-FAGAT đấu tay đôi với `TemporalStanceMLP` (mô hình không dùng đồ thị nhưng có đầy đủ vector nút + thống kê $s_{uv}, h_{uv}$) để chứng minh cấu trúc đồ thị thực sự mang lại giá trị.
4. **Kiểm định thống kê chặt (`Holm-Bonferroni` + Sample-level Bootstrap + `auto` Comparator)**: So sánh với đối thủ mạnh nhất (`--comparator-model auto`), tính Bootstrap từng mẫu trên toàn bộ 13 thước đo (kể cả `R2`, `Spearman`, `Macro-F1`, `AUC-ROC`) và hiệu chỉnh đa kiểm định Holm-Bonferroni.
5. **Đo đồng thuận thực thụ (`compute_kappa.py`)**: Bổ sung bảng Pilot 30 câu để nhóm 2 bạn thống nhất định nghĩa trước khi chấm mù 300 câu, đo cả Human-vs-Human $\kappa$ lẫn Machine-vs-Gold và kiểm chứng ngoài trên RumourEval SDQC.

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
