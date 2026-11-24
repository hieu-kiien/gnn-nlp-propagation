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

| Ký hiệu | Tên gọi trong code / bài báo | Hiểu nôm na bằng đời thường | Nằm ở file nào? |
| :--- | :--- | :--- | :--- |
| $K = 20$ | Early Horizon Window | "Cửa sổ sớm": AI **chỉ được nhìn** bài gốc + tối đa 20 bình luận đến sớm nhất theo giờ UTC. Bình luận thứ 21 trở đi bị khóa kín để tính đáp án, **cấm rò rỉ**. | `src/data/pyg_dataset.py` |
| **SpanPool** (`770` + `10`) | Ancestor-Conditioned Node & `struct_x` | Đọc `[root | parent | reply]` qua encoder đóng băng (`uitnlp/visobert` / `cardiffnlp/twitter-roberta-base`) và chỉ lấy vector ở phần `reply` để hiểu câu ngắn kiểu *"Đúng rồi!"*, ghép thêm 10 chỉ số cấu trúc–thời gian nhân quả `struct_x`. | `src/nlp/extract_features.py` & `TwoStreamNodeStem` |
| $\mathbf{a}_{uv} = [s_{uv}, h_{uv}]^\top$ | 2D Stance–Hostility Edge Vector | Mô hình thầy NLI (`mDeBERTa-v3-xnli` với *Symmetric Verbalizer*) + thầy Độ gay gắt (`UIT-ViHSD` / `TweetEval`) chưng cất về Student tách đoạn `DualHeadCrossEncoder`. | `src/nlp/dual_edge_scorer.py` |
| **TD / BU + Dir-GNN** | Top-Down & Bottom-Up Mixing | Hai luồng truyền tin hai hướng: **TD** (từ gốc xuống lá) và **BU** (từ bình luận lá dội ngược lên gốc), có ma trận trộn chéo giữa 2 hướng qua từng tầng. | `src/models/sbi_fagat.py` |
| $\gamma_{uv}, \alpha_{uv}^{\text{exc}}$ | 2-Channel Adaptive Gates | **Trái tim của SBi-FAGAT**: Nửa kênh đầu dùng van có dấu $\gamma_{uv} \in (-1, +1)$ để **đẩy hai phe đối lập tách xa nhau**; nửa kênh sau dùng van không dấu $\alpha_{uv}^{\text{exc}} \in (0, 1)$ đo mức **kích thích / gay gắt** làm câu tranh luận lan rộng hơn. | `StanceGuidedFAConv` trong `src/models/sbi_fagat.py` |
| **BFS Dipole** (`max_hops=64`) | Root-Aligned Camp Readout | Lần ngược dấu cạnh từ gốc $r$: *"phản đối của phản đối = ủng hộ gốc"*, nhờ vậy gom đúng 2 cụm Ủng hộ gốc vs. Phản đối gốc ở tầng sâu. | `_propagate_root_aligned_edge_signal` |
| $y_{\text{polar}} \in [0, 1]$ | Polarization Target (biên độ tuyến tính) | Mức độ chia rẽ hai phe ở toàn cây tương lai ($t > K$), kết hợp $\sqrt{\sigma_{\tilde{s}}^2}$ và cân bằng 2 phe $2\sqrt{\frac{n}{n-1} m_+ m_-}$. | `_compute_full_tree_polarization` |
| **Tier 2 Macro** | Mạng người dùng có dấu vĩ mô | Chiếu các tương tác $(u, v)$ xuyên chủ đề để đo *Structural Balance*, $Q_{\text{signed}}$, *RWC* và dự báo liên kết đối đầu theo thời gian (ẩn đặc trưng tương lai). | `src/data/user_network.py` & `scripts/run_macro_signed_network.py` |

---

## 3. Tại sao GCN truyền thống thất bại, còn SBi-FAGAT lại hiệu quả?

* **Bệnh "San phẳng quan điểm" (Over-smoothing) của GCN thường (`StandardGCN`, `BiGCN`)**:  
  GNN thông thường mặc định **cộng trung bình** vector của bạn với người phản bác bạn. Khi Phe Ủng hộ ($+1$) và Phe Phản đối ($-1$) cãi nhau kịch liệt ở cùng một nhánh, GCN cộng trung bình lại thành $0$ (**trung lập giả tạo**)!
* **Cách SBi-FAGAT (2D hợp nhất) giải quyết**:  
  Nhờ đọc trực tiếp cặp $[s_{uv}, h_{uv}]^\top$, kênh phân cực của `StanceGuidedFAConv` sinh cổng $\gamma_{uv} < 0$ trên cạnh phản bác để trừ ngược hướng phe đối lập (kèm hàm kích hoạt lẻ đối xứng $x + 0.2\tanh(x)$), đồng thời kênh kích thích $\alpha_{uv}^{\text{exc}}$ giữ lại năng lượng tranh cãi giúp dự báo bùng nổ quy mô $y_{\text{size}}$ và độ chia rẽ $y_{\text{polar}}$ ngay từ 20 bình luận đầu!

👉 **Tự chạy minh họa 5 giây ngay trên máy CPU (không cần GPU hay tải dữ liệu lớn):**

```bash
python -m scripts.demo_research_walkthrough
python -m scripts.predict_live_thread --demo-offline
python -m scripts.benchmark_efficiency --use-cascade-stats --hidden-dim 64 --edge-dim 2
```

---

## 4. Điểm chuẩn Khoa học & Kiểm soát Lỗi của Hệ thống

1. **Độ phủ 100% Teacher sạch**: Bỏ hoàn toàn mô hình 27B nặng nề dễ lỗi OOM/timeout; thay bằng bộ thầy NLI + ViHSD/TweetEval độ phủ 100%, chưng cất cô lập trên tập Train của `topic_splits.json`.
2. **Chặt chẽ `min_total_replies = 25` & Không rò rỉ khi quét $K=50$**: Mọi cây đều có tối thiểu $\ge 5$ bình luận tương lai ngoài $K=20$; `EarlyCascadeDataset` và `run_ablation.py` luôn khóa cứng phân hoạch từ `topic_splits.json` dù ở mức lọc `K=50`.
3. **Thi đấu Công bằng 8 Mô hình ở `hidden_dim = 64`**: Không tune riêng siêu tham số cho `SBiFAGAT`; cả 8 mô hình cùng chia sẻ `TwoStreamNodeStem`, đầu ra 128 chiều, `cascade_stats` 8 chiều và hàm mất mát **KL-calibrated BCE + Smooth-L1 ($\beta = 0.05$)** cho $y_{\text{polar}}$ (kèm Huber cho $y_{\text{size}}$), so sánh thêm với baseline không cần học **Early-K Persistence**.
4. **Kiểm định Người–Máy Nghiêm ngặt**: `compute_kappa.py` hỗ trợ cả `--strict-adjudication` (2 người chấm chéo) lẫn `--allow-single-gold` (tiền chú thích + tác giả duyệt, ghi `IAA = --`), kết hợp đối chứng tiếng Anh `RumourEval SDQC` (`--context-mode root`).

---

## 5. Lộ trình 4 bước đọc code & thực hành

1. **Bước 1 — Chạy kiểm thử & Demo (5 phút, trên laptop CPU)**:  
   Chạy `pytest -q`, `python -m scripts.demo_research_walkthrough` và `python -m scripts.run_macro_signed_network --smoke-test`.
2. **Bước 2 — Đọc luồng dữ liệu chống rò rỉ (15 phút)**:  
   Đọc `src/data/schema.py` $\to$ `src/data/pheme_loader.py` $\to$ `src/data/arctic_shift_vn.py` $\to$ `src/data/pyg_dataset.py` $\to$ `src/data/user_network.py`.
3. **Bước 3 — Đọc mô hình SBi-FAGAT (15 phút)**:  
   Đọc `src/models/baselines.py` (`TwoStreamNodeStem`, `MultiTaskHead`, `BiGCN`) rồi sang `src/models/sbi_fagat.py` (`StanceGuidedFAConv`).
4. **Bước 4 — Chạy thực nghiệm thật trên Kaggle T4x2 & Kiểm định người**:  
   Chạy 2 sổ tay song song trong `notebooks/` (kèm `kaggle_macro_signed_gnn_t4.ipynb` cho Tầng 2) $\to$ chấm mù/kiểm định phiếu tinh gọn `20 pilot + 120 cặp Test + 30 cây Test VN` qua `scripts/human_audit_ui.py` $\to$ ghép bảng LaTeX thật vào `paper/main.tex`!
