# Hướng dẫn Chạy Song Song trên Kaggle `GPU T4 x 2` (32 GB VRAM) — Phiên bản v2

Tài liệu này hướng dẫn trực tiếp cho **2 thành viên trong nhóm** cách đưa mã nguồn mới nhất từ GitHub lên **Kaggle `T4 x 2`** (mỗi bạn có **30 giờ GPU miễn phí/tuần**, tổng **60 giờ GPU/tuần**) để chạy song song không phải chờ nhau.

---

## 1. Hệ thống v2 mới nâng cấp có gì bên trong?

Hệ thống AI của nhóm hoạt động theo **dây chuyền 3 mô hình nối tiếp nhau**:

1. **Mô hình 1 — Thầy Thông minh (`Qwen/Qwen3.8-27B` gắn mảnh ghép thích nghi `QLoRA`):**
   - Nạp chế độ nén **4-bit NF4 + `float16`** trải đều trên cả **2 card T4 (32 GB VRAM)**, đóng băng phần xử lý ảnh để tiết kiệm bộ nhớ.
   - Đọc đầy đủ *(Bài gốc + Bình luận trước + Bình luận trả lời)* để phân loại **5 nhóm hành vi thực tế** (không gộp thô thành ủng hộ/phản đối nữa):
     1. `support`: Tán thành / ủng hộ hợp lý
     2. `neutral`: Trung lập / hỏi thông tin
     3. `constructive_oppose`: Phản biện văn minh có lý lẽ
     4. `irrational_fallacy`: Bình luận vô lý, ngụy biện, nói cùn, bẻ lái phá đám
     5. `hostile_conflict`: Chống đối gay gắt, công kích, kích động xung đột
   - Xuất 4 chỉ số liên tục `[s_uv, h_uv, r_uv, c_uv]` làm nhãn dạy cho Mô hình 2 (tuyệt đối không chạm vào tập Kiểm tra `Test`).

2. **Mô hình 2 — Bộ Đọc Nhanh Trực Chiến NLP (`ViSoBERT` cho Tiếng Việt / `DeBERTa-v3-base` cho Tiếng Anh):**
   - Chưng cất từ Mô hình 1 **chỉ trên tập Học (`70%` chủ đề)** bằng cả 2 card T4 song song (`DataParallel`).
   - Nhỏ gọn, suy luận cực nhanh trong thực tế để gắn nhãn 5 hành vi và 4 chỉ số trên từng mũi tên bình luận.

3. **Mô hình 3 — Mạng Nơ-ron Đồ thị `SBi-FAGAT v2` (Trọng tâm bài báo):**
   - Nhận dữ liệu từ Mô hình 2 + **Đồng hồ tốc độ & hình dáng 8 chiều của 20 câu đầu** (tốc độ dồn dập, gia tốc tăng/giảm, số nhánh trực tiếp, tỷ lệ câu có nội dung thật, chỉ số kích động 1 chiều) + **Đầu đo Độ tự tin dự báo** (tự cảnh báo khi 20 câu đầu quá nhạt cần chờ thêm).
   - Chạy vòng lặp tự tối ưu trên tập Validation (`run_self_auditing_validation_loop`), xuất báo cáo các ca đoán lệch nhiều nhất (`worst_val_cases_for_review.json`), rồi chạy thi đấu 8 mô hình × 5 hạt giống ngẫu nhiên và xuất biểu đồ vector.

---

## 2. Cách chia 2 Bạn chạy Song Song trên Kaggle `T4 x 2`

### Cách A — Chia theo Track Dữ liệu (Khuyên dùng: Hai bạn chạy độc lập 100% cùng lúc)
* **Thành viên 1** mở sổ tay [kaggle_member1_nlp_and_vn_t4x2.ipynb](../notebooks/kaggle_member1_nlp_and_vn_t4x2.ipynb), chọn `ROLE = "member1_vn"`:
  - Thu thập **1.500 cây thảo luận Tiếng Việt** (`r/vozforums`, `r/VietNam`).
  - Chạy trọn gói `Qwen3.8-27B QLoRA -> ViSoBERT 4D -> SBi-FAGAT v2` cho Tiếng Việt.
  - Tải về `sbifagat_member1_vn_bundle.zip`.
* **Thành viên 2** mở sổ tay [kaggle_member2_gnn_and_en_t4x2.ipynb](../notebooks/kaggle_member2_gnn_and_en_t4x2.ipynb), chọn `ROLE = "member2_en"`:
  - Tải bộ chuẩn quốc tế **PHEME** (có kiểm tra MD5).
  - Chạy trọn gói `Qwen3.8-27B QLoRA -> DeBERTa-v3-base 4D -> SBi-FAGAT v2 (Inductive + Leave-One-Event-Out + Ablation + UPFD)`.
  - Tải về `sbifagat_member2_en_bundle.zip`.

### Cách B — Chia mỗi người chuyên trách một loại Mô hình (NLP vs GNN)
* **Thành viên 1 (Chuyên NLP):** Đổi `ROLE = "model_nlp_only"` trong sổ tay 1 để train Mô hình 1 + Mô hình 2 và xuất `data/processed/{vn,pheme}/nlp_features.pt`.
* **Thành viên 2 (Chuyên GNN):** Đổi `ROLE = "model_gnn_only"` trong sổ tay 2 để nạp `nlp_features.pt` (hoặc bật thêm `--bootstrap-if-missing` để chạy ngay không cần chờ), chuyên tâm chạy vòng lặp tối ưu `SBi-FAGAT v2` + 7 mô hình đối chứng + vẽ biểu đồ bài báo.

---

## 3. Tự kiểm tra THẬT bằng link bài viết trên mạng (Sau khi train xong)

Sau khi có checkpoint, dùng trực tiếp công cụ [predict_live_thread.py](../scripts/predict_live_thread.py) để thử với bất kỳ bài thảo luận Reddit mới nào vừa đăng trên mạng:

```bash
# Chạy thử mẫu có sẵn không cần mạng:
python -m scripts.predict_live_thread --demo-offline

# Kiểm tra trực tiếp bằng đường link bài viết thật trên Reddit:
python -m scripts.predict_live_thread \
    --reddit-url "https://www.reddit.com/r/vozforums/comments/<MA_BAI_VIET>/" \
    --cross-encoder-ckpt checkpoints/dual_ce_vn_v2 \
    --gnn-ckpt outputs/auto_loop_vn/best_validated_sbifagat.pt
```

Hệ thống sẽ tự động **bịt mắt từ bình luận thứ 21 trở đi**, phân loại 5 hành vi cho 20 câu đầu, chỉ điểm những câu vô lý/chống đối nguy hiểm nhất, dự báo quy mô + phân cực + độ tự tin, rồi đặt cạnh kết quả thực tế (nếu bài đó đã vượt mốc 20 bình luận) để bạn kiểm chứng tận mắt!

---

## 4. Vòng lặp kiểm tra cùng AI sau mỗi mẻ chạy Kaggle

Sau khi chạy xong trên Kaggle, chỉ cần tải về và gửi cho AI đọc 3 file gọn nhẹ (vài chục KB):
1. `outputs/auto_loop_vn/loop_diagnostics.json` (và bản `auto_loop_pheme` của Thành viên 2)
2. `outputs/auto_loop_vn/worst_val_cases_for_review.json` (danh sách các ca trên tập Validation bị đoán lệch nhất để AI chấm lỗi và tinh chỉnh thuật toán)
3. `paper/results_main_vn.json` & `paper/results_main_pheme.json`

Và **2 bạn nhớ tự chấm chéo độc lập** 300 cặp câu trong `human_audit_vi_300.csv` (mở bằng Excel/Google Sheets) rồi chạy `python -m scripts.compute_kappa` để hoàn thiện con số xác thực người bản xứ cho bài báo!
