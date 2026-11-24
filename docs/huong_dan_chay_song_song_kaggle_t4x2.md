# Hướng dẫn Chạy Song Song trên Kaggle `GPU T4 x 2` (32 GB VRAM) — Pipeline 2D Hợp nhất

Tài liệu này hướng dẫn trực tiếp cho **2 thành viên trong nhóm** cách đưa mã nguồn mới nhất từ GitHub lên **Kaggle `T4 x 2`** (mỗi bạn có **30 giờ GPU miễn phí/tuần**, tổng **60 giờ GPU/tuần**) để chạy song song không phải chờ nhau trên **kiến trúc hai tầng 2D hợp nhất** (`hidden_dim = 64`, VN `K=20, min_total_replies=25` và PHEME giảm cửa sổ quan sát `K=10, min_total_replies=15`, kèm sổ tay Tầng 2 `kaggle_macro_signed_gnn_t4.ipynb` cho Signed Link Prediction GPU).

---

## 1. Kiến trúc Hợp nhất 2D bên trong hệ thống có gì?

Hệ thống AI của nhóm hoạt động theo **dây chuyền 3 tầng sạch và gọn (phủ 100% nhãn Teacher)**:

1. **Tầng 1 — Teacher Kép NLI & Hostility Độ phủ 100% (`dual_nli`):**
   - **Thầy Lập trường (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`)**: Đọc cặp `(Parent, Reply)` trong ngân sách `384` token (bảo toàn trọn vẹn phần `Reply`) bằng **Symmetric 2-Hypothesis Stance Verbalizer** để tính xác suất `[p_support, p_neutral, p_oppose]` và $s_{uv} \in [-1, 1]$ (đồng thời tính luôn `root_s_uv` cho các bình luận sâu `depth >= 2`).
   - **Thầy Độ gay gắt (`h_uv`)**: Tinh chỉnh `uitnlp/visobert` với trọng số lớp cân bằng trên **UIT-ViHSD** (cho Tiếng Việt), hoặc dùng `cardiffnlp/twitter-roberta-base-offensive` (cho Tiếng Anh).
   - Toàn bộ cặp được chấm dựa trên phân hoạch chuẩn hóa `topic_splits.json` (`--min-total-replies 25`, tuyệt đối không chạm vào tập Kiểm tra `Test`; nhánh PHEME LOO dùng cô lập `edge_source="teacher"`).

2. **Tầng 2 — Student `DualHeadCrossEncoder` & Node *Ancestor-Conditioned SpanPool* (`ViSoBERT` VN / `twitter-roberta-base` EN):**
   - Chưng cất `DualHeadCrossEncoder` 2D **chỉ trên tập Học (`70%` chủ đề chuẩn hóa)** bằng kỹ thuật tách đoạn (`h_uv` chỉ pool trên token thuộc `Reply`, tránh lời gay gắt ở bình luận cha lây sang bình luận con), hàm mất mát `BCEWithLogitsLoss + KL (T=2.0)`, và tăng cường cặp sâu `(root, reply)`.
   - Trích xuất nhúng nút 768 chiều bằng **Ancestor-Conditioned SpanPool** đóng băng (`[CLS] root [SEP] parent [SEP] reply [SEP]`, chỉ pool trên token `reply`) kèm `struct_x` 10 chiều và `cascade_stats` 8 chiều xuống tệp `.pt`.

3. **Tầng 3 — Mạng Nơ-ron Đồ thị `SBi-FAGAT` (`hidden_dim = 64`) & Kiểm định Công bằng 8 Mô hình:**
   - Thi đấu công bằng **8 mô hình** với cùng `hidden_dim = 64`, `TwoStreamNodeStem` (`struct_x` 10 chiều), `cascade_stats` 8 chiều, và hàm mất mát tối ưu (**KL-calibrated `BCEWithLogitsLoss` + Smooth-L1** cho $y_{\text{polar}}$, Warmup + `ReduceLROnPlateau` + EMA) qua 5 hạt giống `[42, 100, 2024, 777, 999]` + **Early-K Persistence Baseline**.
   - Xuất tệp nén `.zip` kết quả gọn nhẹ `sbifagat_{role}_bundle.zip` (luôn chặn không đóng gói `*_machine_key.csv` hay `.secret`/`.env` kể cả khi truyền `--bundle-weights`), đồng thời đóng gói riêng file đáp án chấm mù vào tệp niêm phong `sbifagat_{role}_sealed_machine_keys.zip` để không bị mất khi tắt máy ảo Kaggle. Tùy chọn bật `--run-auto-loop` để xuất thêm chẩn đoán các ca lệch nhiều nhất trên tập Validation mà không chỉnh thiên vị siêu tham số cho riêng `SBiFAGAT`.

---

## 2. Cách chia 2 Bạn chạy Song Song trên Kaggle `T4 x 2`

### Cách A — Chia theo Track Dữ liệu (Khuyên dùng: Hai bạn chạy độc lập 100% cùng lúc)
* **Thành viên 1** mở sổ tay [kaggle_member1_nlp_and_vn_t4x2.ipynb](../notebooks/kaggle_member1_nlp_and_vn_t4x2.ipynb), chọn `ROLE = "member1_vn"` *(có hỗ trợ biến `SBIF_BRANCH`)*:
  - Thu thập **1.500 cây thảo luận Tiếng Việt** (`r/vozforums`, `r/VietNam` với `--min-comments 25 --min-vi-ratio 0.65 --max-pages-per-sub 300`, hỗ trợ resume con trỏ qua `.state.json`).
  - Chạy trọn gói `Dual-Teacher (mDeBERTa-v3-xnli + ViHSD) -> ViSoBERT DualHeadCrossEncoder 2D + SpanPool -> SBi-FAGAT (hidden_dim=64)` cho Tiếng Việt, tự xuất phiếu chấm mù tinh gọn **`20` pilot + `120` cặp Test + `30` cây Test** (`audit_vn_50_trees.py`).
  - Tải về `sbifagat_member1_vn_bundle.zip` và `sbifagat_member1_vn_sealed_machine_keys.zip`.
* **Thành viên 2** mở sổ tay [kaggle_member2_gnn_and_en_t4x2.ipynb](../notebooks/kaggle_member2_gnn_and_en_t4x2.ipynb), chọn `ROLE = "member2_en"`:
  - Tải bộ chuẩn quốc tế **PHEME** (có kiểm tra MD5 + fast-path `curl` + fallback `/kaggle/input`).
  - Chạy trọn gói `Dual-Teacher -> Twitter-RoBERTa DualHeadCrossEncoder 2D + SpanPool -> SBi-FAGAT (Inductive + LOO + Ablation + EDA + Efficiency)` cho Tiếng Anh (hỗ trợ `PHEME_K_HORIZON=10`, `PHEME_MIN_REPLIES=15`, kiểm định đối chứng EN qua `scripts.eval_rumoureval_sdqc --context-mode root` và `RUN_UPFD`).
  - Tải về `sbifagat_member2_en_bundle.zip` và `sbifagat_member2_en_sealed_machine_keys.zip`.
* **Sổ tay Tầng 2 (Macro Signed User Network GPU):** Dùng [kaggle_macro_signed_gnn_t4.ipynb](../notebooks/kaggle_macro_signed_gnn_t4.ipynb) khi có sẵn `data/processed/{vn,pheme}/nlp_features.pt` để chạy Structural Balance + $Q_{\text{signed}}$ + RWC + 5-seed Temporal Signed Link Prediction (`UserTextMLP, StandardGCN, StandardGATv2, SGCN, FAGCN, SBiFAGAT-Link`) và xuất `sbifagat_macro_signed_bundle.zip`.
* **Gộp kết quả 2 track sau khi tải cả 2 gói về máy cục bộ:** Giải nén `paper/` từ cả hai gói vào chung thư mục `paper/` rồi chạy:
  ```bash
  python -m scripts.assemble_paper_artifacts --paper-dir paper
  ```
  Lệnh này gộp bảng EDA (`tables_dataset_eda.tex`), bảng mạng xã hội vĩ mô (`tables_macro_network.tex`), bảng hiệu năng (`tables_efficiency.tex`), bảng kiểm định con người (`tables_human_audit.tex` nếu đã có JSON báo cáo), kiểm tra sạch lỗi `_` LaTeX và xuất báo cáo tồn kho `paper/assembly_manifest.json`.

### Cách B — Chia mỗi người chuyên trách một loại Mô hình (NLP vs GNN)
* **Thành viên 1 (Chuyên NLP):** Đổi `ROLE = "model_nlp_only"` trong sổ tay 1 để train Teacher/Student 2D và xuất `data/processed/{vn,pheme}/nlp_features.pt`.
* **Thành viên 2 (Chuyên GNN):** Đổi `ROLE = "model_gnn_only"` trong sổ tay 2 để nạp `nlp_features.pt` (hoặc bật thêm `--bootstrap-if-missing`), chuyên tâm chạy benchmark 8 mô hình × 5 hạt giống + ablation `K in {10, 20, 30, 50}` + vẽ biểu đồ bài báo.

---

## 3. Tự kiểm tra THẬT bằng link bài viết trên mạng (Sau khi train xong)

Sau khi có checkpoint, dùng trực tiếp công cụ [predict_live_thread.py](../scripts/predict_live_thread.py) để thử với bất kỳ bài thảo luận Reddit nào:

```bash
# Chạy thử mẫu có sẵn không cần mạng:
python -m scripts.predict_live_thread --demo-offline

# Kiểm tra trực tiếp bằng đường link bài viết thật trên Reddit:
python -m scripts.predict_live_thread \
    --reddit-url "https://www.reddit.com/r/vozforums/comments/<MA_BAI_VIET>/" \
    --cross-encoder-ckpt checkpoints/dual_ce_vn \
    --gnn-ckpt checkpoints/best_sbifagat_vn.pt
```

Hệ thống sẽ tự động **bịt mắt từ bình luận thứ 21 trở đi**, phân tích lập trường & độ gay gắt cho 20 bình luận đầu, chỉ điểm các bình luận kích hoạt xung đột mạnh nhất, dự báo quy mô + phân cực, rồi đối chiếu kết quả thực tế!

---

## 4. Kiểm tra chéo con người (Human Audit) sau mỗi mẻ chạy Kaggle

Sau khi chạy xong trên Kaggle, trước khi tắt máy ảo hãy tải đồng thời `sbifagat_{role}_bundle.zip` và gói niêm phong `sbifagat_{role}_sealed_machine_keys.zip` (giữ niêm phong, tuyệt đối không mở trước khi chấm). Sau khi hoàn tất chấm mù qua [human_audit_ui.py](../scripts/human_audit_ui.py) (`20` pilot + `120` cặp Test trong `data/interim/human_audit_vi_120.csv` và `30` cây Test trong `data/interim/human_audit_vn_30_trees.csv`), giải nén `*_machine_key.csv` từ gói niêm phong về `data/interim/` rồi chạy:

```bash
# Nếu 2 người chấm độc lập & đã hòa giải gold_adjudicated_*:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --strict-adjudication --output-json paper/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --strict-adjudication --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json paper/results_human_audit_vn_30_trees.json

# Nếu dùng quy trình tiền chú thích AI + kiểm định gold đơn lẻ có xác nhận tác giả:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --allow-single-gold --output-json paper/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --allow-single-gold --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json paper/results_human_audit_vn_30_trees.json

# Đối chứng lập trường tiếng Anh RumourEval-2019 SDQC (ngữ cảnh gốc root):
python -m scripts.eval_rumoureval_sdqc --context-mode root --output-json paper/results_rumoureval_sdqc.json
```

Cờ `--strict-adjudication` đảm bảo mọi cặp 2 người chấm lệch nhau đều đã được hòa giải (`gold_adjudicated_*`), còn `--allow-single-gold` đánh giá trung thực trên nhãn vàng duy nhất đã xác nhận (ghi `IAA = --`) trước khi ghép vào bài báo!
