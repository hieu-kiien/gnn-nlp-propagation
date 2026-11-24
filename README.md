# SBi-FAGAT — Phân cực Quan điểm & Lan truyền Thông tin Sớm trên Mạng Xã hội

Kho mã nguồn nghiên cứu cho đề tài khoa học:  
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"**.

Kiến trúc hai tầng hợp nhất **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network* + *Macro Signed User-Interaction Social Network*): **Tầng 1 (Early Cascade)** (`hidden_dim = 64`, VN `K = 20, min_total_replies = 25` và PHEME Tier 2A `K = 10, min_total_replies = 15`) dự đoán đồng thời quy mô lan truyền cuối kỳ ($y_{\text{size}}$), mức độ phân cực cộng đồng ($y_{\text{polar}}$) và phân loại tin đồn (PHEME) chỉ từ bài đăng gốc và $K$ phản hồi sớm nhất; **Tầng 2 (Macro Social Network)** chiếu mọi tương tác $(u, v)$ xuyên chủ đề lên đồ thị người dùng có dấu (`13,718` người dùng VN, $Q_{\text{signed}} = 0.4280$; `50,062` người dùng PHEME, $Q_{\text{signed}} = 0.8232$, RWC $= 0.6218$) và chạy Temporal Signed Link Prediction chống rò rỉ.

> **Lộ trình Bàn giao 7 ngày (Scope Freeze — Xem chi tiết tại [TODO_ROADMAP.md](docs/project-management/TODO_ROADMAP.md)):**
> - **Không cào thêm dữ liệu mới / Không đập lõi làm lại:** Đã hoàn tất Track 1 VN, PHEME Tier 2A (`K=10, N>=15`), kiểm định VN Verified Gold (`120` cặp + `30` cây) kèm phân tích *Target-Misattributed Profanity* (`57.1%` câu `support+uncivil` chửi bên thứ 3 bị NLI gán nhầm `oppose`), `RumourEval --context-mode root` (`Macro-F1 = 0.3311`), và Tầng 2 Macro (CPU + GPU 5-seed Signed Link Prediction cho cả VN & PHEME: `SBiFAGAT-Link` dẫn đầu `Trans. ROC-AUC = 0.9181` trên PHEME).
> - **3 sản phẩm nộp cuối:** (1) Bài báo tiếng Anh chuẩn IEEE ([paper/main.pdf](paper/main.pdf)), (2) **Giao diện Demo Web MVP** chạy cục bộ (lọc rác bảo toàn cây, tô 3 màu lập trường *Tán thành / Phản đối / Trung lập*, xem mạng 2 phe Tầng 2), và (3) **Báo cáo đề tài tiếng Việt 4 chương** + Slide/Hỏi–Đáp bảo vệ đồng bộ 100% số liệu.

> **Mới bắt đầu hoặc muốn chạy trên GPU miễn phí (Google Colab / Kaggle `T4 x 2`)?**  
> - **Chạy benchmark GNN Tầng 1 nhanh trên Colab / Kaggle T4:** Dùng [notebooks/colab_gpu_v3_gnn_benchmarks.ipynb](notebooks/colab_gpu_v3_gnn_benchmarks.ipynb) (Temporal Holdout VN, Ablation/$K$-Curve và PHEME $K=10, N \ge 15$).
> - **Chạy benchmark Tầng 2 Macro Signed Network trên GPU:** Dùng [notebooks/kaggle_macro_signed_gnn_t4.ipynb](notebooks/kaggle_macro_signed_gnn_t4.ipynb).
> - **Chạy full pipeline 2 người trên Kaggle `GPU T4 x 2`:** Đọc [Hướng dẫn Chạy Song Song trên Kaggle T4 x 2](docs/huong_dan_chay_song_song_kaggle_t4x2.md) và mở `kaggle_member1_nlp_and_vn_t4x2.ipynb`, `kaggle_member2_gnn_and_en_t4x2.ipynb`.
> - Đọc [Hướng dẫn Nghiên cứu Dễ hiểu từ Con số 0](docs/huong_dan_nghien_cuu_de_hieu.md) và chạy thử minh họa trên máy CPU (<5 giây):
> ```bash
> python -m scripts.demo_research_walkthrough
> python -m scripts.predict_live_thread --demo-offline
> python -m scripts.benchmark_efficiency --use-cascade-stats --hidden-dim 64 --edge-dim 2
> python -m scripts.run_macro_signed_network --smoke-test
> ```

---

## 1. Quản trị Dự án & Năm Luật Thép

- **Bảng điều phối Scrum (Jira):** [`SBIF` Board — SBi-FAGAT Research](https://sbifagat-research.atlassian.net/jira/software/projects/SBIF/boards/2)
- **Tài liệu quy chuẩn bắt buộc:** Xem chi tiết trong [GEMINI.md](GEMINI.md) và hồ sơ [docs/project-management/](docs/project-management/PROJECT_CONTEXT.md). Mọi đóng góp mã nguồn và thực nghiệm phải tuân thủ tuyệt đối **5 Luật Thép**:
  1. **Cấm dữ liệu giả trong báo cáo:** Không dùng dữ liệu sinh ngẫu nhiên (mock/synthetic) để xuất bảng hay biểu đồ kết quả thực nghiệm (tensor giả lập chỉ được dùng trong `tests/`).
  2. **Chống rò rỉ dữ liệu (Zero Data Leakage):** Chia tập Train / Val / Test (`70/15/15`) nhất quán bằng `topic_splits.json` (`min_total_replies = 25`) theo sự kiện/chủ đề gốc (*Inductive Split*, giữ nguyên cho mọi $K \in \{10, 20, 30, 50\}$; LOO dùng `edge_source="teacher"`); đầu vào chỉ giữ bài gốc + tối đa $K = 20$ bình luận sớm nhất theo `timestamp`; nhãn mục tiêu không lọt vào đặc trưng nút/cạnh.
  3. **Chống ảo giác tài liệu tham khảo:** Mọi công trình trích dẫn trong `docs/` và `paper/references.bib` phải được xác minh tác giả, năm, hội nghị/tạp chí và DOI/arXiv thật.
  4. **Tối ưu hạ tầng GPU 16GB VRAM / Kaggle `T4 x 2`:** Tách rời trích xuất NLP ngoại tuyến (nhúng nút 768 chiều từ *Ancestor-Conditioned SpanPool* đóng băng + đặc trưng cấu trúc nhân quả 10 chiều `struct_x` + thống kê cụm 8 chiều `cascade_stats` + cạnh 2D `[s_uv, h_uv]` chưng cất cô lập phân đoạn ra `.pt`) khỏi huấn luyện GNN (`hidden_dim = 64`); bật Mixed Precision (`torch.amp`); chạy lặp lại trên đúng 5 hạt giống `[42, 100, 2024, 777, 999]` với Early Stopping theo `task_loss` và chọn checkpoint tốt nhất theo Validation.
  5. **Kiểm thử trước khi bàn giao (Test-Gated):** Mọi thay đổi mã nguồn phải có unit test tương ứng và vượt qua toàn bộ `pytest` trước khi commit/merge.

---

## 2. Cấu trúc Thư mục Dự án

```text
.
├── .github/                    # GitHub Actions CI & mẫu Pull Request kiểm tra 5 Luật Thép
├── GEMINI.md                   # Quy tắc hệ thống, 5 Luật Thép & ADRs dự án
├── pyproject.toml              # Khai báo package (sbi-fagat) & cấu hình pytest
├── requirements.txt            # Thư viện Python (>=3.10), PyTorch, PyG, Transformers, tzdata (Windows)
├── run_all.sh                  # Pipeline kiểm thử + thực nghiệm Giai đoạn 3 & UPFD (2D thống nhất)
├── configs/                    # Chứa tệp cấu hình thực nghiệm (giữ chỗ .gitkeep)
├── data/                       # Dữ liệu local (raw/interim/processed bị ignore trên Git)
├── docs/
│   ├── project-management/     # Hồ sơ quản trị dự án, kiểm toán kỹ thuật & lộ trình
│   ├── huong_dan_chay_song_song_kaggle_t4x2.md # Hướng dẫn chia 2 người chạy Kaggle T4x2
│   ├── huong_dan_nghien_cuu_de_hieu.md  # Giải thích đề tài, ký hiệu & code từ con số 0
│   ├── literature_review.md    # Tổng quan công trình nền tảng đã xác minh
│   └── math_spec.md            # Đặc tả toán học SBi-FAGAT & Hợp đồng dữ liệu
├── notebooks/
│   ├── colab_gpu_v3_gnn_benchmarks.ipynb    # Sổ tay Colab/Kaggle T4 GPU cho Temporal, Ablation & PHEME K=10/N>=15
│   ├── kaggle_macro_signed_gnn_t4.ipynb     # Sổ tay Tầng 2 Macro Signed Network + Temporal Signed Link GNN (T4 GPU)
│   ├── kaggle_member1_nlp_and_vn_t4x2.ipynb # Sổ tay Track VN (`member1_vn`)
│   └── kaggle_member2_gnn_and_en_t4x2.ipynb # Sổ tay Track EN (`member2_en`, PHEME + RumourEval SDQC)
├── paper/
│   ├── main.tex                # Bản thảo bài báo chuẩn IEEE (IEEEtran) hai tầng Micro + Macro
│   ├── main.pdf                # PDF bài báo đã biên dịch trực tiếp bằng Tectonic
│   ├── references.bib          # Thư viện trích dẫn chuẩn BibTeX (24 tài liệu đã xác minh)
│   └── figures/                # Thư mục xuất biểu đồ vector phục vụ bài báo
├── scripts/
│   ├── kaggle_parallel_runner.py # Điều phối chạy song song 4 role (kiểm tra cohort min_total_replies_extracted)
│   ├── export_dataset_stats.py # Xuất Bảng 1 EDA LaTeX động theo cohort & Early-K Persistence Baseline
│   ├── predict_live_thread.py  # Dự báo trực tiếp từ URL/ID bài viết thật trên mạng
│   ├── run_phase2.sh           # Pipeline Dual-Teacher NLI + Hostility -> DualHeadCrossEncoder & .pt (2D)
│   ├── run_macro_signed_network.py # Phân tích mạng người dùng có dấu vĩ mô (Balance/Q_signed/RWC) & Link GNN
│   ├── eval_rumoureval_sdqc.py # Kiểm chứng lập trường NLI tiếng Anh trên SemEval-2019 RumourEval (--context-mode root)
│   ├── compute_kappa.py        # Tính Cohen's Kappa / Verified Gold (--allow-single-gold), ROC-AUC/PR-AUC & LaTeX
│   ├── audit_vn_50_trees.py    # Xuất phiếu chấm mù 30 cây Test VN & đánh giá tương quan y_polar (Kendall tau_b)
│   ├── human_audit_ui.py       # Phòng chấm tay Web UI tiếng Việt chạy cục bộ (--lock-role, hỗ trợ VN 20/120/30 & EN)
│   ├── analyze_mechanisms.py   # Phân tầng xung đột & phân tích độ nhạy nhiễu đặc trưng cạnh (không khẳng định nhân quả)
│   ├── assemble_paper_artifacts.py # Gộp bảng EDA/Macro/Efficiency từ 2 track Kaggle & kiểm tra cú pháp LaTeX
│   ├── benchmark_efficiency.py # Đo #Params, tốc độ suy luận/huấn luyện & RAM/VRAM 8 mô hình (LaTeX/JSON)
│   ├── demo_research_walkthrough.py # Demo minh họa cơ chế GCN vs SBi-FAGAT trên CPU (<5s)
│   ├── run_experiments.py      # Huấn luyện công bằng 8 mô hình (`hidden_dim=64`) + Early-K Persistence qua 5 seeds
│   ├── run_ablation.py         # Thực nghiệm cắt bỏ thành phần (+ BiEdgeGATv2) & quét K cửa sổ tương lai cố định
│   └── run_upfd_extension.py   # Thực nghiệm chuyển giao cấu trúc phi văn bản trên UPFD (Accuracy, Macro-F1, ROC-AUC)
├── src/
│   ├── data/                   # schema.py, pheme_loader.py, arctic_shift_vn.py, pyg_dataset.py, user_network.py
│   ├── nlp/                    # discourse_labels.py, dual_edge_scorer.py, extract_features.py
│   ├── models/                 # baselines.py (TwoStreamNodeStem, MultiTaskHead, 7 baselines), sbi_fagat.py
│   ├── training/               # metrics.py (KL-calibrated BCE + Smooth-L1 & Huber), trainer.py, auto_loop.py, reporting.py
│   └── utils/                  # seed.py, text.py, visualize.py (lazy UMAP import)
└── tests/                      # Bộ unit test kiểm chứng toàn diện
```

---

## 3. Hướng dẫn Cài đặt & Chạy Thực nghiệm

Yêu cầu hệ thống: **Python 3.10+**, Git. Nếu chạy trên GPU NVIDIA / Colab / Kaggle, cài đặt PyTorch tương thích CUDA trước khi cài các gói còn lại. Trên Windows, `tzdata` được tự động cài cùng `requirements.txt`.

```powershell
git clone https://github.com/hieu-kiien/gnn-nlp-propagation.git
cd gnn-nlp-propagation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

### Chạy Pipeline Thực nghiệm Hợp nhất 2D (`hidden_dim = 64`, `min_total_replies = 25`)

```bash
# Bước 1: Sinh teacher/student 2D [s_uv, h_uv] & xuất nlp_features.pt
export VN_BEFORE_EPOCH=1763164800
bash scripts/run_phase2.sh

# Bước 2: Chạy pytest, 8 mô hình x 5 seeds, ablation K={10,20,30,50}, EDA, efficiency & UPFD
bash run_all.sh
```

### Dự báo Luồng Trực tiếp & Mạng Xã hội Vĩ mô (CPU / GPU)

```bash
# Kiểm thử phân tích mạng xã hội có dấu Tầng 2 + GNN liên kết (~5 giây):
python -m scripts.run_macro_signed_network --smoke-test

# Chạy phân tích mạng xã hội có dấu trên dữ liệu thật VN (thuần CPU, không train GNN):
python -m scripts.run_macro_signed_network --data-dir data/processed/vn --dataset-name vn --analysis-only

# Chạy demo dự báo luồng trực tiếp offline trên CPU không cần checkpoint (<3 giây):
python -m scripts.predict_live_thread --demo-offline

# Dự báo luồng Reddit thật (bắt buộc truyền checkpoint):
python -m scripts.predict_live_thread --reddit-url "https://www.reddit.com/r/VietNam/comments/..." --gnn-ckpt checkpoints/best_sbifagat_vn.pt

# Chạy kiểm thử cấu trúc (dry-run) bằng trọng số ngẫu nhiên:
python -m scripts.predict_live_thread --url "https://www.reddit.com/r/VietNam/comments/..." --allow-untrained
```

### Chấm tay & Kiểm định Mù (Human Audit Web UI & Verified Gold)

```bash
# Mở phòng chấm tay Web UI tiếng Việt cục bộ tại http://127.0.0.1:8765/
python -m scripts.human_audit_ui
# Hoặc khóa cứng từng vai trò cho mỗi người chấm:
python -m scripts.human_audit_ui --lock-role annotator_1 --port 8765

# Nếu dùng quy trình 2 người chấm độc lập + hòa giải:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --strict-adjudication --output-json paper/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --strict-adjudication --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json paper/results_human_audit_vn_30_trees.json

# Nếu dùng quy trình tiền chú thích AI + tác giả duyệt vàng đơn lẻ:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --allow-single-gold --output-json paper/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --allow-single-gold --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json paper/results_human_audit_vn_30_trees.json
```

---

## 4. Quy ước Làm việc Nhóm (Git & Jira)

1. **Tạo nhánh theo mã việc Jira:** Nhánh `main` được bảo vệ bằng Ruleset & GitHub Actions CI. Tạo nhánh mới theo định dạng `<type>/<MA-JIRA>-<mo-ta-ngan>`.
2. **Kiểm thử & Review chéo (Test-Gated):** Mọi Pull Request vào `main` phải vượt qua toàn bộ `pytest` và kiểm tra tệp cấm trên CI.
3. **Bảo mật & Dữ liệu Public:** Kho đang ở chế độ **Public**. Không commit `.parquet`, trọng số `.pt`/`.pth`/`.ckpt`/`.bin`/`.safetensors`, `checkpoints/`/`outputs/`/`results/`/`logs/`, `*.zip`, đáp án chấm mù `*_machine_key.csv`, hoặc muối/khóa bí mật (`.env`, `*.secret`, `HF_TOKEN`).
