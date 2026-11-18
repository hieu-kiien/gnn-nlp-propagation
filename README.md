# SBi-FAGAT — Phân cực Quan điểm & Lan truyền Thông tin Sớm trên Mạng Xã hội

Kho mã nguồn nghiên cứu cho đề tài khoa học:  
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"**.

Kiến trúc đề xuất **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network*) thực hiện dự đoán sớm đa nhiệm gồm quy mô lan truyền cuối kỳ ($y_{\text{size}}$), mức độ phân cực cộng đồng ($y_{\text{polar}}$) và phân loại tin đồn (trên bộ PHEME) chỉ từ bài đăng gốc và cửa sổ sớm $K = 20$ phản hồi đầu tiên theo thời gian.

> **Mới bắt đầu hoặc chạy song song 2 người trên Kaggle `GPU T4 x 2`?**  
> - Đọc [Hướng dẫn Chạy Song Song trên Kaggle T4 x 2](docs/huong_dan_chay_song_song_kaggle_t4x2.md) và mở 2 sổ tay sẵn dùng trong `notebooks/` (`kaggle_member1_nlp_and_vn_t4x2.ipynb`, `kaggle_member2_gnn_and_en_t4x2.ipynb`).
> - Đọc [Hướng dẫn Nghiên cứu Dễ hiểu từ Con số 0](docs/huong_dan_nghien_cuu_de_hieu.md) và chạy thử minh họa trên máy CPU (<5 giây):
> ```bash
> python -m scripts.demo_research_walkthrough
> python -m scripts.predict_live_thread --demo-offline
> python -m scripts.benchmark_efficiency --v2 --edge-dim 4
> ```

---

## 1. Quản trị Dự án & Năm Luật Thép

- **Bảng điều phối Scrum (Jira):** [`SBIF` Board — SBi-FAGAT Research](https://sbifagat-research.atlassian.net/jira/software/projects/SBIF/boards/2)
- **Tài liệu quy chuẩn bắt buộc:** Xem chi tiết trong [GEMINI.md](GEMINI.md) và hồ sơ [docs/project-management/](docs/project-management/PROJECT_CONTEXT.md). Mọi đóng góp mã nguồn và thực nghiệm phải tuân thủ tuyệt đối **5 Luật Thép**:
  1. **Cấm dữ liệu giả trong báo cáo:** Không dùng dữ liệu sinh ngẫu nhiên (mock/synthetic) để xuất bảng hay biểu đồ kết quả thực nghiệm (tensor giả lập chỉ được dùng trong `tests/`).
  2. **Chống rò rỉ dữ liệu (Zero Data Leakage):** Chia tập Train / Val / Test (`70/15/15`) nhất quán bằng `topic_splits.json` (`min_total_replies = 25`) theo sự kiện/chủ đề gốc (*Inductive Split*, hoặc `edge_source="teacher"` cho PHEME LOO); đầu vào chỉ giữ bài gốc + tối đa $K = 20$ bình luận sớm nhất theo `timestamp`; nhãn mục tiêu không lọt vào đặc trưng nút/cạnh.
  3. **Chống ảo giác tài liệu tham khảo:** Mọi công trình trích dẫn trong `docs/` và `paper/references.bib` phải được xác minh tác giả, năm, hội nghị/tạp chí và DOI/arXiv thật.
  4. **Tối ưu hạ tầng GPU 16GB VRAM / Kaggle `T4 x 2`:** Tách rời bước trích xuất đặc trưng NLP (lưu `.pt` 768 chiều + 2D `[s_uv, h_uv]` ở Profile `v1` hoặc 4D `[s_uv, h_uv, r_uv, c_uv]` ở Profile `v2`) khỏi bước huấn luyện GNN; bật Mixed Precision (`torch.amp`); chạy lặp lại trên đúng 5 hạt giống `[42, 100, 2024, 777, 999]` với Early Stopping theo `task_loss`.
  5. **Kiểm thử trước khi bàn giao (Test-Gated):** Mọi thay đổi mã nguồn phải có unit test tương ứng và vượt qua toàn bộ `pytest` trước khi commit/merge.

---

## 2. Cấu trúc Thư mục Dự án

```text
.
├── .github/                    # GitHub Actions CI & mẫu Pull Request kiểm tra 5 Luật Thép
├── GEMINI.md                   # Quy tắc hệ thống, 5 Luật Thép & ADRs dự án
├── pyproject.toml              # Khai báo package (sbi-fagat) & cấu hình pytest
├── requirements.txt            # Thư viện Python (>=3.10), PyTorch, PyG, Transformers, tzdata (Windows)
├── requirements-kaggle-t4x2.txt # Bổ sung PEFT/bitsandbytes/TRL cho Qwen QLoRA trên T4x2
├── run_all.sh                  # Pipeline kiểm thử + thực nghiệm Giai đoạn 3 & UPFD (v1|v2)
├── configs/                    # Chứa tệp cấu hình thực nghiệm (giữ chỗ .gitkeep)
├── data/                       # Dữ liệu local (raw/interim/processed bị ignore trên Git)
├── docs/
│   ├── project-management/     # Hồ sơ quản trị dự án, kiểm toán kỹ thuật & lộ trình
│   ├── huong_dan_chay_song_song_kaggle_t4x2.md # Hướng dẫn chia 2 người chạy Kaggle T4x2
│   ├── huong_dan_nghien_cuu_de_hieu.md  # Giải thích đề tài, ký hiệu & code từ con số 0
│   ├── literature_review.md    # Tổng quan công trình nền tảng đã xác minh
│   └── math_spec.md            # Đặc tả toán học SBi-FAGAT & Hợp đồng dữ liệu
├── notebooks/
│   ├── kaggle_member1_nlp_and_vn_t4x2.ipynb # Sổ tay Track VN / NLP
│   └── kaggle_member2_gnn_and_en_t4x2.ipynb # Sổ tay Track EN / GNN
├── paper/
│   ├── main.tex                # Khung bản thảo bài báo chuẩn IEEE (IEEEtran)
│   ├── references.bib          # Thư viện trích dẫn chuẩn BibTeX (23 công trình xác minh)
│   └── figures/                # Thư mục xuất biểu đồ vector phục vụ bài báo
├── scripts/
│   ├── kaggle_parallel_runner.py # Điều phối chạy song song 4 role + auto_loop Kaggle T4x2
│   ├── export_dataset_stats.py # Xuất Bảng 1 EDA LaTeX & Early-K Persistence Baseline
│   ├── predict_live_thread.py  # Dự báo trực tiếp từ URL/ID bài viết thật trên mạng
│   ├── run_phase2.sh           # Pipeline Frozen-Teacher / Qwen QLoRA & xuất NLP .pt (v1|v2)
│   ├── compute_kappa.py        # Tính Cohen's Kappa (3-class & 5-class behavior), Machine-vs-Gold & SDQC
│   ├── benchmark_efficiency.py # Đo #Params, tốc độ suy luận/huấn luyện & RAM/VRAM 8 mô hình (LaTeX/JSON)
│   ├── demo_research_walkthrough.py # Demo minh họa cơ chế GCN vs SBi-FAGAT trên CPU (<5s)
│   ├── run_experiments.py      # Huấn luyện 8 mô hình công bằng qua 5 seeds & kiểm định thống kê
│   ├── run_ablation.py         # Thực nghiệm cắt bỏ thành phần & quét cửa sổ K đồng bộ
│   └── run_upfd_extension.py   # Thực nghiệm mở rộng trên UPFD (GossipCop / PolitiFact)
├── src/
│   ├── data/                   # schema.py, pheme_loader.py, arctic_shift_vn.py, pyg_dataset.py
│   ├── nlp/                    # qwen_teacher.py, dual_edge_scorer.py, extract_features.py
│   ├── models/                 # baselines.py, sbi_fagat.py
│   ├── training/               # metrics.py, trainer.py, auto_loop.py, reporting.py
│   └── utils/                  # seed.py, text.py, visualize.py
└── tests/                      # Bộ unit test (schema, tree, no_leakage, overfit, sota_v2, live_predictor_audit...)
```

---

## 3. Hướng dẫn Cài đặt Nhanh

Yêu cầu hệ thống: **Python 3.10+**, Git. Nếu chạy trên máy có GPU NVIDIA hoặc Colab/Kaggle, cài đặt PyTorch tương thích phiên bản CUDA trước khi cài các gói còn lại. Trên Windows, `tzdata` được tự động cài đặt cùng `requirements.txt` để hỗ trợ phân tích định dạng UTC của PyArrow.

### Windows (PowerShell)

```powershell
git clone https://github.com/hieu-kiien/gnn-nlp-propagation.git
cd gnn-nlp-propagation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

### Linux / macOS

```bash
git clone https://github.com/hieu-kiien/gnn-nlp-propagation.git
cd gnn-nlp-propagation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

### Google Colab / Kaggle (GPU 16GB: T4 / P100)

```bash
!git clone https://github.com/hieu-kiien/gnn-nlp-propagation.git
%cd gnn-nlp-propagation
!pip install -q -r requirements.txt
!pytest
```

### Chạy Pipeline Thực nghiệm (Profile `v1` MVP vs. Profile `v2` Mở rộng)

```bash
# 1. Profile v1 (Mặc định MVP - Teacher legacy_nli 2D [s_uv, h_uv], min_total_replies=25):
export VN_BEFORE_EPOCH=1763164800
bash scripts/run_phase2.sh
bash run_all.sh

# 2. Profile v2 (Mở rộng Kaggle T4 x 2 - Teacher qwen_qlora 4D [s,h,r,c], 8D burst stats, beta-NLL):
export PIPELINE_VERSION=v2
export VN_BEFORE_EPOCH=1763164800
bash scripts/run_phase2.sh
bash run_all.sh
```

### Dự báo Luồng Trực tiếp (Live Thread Predictor)

`scripts/predict_live_thread.py` chặn chạy suy luận trên bài đăng thật nếu không có checkpoint đã huấn luyện:

```bash
# Chạy demo offline CPU không cần checkpoint (<3 giây):
python -m scripts.predict_live_thread --demo-offline

# Dự báo luồng Reddit thật (bắt buộc truyền checkpoint):
python -m scripts.predict_live_thread --reddit-url "https://www.reddit.com/r/VietNam/comments/..." --gnn-ckpt checkpoints/best_sbifagat_vn.pt

# Chạy kiểm thử cấu trúc (dry-run) bằng trọng số ngẫu nhiên:
python -m scripts.predict_live_thread --url "https://www.reddit.com/r/VietNam/comments/..." --allow-untrained
```

---

## 4. Quy ước Làm việc Nhóm (Git & Jira)

1. **Tạo nhánh theo mã việc Jira:** Nhánh `main` được bảo vệ bằng Ruleset & GitHub Actions CI. Luôn kéo `main` mới nhất trước khi làm việc và tạo nhánh với định dạng `<type>/<MA-JIRA>-<mo-ta-ngan>`:
   - `feat/SBIF-14-crawl-data`
   - `fix/SBIF-20-bootstrap-pvalues`
   - `docs/SBIF-23-latex-draft`
2. **Kiểm thử & Review chéo (Test-Gated):** Mọi Pull Request vào `main` phải vượt qua GitHub Actions CI (`pytest` 100% + kiểm tra tệp cấm) và được thành viên còn lại review chấp thuận.
3. **Bảo mật & Dữ liệu Public:** Kho này đang ở chế độ **Public**. Tuyệt đối **không** commit tệp dữ liệu `.parquet`, trọng số `.pt`/`.pth`/`.ckpt`/`.bin`/`.safetensors`, thư mục `checkpoints/`/`outputs/`/`results/`/`logs/`, tệp nén `*.zip`/`*.tar*`, đáp án chấm mù `*_machine_key.csv`, hay muối/khóa bí mật (`.env`, `*.secret`, `HF_TOKEN`) — đã được cấu hình loại trừ trong `.gitignore` và kiểm tra tự động trên CI.
