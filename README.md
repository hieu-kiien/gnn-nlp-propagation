# SBi-FAGAT — Phân cực Quan điểm & Lan truyền Thông tin Sớm trên Mạng Xã hội

Kho mã nguồn nghiên cứu cho đề tài khoa học:  
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"**.

Kiến trúc đề xuất **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network*) thực hiện dự đoán sớm đa nhiệm gồm quy mô lan truyền cuối kỳ ($y_{\text{size}}$), mức độ phân cực cộng đồng ($y_{\text{polar}}$) và phân loại tin đồn (trên bộ PHEME) chỉ từ bài đăng gốc và cửa sổ sớm $K = 20$ phản hồi đầu tiên theo thời gian.

> **Mới bắt đầu hoặc chạy song song 2 người trên Kaggle `GPU T4 x 2` (32 GB VRAM)?**  
> - Đọc [Hướng dẫn Chạy Song Song trên Kaggle T4 x 2 (v2)](docs/huong_dan_chay_song_song_kaggle_t4x2.md) và mở 2 sổ tay sẵn dùng trong `notebooks/` (`kaggle_member1_nlp_and_vn_t4x2.ipynb`, `kaggle_member2_gnn_and_en_t4x2.ipynb`).
> - Đọc [Hướng dẫn Nghiên cứu Dễ hiểu từ Con số 0](docs/huong_dan_nghien_cuu_de_hieu.md) và chạy thử minh họa trên máy CPU (<5 giây):
> ```bash
> python -m scripts.demo_research_walkthrough
> python -m scripts.predict_live_thread --demo-offline
> python -m scripts.benchmark_efficiency
> ```

---

## 1. Quản trị Dự án & Năm Luật Thép

- **Bảng điều phối Scrum (Jira):** [`SBIF` Board — SBi-FAGAT Research](https://sbifagat-research.atlassian.net/jira/software/projects/SBIF/boards/2)
- **Tài liệu quy chuẩn bắt buộc:** Xem chi tiết trong [GEMINI.md](GEMINI.md). Mọi đóng góp mã nguồn và thực nghiệm phải tuân thủ tuyệt đối **5 Luật Thép**:
  1. **Cấm dữ liệu giả trong báo cáo:** Không dùng dữ liệu sinh ngẫu nhiên (mock/synthetic) để xuất bảng hay biểu đồ kết quả thực nghiệm (tensor giả lập chỉ được dùng trong `tests/`).
  2. **Chống rò rỉ dữ liệu (Zero Data Leakage):** Chia tập Train / Val / Test (`70/15/15`) độc lập theo từng sự kiện/chủ đề gốc (*Inductive Split*); đầu vào chỉ giữ bài gốc + tối đa $K = 20$ bình luận sớm nhất theo `timestamp`; nhãn mục tiêu không lọt vào đặc trưng nút/cạnh.
  3. **Chống ảo giác tài liệu tham khảo:** Mọi công trình trích dẫn trong `docs/` và `paper/references.bib` phải được xác minh tác giả, năm, hội nghị/tạp chí và DOI/arXiv thật.
  4. **Tối ưu hạ tầng GPU 16GB VRAM (Colab T4 / Kaggle P100):** Tách rời bước trích xuất đặc trưng NLP (lưu `.pt` ra đĩa) khỏi bước huấn luyện GNN; bật Mixed Precision (`torch.cuda.amp`) khi tinh chỉnh Transformer; chạy thực nghiệm lặp lại trên đúng 5 hạt giống `[42, 100, 2024, 777, 999]`.
  5. **Kiểm thử trước khi bàn giao (Test-Gated):** Mọi thay đổi mã nguồn phải có unit test tương ứng và vượt qua toàn bộ `pytest` trước khi commit/merge.

---

## 2. Cấu trúc Thư mục Dự án

```text
.
├── GEMINI.md                   # Quy tắc hệ thống & 5 Luật Thép dự án
├── pyproject.toml              # Khai báo package (sbi-fagat) & cấu hình pytest
├── requirements.txt            # Thư viện Python (>=3.10), PyTorch, PyG, Transformers
├── requirements-kaggle-t4x2.txt # Bổ sung PEFT/bitsandbytes/TRL cho Qwen3.8-27B QLoRA T4x2
├── run_all.sh                  # Pipeline kiểm thử + thực nghiệm Giai đoạn 3 & UPFD
├── configs/                    # Chứa tệp cấu hình thực nghiệm (giữ chỗ .gitkeep)
├── data/                       # Dữ liệu local (raw/interim/processed bị ignore trên Git)
├── docs/
│   ├── huong_dan_chay_song_song_kaggle_t4x2.md # Hướng dẫn chia 2 người chạy Kaggle T4x2
│   ├── huong_dan_nghien_cuu_de_hieu.md  # Giải thích đề tài, ký hiệu & code từ con số 0
│   ├── literature_review.md    # Tổng quan công trình nền tảng đã xác minh
│   └── math_spec.md            # Đặc tả toán học SBi-FAGAT & Hợp đồng dữ liệu
├── notebooks/
│   ├── kaggle_member1_nlp_and_vn_t4x2.ipynb # Sổ tay Thành viên 1 (Track VN / NLP)
│   └── kaggle_member2_gnn_and_en_t4x2.ipynb # Sổ tay Thành viên 2 (Track EN / GNN)
├── paper/
│   ├── references.bib          # Thư viện trích dẫn chuẩn BibTeX
│   └── figures/                # Thư mục xuất biểu đồ vector phục vụ bài báo
├── scripts/
│   ├── kaggle_parallel_runner.py # Điều phối chạy song song 4 role + auto_loop Kaggle T4x2
│   ├── predict_live_thread.py  # Dự báo trực tiếp từ URL/ID bài viết thật trên mạng
│   ├── run_phase2.sh           # Pipeline Frozen-Teacher Zero-Leakage & xuất NLP .pt
│   ├── compute_kappa.py        # Tính Cohen's Kappa, Machine-vs-Gold & RumourEval SDQC
│   ├── benchmark_efficiency.py # Đo #Params, tốc độ suy luận/huấn luyện & RAM/VRAM 8 mô hình
│   ├── demo_research_walkthrough.py # Demo minh họa cơ chế GCN vs SBi-FAGAT trên CPU (<5s)
│   ├── run_experiments.py      # Huấn luyện 8 mô hình qua 5 seeds & kiểm định thống kê
│   ├── run_ablation.py         # Thực nghiệm cắt bỏ thành phần & quét cửa sổ K đồng bộ
│   └── run_upfd_extension.py   # Thực nghiệm mở rộng trên UPFD GossipCop
├── src/
│   ├── data/                   # schema.py, pheme_loader.py, arctic_shift_vn.py, pyg_dataset.py
│   ├── nlp/                    # qwen_teacher.py, dual_edge_scorer.py, extract_features.py
│   ├── models/                 # baselines.py, sbi_fagat.py
│   ├── training/               # metrics.py, trainer.py, auto_loop.py, reporting.py
│   └── utils/                  # seed.py, visualize.py
└── tests/                      # Bộ unit test (schema, tree, no_leakage, overfit, viz...)
```

---

## 3. Hướng dẫn Cài đặt Nhanh

Yêu cầu hệ thống: **Python 3.10+**, Git. Nếu chạy trên máy có GPU NVIDIA hoặc Colab/Kaggle, cài đặt PyTorch tương thích phiên bản CUDA trước khi cài các gói còn lại.

### Windows (PowerShell)

```powershell
git clone https://github.com/hieu-kiien/gnn-nlp-propagation.git
cd gnn-nlp-propagation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
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

Khi chạy toàn bộ pipeline xử lý dữ liệu/NLP và thực nghiệm GNN (trên môi trường bash hỗ trợ GPU):

```bash
bash scripts/run_phase2.sh
bash run_all.sh
```

---

## 4. Quy ước Làm việc Nhóm (Git & Jira)

1. **Tạo nhánh theo mã việc Jira:** Luôn kéo `main` mới nhất trước khi làm việc và tạo nhánh với định dạng `<type>/<MA-JIRA>-<mo-ta-ngan>`:
   - `feat/SBIF-14-crawl-data`
   - `fix/SBIF-20-bootstrap-pvalues`
   - `docs/SBIF-23-latex-draft`
2. **Kiểm thử trước khi đẩy code (Test-Gated):** Bắt buộc chạy `pytest` đạt **100% pass** và kiểm tra `git status` trước khi `git commit` / mở Pull Request vào `main`.
3. **Bảo mật & Dữ liệu Public:** Kho này đang ở chế độ **Public**. Tuyệt đối **không** commit tệp dữ liệu `.parquet`, trọng số `.pt`/`.pth`/`.ckpt`/`.bin`/`.safetensors`, thư mục `checkpoints/`/`outputs/`/`results/`/`logs/`, hay khóa/token cá nhân (`.env`, `HF_TOKEN`, Atlassian API token) lên GitHub — đã được cấu hình loại trừ trong `.gitignore`.
