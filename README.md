# SBi-FAGAT — Phân cực Quan điểm & Lan truyền Thông tin Sớm trên Mạng Xã hội

Kho mã nguồn nghiên cứu cho đề tài khoa học:  
**"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"**.

Kiến trúc hai tầng hợp nhất **SBi-FAGAT** (*Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network* + *Macro Signed User-Interaction Social Network*): **Tầng 1 (Early Cascade)** (`hidden_dim = 64`, VN `K = 20, min_total_replies = 25` và PHEME reduced-horizon `K = 10, min_total_replies = 15`, `polar_target_mode = "future_only"`) dự đoán đồng thời quy mô lan truyền cuối kỳ ($y_{\text{size}}$), mức độ phân cực cộng đồng ($y_{\text{polar}}$) và phân loại tin đồn (PHEME) chỉ từ bài đăng gốc và $K$ phản hồi sớm nhất, kèm ước lượng bất định dị phương sai $\beta$-NLL (`predict_uncertainty=True`) cho *selective abstention* và bộ phân tích bài lớn + phân loại chủ đề Zero-Shot (`src/nlp/topic_post_analyzer.py`); **Tầng 2 (Macro Social Network)** chiếu mọi tương tác $(u, v)$ xuyên chủ đề lên đồ thị người dùng có dấu (`13,718` người dùng VN, Gómez $Q_{\text{signed}} = 0.1988$ không khác biệt có ý nghĩa so với null $0.1977, z = +0.28, p = 0.333$, RWC $= -0.0925$, cân bằng yếu Davis `0.9405` với $z = +4.21, p = 0.0099$; `50,062` tổng / `36,606` người dùng tham gia cặp hợp lệ trên PHEME sau lọc `24,354` hàng vector 0, Gómez $Q_{\text{signed}} = 0.4000$ thấp hơn null $0.4716, z = -39.72, p = 1.0$, RWC $= 0.3181$, cân bằng yếu Davis `0.8774` với $z = +9.70, p = 0.0099$) và chạy Temporal Signed Link Prediction công bằng trên 10 mô hình qua 5 hạt giống.

> **Tài sản Dự án & Kết quả Thực nghiệm (`results/` & `archive/docs_v1/`):**
> - **Kết quả Thực nghiệm đã Kiểm chứng (`results/`):** Chứa các tệp JSON, bảng LaTeX/Markdown và biểu đồ vector sạch (`fig5_macro_signed_network.{svg,pdf}`, `fig7_uncertainty_risk_coverage.{svg,pdf}`) từ pipeline chuẩn hóa: 10 mô hình Tầng 2 VN/PHEME 5 seeds, benchmark Tầng 1 các split (`results_main_*.json` chạy kèm `include_tier2_prior=True`, còn `tier1_tabular_*.json` chạy tách `include_tier2_prior=False`), bất định $\beta$-NLL Selective Abstention 5 seeds (`results_uncertainty_abstention.json`), kiểm định người chấm VN (`human_audit_vi`), và đo hiệu năng E2E đầy đủ Dual-Head Cross-Encoder (`results_efficiency_vn.json`: `146.04 ms` trên Kaggle Tesla T4 FP16 và `efficiency_e2e.json`: `2,206.36 ms` trên GTX 1660 SUPER FP32).
> - **Tài liệu v1 Lưu trữ Trên GitHub (`archive/docs_v1/`):** Toàn bộ bản thảo tiếng Anh (`archive/docs_v1/paper/`), báo cáo tiếng Việt 4 chương & bộ slide bảo vệ (`archive/docs_v1/reports/`), cùng tài liệu hướng dẫn v1 (`archive/docs_v1/docs/`) được lưu giữ nguyên trạng trên kho GitHub gốc (tách khỏi gói `Program Source` nộp gọn `< 25 MB`).
> - **Giao diện Demo Web MVP Offline-First & Mega-Thread 10k Scanner (`scripts/demo_web_app.py`, `web_demo/`):** Khởi chạy cục bộ `http://127.0.0.1:8765/` (0 dependency ngoài, 0 CDN), đọc trực tiếp số liệu Tầng 2 từ `results/`, hỗ trợ bài Test VN thật từ `data/processed/vn/` + fallback ngoại tuyến trên CI, **Path Contraction** (`junk_mode="contract"`), sơ đồ cây 3 màu lập trường tách bạch chỉ báo công kích, bản đồ 2 phe Tầng 2, và **Mega-Thread 10k Scanner** bất đồng bộ (`/api/tier1/scan_jobs`, `--scan-strategy stratified|full`, `--max-full-comments 10000`) sử dụng lấy mẫu phân tầng 10 phân vị có neo tổ tiên kèm khoảng tin cậy Bootstrap 95% và cảnh báo ngoại suy quy mô (`ood_scale_warning` khi `> 345` phản hồi). Kiểm thử tải 10.000 bình luận thật trên GPU `GTX 1660 SUPER` (`stratified`, `sample_budget=600`, `DualHeadCrossEncoder` + `best_sbifagat_vn.pt`): hoàn tất trong **`39.05 s`** với **`809.79 MB`** đỉnh VRAM (`9,656` bình luận hợp lệ, `600/9,656` cạnh chấm điểm NLP, `bootstrap_ci_95=[0.1349, 0.1493]`).

> **Mới bắt đầu hoặc muốn chạy trên GPU miễn phí (Google Colab / Kaggle `T4 x 2`)?**  
> - **Hướng dẫn thực thi Cloud trọn gói:** Đọc [notebooks/CLOUD_EXECUTION_GUIDE.md](notebooks/CLOUD_EXECUTION_GUIDE.md).
> - **Chạy benchmark GNN Tầng 1 nhanh trên Colab / Kaggle T4:** Dùng [notebooks/colab_gpu_v3_gnn_benchmarks.ipynb](notebooks/colab_gpu_v3_gnn_benchmarks.ipynb) (Temporal Holdout VN, Ablation/$K$-Curve và PHEME $K=10, N \ge 15$).
> - **Chạy benchmark Tầng 2 Macro Signed Network trên GPU:** Dùng [notebooks/kaggle_macro_signed_gnn_t4.ipynb](notebooks/kaggle_macro_signed_gnn_t4.ipynb).
> - **Chạy full pipeline 2 người trên Kaggle `GPU T4 x 2`:** Tham khảo `archive/docs_v1/docs/huong_dan_chay_song_song_kaggle_t4x2.md` trên GitHub và mở `kaggle_member1_nlp_and_vn_t4x2.ipynb`, `kaggle_member2_gnn_and_en_t4x2.ipynb`.
> - Tham khảo `archive/docs_v1/docs/huong_dan_nghien_cuu_de_hieu.md` trên GitHub và chạy thử minh họa/kiểm chứng trên máy CPU:
> ```bash
> python -m scripts.demo_research_walkthrough
> python -m scripts.predict_live_thread --demo-offline
> python -m scripts.benchmark_efficiency --use-cascade-stats --hidden-dim 64 --edge-dim 2
> python -m scripts.run_macro_signed_network --smoke-test
> python -m scripts.run_m5_local_audits --validate-results
> ```

---

## 1. Quản trị Dự án & Năm Luật Thép

- **Bảng điều phối Scrum (Jira):** [`SBIF` Board — SBi-FAGAT Research](https://sbifagat-research.atlassian.net/jira/software/projects/SBIF/boards/2)
- **Tài liệu quy chuẩn bắt buộc:** Xem chi tiết trong [GEMINI.md](GEMINI.md) (và bản lưu trữ v1 `archive/docs_v1/docs/project-management/PROJECT_CONTEXT.md` trên GitHub). Mọi đóng góp mã nguồn và thực nghiệm phải tuân thủ tuyệt đối **5 Luật Thép**:
  1. **Cấm dữ liệu giả trong báo cáo:** Không dùng dữ liệu sinh ngẫu nhiên (mock/synthetic) để xuất bảng hay biểu đồ kết quả thực nghiệm (tensor giả lập chỉ được dùng trong `tests/`).
  2. **Chống rò rỉ dữ liệu (Zero Data Leakage):** Chia tập Train / Val / Test (`70/15/15`) nhất quán bằng `topic_splits.json` (`min_total_replies = 25`) theo sự kiện/chủ đề gốc (*Inductive Split*, giữ nguyên cho mọi $K \in \{10, 20, 30, 50\}$; LOO dùng `edge_source="teacher"`); đầu vào chỉ giữ bài gốc + tối đa $K = 20$ bình luận sớm nhất theo `timestamp`; nhãn mục tiêu không lọt vào đặc trưng nút/cạnh.
  3. **Chống ảo giác tài liệu tham khảo:** Mọi công trình trích dẫn trong `archive/docs_v1/docs/` và `archive/docs_v1/paper/references.bib` phải được xác minh tác giả, năm, hội nghị/tạp chí và DOI/arXiv thật.
  4. **Tối ưu hạ tầng GPU 16GB VRAM / Kaggle `T4 x 2`:** Tách rời trích xuất NLP ngoại tuyến (nhúng nút 768 chiều từ *Ancestor-Conditioned SpanPool* đóng băng + đặc trưng cấu trúc nhân quả 10 chiều `struct_x` + thống kê cụm 8 chiều `cascade_stats` + cạnh 2D `[s_uv, h_uv]` chưng cất cô lập phân đoạn ra `.pt`) khỏi huấn luyện GNN (`hidden_dim = 64`); bật Mixed Precision (`torch.amp`); chạy lặp lại trên đúng 5 hạt giống `[42, 100, 2024, 777, 999]` với Early Stopping theo `task_loss` và chọn checkpoint tốt nhất theo Validation.
  5. **Kiểm thử trước khi bàn giao (Test-Gated):** Mọi thay đổi mã nguồn phải có unit test tương ứng và vượt qua toàn bộ `pytest` trước khi commit/merge.

---

## 2. Cấu trúc Thư mục Dự án

```text
.
├── .github/                    # GitHub Actions CI & mẫu Pull Request kiểm tra 5 Luật Thép
├── GEMINI.md                   # Quy tắc hệ thống, 5 Luật Thép & ADRs dự án
├── pyproject.toml              # Khai báo package (sbi-fagat) & cấu hình pytest
├── requirements.txt            # Thư viện Python (>=3.10), PyTorch, PyG, Transformers, LightGBM, tzdata
├── run_all.sh                  # Pipeline kiểm thử + thực nghiệm Giai đoạn 3 & UPFD (2D thống nhất)
├── archive/
│   └── docs_v1/                # Tài liệu v1 lưu trữ nguyên trạng (R4): paper/, reports/, docs/
├── data/                       # Dữ liệu local (raw/interim/processed bị ignore trên Git)
├── notebooks/
│   ├── CLOUD_EXECUTION_GUIDE.md             # Hướng dẫn thực thi Colab / Kaggle T4 x 2 trọn gói
│   ├── colab_gpu_v3_gnn_benchmarks.ipynb    # Sổ tay Colab/Kaggle T4 GPU cho Temporal, Ablation & PHEME K=10/N>=15
│   ├── kaggle_macro_signed_gnn_t4.ipynb     # Sổ tay Tầng 2 Macro Signed Network + Temporal Signed Link GNN (T4 GPU)
│   ├── kaggle_member1_nlp_and_vn_t4x2.ipynb # Sổ tay Track VN (`member1_vn`)
│   └── kaggle_member2_gnn_and_en_t4x2.ipynb # Sổ tay Track EN (`member2_en`, PHEME + RumourEval SDQC)
├── results/                    # Kết quả thực nghiệm sạch không nhị phân (JSON/MD/LaTeX & figures SVG/PDF)
├── scripts/
│   ├── demo_web_app.py         # Máy chủ HTTP cục bộ (127.0.0.1, stdlib) cho Giao diện Demo Web MVP 2 Tab
│   ├── run_m5_local_audits.py  # Kiểm chứng định lượng dữ liệu thật (RWC/Gómez/Davis/zero-rows) & validate results/
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
│   ├── benchmark_efficiency.py # Đo #Params, tốc độ suy luận/huấn luyện, VRAM & độ trễ E2E (LaTeX/JSON)
│   ├── demo_research_walkthrough.py # Demo minh họa cơ chế GCN vs SBi-FAGAT trên CPU (<5s)
│   ├── run_experiments.py      # Huấn luyện công bằng mô hình nền + SOTA Tầng 1 (`hidden_dim=64`) qua 5 seeds
│   ├── run_ablation.py         # Thực nghiệm cắt bỏ thành phần (+ BiEdgeGATv2) & quét K cửa sổ tương lai cố định
│   ├── eval_large_post_topic.py # Đánh giá Zero-Shot Topic (9 sự kiện PHEME + VN), trajectory đa mốc & early-warning K=20
│   ├── eval_uncertainty_abstention.py # Đánh giá bất định beta-NLL & Selective Abstention (Risk-Coverage) 5 seeds
│   ├── eval_hostility_router_audit.py # Kiểm chứng HostilityGatedTargetRouter trên 140 cặp kiểm định VN (zero-PII)
│   └── run_upfd_extension.py   # Thực nghiệm chuyển giao cấu trúc phi văn bản trên UPFD (Accuracy, Macro-F1, ROC-AUC)
├── src/
│   ├── data/                   # schema.py, pheme_loader.py, arctic_shift_vn.py, pyg_dataset.py, user_network.py
│   ├── nlp/                    # discourse_labels.py, dual_edge_scorer.py, extract_features.py, topic_post_analyzer.py
│   ├── models/                 # baselines.py, sbi_fagat.py, sota_cascades.py, signed_gnns.py
│   ├── training/               # metrics.py (KL-calibrated BCE + Smooth-L1 & Huber), trainer.py, auto_loop.py, reporting.py
│   └── utils/                  # seed.py, text.py, visualize.py (lazy UMAP import)
├── web_demo/                   # Giao diện Demo Web MVP thuần HTML/CSS/SVG/JS (0 CDN, Path Contraction)
└── tests/                      # Bộ kiểm thử đơn vị, hồi quy & E2E 4 tầng (`tests/e2e/`)
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

# Quét Mega-Thread tới 10.000 bình luận với lấy mẫu phân tầng + CI 95% (giữ GNN K=20):
python -m scripts.predict_live_thread --reddit-url "https://www.reddit.com/r/VietNam/comments/..." --gnn-ckpt checkpoints/best_sbifagat_vn.pt --analysis-mode full --scan-strategy stratified --sample-budget 600 --max-full-comments 10000

# Chạy kiểm thử cấu trúc (dry-run) bằng trọng số ngẫu nhiên:
python -m scripts.predict_live_thread --url "https://www.reddit.com/r/VietNam/comments/..." --allow-untrained
```

### Chấm tay, Kiểm định Mù & Giao diện Demo Web MVP (127.0.0.1)

```bash
# Khởi chạy Giao diện Demo Web MVP 2 Tab tại http://127.0.0.1:8765/ (offline-first, 0 CDN)
python scripts/demo_web_app.py --port 8765

# Mở phòng chấm tay Web UI tiếng Việt cục bộ tại http://127.0.0.1:8765/
python -m scripts.human_audit_ui
# Hoặc khóa cứng từng vai trò cho mỗi người chấm:
python -m scripts.human_audit_ui --lock-role annotator_1 --port 8765

# Nếu dùng quy trình 2 người chấm độc lập + hòa giải:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --strict-adjudication --output-json results/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --strict-adjudication --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json results/results_human_audit_vn_30_trees.json

# Nếu dùng quy trình tiền chú thích AI + tác giả duyệt vàng đơn lẻ:
python -m scripts.compute_kappa --input data/interim/human_audit_vi_120.csv --allow-single-gold --output-json results/results_human_audit_vi.json
python -m scripts.audit_vn_50_trees --evaluate --allow-single-gold --audit-csv data/interim/human_audit_vn_30_trees.csv --output-json results/results_human_audit_vn_30_trees.json
```

### Đánh giá Mở rộng: Chủ đề Zero-Shot & Động lực Bài Lớn (`results_large_post_topic.json`)

```bash
python -m scripts.eval_large_post_topic --batch-size 128 --max-length 160
```

- **Phân loại Chủ đề Zero-Shot PHEME ($N=2{,}664$, 9 sự kiện thật):** `mDeBERTa-v3-XNLI` GPU FP16 đạt `58.00%` Acc (`Macro-F1 = 0.4424`, `Weighted-F1 = 0.6655`, `Top-2 = 71.55%`) trên toàn cohort và `70.36%` Acc (`Top-2 = 79.64%`) trên Test-334 — vượt xa `Lexical N-gram` (`43.88%`) và `RoBERTa Cosine` (`36.90%`).
- **Quỹ đạo Bài Lớn & Cảnh báo Sớm $K=20$ (VN Test, $N=225$, khóa ngưỡng F1 trên Val):** Các chuỗi lớn ($\ge 50$ bình luận) leo thang phân cực mạnh hơn (`37.96%` vs `20.00%` trên PHEME; `20.07%` trên VN `large_ge_50` vs `14.22%` ở nhóm trung bình `Medium (25–99)` vốn chồng lấp tập `>=50` do `n=1,336` gồm cả chuỗi `50–99`). Từ tiền tố $K=20$, **`SBiFAGAT`** dẫn đầu phát hiện sớm **High Future Polarization ($\ge Q_{75}^{\text{train}}$)** với `ROC-AUC = 0.6600`, `PR-AUC = 0.3928`, `Val-F1 = 0.4074` (vượt `Ridge` `0.6234 / 0.3607 / 0.4038`), đồng thời dẫn đầu `PR-AUC` & `Val-F1` cho cảnh báo `Large >= 50` (`0.6769 / 0.6477`) và `Viral >= 100` (`0.3911 / 0.4490`) (xem `results/figures/vn/fig6_large_post_topic_trajectory.{pdf,svg}`).
- **Bất định $\beta$-NLL (`polar_logvar` $\in [-11.0, -2.0]$) & Kiểm chứng Router Thái độ Thù địch:** Sau khi mở khóa cận dưới `polar_logvar`, `SBiFAGAT` đạt xếp hạng bất định dương trên VN (`Size` $\rho = 0.328$, giảm `30.8%` MSLE tại coverage `70%`; `Polar` $\rho = 0.201$, giảm `13.5%` RMSE tại coverage `70%`, `PICP@90% = 0.893 / 0.877`; `results/results_uncertainty_abstention.json`). Kiểm chứng `HostilityGatedTargetRouter` trên `140` cặp gán nhãn VN (`results/results_hostility_router_audit.json`) cung cấp bằng chứng định lượng cho giới hạn của bộ định tuyến zero-shot/regex không tinh chỉnh trên tiếng lóng mạng xã hội.

---

## 4. Quy ước Làm việc Nhóm (Git & Jira)

1. **Tạo nhánh theo mã việc Jira:** Nhánh `main` được bảo vệ bằng Ruleset & GitHub Actions CI. Tạo nhánh mới theo định dạng `<type>/<MA-JIRA>-<mo-ta-ngan>`.
2. **Kiểm thử & Review chéo (Test-Gated):** Mọi Pull Request vào `main` phải vượt qua toàn bộ `pytest` và kiểm tra tệp cấm trên CI.
3. **Bảo mật & Dữ liệu Public:** Kho đang ở chế độ **Public**. Chỉ theo dõi các kết quả thực nghiệm không nhị phân trong `results/` (`*.json`, `*.md`, `*.tex`, `results/figures/*/*.{svg,pdf}`); tuyệt đối không commit `.parquet`, trọng số `.pt`/`.pth`/`.ckpt`/`.bin`/`.safetensors`, `checkpoints/`/`outputs/`/`logs/`, `*.zip`, `runner_summary_*.json`, `*.tmp.json`, đáp án chấm mù `*_machine_key.csv`, hoặc muối/khóa bí mật (`.env`, `*.secret`, `HF_TOKEN`).
