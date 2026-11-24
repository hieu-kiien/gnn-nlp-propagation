# SBi-FAGAT: Prioritized Technical Roadmap & 7-Day Delivery Plan

> **7-Day Scope Freeze (`ADR-10`):** Core modeling and data collection are **frozen**. **Zero new data crawling** and **zero core pipeline rewrites** before submission; Kaggle V2 GPU metrics and `paper/main.pdf` are integrated, so remaining effort focuses strictly on delivering the runnable Live Demo Web UI MVP and finalizing the 4-chapter Vietnamese report & defense kit.

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Architecture Upgrade | v3 EGM overhaul complete: PII/spam scrub, linear `y_polar`, ReZero centered Persistence-Residual anchor, masked single-camp Camp Dipole, Smooth-L1 `beta=0.05` + validation-only comparator selection, temporal/cluster splits (`edge_source="teacher"`) | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`pytest`), PR template, Branch Ruleset, dynamic cohort `export_dataset_stats.py`, efficiency LaTeX, Two-Tier `paper/main.tex`, `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`, `notebooks/kaggle_macro_signed_gnn_t4.ipynb` | `SBIF-42`, `SBIF-47` | **Done** |
| **P2** | Real Data Ingestion, Offline NLP & VN Gold Audit | PHEME + Arctic Shift VN ingestion, PII-scrubbed Parquets, RumourEval `parent` baseline (`1,675` pairs), VN Verified Gold (`120` pairs Stance/Incivility + `30` trees via `--allow-single-gold` -> `tables_human_audit.tex`) & VN Target-of-Profanity audit | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Done (`a95537a`)** |
| **P3** | Tier 1 VN & PHEME Tier 2A + Tier 2 Macro CPU Analytics | Track 1 VN Canonical/Temporal/Ablation/Sensitivity, Track 2 PHEME Tier 2A (`K=10, N>=15`, `2,664` eligible, Inductive/LOO/Ablation/Mechanisms), UPFD (GossipCop + PolitiFact), and Tier 2 Macro CPU Analytics for BOTH VN (`13,718` users, $Q_{\text{signed}}=0.4280$) & PHEME (`50,062` users, $Q_{\text{signed}}=0.8232$, RWC `0.6218`) | `SBIF-30..46`, `SBIF-50` | **Done (`a95537a`)** |
| **P4** | Kaggle V2 Integration & English Paper Finalization | Integrated Kaggle V2 `kaggle_macro_signed_gnn_t4` (`results_signed_link_{vn,pheme}.json` + `results_rumoureval_sdqc.json --context-mode root`), synced `paper/main.tex` with `a95537a`/V2 metrics & Target-Misattributed Profanity analysis (fixed `main.tex:98` mask `{3,4,7}` and `F=5` LOO folds), ran `assemble_paper_artifacts.py`, compiled `paper/main.pdf` (`15` pages) | — | **Done** |
| **P5** | Live Demo Web UI MVP | Build runnable local Web Demo using existing trained checkpoints (`checkpoints/best_sbifagat_vn.pt`, `dual_ce_vn`) + offline fallback presets: tree-preserving junk/spam filtering, 3-color stance tree (*Support / Oppose / Neutral*), Tier 1 early forecast, and Tier 2 two-camp network viewer | — | **Planned (Days 3–4)** |
| **P6** | Vietnamese Research Report & Defense Kit | Write full 4-chapter Vietnamese Research Report (`Mở đầu`, `Chương 1 Cơ sở lý thuyết`, `Chương 2 Thiết kế & Phương pháp`, `Chương 3 Thực nghiệm & Thảo luận trung thực`, `Chương 4 Hệ thống Demo`, `Kết luận & Hướng phát triển`) 100% synced with verified paper metrics + defense slides & Q&A handbook | — | **Planned (Days 5–7)** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper & Demo Integration):**
  - GitHub governance, CI/CD, and release tagging.
  - PHEME Figshare ingestion (`member2_en`), unified 2D feature extraction, 5-seed benchmarks, LOO, Tier 2 Macro Signed Network, UPFD extension, efficiency benchmarks, `paper/main.tex` IEEEtran compilation, Live Demo MVP, and Vietnamese Research Report integration.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`, `member1_vn`), blind `20`-pair pilot + `120`-pair + `30`-tree VN audit (`--lock-role` + `compute_kappa.py`), qualitative case study / `fig3_top5_trigger_comments.txt`, and Vietnamese report / `docs/` QA.
