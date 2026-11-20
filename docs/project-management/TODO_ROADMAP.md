# SBi-FAGAT: Prioritized Technical Roadmap & Execution Plan

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Architecture Upgrade | Steps 1–3 complete: Dual-channel Dir-GNN `SBiFAGAT`, 10D `struct_x`, `SpanPool`, Symmetric NLI + Hostility 2D teachers, segment-isolated CE, KL-calibrated BCE + EMA + best-val seed selection, fair 8-model parity (`hidden_dim=64`), canonical split locks | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`pytest`), PR template, Branch Ruleset, `export_dataset_stats.py`, efficiency LaTeX, unified `paper/main.tex` | `SBIF-42`, `SBIF-47` | **Done** |
| **P2** | Real Data Ingestion & Offline NLP (Step 4) | Ingest PHEME + Arctic Shift VN, blind 300-pair + 50-tree VN audit (`compute_kappa.py --strict-adjudication`), extract 2D `.pt` features & canonical splits on Kaggle `T4 x 2` | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Active (Needs GPU)** |
| **P3** | 5-Seed Benchmarks, Ablations & Camera-Ready Paper (Step 4) | Train 8 models across `[42, 100, 2024, 777, 999]` (`hidden_dim=64`), LOO PHEME, $K$ ablation, `Early-K Persistence`, UPFD, figures & compile `paper/main.tex` | `SBIF-30..46`, `SBIF-50` | **Planned (Step 4)** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper Compilation):**
  - GitHub governance, CI/CD, and release tagging.
  - PHEME Figshare ingestion (`member2_en`), unified 2D feature extraction, 5-seed benchmarks, LOO, Persistence baseline, UPFD extension, efficiency benchmarks, and `paper/main.tex` IEEEtran compilation.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`, `member1_vn`), blind 300-pair + 50-tree VN manual audit (`compute_kappa.py --strict-adjudication`), qualitative case study / `fig3_top5_trigger_comments.txt`, and `docs/` QA.
