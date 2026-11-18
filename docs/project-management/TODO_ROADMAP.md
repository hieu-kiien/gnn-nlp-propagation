# SBi-FAGAT: Prioritized Technical Roadmap & Execution Plan

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Reconciliation | Reconcile PR #1 with v2 + 6 EGM fixes (`F1–F6`), enforce `min_total_replies=25`, CLI compatibility | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`test`), PR template, Branch Ruleset, `export_dataset_stats.py`, efficiency LaTeX, `paper/main.tex` | `SBIF-42`, `SBIF-47` | **In Progress / Done** |
| **P2** | Real Data Ingestion & Offline NLP (`Sprint 2`) | Ingest PHEME + Arctic Shift VN, blind 300-pair audit, extract 2D `v1` `.pt` features & canonical splits | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Active (Needs Compute)** |
| **P3** | 5-Seed Benchmarks, Ablations & Opt-In `v2` (`Sprint 3`) | Train 8 models across `[42, 100, 2024, 777, 999]`, $K$ ablation, Persistence baseline, UPFD, figures & camera-ready PDF | `SBIF-30..46`, `SBIF-50` | **Planned** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper Compilation):**
  - GitHub governance, CI/CD, and release tagging (`v2.0-rc1`).
  - PHEME Figshare ingestion, `v1` feature extraction, 5-seed benchmarks, Persistence baseline, UPFD extension, efficiency benchmarks, and `paper/main.tex` IEEEtran compilation.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`), blind 300-pair + 50-thread VN manual audit (`compute_kappa.py`), qualitative case study / `fig3_top5_trigger_comments.txt`, and `docs/` QA.
