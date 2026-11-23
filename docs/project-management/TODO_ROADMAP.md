# SBi-FAGAT: Prioritized Technical Roadmap & Execution Plan

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Architecture Upgrade | v3 EGM overhaul complete: PII/spam scrub, linear `y_polar`, ReZero centered Persistence-Residual anchor, masked single-camp Camp Dipole, Smooth-L1 `beta=0.05` + validation-only comparator selection, temporal/cluster splits (`edge_source="teacher"`) | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`pytest`), PR template, Branch Ruleset, `export_dataset_stats.py`, efficiency LaTeX, unified `paper/main.tex`, `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb` | `SBIF-42`, `SBIF-47` | **Done** |
| **P2** | Real Data Ingestion, Offline NLP & Human Audit | PHEME + Arctic Shift VN ingestion, PII-scrubbed Parquets & Test-only disjoint blind VN audit sheets **Done**. **Remaining:** Gate 1 blind human audit (`20`-pair pilot calibration, `120`-pair binary `clean`/`uncivil` + 3-class stance, `30` VN Test trees via `scripts/human_audit_ui.py --lock-role` -> `compute_kappa.py` & `audit_vn_50_trees.py`) | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Active (Human Audit)** |
| **P3** | 5-Seed Benchmarks, Ablations & Camera-Ready Paper | VN Canonical v3 (`task-1008`), VN Sensitivity, EDA & Efficiency **Done**. **Remaining:** Execute `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb` on Google Colab / Kaggle T4 GPU (VN temporal holdout + refreshed v3 ablation + Track 2 PHEME Tier 2A $K=10, N \ge 15$), merge Gate 1 `tables_human_audit.tex` & compile PDF | `SBIF-30..46`, `SBIF-50` | **Active (Cloud GPU + Paper)** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper Compilation):**
  - GitHub governance, CI/CD, and release tagging.
  - PHEME Figshare ingestion (`member2_en`), unified 2D feature extraction, 5-seed benchmarks, LOO, Persistence baseline, UPFD extension, efficiency benchmarks, and `paper/main.tex` IEEEtran compilation.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`, `member1_vn`), blind `20`-pair pilot + `120`-pair + `30`-tree VN manual audit (`--lock-role` + `compute_kappa.py --strict-adjudication`), qualitative case study / `fig3_top5_trigger_comments.txt`, and `docs/` QA.
