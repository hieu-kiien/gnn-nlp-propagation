# SBi-FAGAT: Prioritized Technical Roadmap & Execution Plan

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Architecture Upgrade | Steps 1–3 complete: Dual-channel Dir-GNN `SBiFAGAT`, 10D `struct_x`, `SpanPool`, Symmetric NLI + Hostility 2D teachers, segment-isolated CE, KL-calibrated BCE + EMA + best-val seed selection, fair 8-model parity (`hidden_dim=64`), canonical split locks | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`pytest`), PR template, Branch Ruleset, `export_dataset_stats.py`, efficiency LaTeX, unified `paper/main.tex` | `SBIF-42`, `SBIF-47` | **Done** |
| **P2** | Real Data Ingestion, Offline NLP & Human Audit | PHEME + Arctic Shift VN ingestion, offline 2D NLP `.pt` & canonical splits **Done** (`member1_vn` + `member2_en`). **Remaining:** Gate 1 blind human audit (`pilot` 30 pairs, `300` pairs, `50` VN Test trees via `scripts/human_audit_ui.py` -> `compute_kappa.py` & `audit_vn_50_trees.py`) | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Active (Human Audit)** |
| **P3** | 5-Seed Benchmarks, Ablations & Camera-Ready Paper | 8-model 5-seed benchmarks, LOO PHEME, $K$ ablation, mechanisms, UPFD (GossipCop & PolitiFact), EDA, efficiency & 8 vector figures **Done & Assembled** (`paper/`). **Remaining:** integrate Gate 1 human audit tables + final prose/compile `paper/main.tex` PDF | `SBIF-30..46`, `SBIF-50` | **Active (Paper Finalization)** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper Compilation):**
  - GitHub governance, CI/CD, and release tagging.
  - PHEME Figshare ingestion (`member2_en`), unified 2D feature extraction, 5-seed benchmarks, LOO, Persistence baseline, UPFD extension, efficiency benchmarks, and `paper/main.tex` IEEEtran compilation.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`, `member1_vn`), blind 300-pair + 50-tree VN manual audit (`compute_kappa.py --strict-adjudication`), qualitative case study / `fig3_top5_trigger_comments.txt`, and `docs/` QA.
