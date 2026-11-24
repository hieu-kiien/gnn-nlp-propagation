# SBi-FAGAT: Prioritized Technical Roadmap & 7-Day Delivery Plan

> **7-Day Scope Freeze (`ADR-10`):** Core modeling and data collection are **frozen**. **Zero new data crawling** and **zero core pipeline rewrites** before submission; Kaggle V2 GPU metrics and `paper/main.pdf` are integrated, so remaining effort focuses strictly on delivering the runnable Live Demo Web UI MVP and finalizing the 4-chapter Vietnamese report & defense kit.

| Priority | Category | Scope | Jira Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Scientific Integrity & Architecture Upgrade | v3 EGM overhaul complete: PII/spam scrub, linear `y_polar`, ReZero centered Persistence-Residual anchor, masked single-camp Camp Dipole, Smooth-L1 `beta=0.05` + validation-only comparator selection, temporal/cluster splits (`edge_source="teacher"`) | `SBIF-27`, `SBIF-28`, `SBIF-47` | **Done** |
| **P1** | Repo Governance & Publication Tooling | GitHub CI (`pytest`), PR template, Branch Ruleset, dynamic cohort `export_dataset_stats.py`, efficiency LaTeX, Two-Tier `paper/main.tex`, `notebooks/colab_gpu_v3_gnn_benchmarks.ipynb`, `notebooks/kaggle_macro_signed_gnn_t4.ipynb` | `SBIF-42`, `SBIF-47` | **Done** |
| **P2** | Real Data Ingestion, Offline NLP & VN Gold Audit | PHEME + Arctic Shift VN ingestion, PII-scrubbed Parquets, RumourEval `parent` baseline (`1,675` pairs), VN Verified Gold (`120` pairs Stance/Incivility + `30` trees via `--allow-single-gold` -> `tables_human_audit.tex`) & VN Target-of-Profanity audit | `SBIF-14..26`, `SBIF-48`, `SBIF-49` | **Done (`a95537a`)** |
| **P3** | Tier 1 VN & PHEME Reduced-Horizon + Tier 2 Macro CPU Analytics | Track 1 VN Canonical/Temporal/Ablation/Sensitivity, Track 2 PHEME Reduced-Horizon (`K=10, N>=15`, `2,664` eligible, Inductive/LOO/Ablation/Mechanisms), UPFD (GossipCop + PolitiFact), and Tier 2 Macro CPU Analytics for BOTH VN (`13,718` users, $Q_{\text{signed}}=0.4280$) & PHEME (`50,062` total / `36,606` active users, $Q_{\text{signed}}=0.8232$, RWC `0.6218`) | `SBIF-30..46`, `SBIF-50` | **Done (`a95537a`)** |
| **P4** | Kaggle V2 Integration, Gate 1 Audit & English Report Finalization | Integrated Kaggle V2 `kaggle_macro_signed_gnn_t4` (`results_signed_link_{vn,pheme}.json` + `results_rumoureval_sdqc.json --context-mode root`), resolved all `B1–B9` + Gate 1 Goldfish math/statistical/self-containment items in `paper/main.tex` (with PHEME `fig2_early_k_curve` updated to `K=10`), recompiled `paper/main.pdf` (`20` pages, `0` Blockers), hardened `predict_live_thread.py`, `arctic_shift_vn.py` (`junk_mode="contract"`), audit overwrite guards, and CI gates | — | **Done (`a234339`, `6f2f4fe`)** |
| **P5** | Live Demo Web UI MVP | Built runnable stdlib local 2-Tab Demo (`scripts/demo_web_app.py`, `web_demo/*`, `tests/test_demo_web_app.py`) with `127.0.0.1` loopback + allowlist enforcement, Real VN Test threads (`data/processed/vn/` + `best_sbifagat_vn.pt` + CI 2-preset fallback), Sandbox *Path Contraction* with non-dismissible Heuristic banner, Tree <-> User-Graph toggle, hidden-by-default profanity indicator, and Tab 2 Macro Two-Camp viewer; Gate 2 live browser QA passed (`0` console errors) | — | **Done** |
| **P6** | Vietnamese Research Report & Defense Kit | Authored full 4-chapter Vietnamese Research Report (`reports/bao_cao_de_tai_vn.md`, compiled `40`-page `reports/bao_cao_de_tai_vn.pdf`) 100% synced with `paper/main.tex` & Demo screenshots + `reports/slide_outline_va_hoi_dap_bao_ve.md` (14 slides + 15 Q&As), cross-audited by Zero-Context Goldfish | — | **Done** |

---

## Execution Ownership (`Sprint 2` & `Sprint 3`)
- **VU HIEU KIEN (Architecture, PHEME, Core GNN, Paper & Demo Integration):**
  - GitHub governance, CI/CD, and release tagging.
  - PHEME Figshare ingestion (`member2_en`), unified 2D feature extraction, 5-seed benchmarks, LOO, Tier 2 Macro Signed Network, UPFD extension, efficiency benchmarks, `paper/main.tex` IEEEtran compilation, Live Demo MVP, and Vietnamese Research Report integration.
- **Vũ Ngọc Anh Thuyên (Vietnamese Corpus, Human Audit, Case Study, Docs/QA):**
  - Arctic Shift VN crawl (`--before-epoch`, `member1_vn`), blind `20`-pair pilot + `120`-pair + `30`-tree VN audit (`--lock-role` + `compute_kappa.py`), qualitative case study / `fig3_top5_trigger_comments.txt`, and Vietnamese report / `docs/` QA.
