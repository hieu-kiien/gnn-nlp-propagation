# SBi-FAGAT: Test & Verification Report

**Platform:** Windows (`win32`, CPU verification pass)  
**Overall Result:** **86 / 86 PASSED (100% Pass Rate, 0 failed, 0 skipped, 127.41s)**

---

## 1. Test Environment Specifications

| Component | Package / Tool | Declared Constraint | Verified Local Version |
| :--- | :--- | :--- | :--- |
| **Python** | CPython | `>=3.10` | `3.13.14` |
| **PyTorch** | `torch` | `>=2.3.0` | `2.15.0+cpu` |
| **PyG** | `torch-geometric` | `>=2.5.0` | `2.8.1` |
| **Transformers** | `transformers` | `>=4.38.0` (`>=5.8.0` Kaggle T4x2) | `5.23.0` |
| **Arrow / Tabular** | `pyarrow`, `pandas` | `>=14.0.0` / `>=2.1.0` | `25.0.1` / `3.1.1` |
| **Scientific** | `numpy`, `scipy`, `scikit-learn` | `>=1.24.0,<2.1.0` / `>=1.11.0` / `>=1.3.0` | `2.4.6` / `1.18.1` / `1.9.1` |
| **Test Runner** | `pytest` | `>=8.0.0` | `9.1.0` |

---

## 2. Pytest Suite Breakdown (`python -m pytest -v`)

| Test Module | Test Cases | Status | Focus |
| :--- | :---: | :---: | :--- |
| `tests/test_egm_fixes.py` | 7 | **PASSED** | EGM 6 hardening fixes (`F1–F6`), `node_backbone`, fair model parity, `min_total_replies=25` |
| `tests/test_kaggle_t4x2_v2_pipeline.py` | 5 | **PASSED** | QLoRA `Qwen3.8-27B` 5-class pipeline, `auto_loop`, `kaggle_parallel_runner`, notebooks |
| `tests/test_live_predictor_audit.py` | 5 | **PASSED** | Live predictor untrained guard, `--allow-untrained`, corrupt checkpoints, `--demo-offline` |
| `tests/test_nlp_and_features.py` | 5 | **PASSED** | Dual teachers, masked mean pooling, finite `.pt` bank extraction |
| `tests/test_no_leakage.py` | 7 | **PASSED** | Strict $K=20$ truncation, `future_only` $y_{\text{polar}}$, zero post-$t_K$ leakage |
| `tests/test_overfit_batch.py` | 6 | **PASSED** | Overfit convergence across `SBiFAGAT` and baselines |
| `tests/test_pheme_tree.py` | 3 | **PASSED** | PHEME tree parser & event-grouped inductive splitting |
| `tests/test_pipeline_and_viz.py` | 3 | **PASSED** | Vector figure generation (`.pdf`/`.svg`), UMAP, top-5 trigger export, reporting |
| `tests/test_round2_research_fixes.py` | 4 | **PASSED** | Loss balancing, statistical tests, Sarle's BC & Signed Assortativity |
| `tests/test_schema.py` | 31 | **PASSED** | `NodeRecord`/`EdgeRecord` Pydantic contracts, UTC enforcement, cycle/leakage rejection |
| `tests/test_sota_v2_upgrades.py` | 7 | **PASSED** | G-Eval logits, $\beta$-NLL, LOO `edge_source="teacher"`, EDA/Persistence & Efficiency LaTeX, CLI aliases |
| `tests/test_vn_tree.py` | 3 | **PASSED** | Arctic Shift VN tree builder & salted SHA-256 pseudonymization |
| **TOTAL** | **86** | **PASSED** | **86 passed, 0 failed, 0 skipped (`127.41s`)** |

---

## 3. Offline Smoke Checks (`exit code 0`)

1. `python -m scripts.demo_research_walkthrough` — Exited `0` (CPU walkthrough completed cleanly).
2. `python -m scripts.predict_live_thread --demo-offline` — Exited `0` (constructed demo cascade, flagged untrained check mode, output 6 dangerous replies and diagnostics).
3. `python -m scripts.kaggle_parallel_runner --role member1_vn --smoke-test` — Exited `0` (completed `distill`, `extract_features`, `nlp_features.pt`, and smoke bundle pipeline offline).
