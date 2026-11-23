# SBi-FAGAT: Test & Verification Report

**Platform:** Windows (`win32`, CPU verification pass) + GitHub Actions Ubuntu (`Python 3.11`)  
**Suite Scope:** 18 test modules in `tests/` — **128 passed, 0 failed** (`563.27s`, zero skipped — covering unified 2D architecture, multi-task KL-BCE optimization, zero data leakage, EGM Round-1/Round-2 remediations, and **v3 EGM Overhaul**: PII/doxxing/spam filtering, linear `y_polar`, ReZero `tanh(w_p)` centered Persistence-Residual anchor, masked single-camp Camp Dipole, Smooth-L1 `beta=0.05` + validation-only comparator selection, `temporal`/`embedding_cluster` splits with `edge_source="teacher"`, binary `clean`/`uncivil` Gate 1 audit, `--lock-role` UI isolation, and non-overwriting multi-corpus LaTeX assembly)

---

## 1. Test Environment Specifications

| Component | Package / Tool | Declared Constraint | Verified Local Version |
| :--- | :--- | :--- | :--- |
| **Python** | CPython | `>=3.10` | `3.13.14` |
| **PyTorch** | `torch` | `>=2.3.0` | `2.15.0+cpu` |
| **PyG** | `torch-geometric` | `>=2.5.0` | `2.8.1` |
| **Transformers** | `transformers` | `>=4.38.0,<5.0.0` | `5.23.0` (local) / `4.x` (Kaggle & CI) |
| **Arrow / Tabular** | `pyarrow`, `pandas` | `>=14.0.0` / `>=2.1.0` | `25.0.1` / `3.1.1` |
| **Scientific** | `numpy`, `scipy`, `scikit-learn` | `>=1.24.0,<3.0.0` / `>=1.11.0` / `>=1.3.0` | `2.4.6` / `1.18.1` / `1.9.1` |
| **Test Runner** | `pytest` | `>=8.0.0` | `9.1.0` |

---

## 2. Pytest Suite Coverage (`python -m pytest -v`)

| Test Module | Focus |
| :--- | :--- |
| `tests/test_egm_audit_remediation.py` | Full EGM audit regressions (`B1–B10`, `M1–M8`): `min_total_replies` LOO teacher coverage, `edge_source="auto"` Student consistency, $O(N)$ JSONL teacher delta resume, crawl `.state.json` + salted PHEME, LaTeX `_` escaping & `table*` + RFC JSON, NLI double-contradiction neutral + 50-tree VN blind audit, direction-aware stars + exact $F=4$ LOO permutation + topic-cluster bootstrap, continuous `y_polar` + fixed future $K$-window, Camp Dipole/Dir-GNN/Dual-Channel ablations + `BiEdgeGATv2` + `TextMLP` topology mask + `assemble_paper_artifacts` + UPFD fault isolation |
| `tests/test_egm_fixes.py` | EGM hardening fixes (`F1–F6`), `node_backbone`, fair model parity, `min_total_replies=25` |
| `tests/test_human_audit_ui.py` | Localhost blind audit Web UI: `127.0.0.1` binding, `403` 5-CSV allowlist, role blinding, atomic UTF-8 BOM saves + `.bak`, `compute_kappa`/`audit_vn_50_trees` strict-adjudication compatibility, native `<dialog>` & IME-safe UI invariants |
| `tests/test_kaggle_t4x2_v2_pipeline.py` | Unified 2D discourse labels, NLI verbalizer simplex, `DualHeadCrossEncoder`, `auto_loop`, smoke runner & notebooks |
| `tests/test_live_predictor_audit.py` | Live predictor untrained guard, `--allow-untrained`, corrupt checkpoints, `--demo-offline` |
| `tests/test_mechanisms_and_optimizations.py` | Lossless NLP length-bucketing/DP wrappers, mechanistic counterfactual edge/hop/polarity interventions, LaTeX & underscore validation |
| `tests/test_nlp_and_features.py` | Dual teachers, *Ancestor-Conditioned SpanPool*, finite `.pt` bank extraction |
| `tests/test_no_leakage.py` | Strict $K=20$ truncation, `future_only` $y_{\text{polar}}$, zero post-$t_K$ leakage |
| `tests/test_overfit_20.py` | Step 1 20-cascade overfit, 50/50 dipole cancellation, `TwoStreamNodeStem` topology mask, fair `hidden_dim=64` |
| `tests/test_overfit_batch.py` | Overfit convergence across `SBiFAGAT` and baselines |
| `tests/test_pheme_tree.py` | PHEME tree parser & event-grouped inductive splitting |
| `tests/test_pipeline_and_viz.py` | Vector figure generation (`.pdf`/`.svg`), UMAP, top-5 trigger export, reporting |
| `tests/test_round2_research_fixes.py` | Loss balancing, statistical tests, Sarle's BC & Signed Assortativity |
| `tests/test_schema.py` | `NodeRecord`/`EdgeRecord` Pydantic contracts, UTC enforcement, cycle/leakage rejection |
| `tests/test_sota_v2_upgrades.py` | LOO `edge_source="teacher"`, EDA/Persistence & Efficiency LaTeX, CLI aliases, fair runner config |
| `tests/test_step3_training_and_audit_fixes.py` | KL-BCE zero min & tail gradient, EMA/best-val seed selection, `EarlyStopping` NaN/Inf guard, sealed key bundles, canonical split locks, dynamic LaTeX labels, NLI reply preservation |
| `tests/test_v3_egm_overhaul.py` | v3 EGM overhaul: PII/doxxing/LMGX spam redaction, linear `y_polar`, `early_y_polar` & chronological/embedding splits, ReZero `tanh(w_p)` centered Persistence anchor + `persist_ep_logit_ref` + consistent mechanism perturbation clone, `predict_uncertainty` parity & validation-only comparator selection, binary `clean`/`uncivil` Gate 1 (`ROC-AUC`/`PR-AUC`), `--lock-role` 403 UI isolation, Kendall $\tau_b$/AUROC $@\ge 2$ thread audit, multi-corpus LaTeX merge |
| `tests/test_vn_tree.py` | Arctic Shift VN tree builder & salted SHA-256 pseudonymization |

---

## 3. Offline CLI & Smoke Verification (`exit code 0`)

1. `python -u -m scripts.demo_research_walkthrough` — Exited `0` (CPU walkthrough comparing `StandardGCN` vs. `SBiFAGAT`).
2. `python -u -m scripts.predict_live_thread --demo-offline` — Exited `0` (verified live safety guard + offline diagnostics).
3. `python -u -m scripts.benchmark_efficiency --use-cascade-stats --hidden-dim 64 --edge-dim 2` — Exited `0` (`SBiFAGAT = 137,086` params, `1.1843 ms/cascade`, `155.79 ms/batch`, `0.03 MB` peak alloc RAM on CPU).
4. `python -u -m scripts.kaggle_parallel_runner --role member1_vn --smoke-test` — Exited `0` (verified temp-isolated 2D VN distillation + sealed machine-key separation).
5. `python -u -m scripts.kaggle_parallel_runner --role member2_en --smoke-test` — Exited `0` (verified temp-isolated 2D EN distillation + LOO/UPFD path).
